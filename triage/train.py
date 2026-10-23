"""Fine-tune ResNet18 on PneumoniaMNIST on CPU, then evaluate and save artifacts.

Usage:  python -m triage.train --epochs 6
Writes artifacts/model.pt, artifacts/metrics.json and artifacts/confusion_matrix.png.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch import nn

from triage.config import ARTIFACTS, IMAGE_SIZE, METRICS_FILE, MODEL_VERSION, MODEL_WEIGHTS
from triage.data import load_split
from triage.model import Classifier, build_model, to_tensor
from triage.routing import pick_threshold, routing_report, stratified_halves


def augment(x: torch.Tensor) -> torch.Tensor:
    """Small random shifts and horizontal flips; enough for 64x64 X-rays."""
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    dx, dy = np.random.randint(-4, 5, size=2)
    return torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))


def train(epochs: int, batch_size: int, lr: float, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")

    model = build_model(pretrained=True)
    # Weight the minority class (normal) so the model does not lean on the 3:1 imbalance.
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_auc, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(x_train))
        start, total = time.time(), 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            xb = augment(to_tensor(x_train[idx]))
            yb = torch.from_numpy(y_train[idx]).float()
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(to_tensor(x_val)).squeeze(1)).numpy()
        val_auc = roc_auc_score(y_val, val_prob)
        val_acc = accuracy_score(y_val, val_prob >= 0.5)
        history.append({"epoch": epoch, "train_loss": round(total / len(order), 4),
                        "val_auc": round(val_auc, 4), "val_accuracy": round(val_acc, 4),
                        "seconds": round(time.time() - start, 1)})
        print(history[-1], flush=True)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    ARTIFACTS.mkdir(exist_ok=True)
    # Half precision halves the file size; inference casts back to float32.
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}, MODEL_WEIGHTS)
    return {"epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed, "history": history}


def evaluate(target_accuracy: float) -> dict:
    clf = Classifier()
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")
    val_prob = clf.probabilities(x_val)
    test_prob = clf.probabilities(x_test)
    test_pred = (test_prob >= 0.5).astype(int)

    # PneumoniaMNIST's validation split is easier than its test split, so a threshold tuned on
    # validation auto-triages almost everything. Instead the test split is cut into two stratified
    # halves: the threshold is picked on the calibration half and routing is reported on the other.
    calib, held = stratified_halves(y_test, seed=0)
    threshold = pick_threshold(test_prob[calib], y_test[calib], target_accuracy=target_accuracy)
    val_threshold = pick_threshold(val_prob, y_val, target_accuracy=target_accuracy)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    return {
        "test_set": {"name": "PneumoniaMNIST test split", "cases": int(len(y_test)),
                     "normal": int((y_test == 0).sum()), "pneumonia": int((y_test == 1).sum())},
        "test_metrics": {
            "accuracy": round(accuracy_score(y_test, test_pred), 4),
            "precision_pneumonia": round(precision_score(y_test, test_pred), 4),
            "recall_pneumonia": round(recall_score(y_test, test_pred), 4),
            "f1_pneumonia": round(f1_score(y_test, test_pred), 4),
            "specificity": round(cm[0, 0] / cm[0].sum(), 4),
            "roc_auc": round(roc_auc_score(y_test, test_prob), 4),
        },
        "confusion_matrix": {"labels": ["normal", "pneumonia"], "rows_true_cols_pred": cm.tolist()},
        "routing": {
            "threshold": threshold,
            "target_accuracy": target_accuracy,
            "threshold_source": f"calibration half of the test split ({len(calib)} images, stratified, seed 0): "
                                f"lowest confidence threshold with >= {target_accuracy:.0%} accuracy on "
                                "auto-triaged cases",
            "calibration_half": routing_report(test_prob[calib], y_test[calib], threshold),
            "evaluation_half": routing_report(test_prob[held], y_test[held], threshold),
            "validation_reference": {
                "note": "threshold picked on the validation split instead, shown for comparison; "
                        "it auto-triages almost everything because validation is easier than test",
                "threshold": val_threshold,
                "evaluation_half": routing_report(test_prob[held], y_test[held], val_threshold),
            },
        },
    }


def save_confusion_png(cm: list[list[int]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=150)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black")
    ax.set_xticks([0, 1], ["normal", "pneumonia"])
    ax.set_yticks([0, 1], ["normal", "pneumonia"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "confusion_matrix.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-accuracy", type=float, default=0.98)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    started = time.time()
    training = None
    if not args.eval_only:
        training = train(args.epochs, args.batch_size, args.lr, args.seed)
    elif METRICS_FILE.exists():
        training = json.loads(METRICS_FILE.read_text()).get("training")

    results = evaluate(args.target_accuracy)
    results = {"model_version": MODEL_VERSION, "image_size": IMAGE_SIZE,
               "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "training": training, **results,
               "wall_clock_seconds": round(time.time() - started, 1)}
    METRICS_FILE.write_text(json.dumps(results, indent=2) + "\n")
    save_confusion_png(results["confusion_matrix"]["rows_true_cols_pred"])
    print(json.dumps({k: results[k] for k in ("test_metrics", "confusion_matrix", "routing")}, indent=2))


if __name__ == "__main__":
    main()
