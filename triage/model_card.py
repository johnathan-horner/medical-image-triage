"""Render MODEL_CARD.md from artifacts/metrics.json so the card never drifts from measured numbers.

Usage:  python -m triage.model_card
"""

from __future__ import annotations

import json

from triage.config import DISCLAIMER, METRICS_FILE, ROOT


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def render(m: dict) -> str:
    t, cm, r = m["test_metrics"], m["confusion_matrix"]["rows_true_cols_pred"], m["routing"]
    ts = m["test_set"]
    hist = m["training"]["history"] if m.get("training") else []
    best = max(hist, key=lambda h: h["val_auc"]) if hist else None
    lanes = []
    for name, key in (("test calibration half", "calibration_half"), ("test evaluation half", "evaluation_half")):
        for lane in ("auto_triage", "human_review"):
            d = r[key][lane]
            lanes.append(f"| {name} | {lane} | {d['cases']} | {pct(d['share'])} | {pct(d['accuracy'])} "
                         f"| {d['missed_pneumonia']} |")
    ref = r["validation_reference"]
    ref_auto = ref["evaluation_half"]["auto_triage"]
    lanes_md = "\n".join(lanes)
    training_md = ""
    if hist:
        training_md = (f"- Epochs: {len(hist)} (kept epoch {best['epoch']}, best validation ROC AUC "
                       f"{best['val_auc']:.3f})\n"
                       f"- CPU time per epoch: about {sum(h['seconds'] for h in hist) / len(hist) / 60:.0f} "
                       f"minutes on a 2-core Intel laptop\n")

    return f"""# Model card: {m['model_version']}

**{DISCLAIMER}** Research demo trained on a public dataset. Not validated for clinical use.

## Model

- Architecture: ResNet18 pretrained on ImageNet (torchvision `IMAGENET1K_V1`), final layer replaced
  with one output (probability of pneumonia), all layers fine-tuned.
- Input: one grayscale chest X-ray resized to {m['image_size']}x{m['image_size']}, repeated to 3 channels.
- Output: probability of pneumonia `p`. Label is pneumonia when `p >= 0.5`.
  Confidence is `max(p, 1 - p)`.
- Weights: `artifacts/model.pt` (half precision, about 22 MB).

## Training

- Data: PneumoniaMNIST (MedMNIST v2) official train split, 4,708 images, about 74% pneumonia.
- Loss: binary cross-entropy with the positive class down-weighted to offset the imbalance.
- Optimizer: AdamW, learning rate {m['training']['lr'] if m.get('training') else 'n/a'}, cosine schedule,
  batch size {m['training']['batch_size'] if m.get('training') else 'n/a'}, seed {m['training']['seed'] if m.get('training') else 'n/a'}.
- Augmentation: random horizontal flip and shifts of up to 4 pixels.
- Checkpoint: the epoch with the best ROC AUC on the official validation split (524 images).
{training_md}- Hardware: CPU only.

## Evaluation (held-out test split)

Test set: {ts['name']}, {ts['cases']} images ({ts['normal']} normal, {ts['pneumonia']} pneumonia).
The test split was not used for training or checkpoint choice. Half of it was used to pick the
routing threshold (see below); the classification metrics here use the fixed 0.5 cut-off.

| Metric | Value |
|---|---|
| Accuracy | {t['accuracy']:.3f} |
| Precision (pneumonia) | {t['precision_pneumonia']:.3f} |
| Recall (pneumonia) | {t['recall_pneumonia']:.3f} |
| Specificity (normal) | {t['specificity']:.3f} |
| F1 (pneumonia) | {t['f1_pneumonia']:.3f} |
| ROC AUC | {t['roc_auc']:.3f} |

Confusion matrix (rows are the true label, columns the prediction):

| | predicted normal | predicted pneumonia |
|---|---|---|
| **true normal** | {cm[0][0]} | {cm[0][1]} |
| **true pneumonia** | {cm[1][0]} | {cm[1][1]} |

## Routing threshold

- Threshold: **{r['threshold']:.3f}** on confidence.
- How it was picked: {r['threshold_source']}.
- Routing results are reported on the other half of the test split, which played no part in
  picking the threshold.
- Cases at or above the threshold go to auto-triage. Everything else goes to the human review queue.

| Data | Lane | Cases | Share | Accuracy | Missed pneumonia |
|---|---|---|---|---|---|
{lanes_md}

Why not the validation split: PneumoniaMNIST's validation split is easier than its test split. A threshold
picked on validation ({ref['threshold']:.3f}) would auto-triage {pct(ref_auto['share'])} of the evaluation
half at only {pct(ref_auto['accuracy'])} accuracy.

## Intended use

- Showing how a classifier, a confidence threshold, a human review queue and an audit log fit
  together in a triage workflow.
- Teaching and portfolio use.

## Not intended for

- Diagnosis, treatment decisions or any clinical use.
- Adult X-rays, other views, other conditions, or images from other hospitals. The data is pediatric
  chest X-rays from one center (Guangzhou Women and Children's Medical Center).

## Known limitations

- The validation and test splits come from different distributions in PneumoniaMNIST. Validation
  scores are higher than test scores, and a threshold tuned on validation is less reliable on test.
- 64x64 images remove fine detail a radiologist would use.
- Binary labels only (normal or pneumonia). Bacterial and viral pneumonia are merged.
- The model is overconfident on some wrong answers; see the missed pneumonia counts in the
  auto-triage lane.
- No calibration step, no subgroup analysis (the dataset has no demographic fields).

## Data license

PneumoniaMNIST is released under CC BY 4.0. Source images: Kermany et al., Cell 2018 (CC BY 4.0).
ImageNet-pretrained weights come from torchvision (BSD-3-Clause code).
"""


def main() -> None:
    (ROOT / "MODEL_CARD.md").write_text(render(json.loads(METRICS_FILE.read_text())))
    print("wrote MODEL_CARD.md")


if __name__ == "__main__":
    main()
