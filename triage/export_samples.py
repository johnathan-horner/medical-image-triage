"""Copy a fixed, stratified slice of the PneumoniaMNIST test split into samples/ as PNGs.

These images ship with the repo (CC BY 4.0, attribution in samples/README.md) so the app
and CI can run real inference without downloading the dataset.
Usage:  python -m triage.export_samples --per-class 16
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
from PIL import Image

from triage.config import CLASS_NAMES, SAMPLES
from triage.data import load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    x, y = load_split("test")
    rng = np.random.default_rng(args.seed)
    SAMPLES.mkdir(exist_ok=True)
    rows = []
    for label in (0, 1):
        for idx in sorted(rng.choice(np.where(y == label)[0], size=args.per_class, replace=False)):
            name = f"test_{idx:04d}_{CLASS_NAMES[label]}.png"
            Image.fromarray(x[idx]).save(SAMPLES / name)
            rows.append({"file": name, "test_index": int(idx), "true_label": CLASS_NAMES[label]})
    with (SAMPLES / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "test_index", "true_label"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} samples to {SAMPLES}")


if __name__ == "__main__":
    main()
