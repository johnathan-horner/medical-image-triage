"""Confidence-based routing: auto-triage vs. human review queue."""

from __future__ import annotations

import numpy as np

from triage.config import ROUTE_AUTO, ROUTE_REVIEW


def route(confidence: float, threshold: float) -> str:
    """Cases at or above the threshold go to auto-triage, the rest to a person."""
    return ROUTE_AUTO if confidence >= threshold else ROUTE_REVIEW


def confidence_of(prob: np.ndarray) -> np.ndarray:
    return np.maximum(prob, 1.0 - prob)


def pick_threshold(prob: np.ndarray, labels: np.ndarray, target_accuracy: float = 0.98,
                   min_auto_cases: int = 20) -> float:
    """Lowest confidence threshold whose auto-triaged cases reach the target accuracy.

    Candidate thresholds are the observed confidence values. Run this on calibration data
    that is separate from the data you report results on.
    Returns 1.01 (nothing auto-triaged) if no threshold reaches the target.
    """
    conf = confidence_of(prob)
    correct = (prob >= 0.5).astype(int) == labels
    for t in np.unique(np.round(conf, 4)):
        auto = conf >= t
        if auto.sum() < min_auto_cases:
            break
        if correct[auto].mean() >= target_accuracy:
            return round(float(t), 4)
    return 1.01


def stratified_halves(labels: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into two halves with the same class balance (calibration, evaluation)."""
    rng = np.random.default_rng(seed)
    first, second = [], []
    for cls in np.unique(labels):
        idx = rng.permutation(np.where(labels == cls)[0])
        first.append(idx[: len(idx) // 2])
        second.append(idx[len(idx) // 2:])
    return np.sort(np.concatenate(first)), np.sort(np.concatenate(second))


def routing_report(prob: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """How the threshold splits a labeled set and how accurate each lane is."""
    conf = confidence_of(prob)
    pred = (prob >= 0.5).astype(int)
    auto = conf >= threshold
    review = ~auto

    def lane(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {"cases": 0, "share": 0.0, "accuracy": None, "missed_pneumonia": 0}
        return {
            "cases": n,
            "share": round(n / len(prob), 4),
            "accuracy": round(float((pred[mask] == labels[mask]).mean()), 4),
            "missed_pneumonia": int(((pred == 0) & (labels == 1) & mask).sum()),
        }

    return {"threshold": threshold, "auto_triage": lane(auto), "human_review": lane(review)}
