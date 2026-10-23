"""Runs the committed model on the bundled PneumoniaMNIST test images."""
import csv

import numpy as np
import pytest
from PIL import Image

from triage.audit import AuditLog
from triage.config import DISCLAIMER, METRICS_FILE, MODEL_WEIGHTS, SAMPLES
from triage.model import Classifier, image_to_array, make_prediction
from triage.pipeline import load_threshold, triage_case


@pytest.fixture(scope="module")
def clf():
    assert MODEL_WEIGHTS.exists(), "artifacts/model.pt missing; run python -m triage.train"
    return Classifier()


@pytest.fixture(scope="module")
def samples():
    with (SAMPLES / "labels.csv").open() as f:
        return list(csv.DictReader(f))


def test_prediction_fields():
    p = make_prediction(0.2)
    assert p.label == "normal" and p.confidence == pytest.approx(0.8)


def test_probabilities_in_range_and_deterministic(clf, samples):
    imgs = np.stack([image_to_array(Image.open(SAMPLES / s["file"])) for s in samples])
    a, b = clf.probabilities(imgs), clf.probabilities(imgs)
    assert a.shape == (len(samples),)
    assert np.all((a >= 0) & (a <= 1))
    np.testing.assert_allclose(a, b)


def test_accuracy_on_bundled_samples(clf, samples):
    imgs = np.stack([image_to_array(Image.open(SAMPLES / s["file"])) for s in samples])
    truth = np.array([s["true_label"] == "pneumonia" for s in samples])
    acc = ((clf.probabilities(imgs) >= 0.5) == truth).mean()
    assert acc >= 0.75, acc


def test_upload_of_other_size_and_mode(clf):
    rgb = Image.new("RGB", (300, 200), (120, 120, 120))
    assert 0.0 <= clf.predict(rgb).pneumonia_probability <= 1.0


def test_threshold_picked_on_calibration_half():
    import json

    r = json.loads(METRICS_FILE.read_text())["routing"]
    assert 0.5 <= load_threshold() <= 1.01
    assert "calibration half" in r["threshold_source"]
    # Reported routing must come from the half that did not pick the threshold.
    held = r["evaluation_half"]["auto_triage"]
    assert held["cases"] > 0 and held["missed_pneumonia"] == 0


def test_triage_case_end_to_end(clf, samples, tmp_path):
    audit = AuditLog(tmp_path / "a.sqlite3")
    s = samples[0]
    result = triage_case(Image.open(SAMPLES / s["file"]), clf, audit, load_threshold(), source=s["file"])
    assert result.disclaimer == DISCLAIMER
    assert result.summary.text.endswith(DISCLAIMER)
    (row,) = audit.events(result.case_id)
    assert row["route"] == result.route
    assert row["model_version"] == clf.version
