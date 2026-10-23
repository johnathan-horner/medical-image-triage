"""Shared settings for the triage pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
SAMPLES = ROOT / "samples"
DATA_CACHE = ROOT / ".data"
AUDIT_DB = ROOT / "audit_log.sqlite3"

MODEL_VERSION = "resnet18-pneumoniamnist64-v1"
MODEL_WEIGHTS = ARTIFACTS / "model.pt"
METRICS_FILE = ARTIFACTS / "metrics.json"

IMAGE_SIZE = 64
CLASS_NAMES = ("normal", "pneumonia")

ROUTE_AUTO = "auto_triage"
ROUTE_REVIEW = "human_review"

DISCLAIMER = "Decision support only, not a diagnosis."
