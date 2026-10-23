"""One case end to end: score, route, summarize, audit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from PIL import Image

from triage.audit import AuditLog, new_case_id
from triage.config import DISCLAIMER, METRICS_FILE
from triage.model import Classifier
from triage.routing import route
from triage.summary import Summary, model_payload, summarize


def load_threshold() -> float:
    return float(json.loads(METRICS_FILE.read_text())["routing"]["threshold"])


@dataclass
class CaseResult:
    case_id: str
    label: str
    pneumonia_probability: float
    confidence: float
    route: str
    threshold: float
    model_version: str
    summary: Summary
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


def triage_case(image: Image.Image, classifier: Classifier, audit: AuditLog, threshold: float,
                use_llm: bool = False, source: str = "", llm_client=None) -> CaseResult:
    pred = classifier.predict(image)
    lane = route(pred.confidence, threshold)
    case_id = new_case_id()
    audit.record_prediction(case_id, pred.model_version, pred.pneumonia_probability, pred.confidence,
                            pred.label, lane, threshold, source)
    payload = model_payload(pred.label, pred.pneumonia_probability, pred.confidence, lane, threshold,
                            pred.model_version)
    return CaseResult(case_id, pred.label, pred.pneumonia_probability, pred.confidence, lane, threshold,
                      pred.model_version, summarize(payload, use_llm, llm_client))
