import pytest

from triage.audit import AuditLog
from triage.config import ROUTE_AUTO, ROUTE_REVIEW


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.sqlite3")


def test_prediction_fields_logged(audit):
    audit.record_prediction("case-1", "v1", 0.97, 0.97, "pneumonia", ROUTE_AUTO, 0.9, "sample:x.png")
    (row,) = audit.events("case-1")
    assert row["model_version"] == "v1"
    assert row["score"] == 0.97
    assert row["route"] == ROUTE_AUTO
    assert row["reviewer_action"] == "none"
    assert row["created_at"].endswith("+00:00")


def test_review_queue_and_actions_are_append_only(audit):
    audit.record_prediction("case-a", "v1", 0.6, 0.6, "pneumonia", ROUTE_REVIEW, 0.9)
    audit.record_prediction("case-b", "v1", 0.3, 0.7, "normal", ROUTE_REVIEW, 0.9)
    audit.record_prediction("case-c", "v1", 0.99, 0.99, "pneumonia", ROUTE_AUTO, 0.9)
    assert [c["case_id"] for c in audit.review_queue()] == ["case-a", "case-b"]

    assert audit.record_review("case-a", "dr-x", "pneumonia") == "confirmed"
    assert audit.record_review("case-b", "dr-x", "pneumonia") == "overridden"
    assert audit.review_queue() == []

    events = audit.events("case-b")
    assert [e["event_type"] for e in events] == ["prediction", "review"]
    assert events[0]["reviewer_action"] == "pending"   # original row untouched
    assert events[1]["reviewer_label"] == "pneumonia"
    assert events[1]["reviewer"] == "dr-x"


def test_review_unknown_case(audit):
    with pytest.raises(KeyError):
        audit.record_review("nope", "dr-x", "normal")
