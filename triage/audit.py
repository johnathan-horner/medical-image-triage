"""Append-only audit log in SQLite.

Every prediction and every reviewer action is a new row; nothing is updated in place.
The current state of a case is its latest event.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from triage.config import AUDIT_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         TEXT NOT NULL,
    event_type      TEXT NOT NULL CHECK (event_type IN ('prediction', 'review')),
    model_version   TEXT NOT NULL,
    score           REAL NOT NULL,      -- model probability of pneumonia
    confidence      REAL NOT NULL,
    predicted_label TEXT NOT NULL,
    route           TEXT NOT NULL,
    threshold       REAL NOT NULL,
    reviewer_action TEXT NOT NULL,      -- 'none', 'pending', 'confirmed', 'overridden'
    reviewer_label  TEXT,
    reviewer        TEXT,
    source          TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_events(case_id);
"""

COLUMNS = ("event_id", "case_id", "event_type", "model_version", "score", "confidence", "predicted_label",
           "route", "threshold", "reviewer_action", "reviewer_label", "reviewer", "source", "created_at")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_case_id() -> str:
    return "case-" + uuid.uuid4().hex[:12]


class AuditLog:
    def __init__(self, path: Path | str = AUDIT_DB):
        self.path = str(path)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _insert(self, row: dict) -> None:
        cols = [c for c in COLUMNS if c != "event_id"]
        with self._connect() as conn:
            conn.execute(f"INSERT INTO audit_events ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                         [row.get(c) for c in cols])

    def record_prediction(self, case_id: str, model_version: str, score: float, confidence: float,
                          predicted_label: str, route: str, threshold: float, source: str = "") -> None:
        from triage.config import ROUTE_REVIEW

        self._insert({
            "case_id": case_id, "event_type": "prediction", "model_version": model_version,
            "score": score, "confidence": confidence, "predicted_label": predicted_label,
            "route": route, "threshold": threshold,
            "reviewer_action": "pending" if route == ROUTE_REVIEW else "none",
            "source": source, "created_at": _now(),
        })

    def record_review(self, case_id: str, reviewer: str, reviewer_label: str) -> str:
        """Log a person's decision on a queued case. Returns 'confirmed' or 'overridden'."""
        latest = self.latest(case_id)
        if latest is None:
            raise KeyError(f"unknown case {case_id}")
        action = "confirmed" if reviewer_label == latest["predicted_label"] else "overridden"
        self._insert({**latest, "event_type": "review", "reviewer_action": action,
                      "reviewer_label": reviewer_label, "reviewer": reviewer, "created_at": _now()})
        return action

    def events(self, case_id: str | None = None) -> list[dict]:
        sql = f"SELECT {', '.join(COLUMNS)} FROM audit_events"
        args: tuple = ()
        if case_id:
            sql += " WHERE case_id = ?"
            args = (case_id,)
        with self._connect() as conn:
            return [dict(zip(COLUMNS, r)) for r in conn.execute(sql + " ORDER BY event_id", args)]

    def latest(self, case_id: str) -> dict | None:
        rows = self.events(case_id)
        return rows[-1] if rows else None

    def review_queue(self) -> list[dict]:
        """Cases whose latest event is still waiting for a person."""
        sql = f"""
            SELECT {', '.join('e.' + c for c in COLUMNS)} FROM audit_events e
            JOIN (SELECT case_id, MAX(event_id) AS last_id FROM audit_events GROUP BY case_id) l
              ON e.event_id = l.last_id
            WHERE e.reviewer_action = 'pending' ORDER BY e.event_id
        """
        with self._connect() as conn:
            return [dict(zip(COLUMNS, r)) for r in conn.execute(sql)]
