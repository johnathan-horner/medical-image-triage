"""Smoke test of the Streamlit UI with Streamlit's own test harness."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from triage.config import DISCLAIMER


def test_app_runs_a_case(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIAGE_AUDIT_DB", str(tmp_path / "audit.sqlite3"))
    monkeypatch.delenv("TRIAGE_USE_BEDROCK", raising=False)
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert any(DISCLAIMER in e.value for e in at.error)
    at.button[0].click().run()
    assert not at.exception
    assert any(DISCLAIMER in m.value for m in list(at.markdown) + list(at.caption))
    assert at.subheader[0].value.startswith("Case case-")


def test_llm_toggle_is_off_without_aws_secrets(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("TRIAGE_USE_BEDROCK", raising=False)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120).run()
    assert not at.exception
    toggle = at.sidebar.toggle[0]
    assert toggle.value is False and toggle.disabled is True
    assert any("No AWS credentials" in c.value for c in at.sidebar.caption)
