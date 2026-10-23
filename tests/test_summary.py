import json
from types import SimpleNamespace

from triage.config import DISCLAIMER, ROUTE_AUTO, ROUTE_REVIEW
from triage import summary as S

PAYLOAD = S.model_payload("pneumonia", 0.93, 0.93, ROUTE_AUTO, 0.9, "v1")


class FakeClient:
    def __init__(self, text="The model leans pneumonia at 93%. It was auto-triaged.", stop="end_turn"):
        self.calls = []
        self.messages = self
        self._text, self._stop = text, stop

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason=self._stop,
                               content=[SimpleNamespace(type="text", text=self._text)],
                               usage=SimpleNamespace(input_tokens=200, output_tokens=50))


def test_template_always_has_disclaimer():
    for route in (ROUTE_AUTO, ROUTE_REVIEW):
        s = S.summarize(S.model_payload("normal", 0.2, 0.8, route, 0.9, "v1"), use_llm=False)
        assert s.source == "template"
        assert s.text.endswith(DISCLAIMER)


def test_llm_sees_only_model_output(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "COST_LOG", tmp_path / "cost.csv")
    client = FakeClient()
    s = S.summarize(PAYLOAD, use_llm=True, client=client)
    assert s.source == "bedrock"
    assert s.text.endswith(DISCLAIMER)
    sent = json.loads(client.calls[0]["messages"][0]["content"])
    assert set(sent) == {"model_version", "model_leaning", "pneumonia_probability", "confidence",
                         "routing_threshold", "route"}
    assert s.cost_usd == round((200 * S.PRICE_IN + 50 * S.PRICE_OUT) / 1e6, 6)


def test_llm_failure_falls_back_to_template():
    s = S.summarize(PAYLOAD, use_llm=True, client=FakeClient(stop="refusal"))
    assert s.source == "template"
    assert DISCLAIMER in s.text


def test_cost_log_written(tmp_path):
    path = tmp_path / "cost.csv"
    S.log_cost("m", 10, 5, 0.000140, path)
    S.log_cost("m", 10, 5, 0.000140, path)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("timestamp,model")
    assert len(lines) == 3


def test_default_model_is_haiku_45():
    assert S.BEDROCK_MODEL == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_make_client_uses_keys_from_secrets(monkeypatch):
    import anthropic

    seen = {}
    monkeypatch.delenv("EXPECTED_AWS_ACCOUNT", raising=False)
    monkeypatch.setattr(anthropic, "AnthropicBedrock", lambda **kw: seen.update(kw) or "client")
    aws = {"access_key_id": "FAKEKEYID000", "secret_access_key": "fake-secret", "region": "us-east-1"}
    assert S.make_client(aws) == "client"
    assert seen == {"aws_access_key": "FAKEKEYID000", "aws_secret_key": "fake-secret", "aws_region": "us-east-1"}
