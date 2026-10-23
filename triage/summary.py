"""Short plain-language summary of a model result.

The LLM sees only the model's structured output (label, probability, route, threshold),
never the image or any patient data. If Bedrock is not configured, a fixed template is used.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from triage.config import DISCLAIMER, ROOT, ROUTE_AUTO

# Claude Sonnet 4.6 through a US cross-region inference profile. Override with TRIAGE_BEDROCK_MODEL.
BEDROCK_MODEL = os.environ.get("TRIAGE_BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6")
BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")
COST_LOG = ROOT / "llm_cost_log.csv"
# USD per million tokens (Claude Sonnet 4.6 on-demand price). Used to estimate cost per call.
PRICE_IN, PRICE_OUT = 3.00, 15.00

SYSTEM_PROMPT = (
    "You write a two-sentence note for a chest X-ray worklist about an image classifier's output. "
    "Use only the JSON fields you are given. Do not describe image findings, do not name a diagnosis, "
    "and do not suggest treatment. State the model's leaning, its probability as a percentage, and "
    "where the case was routed and why (confidence versus threshold). Plain wording, no lists."
)


@dataclass(frozen=True)
class Summary:
    text: str
    source: str            # "bedrock" or "template"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def model_payload(label: str, pneumonia_probability: float, confidence: float, route: str,
                  threshold: float, model_version: str) -> dict:
    return {
        "model_version": model_version,
        "model_leaning": label,
        "pneumonia_probability": round(pneumonia_probability, 4),
        "confidence": round(confidence, 4),
        "routing_threshold": round(threshold, 4),
        "route": "auto-triage" if route == ROUTE_AUTO else "human review queue",
    }


def template_summary(payload: dict) -> str:
    auto = payload["route"] == "auto-triage"
    lane = "auto-triage" if auto else "the human review queue"
    cmp = "at or above" if auto else "below"
    return (f"The model leans {payload['model_leaning']} with a pneumonia probability of "
            f"{payload['pneumonia_probability']:.0%}. Its confidence of {payload['confidence']:.0%} is {cmp} "
            f"the {payload['routing_threshold']:.0%} threshold, so the case went to {lane}.")


def with_disclaimer(text: str) -> str:
    return f"{text.strip()} {DISCLAIMER}"


def check_aws_account() -> None:
    """If EXPECTED_AWS_ACCOUNT is set, refuse to call AWS from any other account."""
    expected = os.environ.get("EXPECTED_AWS_ACCOUNT")
    if not expected:
        return
    import boto3

    actual = boto3.client("sts", region_name=BEDROCK_REGION).get_caller_identity()["Account"]
    if actual != expected:
        raise RuntimeError(f"AWS account {actual} is not the expected account {expected}")


def bedrock_summary(payload: dict, client=None) -> Summary:
    if client is None:
        check_aws_account()
        from anthropic import AnthropicBedrock

        client = AnthropicBedrock(aws_region=BEDROCK_REGION)
    response = client.messages.create(
        model=BEDROCK_MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("model declined to summarize")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise RuntimeError("empty summary")
    usage = response.usage
    cost = (usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT) / 1_000_000
    log_cost(BEDROCK_MODEL, usage.input_tokens, usage.output_tokens, cost)
    return Summary(text, "bedrock", usage.input_tokens, usage.output_tokens, round(cost, 6))


def summarize(payload: dict, use_llm: bool, client=None) -> Summary:
    """Bedrock summary when asked for and available, otherwise the template. Always ends with the disclaimer."""
    if use_llm:
        try:
            s = bedrock_summary(payload, client)
            return Summary(with_disclaimer(s.text), s.source, s.input_tokens, s.output_tokens, s.cost_usd)
        except Exception as exc:  # noqa: BLE001 - any failure falls back to the template
            return Summary(with_disclaimer(template_summary(payload)) + f" (LLM unavailable: {type(exc).__name__})",
                           "template")
    return Summary(with_disclaimer(template_summary(payload)), "template")


def log_cost(model: str, input_tokens: int, output_tokens: int, cost: float, path: Path | None = None) -> None:
    path = path or COST_LOG
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "model", "input_tokens", "output_tokens", "estimated_cost_usd"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), model,
                    input_tokens, output_tokens, f"{cost:.6f}"])
