"""Streamlit demo: score a chest X-ray, route it, summarize it, and log every step."""

from __future__ import annotations

import csv
import json
import os

import pandas as pd
import streamlit as st
from PIL import Image

from triage.audit import AuditLog
from triage.config import ARTIFACTS, CLASS_NAMES, DISCLAIMER, METRICS_FILE, ROUTE_AUTO, SAMPLES
from triage.model import Classifier
from triage.pipeline import load_threshold, triage_case
from triage.summary import BEDROCK_MODEL, make_client

st.set_page_config(page_title="Chest X-ray Triage Demo", layout="wide")

MAX_LLM_CALLS_PER_SESSION = 10


def aws_secrets() -> dict | None:
    """The [aws] block from Streamlit secrets (access_key_id, secret_access_key, region), or None."""
    try:
        aws = st.secrets["aws"]
        if aws.get("access_key_id") and aws.get("secret_access_key"):
            return {"access_key_id": aws["access_key_id"], "secret_access_key": aws["secret_access_key"],
                    "region": aws.get("region") or "us-east-1"}
    except Exception:
        pass
    return None


def llm_available() -> bool:
    # On Streamlit Cloud only the [aws] secrets count; TRIAGE_USE_BEDROCK=1 allows the local default chain.
    return aws_secrets() is not None or os.environ.get("TRIAGE_USE_BEDROCK") == "1"


@st.cache_resource
def get_classifier() -> Classifier:
    return Classifier()


@st.cache_resource
def get_audit() -> AuditLog:
    return AuditLog(os.environ.get("TRIAGE_AUDIT_DB", "audit_log.sqlite3"))


@st.cache_data
def get_metrics() -> dict:
    return json.loads(METRICS_FILE.read_text())


@st.cache_data
def get_samples() -> list[dict]:
    with (SAMPLES / "labels.csv").open() as f:
        return list(csv.DictReader(f))


def route_badge(route: str) -> str:
    return "Auto-triage" if route == ROUTE_AUTO else "Human review queue"


def triage_tab(clf: Classifier, audit: AuditLog, threshold: float, use_llm: bool) -> None:
    samples = get_samples()
    left, right = st.columns([1, 2])
    with left:
        choice = st.radio("Image source", ["Bundled test image", "Upload"], horizontal=True)
        image, source, truth = None, "", None
        if choice == "Bundled test image":
            names = [s["file"] for s in samples]
            name = st.selectbox("PneumoniaMNIST test image", names)
            image = Image.open(SAMPLES / name)
            source = f"sample:{name}"
            truth = next(s["true_label"] for s in samples if s["file"] == name)
        else:
            upload = st.file_uploader("Chest X-ray (PNG or JPEG)", type=["png", "jpg", "jpeg"])
            if upload:
                image = Image.open(upload)
                source = f"upload:{upload.name}"
        if image is not None:
            st.image(image.convert("L").resize((256, 256)), caption=source, width=256)
            if truth:
                st.caption(f"Dataset label (hidden from the model): {truth}")
        run = st.button("Run triage", type="primary", disabled=image is None)

    with right:
        if run and image is not None:
            used = st.session_state.get("llm_calls", 0)
            capped = use_llm and used >= MAX_LLM_CALLS_PER_SESSION
            st.session_state["llm_note"] = (f"Session cap reached ({MAX_LLM_CALLS_PER_SESSION} LLM summaries per "
                                            "session), so the template wrote this summary." if capped else None)
            llm = use_llm and not capped
            with st.spinner("Scoring"):
                client = None
                if llm:
                    st.session_state["llm_calls"] = used + 1
                    try:
                        client = make_client(aws_secrets())
                    except Exception as exc:  # noqa: BLE001 - fall back to the template, say why
                        llm = False
                        st.session_state["llm_note"] = f"LLM unavailable ({type(exc).__name__}); template summary."
                st.session_state["last_result"] = triage_case(image, clf, audit, threshold, llm, source,
                                                              llm_client=client)
            st.rerun()  # refresh the sidebar queue count
        result = st.session_state.get("last_result")
        if result is None:
            st.info("Pick an image and press Run triage.")
            return
        st.subheader(f"Case {result.case_id}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model leaning", result.label)
        c2.metric("Pneumonia probability", f"{result.pneumonia_probability:.1%}")
        c3.metric("Confidence", f"{result.confidence:.1%}", help="max(p, 1 - p)")
        if result.route == ROUTE_AUTO:
            st.success(f"Route: {route_badge(result.route)} (confidence {result.confidence:.1%} "
                       f">= threshold {result.threshold:.1%})")
        else:
            st.warning(f"Route: {route_badge(result.route)} (confidence {result.confidence:.1%} "
                       f"< threshold {result.threshold:.1%})")
        st.markdown(f"**Summary** ({result.summary.source})")
        st.write(result.summary.text)
        if st.session_state.get("llm_note"):
            st.caption(st.session_state["llm_note"])
        if result.summary.source == "bedrock":
            st.caption(f"{result.summary.input_tokens} input / {result.summary.output_tokens} output tokens, "
                       f"about ${result.summary.cost_usd:.4f}")
        st.caption(f"Model version {result.model_version}. {DISCLAIMER}")


def review_tab(audit: AuditLog, reviewer: str) -> None:
    queue = audit.review_queue()
    st.write(f"{len(queue)} case(s) waiting for review.")
    for case in queue:
        with st.container(border=True):
            cols = st.columns([1, 3])
            src = case["source"] or ""
            if src.startswith("sample:"):
                cols[0].image(Image.open(SAMPLES / src.split(":", 1)[1]).resize((160, 160)))
            cols[1].markdown(f"**{case['case_id']}**  model leans **{case['predicted_label']}**, "
                             f"p(pneumonia) {case['score']:.1%}, confidence {case['confidence']:.1%}")
            label = cols[1].radio("Reviewer label", CLASS_NAMES, key=f"lbl-{case['case_id']}", horizontal=True,
                                  index=CLASS_NAMES.index(case["predicted_label"]))
            if cols[1].button("Submit review", key=f"btn-{case['case_id']}"):
                action = audit.record_review(case["case_id"], reviewer, label)
                st.toast(f"{case['case_id']}: {action}")
                st.rerun()


def audit_tab(audit: AuditLog) -> None:
    events = pd.DataFrame(audit.events())
    if events.empty:
        st.info("No cases yet.")
        return
    st.dataframe(events, width="stretch", hide_index=True)
    st.download_button("Download CSV", events.to_csv(index=False), "audit_log.csv", "text/csv")


def evaluation_tab(metrics: dict) -> None:
    m = metrics["test_metrics"]
    ts = metrics["test_set"]
    st.markdown(f"Held-out set: **{ts['name']}**, {ts['cases']} images "
                f"({ts['normal']} normal, {ts['pneumonia']} pneumonia). Threshold 0.5 on p(pneumonia).")
    cols = st.columns(5)
    for col, (k, label) in zip(cols, [("accuracy", "Accuracy"), ("precision_pneumonia", "Precision"),
                                      ("recall_pneumonia", "Recall"), ("specificity", "Specificity"),
                                      ("roc_auc", "ROC AUC")]):
        col.metric(label, f"{m[k]:.3f}")
    left, right = st.columns(2)
    left.image(str(ARTIFACTS / "confusion_matrix.png"), width=360)
    r = metrics["routing"]
    right.markdown(f"**Routing threshold: {r['threshold']:.3f}** on confidence  \n{r['threshold_source']}. "
                   "Routing results below are on the other half of the test split, which played no part "
                   "in picking the threshold.")
    rows = []
    for name, rep in (("test calibration half", r["calibration_half"]), ("test evaluation half", r["evaluation_half"])):
        for lane in ("auto_triage", "human_review"):
            d = rep[lane]
            rows.append({"data": name, "lane": lane, "cases": d["cases"], "share": d["share"],
                         "accuracy": d["accuracy"], "missed pneumonia": d["missed_pneumonia"]})
    right.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ref = r["validation_reference"]
    right.caption(f"For comparison, a threshold picked on the validation split ({ref['threshold']:.3f}) would "
                  f"auto-triage {ref['evaluation_half']['auto_triage']['share']:.0%} of the evaluation half at "
                  f"{ref['evaluation_half']['auto_triage']['accuracy']:.1%} accuracy.")
    if metrics.get("training"):
        st.markdown("**Training history (CPU)**")
        st.dataframe(pd.DataFrame(metrics["training"]["history"]), hide_index=True)


def main() -> None:
    st.title("Chest X-ray Triage Demo")
    st.error(DISCLAIMER + " Research demo on a public dataset. Not for clinical use.")
    clf, audit, metrics = get_classifier(), get_audit(), get_metrics()
    threshold = load_threshold()

    with st.sidebar:
        st.markdown(f"**Model** `{clf.version}`")
        st.markdown(f"**Routing threshold** {threshold:.3f}")
        st.markdown(f"**Waiting for review** {len(audit.review_queue())}")
        reviewer = st.text_input("Reviewer name", value="demo-reviewer")
        available = llm_available()
        use_llm = st.toggle("LLM summary via Amazon Bedrock", value=available, disabled=not available,
                            help="Off: a fixed template writes the summary. On: Claude Haiku 4.5 on Bedrock writes "
                                 "it from the model output only.")
        if available:
            st.caption(f"Bedrock model `{BEDROCK_MODEL}` · LLM summaries this session: "
                       f"{st.session_state.get('llm_calls', 0)}/{MAX_LLM_CALLS_PER_SESSION}")
        else:
            st.caption("No AWS credentials are set for this app, so a fixed template writes every summary.")

    tabs = st.tabs(["Triage a case", "Review queue", "Audit log", "Model evaluation"])
    with tabs[0]:
        triage_tab(clf, audit, threshold, use_llm)
    with tabs[1]:
        review_tab(audit, reviewer)
    with tabs[2]:
        audit_tab(audit)
    with tabs[3]:
        evaluation_tab(metrics)


main()
