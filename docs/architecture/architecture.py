"""Medical Image Triage: what runs today, traced from origin/main (e73b006).

Run from this directory:  python architecture.py
Writes architecture.png (300 DPI) and architecture-linkedin.png (1200x1500).
"""
from diagrams.aws.general import User
from diagrams.aws.ml import Bedrock
from diagrams.aws.security import IAMAWSSts

from archstyle import arch, aws_cloud, box, e, layer, linkedin, outside


def layer_r(label: str, margin: int = 30):
    """layer() with its label top-right, for layers whose arrows arrive from the upper left."""
    c = layer(label, margin)
    c.dot.graph_attr["labeljust"] = "r"
    return c

OUT = "architecture"

with arch("Medical Image Triage", "What runs today", OUT):
    user = User("Visitor / reviewer")

    with outside("Streamlit Community Cloud"):
        with layer("App (Streamlit, app.py)", margin=64):
            triage = box("Triage tab\n(sample or upload)")
            review = box("Review queue tab\n(confirm or override)")

        with layer("Image model (CPU)", margin=50):
            clf = box("ResNet18 classifier\n(PyTorch, 64x64 grayscale)")
            weights = box("model.pt\n(fp16 weights, 22 MB)")

        with layer("Routing", margin=44):
            metrics = box("metrics.json\n(threshold 0.9903)")
            router = box("Confidence router\nconf >= 0.9903: auto-triage\nelse: human review")

        with layer_r("Storage", margin=40):
            audit = box("SQLite audit log\n(append-only rows)")

        with layer("Summary", margin=34):
            secrets = box("Streamlit secrets\n[aws] block")
            summ = box("Summary builder\n6 JSON fields, no image\ncap 10 LLM calls / session")
            tmpl = box("Fixed template\n(fallback)")
            cost = box("llm_cost_log.csv\n(tokens, est. cost)")

    with aws_cloud():
        with layer("Generative AI", margin=48):
            llm = Bedrock("Amazon Bedrock\nClaude Haiku 4.5\n(us. inference profile)")
        with layer_r("Account check", margin=40):
            sts = IAMAWSSts("AWS STS")

    user >> e("HTTPS: run triage") >> triage
    user >> e("HTTPS: review a case") >> review
    triage >> e("image") >> clf
    clf >> e("loads weights (cached)") >> weights
    clf >> e("probability,\nconfidence") >> router
    metrics >> e("threshold") >> router
    router >> e("writes prediction row") >> audit
    review >> e("reads pending cases,\nwrites review row") >> audit
    router >> e("label, probability, route") >> summ
    summ >> e("reads [aws] keys") >> secrets
    summ >> e("InvokeModel (SigV4)") >> llm
    summ >> e("GetCallerIdentity\n(optional)", style="dashed") >> sts
    summ >> e("on error, cap or no keys") >> tmpl
    summ >> e("appends token + cost row") >> cost

linkedin(OUT)
