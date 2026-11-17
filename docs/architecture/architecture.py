"""Medical Image Triage: what runs today (live on Streamlit Community Cloud)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagrams import Diagram, Edge
from diagrams.aws.ml import Bedrock
from diagrams.aws.security import IAM
from diagrams.onprem.client import User

from style import EDGE, GRAPH, NODE, area, flow, icon, primary, soft

with Diagram("Medical Image Triage  |  Live architecture",
             filename=str(Path(__file__).with_name("architecture")), outformat="png", direction="TB",
             show=False, graph_attr=GRAPH, node_attr=NODE, edge_attr=EDGE):
    user = User("Reviewer\n(browser)")

    with area("Streamlit Community Cloud", "host"):
        app = icon("streamlit", "Streamlit app\ntriage, review queue,\naudit log, evaluation")

        with area("Model"):
            model = icon("pytorch", "ResNet18 classifier\nPyTorch on CPU")
            router = icon("route-ink", "Confidence router\nthreshold 0.990")

        with area("Routing lanes"):
            auto = icon("list-checks-ink", "Auto-triage\n77% of test cases")
            queue = icon("user-check-ink", "Human review queue\nconfirm or override")

        with area("LLM summary"):
            summary = icon("file-text-ink", "Summary writer\nnumbers only, no image\n10 per session")
        with area("Audit"):
            audit = icon("sqlite", "Audit log (SQLite)\nappend-only, every event")

    with area("AWS Cloud  (us-east-1)", "aws"):
        iam = IAM("IAM user\nallowed: Haiku 4.5 only")
        bedrock = Bedrock("Amazon Bedrock\nClaude Haiku 4.5\nUS inference profile")

    user >> flow("X-ray, review decisions") >> app
    app >> flow("image") >> model
    model >> flow("score + confidence") >> router
    router >> flow("at or above") >> auto
    router >> flow("below") >> queue
    model >> flow("label, score, route") >> summary
    summary >> primary("InvokeModel\n(Anthropic SDK)") >> bedrock
    iam >> soft() >> bedrock
    auto >> flow("prediction") >> audit
    queue >> flow("prediction + review") >> audit
