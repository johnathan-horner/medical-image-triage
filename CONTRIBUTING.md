# Contributing

## Getting Started
1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Run locally: streamlit run app.py

## Architecture
See [docs/decisions/](docs/decisions/) for architecture decision records explaining why each service and pattern was chosen.

## MCP Server
The MCP server exposes tools, resources, and prompts for AI assistant integration. See README.md for the full list.

## API Documentation
Run the FastAPI server and visit /docs for interactive Swagger documentation.

## Testing
Push to main triggers GitHub Actions CI pipeline. See .github/workflows/ci.yml.

## Model Card
See MODEL_CARD.md for model documentation including training data, evaluation metrics, and compliance information.