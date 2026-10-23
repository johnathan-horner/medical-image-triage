# Contributing

1. Create a virtual environment with Python 3.11.
2. `pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu`
3. `ruff check triage tests app.py` and `pytest` must pass. CI runs both, plus a gitleaks scan.
4. If you retrain, run `python -m triage.model_card` so MODEL_CARD.md matches `artifacts/metrics.json`.
5. Never commit AWS credentials, `.env` files or `audit_log.sqlite3`.
