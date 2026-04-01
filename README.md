# Model Service

This directory contains the Python ML and AI-engine codebase, separated from the frontend and NestJS backend.

Core folders:
- [`src/crypto_signal_ml/`](C:/Users/Tolul/Desktop/learning-python/model-service/src/crypto_signal_ml): package source
- [`scripts/`](C:/Users/Tolul/Desktop/learning-python/model-service/scripts): runnable entrypoints
- [`tests/`](C:/Users/Tolul/Desktop/learning-python/model-service/tests): pytest coverage
- [`data/`](C:/Users/Tolul/Desktop/learning-python/model-service/data): raw and processed market data
- [`models/`](C:/Users/Tolul/Desktop/learning-python/model-service/models): trained artifacts
- [`outputs/`](C:/Users/Tolul/Desktop/learning-python/model-service/outputs): generated snapshots and reports

Typical local setup from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r model-service/requirements.txt
python model-service/scripts/runAiEngine.py
```
