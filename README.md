# Model Service

`model-service` is the internal Python engine behind the project. It handles market-data refreshes, model training, signal generation, live engine endpoints, chart data, assistant workflows, trader portfolio state, and optional RAG storage.

This service is intended to sit behind `backend-ts`. The frontend should talk to the backend, not directly to the Python service.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current layer map, legacy shims, and the non-invasive cleanup approach.

## Quick Start

Run these commands from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r model-service/requirements.txt
Copy-Item model-service/.env.example model-service/.env
python model-service
```

Canonical startup command:

```powershell
python model-service
```

Equivalent npm shortcut:

```powershell
npm run model-service:start
```

Legacy compatibility command:

```powershell
python model-service/scripts/runAiEngine.py
```

## What `python model-service` Starts

The unified startup command runs the model service in one process. By default it:

- loads settings from `model-service/.env`
- performs an initial signal-generation cycle
- starts the scheduled background signal refresh loop
- serves the internal FastAPI engine

Default bind settings:

- host: `127.0.0.1`
- port: `8100`

Override them with:

- `AI_ENGINE_HOST`
- `AI_ENGINE_PORT`

## Verify the Service

After startup, these endpoints are the quickest checks:

- engine health: `http://127.0.0.1:8100/api/health`
- engine landing payload: `http://127.0.0.1:8100/api/landing`
- model summary: `http://127.0.0.1:8100/api/model`
- cached signal overview: `http://127.0.0.1:8100/api/overview`
- live signal overview: `http://127.0.0.1:8100/api/live/overview`

PowerShell example:

```powershell
Invoke-RestMethod http://127.0.0.1:8100/api/health
```

If `AI_ENGINE_INTERNAL_API_KEY` is set, include the same value in the `x-ai-engine-key` request header when calling the Python service directly.

## Main Workflows

Run these from the repo root.

### Service and APIs

- Start the unified model service: `python model-service`
- Start the cached signal API only: `python model-service/scripts/runSignalApi.py`

### Data Refresh

- Refresh the current configured market-data batch: `python model-service/scripts/refreshMarketData.py`
- Refresh all remaining market-data batches: `python model-service/scripts/refreshAllMarketData.py`
- Refresh CoinMarketCal events for the tracked universe: `python model-service/scripts/refreshMarketEvents.py`

### Model Training and Evaluation

- Train the current signal model: `python model-service/scripts/trainModel.py`
- Train the standalone regime model: `python model-service/scripts/trainRegimeModel.py`
- Compare configured model classes on the same split: `python model-service/scripts/compareModels.py`
- Run a trading backtest: `python model-service/scripts/runBacktest.py`
- Run walk-forward validation: `python model-service/scripts/runWalkForwardValidation.py`
- Tune labeling and confidence settings: `python model-service/scripts/tuneSignalParameters.py`

#### Model Families and When to Use Them

Model selection is configuration-driven. The trading engine stays authoritative; model families only change inference.

- Baseline (current default): scikit-learn classifiers tuned for stable, explainable tabular signals.
- LightGBM classifier: primary challenger for stronger non-linear tabular performance.
- LightGBM ranker: ranking-only mode for opportunity ordering, not default signal inference.
- XGBoost classifier: alternative ensemble benchmark against LightGBM.
- River online model: incremental updates, watchlist progression scoring, and drift-aware experimentation.
- TFT (PyTorch Forecasting): experimental sequence modeling; only enable when explicitly configured.

Config switches to focus on:

- `SIGNAL_MODEL_FAMILY` (e.g., `baseline_current`, `lightgbm`, `xgboost`, `river`, `tft`)
- `SIGNAL_MODEL_VARIANT` (e.g., `classifier`, `ranker`, `online_scorer`, `sequence`)
- `ENABLE_TFT_EXPERIMENTS` to gate TFT usage

### Signal Publishing

- Generate signal files from the latest data and model artifact: `python model-service/scripts/generateSignals.py`
- Refresh data, retrain, and publish a fresh snapshot in one command: `python model-service/scripts/runProductionCycle.py`

### Tests

- Run the test suite: `pytest model-service/tests`

## Directory Layout

- `src/crypto_signal_ml/`: package source
- `src/crypto_signal_ml/service/`: service/API facade for monitor and FastAPI entrypoints
- `src/crypto_signal_ml/application/`: orchestration facade for training, refresh, validation, and publishing flows
- `src/crypto_signal_ml/ml/`: ML facade for data loading, features, labels, models, and pipeline builders
- `src/crypto_signal_ml/chat/`: assistant/session facade
- `src/crypto_signal_ml/retrieval/`: retrieval and knowledge-store facade
- `src/crypto_signal_ml/tools/`: stable structured tool layer for LLM-callable signal, trader, model, and retrieval access
- `src/crypto_signal_ml/llm/`: provider-specific adapters that consume tool schemas instead of trading logic
- `src/crypto_signal_ml/memory/`: session-memory facade
- `src/crypto_signal_ml/trading/`: trading, signal, portfolio, policy, and trader-brain modules
- `src/crypto_signal_ml/storage/`: storage adapters
- `scripts/`: command-line entrypoints
- `tests/`: pytest coverage
- `data/raw/`: downloaded market data and enrichment caches
- `data/processed/`: prepared datasets
- `models/`: trained model artifacts
- `outputs/`: published snapshots, reports, and local SQLite databases

## Important Outputs

Common files written by this service:

- `data/raw/marketPrices.csv`: raw market data
- `data/raw/coinMarketCapContext.csv`: cached CoinMarketCap context
- `data/raw/coinMarketCapMarketIntelligence.csv`: cached market-intelligence snapshot
- `data/raw/coinMarketCalEvents.csv`: cached market events
- `data/processed/marketFeaturesAndLabels.csv`: prepared training dataset
- `models/*.pkl`: trained model artifacts
- `models/*.metadata.json`: model metadata sidecars
- `outputs/frontendSignalSnapshot.json`: frontend-ready signal snapshot
- `outputs/latestSignal.json`: current primary signal
- `outputs/latestSignals.json`: latest signal set
- `outputs/actionableSignals.json`: filtered actionable signals
- `outputs/*.sqlite3`: local assistant, portfolio, and RAG stores when PostgreSQL is not configured

## Storage

If `DATABASE_URL` is set, the service uses PostgreSQL-backed stores where supported.

Per-store overrides:

- `ASSISTANT_DATABASE_URL`
- `PORTFOLIO_DATABASE_URL`
- `RAG_DATABASE_URL`

If those are not set, the service falls back to local SQLite files under `model-service/outputs/`.

## Environment Variables You Will Likely Touch First

Start with `model-service/.env.example`, then adjust these values as needed:

- `COINMARKETCAP_API_KEY`: required for CoinMarketCap-backed refresh flows
- `MARKET_DATA_SOURCE`: market-data provider selection
- `AI_ENGINE_INTERNAL_API_KEY`: shared secret used by the backend when the engine is protected
- `LIVE_FETCH_ALL_QUOTE_PRODUCTS`: whether live workflows use the full quote-currency universe
- `LIVE_MAX_PRODUCTS`: cap for live product coverage
- `LIVE_PRODUCT_IDS`: explicit live symbols when not using full-universe mode
- `MARKET_PRODUCT_BATCH_ROTATION_ENABLED`: rotate market batches between refreshes
- `SIGNAL_MONITOR_REFRESH_INTERVAL_SECONDS`: background refresh cadence
- `SIGNAL_TRACK_GENERATED_TRADES`: automatically track generated trades
- `SIGNAL_GENERATED_TRADE_STATUS`: initial status for generated trades
- `DATABASE_URL`: shared database connection string

Restart the Python process after editing `.env` so a fresh `TrainingConfig` is loaded.

## Notes for Developers

- The canonical service entrypoint is `model-service/main.py`, exposed through `python model-service`.
- New internal imports should prefer `crypto_signal_ml.service`, `crypto_signal_ml.application`, `crypto_signal_ml.ml`, `crypto_signal_ml.chat`, and `crypto_signal_ml.retrieval` where those facades fit.
- LLM-facing integrations should prefer `crypto_signal_ml.tools` as the structured contract and keep provider code isolated in `crypto_signal_ml.llm`.
- `scripts/runAiEngine.py` is kept as a compatibility wrapper.
- Some legacy top-level modules still exist as shims that re-export newer package modules from `src/crypto_signal_ml/trading/` and related subpackages.

## Troubleshooting

- If `python model-service` starts but `/api/health` does not return `ok`, check that the service is listening on `AI_ENGINE_HOST` and `AI_ENGINE_PORT`.
- If requests to the Python service return `403`, make sure the caller is sending the `x-ai-engine-key` header with the same value configured in `AI_ENGINE_INTERNAL_API_KEY`.
- If signal generation starts slowly on a fresh setup, the service may be refreshing market data and building the first published snapshot before the API becomes fully useful.
- If PostgreSQL is not configured, inspect the SQLite files under `outputs/` first when debugging assistant, portfolio, or RAG persistence.#   c r y p t o T r a d i n g - m o d e l  
 #   c r y p t o T r a d i n g - m o d e l  
 
#   c r y p t o T r a d i n g - m o d e l  
 
#   c r y p t o T r a d i n g - m o d e l  
 
#   c r y p t o T r a d i n g - m o d e l  
 