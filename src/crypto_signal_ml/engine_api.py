"""FastAPI engine service for model, live-signal, assistant, and RAG operations."""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .application import PublishedSignalViewService
from .chat import TradingAssistantService
from .charting import MarketChartService
from .config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    TrainingConfig,
    is_coinmarketcap_market_data_source,
)
from .frontend import SignalSnapshotStore
from .live import LiveSignalEngine
from .memory import ConversationSessionStore
from .modeling import BaseSignalModel, get_model_class
from .rag import RagKnowledgeStore
from .tools import ModelToolService, RetrievalToolService, SignalToolService, ToolRegistry, TraderToolService
from .trading.portfolio import TradingPortfolioStore
from .trading.signal_store import TradingSignalStore

INTERNAL_API_KEY_HEADER = "x-ai-engine-key"


def _timestamp_to_isoformat(timestamp_value: float) -> str:
    """Convert a filesystem timestamp into a readable UTC ISO string."""

    return datetime.fromtimestamp(timestamp_value, tz=timezone.utc).isoformat()


def _parse_iso_timestamp(timestamp_value: str | None) -> datetime | None:
    """Parse an ISO timestamp into a timezone-aware datetime when possible."""

    if not timestamp_value:
        return None

    normalized_timestamp = str(timestamp_value).replace("Z", "+00:00")
    try:
        parsed_timestamp = datetime.fromisoformat(normalized_timestamp)
    except ValueError:
        return None

    if parsed_timestamp.tzinfo is None:
        return parsed_timestamp.replace(tzinfo=timezone.utc)

    return parsed_timestamp.astimezone(timezone.utc)


def _age_hours_from_timestamp(timestamp_value: str | None) -> float | None:
    """Return the elapsed hours since the provided ISO timestamp."""

    parsed_timestamp = _parse_iso_timestamp(timestamp_value)
    if parsed_timestamp is None:
        return None

    return max((datetime.now(timezone.utc) - parsed_timestamp).total_seconds() / 3600, 0.0)


def _classify_freshness(age_hours: float | None, max_age_hours: float) -> str:
    """Classify an artifact as fresh or stale against the configured age budget."""

    if age_hours is None:
        return "unknown"

    return "fresh" if age_hours <= max_age_hours else "stale"


MODEL_RESEARCH_ROADMAP: dict[str, Any] = {
    "summary": (
        "The largest gains usually come from data handling and problem definition, "
        "not from swapping in a more complex algorithm."
    ),
    "tracks": [
        {
            "theme": "Labeling",
            "title": "Triple-Barrier Method",
            "problem": (
                "Fixed-horizon labels can mark a trade as successful even when the path "
                "would likely have hit a stop first."
            ),
            "proposal": (
                "Label each candle by the first barrier touched so the model learns "
                "tradable paths instead of abstract future returns."
            ),
            "details": [
                "Upper barrier: Take profit, such as +5%",
                "Lower barrier: Stop loss, such as -3%",
                "Vertical barrier: Max holding window, such as 2 days",
            ],
            "impact": "Higher-quality targets that better reflect real execution risk.",
        },
        {
            "theme": "Validation",
            "title": "Purged Cross-Validation",
            "problem": (
                "Neighboring financial samples are correlated, so standard folds can "
                "leak future information into validation."
            ),
            "proposal": (
                "Purge the training rows around each validation block so every score is "
                "measured on genuinely unseen market conditions."
            ),
            "details": [
                "Remove rows that overlap the test window",
                "Add a buffer before and after each validation fold",
                "Use the purged folds alongside the current walk-forward workflow",
            ],
            "impact": "More honest offline metrics and fewer live-market surprises.",
        },
        {
            "theme": "Features",
            "title": "Fractional Differentiation",
            "problem": (
                "Standard differencing helps stationarity but often destroys the "
                "long-memory structure that matters in market regimes."
            ),
            "proposal": (
                "Use fractional differentiation so the model keeps more macro context "
                "while still reducing non-stationarity."
            ),
            "details": [
                "Target a fractional order such as 0.4",
                "Keep more support, resistance, and regime memory",
                "Apply only where stationarity pressure is blocking model quality",
            ],
            "impact": "More informative features without fully discarding historical structure.",
        },
        {
            "theme": "Tuning",
            "title": "Optuna Hyperparameter Search",
            "problem": (
                "Grid search and random search spend too much time evaluating weak "
                "regions of the parameter space."
            ),
            "proposal": (
                "Use Optuna to optimize model and threshold settings with a smarter "
                "search loop guided by previous trials."
            ),
            "details": [
                "Tune tree depth, estimators, learning rate, and thresholds",
                "Score trials with balanced accuracy or backtest-aware objectives",
                "Focus compute budget on the most promising settings",
            ],
            "impact": "Better model configurations in fewer training runs.",
        },
        {
            "theme": "Regimes",
            "title": "Stacked Ensemble",
            "problem": (
                "One model rarely performs well across both trend-following and "
                "mean-reverting conditions."
            ),
            "proposal": (
                "Blend specialized base models with a meta-model that also sees regime "
                "features such as volatility."
            ),
            "details": [
                "Model A: trend signals such as moving averages and ADX",
                "Model B: mean-reversion signals such as RSI and Bollinger Bands",
                "Meta-model: combine both outputs with a regime feature",
            ],
            "impact": "A more adaptive decision layer across changing market regimes.",
        },
    ],
    "focusQuestion": (
        "Current bottleneck to attack first: labeling, validation, or feature engineering?"
    ),
}


class ChatSessionCreateRequest(BaseModel):
    """Request payload for creating an assistant session."""

    title: str | None = None


class ChatMessageRequest(BaseModel):
    """Request payload for one user message to the trading assistant."""

    message: str
    productId: str | None = None
    forceRefresh: bool = False


class RagTextDocumentRequest(BaseModel):
    """Request payload for ingesting a raw text document into the knowledge store."""

    title: str
    content: str
    sourceUri: str | None = None
    metadata: dict[str, Any] | None = None


class RagUrlDocumentRequest(BaseModel):
    """Request payload for ingesting a fetched URL into the knowledge store."""

    url: str
    title: str | None = None
    metadata: dict[str, Any] | None = None


class RagFileDocumentRequest(BaseModel):
    """Request payload for ingesting a local file into the knowledge store."""

    path: str
    title: str | None = None
    metadata: dict[str, Any] | None = None


class RagSearchRequest(BaseModel):
    """Request payload for running a retrieval query against the knowledge store."""

    query: str
    limit: int | None = None


class TraderCapitalRequest(BaseModel):
    """Request payload for updating the tracked portfolio capital."""

    capital: float


class TraderPositionRequest(BaseModel):
    """Request payload for creating or updating one tracked position."""

    productId: str
    quantity: float = 0.0
    entryPrice: float | None = None
    currentPrice: float | None = None
    positionFraction: float | None = None
    openedAt: str | None = None
    metadata: dict[str, Any] | None = None


class TraderExecutionRequest(BaseModel):
    """Request payload for recording one executed spot fill."""

    productId: str
    side: str
    quantity: float
    price: float
    fee: float = 0.0
    currentPrice: float | None = None
    executedAt: str | None = None
    metadata: dict[str, Any] | None = None


class TraderTrackedTradeRequest(BaseModel):
    """Request payload for recording one tracked trade idea or live entry."""

    productId: str
    entryPrice: float
    takeProfitPrice: float | None = None
    stopLossPrice: float | None = None
    quantity: float | None = None
    currentPrice: float | None = None
    signalName: str | None = None
    status: str = "planned"
    openedAt: str | None = None
    metadata: dict[str, Any] | None = None


class TraderTrackedTradeFromSignalRequest(BaseModel):
    """Request payload for promoting a cached or live signal into a tracked trade."""

    live: bool = False
    forceRefresh: bool = False
    entryPrice: float | None = None
    quantity: float | None = None
    status: str = "planned"
    metadata: dict[str, Any] | None = None


class TraderTrackedTradeCloseRequest(BaseModel):
    """Request payload for closing a tracked trade and recording the outcome."""

    exitPrice: float
    closedAt: str | None = None
    closeReason: str | None = None
    currentPrice: float | None = None
    outcome: str | None = None
    metadata: dict[str, Any] | None = None


class ModelArtifactStore:
    """Load and summarize the latest trained model artifact if one exists."""

    def __init__(
        self,
        model_dir: Path,
        config: TrainingConfig | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = config or TrainingConfig()

    @staticmethod
    def _metadata_path(model_path: Path) -> Path:
        """Return the metadata sidecar path for a saved model artifact."""

        return model_path.with_suffix(".metadata.json")

    def _load_metadata(self, model_path: Path) -> dict[str, Any]:
        """Load model metadata when the sidecar exists and is readable."""

        metadata_path = self._metadata_path(model_path)
        if not metadata_path.exists():
            return {}

        try:
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                return json.load(metadata_file)
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _resolve_quote_currency(config: TrainingConfig) -> str:
        """Return the active quote currency for the configured market source."""

        if is_coinmarketcap_market_data_source(config.market_data_source):
            return config.coinmarketcap_quote_currency

        return config.coinbase_quote_currency

    @staticmethod
    def _resolve_product_mode(config: TrainingConfig) -> str:
        """Return the active product-mode label for the configured market source."""

        if is_coinmarketcap_market_data_source(config.market_data_source):
            fetch_all = bool(config.coinmarketcap_fetch_all_quote_products)
            quote_currency = config.coinmarketcap_quote_currency
        else:
            fetch_all = bool(config.coinbase_fetch_all_quote_products)
            quote_currency = config.coinbase_quote_currency

        return (
            f"all-{quote_currency.lower()} markets"
            if fetch_all
            else "explicit product list"
        )

    def get_summary(self) -> dict[str, Any]:
        """Return a frontend-friendly description of the latest saved model."""

        model_path = self._resolve_model_path()
        if not model_path.exists():
            return {
                "status": "missing",
                "path": str(model_path),
                "message": "No trained model artifact found yet. Run `python model-service/scripts/trainModel.py` first.",
            }

        try:
            model = BaseSignalModel.load(model_path)
        except Exception as error:
            return {
                "status": "error",
                "path": str(model_path),
                "message": f"Could not load the saved model artifact: {error}",
            }

        feature_importance_rows = [
            {
                "name": feature_name,
                "importance": float(importance),
            }
            for feature_name, importance in list(model.get_feature_importance().items())[:8]
        ]
        metadata = self._load_metadata(model_path)
        artifact_created_at = metadata.get("artifactCreatedAt") or _timestamp_to_isoformat(model_path.stat().st_mtime)
        model_age_hours = _age_hours_from_timestamp(artifact_created_at)
        metadata_path = self._metadata_path(model_path)
        training_data_path = Path(str(metadata.get("trainingDataPath") or model.config.data_file))
        training_data_last_modified = (
            _timestamp_to_isoformat(training_data_path.stat().st_mtime) if training_data_path.exists() else None
        )
        newer_data_available = training_data_path.exists() and (
            training_data_path.stat().st_mtime > model_path.stat().st_mtime
        )
        model_freshness = _classify_freshness(
            age_hours=model_age_hours,
            max_age_hours=self.config.production_model_max_age_hours,
        )
        retraining_due = bool(newer_data_available or model_freshness == "stale")

        return {
            "status": "ready",
            "path": str(model_path),
            "lastModified": _timestamp_to_isoformat(model_path.stat().st_mtime),
            "modelType": model.model_type,
            "featureCount": len(model.feature_columns),
            "featurePreview": list(model.feature_columns[:10]),
            "topFeatures": feature_importance_rows,
            "trainingMetrics": metadata.get("metrics", {}),
            "split": {
                "trainRows": metadata.get("trainRows"),
                "testRows": metadata.get("testRows"),
            },
            "lifecycle": {
                "artifactCreatedAt": artifact_created_at,
                "metadataPath": str(metadata_path) if metadata_path.exists() else None,
                "ageHours": model_age_hours,
                "freshness": model_freshness,
                "retrainingDue": retraining_due,
                "newerDataAvailable": newer_data_available,
                "trainingDataPath": str(training_data_path),
                "trainingDataLastModified": training_data_last_modified,
                "recommendedAction": (
                    "Refresh market data, retrain the model, and regenerate the frontend snapshot."
                    if retraining_due
                    else "Current artifact is fresh enough for the active production cadence."
                ),
            },
            "settings": {
                "labelingStrategy": model.config.labeling_strategy,
                "predictionHorizon": int(model.config.prediction_horizon),
                "buyThreshold": float(model.config.buy_threshold),
                "sellThreshold": float(model.config.sell_threshold),
                "recencyWeightingEnabled": bool(model.config.recency_weighting_enabled),
                "recencyWeightingHalflifeHours": float(model.config.recency_weighting_halflife_hours),
                "walkForwardPurgeGapTimestamps": (
                    int(model.config.walkforward_purge_gap_timestamps)
                    if model.config.walkforward_purge_gap_timestamps is not None
                    else int(model.config.prediction_horizon)
                ),
                "trainSize": float(model.config.train_size),
                "marketDataSource": model.config.market_data_source,
                "quoteCurrency": self._resolve_quote_currency(model.config),
                "productMode": self._resolve_product_mode(model.config),
            },
        }

    def _resolve_model_path(self) -> Path:
        """Choose the configured deployment artifact, falling back to the newest model."""

        default_model_class = get_model_class(self.config.model_type)
        default_model_path = self.model_dir / default_model_class.default_model_filename
        if default_model_path.exists():
            return default_model_path

        model_candidates = sorted(
            (path for path in self.model_dir.glob("*.pkl") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if model_candidates:
            return model_candidates[0]

        return default_model_path


def create_app(
    snapshot_path: Path | None = None,
    model_dir: Path | None = None,
    config: TrainingConfig | None = None,
    live_signal_engine: LiveSignalEngine | None = None,
    chart_service: MarketChartService | None = None,
    assistant_service: TradingAssistantService | None = None,
    session_store: ConversationSessionStore | None = None,
    portfolio_store: TradingPortfolioStore | None = None,
    signal_store: TradingSignalStore | None = None,
    knowledge_store: RagKnowledgeStore | None = None,
    require_internal_api_key: bool = False,
    internal_api_key: str | None = None,
) -> FastAPI:
    """Create the Python AI engine service."""

    snapshot_path = Path(snapshot_path or (OUTPUTS_DIR / "frontendSignalSnapshot.json"))
    model_dir = Path(model_dir or MODELS_DIR)
    runtime_config = config or TrainingConfig()

    snapshot_store = SignalSnapshotStore(snapshot_path=snapshot_path)
    model_store = ModelArtifactStore(model_dir=model_dir, config=runtime_config)
    live_signal_engine = live_signal_engine or LiveSignalEngine(
        model_dir=model_dir,
        config=runtime_config,
    )
    chart_service = chart_service or MarketChartService(config=runtime_config)
    session_store = session_store or ConversationSessionStore(
        db_path=runtime_config.assistant_store_path,
        database_url=runtime_config.assistant_store_url,
    )
    portfolio_store = portfolio_store or TradingPortfolioStore(
        db_path=runtime_config.portfolio_store_path,
        default_capital=runtime_config.portfolio_default_capital,
        database_url=runtime_config.portfolio_store_url,
    )
    signal_store = signal_store or TradingSignalStore(
        db_path=runtime_config.signal_store_path,
        database_url=runtime_config.signal_store_url,
    )
    knowledge_store = knowledge_store or (
        RagKnowledgeStore(
            db_path=runtime_config.rag_store_path,
            database_url=runtime_config.rag_store_url,
            chunk_size_chars=runtime_config.rag_chunk_size_chars,
            chunk_overlap_chars=runtime_config.rag_chunk_overlap_chars,
            fetch_timeout_seconds=runtime_config.rag_fetch_timeout_seconds,
            fetch_max_chars=runtime_config.rag_fetch_max_chars,
        )
        if runtime_config.rag_enabled
        else None
    )

    def get_cached_snapshot_or_none() -> dict[str, Any] | None:
        """Return the cached snapshot when it exists, otherwise None."""

        try:
            return snapshot_store.get_snapshot()
        except FileNotFoundError:
            return None

    fallback_snapshot_provider = (
        live_signal_engine.get_live_snapshot
        if hasattr(live_signal_engine, "get_live_snapshot")
        else None
    )
    published_signal_view = PublishedSignalViewService(
        signal_store=signal_store,
        snapshot_store=snapshot_store,
        cached_snapshot_provider=get_cached_snapshot_or_none,
        fallback_snapshot_provider=fallback_snapshot_provider,
    )
    signal_tools = SignalToolService(
        live_signal_engine=live_signal_engine,
        cached_snapshot_provider=get_cached_snapshot_or_none,
        published_signal_service=published_signal_view,
    )
    trader_tools = TraderToolService(
        signal_tools=signal_tools,
        portfolio_store=portfolio_store,
        config=runtime_config,
        published_signal_service=published_signal_view,
    )
    model_tools = ModelToolService(model_status_provider=model_store.get_summary)
    retrieval_tools = RetrievalToolService(
        knowledge_store=knowledge_store,
        default_limit=runtime_config.rag_search_limit,
    )
    tool_registry = ToolRegistry(
        signal_tools=signal_tools,
        trader_tools=trader_tools,
        model_tools=model_tools,
        retrieval_tools=retrieval_tools,
    )

    assistant_service = assistant_service or TradingAssistantService(
        live_signal_engine=live_signal_engine,
        session_store=session_store,
        model_summary_provider=model_store.get_summary,
        cached_snapshot_provider=get_cached_snapshot_or_none,
        knowledge_store=knowledge_store,
        portfolio_store=portfolio_store,
        tool_registry=tool_registry,
        published_signal_service=published_signal_view,
        config=runtime_config,
    )

    app = FastAPI(
        title="Crypto Signal AI Engine",
        version="0.1.0",
        description="Internal Python engine for model lifecycle, live inference, assistant chat, and retrieval.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    resolved_internal_api_key = str(internal_api_key or "").strip()

    @app.middleware("http")
    async def require_backend_internal_key(request: Request, call_next):
        """Reject direct callers when the engine is configured for backend-only access."""

        protected_path = request.url.path.startswith("/api") or request.url.path in {
            "/current-signal",
            "/current-signals",
            "/signal-history",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
        if not require_internal_api_key or not protected_path:
            return await call_next(request)

        if not resolved_internal_api_key:
            return JSONResponse(
                status_code=503,
                content={"detail": "Internal model-service API key is not configured."},
            )

        provided_key = str(request.headers.get(INTERNAL_API_KEY_HEADER, "") or "")
        if not secrets.compare_digest(provided_key, resolved_internal_api_key):
            return JSONResponse(
                status_code=403,
                content={"detail": "Python model service only accepts requests from the backend."},
            )

        return await call_next(request)

    def require_knowledge_store() -> RagKnowledgeStore:
        """Return the configured knowledge store or fail when RAG is disabled."""

        if knowledge_store is None:
            raise HTTPException(status_code=503, detail="RAG knowledge store is disabled in the current config.")

        return knowledge_store

    def get_snapshot_status() -> dict[str, Any]:
        """Return the current snapshot state without raising HTTP errors."""

        if not snapshot_path.exists():
            return {
                "status": "missing",
                "path": str(snapshot_path),
                "message": "No cached frontend snapshot found yet. Run `python model-service/scripts/generateSignals.py` first.",
            }

        overview = snapshot_store.get_overview()
        snapshot_age_hours = _age_hours_from_timestamp(overview["generatedAt"])
        return {
            "status": "ready",
            "path": str(snapshot_path),
            "generatedAt": overview["generatedAt"],
            "ageHours": snapshot_age_hours,
            "freshness": _classify_freshness(
                age_hours=snapshot_age_hours,
                max_age_hours=runtime_config.production_snapshot_max_age_hours,
            ),
            "modelType": overview["modelType"],
            "marketState": overview.get("marketState", {}),
            "marketSummary": overview["marketSummary"],
            "primarySignal": overview["primarySignal"],
            "traderBrain": overview.get("traderBrain", {}),
        }

    def require_snapshot() -> dict[str, Any]:
        """Return the loaded snapshot or raise a 404 for API consumers."""

        try:
            return snapshot_store.get_snapshot()
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    def get_current_signal_or_404() -> dict[str, Any]:
        """Return the primary current signal from the database or raise a 404."""

        signal_summary = signal_store.get_current_signal()
        if signal_summary is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No current live signal has been persisted yet. "
                    "Run `python model-service/scripts/generateSignals.py` first."
                ),
            )

        return signal_summary

    def filter_current_signal_rows(
        signal_rows: list[dict[str, Any]],
        action: str,
    ) -> list[dict[str, Any]]:
        """Apply the same public action filter semantics used by the cached snapshot routes."""

        normalized_action = str(action).strip().lower() or "all"
        if normalized_action == "all":
            return list(signal_rows)
        if normalized_action == "actionable":
            return [
                signal_summary
                for signal_summary in signal_rows
                if bool(signal_summary.get("actionable", False))
            ]

        return [
            signal_summary
            for signal_summary in signal_rows
            if str(signal_summary.get("spotAction", "")).strip().lower() == normalized_action
        ]

    def resolve_signal_for_tracking(
        product_id: str,
        *,
        live: bool = False,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Return one cached or live signal row for trade tracking."""

        normalized_product_id = str(product_id).strip().upper()
        try:
            if live:
                signal_summary = live_signal_engine.get_signal_by_product(
                    product_id=normalized_product_id,
                    force_refresh=force_refresh,
                )
            else:
                require_snapshot()
                signal_summary = snapshot_store.get_signal_by_product(normalized_product_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            if live:
                raise HTTPException(status_code=503, detail=f"Live market refresh failed: {error}") from error
            raise

        if signal_summary is None:
            raise HTTPException(
                status_code=404,
                detail=f"No {'live ' if live else ''}signal found for {normalized_product_id}.",
            )

        return signal_summary

    def build_trader_plan_payload(
        *,
        force_refresh: bool = False,
        capital_override: float | None = None,
    ) -> dict[str, Any]:
        """Build a portfolio-aware trader plan from live data or cached signals."""

        tool_payload = trader_tools.get_trader_plan(
            capital=capital_override,
            force_refresh=force_refresh,
        )
        if tool_payload.get("status") == "error":
            error_message = str(tool_payload.get("error") or "Trader plan is unavailable.")
            raise HTTPException(status_code=503, detail=error_message)

        return {
            "source": tool_payload.get("source"),
            "warning": tool_payload.get("warning"),
            "portfolio": tool_payload.get("portfolio"),
            "liveSnapshot": tool_payload.get("liveSnapshot"),
            "traderPlan": tool_payload.get("traderPlan"),
        }

    @app.get("/")
    def engine_root() -> dict[str, Any]:
        """Expose a compact status payload for the internal AI engine."""

        return {
            "status": "online",
            "service": "python-ai-engine",
            "apiBase": "/api",
            "docsPath": "/docs",
        }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Expose a compact health payload for the landing page."""

        snapshot_status = get_snapshot_status()
        model_status = model_store.get_summary()
        live_status = live_signal_engine.get_status()
        current_signal_status = signal_store.get_status()
        rag_status = knowledge_store.get_status() if knowledge_store is not None else {"enabled": False}

        return {
            "status": "ok" if snapshot_status["status"] == "ready" else "waiting_for_snapshot",
            "snapshotStatus": snapshot_status["status"],
            "modelStatus": model_status["status"],
            "liveStatus": live_status["status"],
            "currentSignalStatus": current_signal_status["status"],
            "currentSignalStore": current_signal_status,
            "ragStatus": rag_status,
            "snapshotFreshness": snapshot_status.get("freshness"),
            "modelFreshness": model_status.get("lifecycle", {}).get("freshness"),
            "modelRetrainingDue": model_status.get("lifecycle", {}).get("retrainingDue"),
            "generatedAt": snapshot_status.get("generatedAt"),
            "modelType": snapshot_status.get("modelType") or model_status.get("modelType"),
            "snapshotPath": str(snapshot_path),
            "modelPath": model_status.get("path"),
            "liveCacheAgeSeconds": live_status.get("cacheAgeSeconds"),
        }

    @app.get("/api/model")
    def model_summary() -> dict[str, Any]:
        """Return the latest model artifact details for the landing page."""

        return model_tools.get_model_status()["model"]

    @app.get("/api/tools")
    def tool_catalog() -> dict[str, Any]:
        """Return the stable internal tool catalog for LLM callers and adapters."""

        return {
            "count": len(tool_registry.list_tools()),
            "tools": tool_registry.list_tools(),
        }

    @app.get("/api/tools/market-overview")
    def tool_market_overview(
        force_refresh: bool = Query(default=False),
    ) -> dict[str, Any]:
        """Return the stable market-overview tool payload."""

        return signal_tools.get_market_overview(force_refresh=force_refresh)

    @app.get("/api/tools/signals/{product_id}")
    def tool_signal_detail(
        product_id: str,
        force_refresh: bool = Query(default=False),
    ) -> dict[str, Any]:
        """Return the stable single-signal tool payload."""

        return signal_tools.get_signal(
            product_id=product_id,
            force_refresh=force_refresh,
        )

    @app.get("/api/tools/trader-plan")
    def tool_trader_plan(
        force_refresh: bool = Query(default=False),
        capital: float | None = Query(default=None, gt=0),
    ) -> dict[str, Any]:
        """Return the stable trader-plan tool payload."""

        return trader_tools.get_trader_plan(
            capital=capital,
            force_refresh=force_refresh,
        )

    @app.get("/api/tools/model-status")
    def tool_model_status() -> dict[str, Any]:
        """Return the stable model-status tool payload."""

        return model_tools.get_model_status()

    @app.post("/api/tools/retrieval/search")
    def tool_retrieval_search(request: RagSearchRequest) -> dict[str, Any]:
        """Return the stable retrieval-search tool payload."""

        return retrieval_tools.search_knowledge(
            request.query,
            limit=request.limit or runtime_config.rag_search_limit,
        )

    @app.get("/api/landing")
    def landing_payload() -> dict[str, Any]:
        """Return one payload that the landing page can render immediately."""

        return {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "project": {
                "name": "Crypto Signal ML",
                "headline": "Model operations console for a continuously refreshed crypto signal pipeline",
                "description": (
                    "A frontend-ready view over the active trained model artifact, the cached market snapshot, "
                    "and the operational workflow that keeps both current."
                ),
                "rootPath": str(PROJECT_ROOT),
            },
            "backend": {
                "status": "online",
                "apiBase": "/api",
                "runtime": "typescript-backend -> python-ai-engine",
            },
            "assistant": {
                "name": runtime_config.assistant_system_name,
                "supportsLiveData": True,
                "supportsSessionMemory": True,
                "supportsRetrieval": bool(runtime_config.assistant_enable_retrieval),
                "supportsToolCalls": True,
            },
            "trader": {
                "brainEnabled": bool(runtime_config.brain_enabled),
                "portfolio": portfolio_store.get_portfolio(),
                "supportsLivePlan": True,
            },
            "currentSignals": signal_store.get_status(),
            "rag": knowledge_store.get_status() if knowledge_store is not None else {"enabled": False},
            "model": model_store.get_summary(),
            "modelResearch": MODEL_RESEARCH_ROADMAP,
            "snapshot": get_snapshot_status(),
            "live": live_signal_engine.get_status(),
            "workflow": [
                {
                    "step": "Run the production refresh cycle",
                    "command": "python model-service/scripts/runProductionCycle.py",
                },
                {
                    "step": "Refresh market data manually",
                    "command": "python model-service/scripts/refreshMarketData.py",
                },
                {
                    "step": "Refresh CoinMarketCal events manually",
                    "command": "python model-service/scripts/refreshMarketEvents.py",
                },
                {
                    "step": "Retrain and publish the current artifact",
                    "command": "python model-service/scripts/trainModel.py",
                },
                {
                    "step": "Regenerate the cached signal snapshot",
                    "command": "python model-service/scripts/generateSignals.py",
                },
                {
                    "step": "Start the Python AI engine",
                    "command": "python model-service",
                },
                {
                    "step": "Update tracked portfolio capital",
                    "command": "POST /api/trader/capital",
                },
                {
                    "step": "Sync one open position into the trader brain",
                    "command": "POST /api/trader/positions",
                },
                {
                    "step": "Record an executed buy or sell fill",
                    "command": "POST /api/trader/executions",
                },
                {
                    "step": "Track a trade idea with entry, stop, and take-profit targets",
                    "command": "POST /api/trader/trades or POST /api/trader/trades/from-signal/{productId}",
                },
                {
                    "step": "Close a tracked trade and label the outcome",
                    "command": "POST /api/trader/trades/{tradeId}/close",
                },
                {
                    "step": "Build the current portfolio-aware trader plan",
                    "command": "GET /api/trader/plan?force_refresh=true",
                },
                {
                    "step": "Start the TypeScript backend",
                    "command": "npm run backend:dev",
                },
                {
                    "step": "Ask live trading questions through the assistant",
                    "command": "POST /api/chat/sessions then POST /api/chat/sessions/{sessionId}/messages",
                },
                {
                    "step": "Ingest external sources into the RAG store",
                    "command": "POST /api/rag/documents/url or POST /api/rag/documents/text",
                },
            ],
            "endpoints": [
                {
                    "label": "Health",
                    "path": "/api/health",
                },
                {
                    "label": "Landing payload",
                    "path": "/api/landing",
                },
                {
                    "label": "Model summary",
                    "path": "/api/model",
                },
                {
                    "label": "Tool catalog",
                    "path": "/api/tools",
                },
                {
                    "label": "Signal overview",
                    "path": "/api/overview",
                },
                {
                    "label": "Market state",
                    "path": "/api/market-state",
                },
                {
                    "label": "Cached signal list",
                    "path": "/api/signals?action=all&limit=12",
                },
                {
                    "label": "Current signal",
                    "path": "/current-signal",
                },
                {
                    "label": "Current signals",
                    "path": "/current-signals?limit=12",
                },
                {
                    "label": "Signal history",
                    "path": "/signal-history?limit=25",
                },
                {
                    "label": "Live overview",
                    "path": "/api/live/overview",
                },
                {
                    "label": "Live signals",
                    "path": "/api/live/signals?action=all&limit=12",
                },
                {
                    "label": "Tool market overview",
                    "path": "/api/tools/market-overview?force_refresh=true",
                },
                {
                    "label": "Tool signal detail",
                    "path": "/api/tools/signals/BTC-USD?force_refresh=true",
                },
                {
                    "label": "Tool trader plan",
                    "path": "/api/tools/trader-plan?force_refresh=true",
                },
                {
                    "label": "Tool model status",
                    "path": "/api/tools/model-status",
                },
                {
                    "label": "Trader portfolio",
                    "path": "/api/trader/portfolio",
                },
                {
                    "label": "Trader plan",
                    "path": "/api/trader/plan?force_refresh=true",
                },
                {
                    "label": "Trader journal",
                    "path": "/api/trader/journal?limit=50",
                },
                {
                    "label": "Tracked trades",
                    "path": "/api/trader/trades?limit=50",
                },
                {
                    "label": "TradingView config",
                    "path": "/api/tradingview/config",
                },
                {
                    "label": "TradingView history",
                    "path": "/api/tradingview/history?symbol=BTC-USD&resolution=60&from=0&to=4102444800",
                },
                {
                    "label": "CoinMarketCal events",
                    "path": "/api/events?symbol=BTC-USD&limit=20",
                },
                {
                    "label": "Create chat session",
                    "path": "/api/chat/sessions",
                },
                {
                    "label": "RAG status",
                    "path": "/api/rag/status",
                },
                {
                    "label": "RAG sources",
                    "path": "/api/rag/sources",
                },
            ],
        }

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        """Return the signal snapshot overview block."""

        snapshot = require_snapshot()
        return {
            "generatedAt": snapshot["generatedAt"],
            "modelType": snapshot["modelType"],
            "marketState": snapshot.get("marketState", {}),
            "marketSummary": snapshot["marketSummary"],
            "primarySignal": snapshot["primarySignal"],
            "traderBrain": snapshot.get("traderBrain", {}),
        }

    @app.get("/api/market-state")
    def market_state() -> dict[str, Any]:
        """Return the aggregate market-state block from the cached snapshot."""

        require_snapshot()
        return snapshot_store.get_market_state()

    @app.get("/api/signals")
    def list_signals(
        action: str = Query(default="all"),
        limit: Optional[int] = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        """Return cached signals for the requested action."""

        require_snapshot()
        signal_rows = snapshot_store.list_signals(action=action, limit=limit)
        return {
            "action": action,
            "count": len(signal_rows),
            "signals": signal_rows,
        }

    @app.get("/api/signals/{product_id}")
    def signal_detail(product_id: str) -> dict[str, Any]:
        """Return one cached signal row by product id."""

        require_snapshot()
        signal_summary = snapshot_store.get_signal_by_product(product_id)
        if signal_summary is None:
            raise HTTPException(status_code=404, detail=f"No signal found for {product_id}.")

        return signal_summary

    @app.get("/current-signal")
    @app.get("/api/current-signal")
    def current_signal() -> dict[str, Any]:
        """Return the latest persisted current signal from the database."""

        return get_current_signal_or_404()

    @app.get("/current-signals")
    @app.get("/api/current-signals")
    def current_signals(
        action: str = Query(default="all"),
        limit: Optional[int] = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        """Return the currently published live signal set from the database."""

        return published_signal_view.build_current_signals_response(
            action=action,
            limit=limit or 50,
        )

    @app.get("/signal-history")
    @app.get("/api/signal-history")
    def signal_history(
        limit: Optional[int] = Query(default=100, ge=1, le=500),
        product_id: str | None = Query(default=None, alias="productId"),
    ) -> dict[str, Any]:
        """Return the newest persisted signal events from the database."""

        return published_signal_view.build_signal_history_response(
            limit=limit or 100,
            product_id=product_id,
        )

    @app.get("/api/live/overview")
    def live_overview(
        product_id: str | None = Query(default=None),
        force_refresh: bool = Query(default=False),
    ) -> dict[str, Any]:
        """Return the latest live market snapshot from the realtime signal engine."""

        try:
            return live_signal_engine.get_live_snapshot(
                force_refresh=force_refresh,
                product_id=product_id,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Live market refresh failed: {error}") from error

    @app.get("/api/live/signals")
    def live_signal_list(
        action: str = Query(default="all"),
        limit: Optional[int] = Query(default=50, ge=1, le=500),
        force_refresh: bool = Query(default=False),
    ) -> dict[str, Any]:
        """Return filtered live signals from the realtime signal engine."""

        try:
            signal_rows = live_signal_engine.list_signals(
                action=action,
                limit=limit,
                force_refresh=force_refresh,
            )
            return {
                "action": action,
                "count": len(signal_rows),
                "signals": signal_rows,
            }
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Live market refresh failed: {error}") from error

    @app.get("/api/live/signals/{product_id}")
    def live_signal_detail(
        product_id: str,
        force_refresh: bool = Query(default=False),
    ) -> dict[str, Any]:
        """Return one live signal row by product id."""

        try:
            signal_summary = live_signal_engine.get_signal_by_product(
                product_id=product_id,
                force_refresh=force_refresh,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Live market refresh failed: {error}") from error

        if signal_summary is None:
            raise HTTPException(status_code=404, detail=f"No live signal found for {product_id}.")

        return signal_summary

    @app.get("/api/trader/portfolio")
    def trader_portfolio() -> dict[str, Any]:
        """Return the tracked portfolio capital and open positions."""

        return portfolio_store.get_portfolio()

    @app.post("/api/trader/capital")
    def trader_capital(request: TraderCapitalRequest) -> dict[str, Any]:
        """Update the tracked portfolio capital."""

        try:
            return portfolio_store.set_capital(request.capital)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/trader/positions")
    def trader_upsert_position(request: TraderPositionRequest) -> dict[str, Any]:
        """Create or update one tracked position."""

        try:
            position = portfolio_store.upsert_position(
                product_id=request.productId,
                quantity=request.quantity,
                entry_price=request.entryPrice,
                current_price=request.currentPrice,
                position_fraction=request.positionFraction,
                opened_at=request.openedAt,
                metadata=request.metadata,
            )
            return {
                "status": "upserted",
                "position": position,
                "portfolio": portfolio_store.get_portfolio(),
            }
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.delete("/api/trader/positions/{product_id}")
    def trader_delete_position(product_id: str) -> dict[str, Any]:
        """Delete one tracked position from the portfolio store."""

        deleted = portfolio_store.delete_position(product_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Tracked position not found: {product_id}")

        return {
            "status": "deleted",
            "productId": str(product_id).strip().upper(),
            "portfolio": portfolio_store.get_portfolio(),
        }

    @app.get("/api/trader/journal")
    def trader_journal(
        limit: Optional[int] = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        """Return the newest execution-journal rows."""

        journal_rows = portfolio_store.list_journal(limit=limit or 50)
        return {
            "count": len(journal_rows),
            "executions": journal_rows,
        }

    @app.get("/api/trader/trades")
    def trader_trades(
        limit: Optional[int] = Query(default=50, ge=1, le=500),
        status: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Return tracked trade records for review and post-trade analysis."""

        try:
            trade_rows = portfolio_store.list_trades(limit=limit or 50, status=status)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return {
            "count": len(trade_rows),
            "status": status,
            "trades": trade_rows,
        }

    @app.get("/api/trader/trades/{trade_id}")
    def trader_trade_detail(trade_id: int) -> dict[str, Any]:
        """Return one tracked trade record by id."""

        try:
            trade_row = portfolio_store.get_trade(trade_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        if trade_row is None:
            raise HTTPException(status_code=404, detail=f"Tracked trade not found: {trade_id}")

        return trade_row

    @app.post("/api/trader/trades")
    def trader_create_trade(request: TraderTrackedTradeRequest) -> dict[str, Any]:
        """Create one tracked trade record with entry and target prices."""

        try:
            trade_row = portfolio_store.create_trade(
                product_id=request.productId,
                entry_price=request.entryPrice,
                take_profit_price=request.takeProfitPrice,
                stop_loss_price=request.stopLossPrice,
                quantity=request.quantity,
                current_price=request.currentPrice,
                signal_name=request.signalName,
                status=request.status,
                opened_at=request.openedAt,
                metadata=request.metadata,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return {
            "status": "tracked",
            "trade": trade_row,
            "portfolio": portfolio_store.get_portfolio(),
        }

    @app.post("/api/trader/trades/from-signal/{product_id}")
    def trader_create_trade_from_signal(
        product_id: str,
        request: TraderTrackedTradeFromSignalRequest,
    ) -> dict[str, Any]:
        """Create one tracked trade directly from a cached or live signal row."""

        signal_summary = resolve_signal_for_tracking(
            product_id=product_id,
            live=request.live,
            force_refresh=request.forceRefresh,
        )
        brain = signal_summary.get("brain", {}) if isinstance(signal_summary.get("brain"), dict) else {}
        signal_metadata = {
            "trackedFromSignal": True,
            "signalSource": "live" if request.live else "cached",
            "signalTimestamp": signal_summary.get("timestamp"),
            "signalName": signal_summary.get("signal_name"),
            "confidence": signal_summary.get("confidence"),
            "signalSourceLabel": signal_summary.get("signalSource"),
            "marketDataSource": signal_summary.get("marketDataSource"),
            "brainDecision": brain.get("decision"),
            "brainSummary": brain.get("summaryLine"),
        }
        merged_metadata = {**signal_metadata, **(request.metadata or {})}

        try:
            trade_row = portfolio_store.create_trade(
                product_id=str(signal_summary.get("productId") or product_id),
                entry_price=float(request.entryPrice or signal_summary.get("close") or 0.0),
                take_profit_price=brain.get("takeProfitPrice"),
                stop_loss_price=brain.get("stopLossPrice"),
                quantity=request.quantity,
                current_price=float(signal_summary.get("close") or request.entryPrice or 0.0),
                signal_name=str(signal_summary.get("signal_name") or ""),
                status=request.status,
                opened_at=str(signal_summary.get("timestamp") or ""),
                metadata=merged_metadata,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return {
            "status": "tracked",
            "trade": trade_row,
            "signal": signal_summary,
            "portfolio": portfolio_store.get_portfolio(),
        }

    @app.post("/api/trader/executions")
    def trader_record_execution(request: TraderExecutionRequest) -> dict[str, Any]:
        """Record one executed spot fill and update the tracked portfolio."""

        try:
            return portfolio_store.record_execution(
                product_id=request.productId,
                side=request.side,
                quantity=request.quantity,
                price=request.price,
                fee=request.fee,
                current_price=request.currentPrice,
                executed_at=request.executedAt,
                metadata=request.metadata,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/trader/trades/{trade_id}/close")
    def trader_close_trade(
        trade_id: int,
        request: TraderTrackedTradeCloseRequest,
    ) -> dict[str, Any]:
        """Close one tracked trade and compute the outcome summary."""

        try:
            trade_row = portfolio_store.close_trade(
                trade_id=trade_id,
                exit_price=request.exitPrice,
                closed_at=request.closedAt,
                close_reason=request.closeReason,
                current_price=request.currentPrice,
                outcome=request.outcome,
                metadata=request.metadata,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return {
            "status": "closed",
            "trade": trade_row,
            "portfolio": portfolio_store.get_portfolio(),
        }

    @app.get("/api/trader/plan")
    def trader_plan(
        force_refresh: bool = Query(default=False),
        capital: float | None = Query(default=None, gt=0),
    ) -> dict[str, Any]:
        """Return a live or cached trader-brain plan using tracked portfolio state."""

        return build_trader_plan_payload(
            force_refresh=force_refresh,
            capital_override=capital,
        )

    @app.get("/api/tradingview/config")
    def tradingview_config() -> dict[str, Any]:
        """Return the TradingView UDF capability block."""

        return chart_service.get_udf_config()

    @app.get("/api/tradingview/time")
    def tradingview_time() -> int:
        """Return the current server time for TradingView clients."""

        return chart_service.get_server_time()

    @app.get("/api/tradingview/search")
    def tradingview_search(
        query: str = Query(default=""),
        exchange: str | None = Query(default=None),
        type: str | None = Query(default=None),
        limit: int = Query(default=30, ge=1, le=100),
    ) -> list[dict[str, Any]]:
        """Search the chart symbol catalog."""

        return chart_service.search_symbols(
            query=query,
            exchange=exchange,
            symbol_type=type,
            limit=limit,
        )

    @app.get("/api/tradingview/symbols")
    def tradingview_symbol(symbol: str = Query(...)) -> dict[str, Any]:
        """Resolve one symbol into TradingView metadata."""

        try:
            return chart_service.resolve_symbol(symbol)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/tradingview/history")
    def tradingview_history(
        symbol: str = Query(...),
        resolution: str = Query(...),
        from_value: int = Query(alias="from"),
        to: int = Query(...),
        countback: int | None = Query(default=None, ge=1),
    ) -> dict[str, Any]:
        """Return OHLCV history in TradingView UDF format."""

        try:
            return chart_service.get_history(
                symbol=symbol,
                resolution=resolution,
                from_seconds=from_value,
                to_seconds=to,
                countback=countback,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Chart history request failed: {error}") from error

    @app.get("/api/tradingview/quotes")
    def tradingview_quotes(symbols: str = Query(...)) -> dict[str, Any]:
        """Return the latest quote block for one or more symbols."""

        symbol_list = [value for value in str(symbols).split(",") if value.strip()]
        try:
            return chart_service.get_quotes(symbol_list)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Quote request failed: {error}") from error

    @app.get("/api/tradingview/marks")
    def tradingview_marks(
        symbol: str = Query(...),
        from_value: int = Query(alias="from"),
        to: int = Query(...),
        resolution: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Return CoinMarketCal events as TradingView chart marks."""

        del resolution
        return chart_service.get_marks(
            symbol=symbol,
            from_seconds=from_value,
            to_seconds=to,
        )

    @app.get("/api/tradingview/timescale_marks")
    def tradingview_timescale_marks(
        symbol: str = Query(...),
        from_value: int = Query(alias="from"),
        to: int = Query(...),
        resolution: str | None = Query(default=None),
    ) -> list[dict[str, Any]]:
        """Return CoinMarketCal events as TradingView timescale marks."""

        del resolution
        return chart_service.get_timescale_marks(
            symbol=symbol,
            from_seconds=from_value,
            to_seconds=to,
        )

    @app.get("/api/events")
    def market_events(
        symbol: str | None = Query(default=None),
        from_value: int | None = Query(default=None, alias="from"),
        to: int | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        """Return cached CoinMarketCal events for frontend consumers."""

        event_rows = chart_service.list_events(
            symbol=symbol,
            from_seconds=from_value,
            to_seconds=to,
            limit=limit,
        )
        return {
            "count": len(event_rows),
            "events": event_rows,
        }

    @app.post("/api/chat/sessions")
    def create_chat_session(request: ChatSessionCreateRequest) -> dict[str, Any]:
        """Create a new assistant conversation session."""

        return assistant_service.create_session(title=request.title)

    @app.get("/api/chat/sessions/{session_id}")
    def chat_session(session_id: str) -> dict[str, Any]:
        """Return one chat session and its recent messages."""

        try:
            return assistant_service.get_session_state(session_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/chat/sessions/{session_id}/messages")
    def chat_messages(
        session_id: str,
        limit: Optional[int] = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        """Return recent messages for one assistant session."""

        try:
            session = session_store.get_session(session_id)
            if session is None:
                raise ValueError(f"Assistant session not found: {session_id}")

            return {
                "session": session,
                "messages": session_store.list_messages(session_id=session_id, limit=limit or 50),
            }
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/chat/sessions/{session_id}/messages")
    def send_chat_message(
        session_id: str,
        request: ChatMessageRequest,
    ) -> dict[str, Any]:
        """Answer one trading question using live market data and session memory."""

        try:
            return assistant_service.answer_question(
                session_id=session_id,
                question=request.message,
                product_id=request.productId,
                force_refresh=request.forceRefresh,
            )
        except ValueError as error:
            status_code = 404 if "session not found" in str(error).lower() else 400
            raise HTTPException(status_code=status_code, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Assistant response failed: {error}") from error

    @app.get("/api/rag/status")
    def rag_status() -> dict[str, Any]:
        """Return the current knowledge-store status."""

        return require_knowledge_store().get_status()

    @app.get("/api/rag/sources")
    def rag_sources(
        limit: Optional[int] = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        """List recently ingested knowledge sources."""

        store = require_knowledge_store()
        sources = store.list_sources(limit=limit or 50)
        return {
            "count": len(sources),
            "sources": sources,
        }

    @app.post("/api/rag/search")
    def rag_search(request: RagSearchRequest) -> dict[str, Any]:
        """Search the knowledge store for chunks relevant to the query."""

        store = require_knowledge_store()
        results = store.search(
            request.query,
            limit=request.limit or runtime_config.rag_search_limit,
        )
        return {
            "query": request.query,
            "count": len(results),
            "results": results,
        }

    @app.post("/api/rag/documents/text")
    def rag_ingest_text(request: RagTextDocumentRequest) -> dict[str, Any]:
        """Ingest a raw text source into the knowledge store."""

        try:
            source = require_knowledge_store().ingest_text(
                title=request.title,
                content=request.content,
                source_uri=request.sourceUri,
                metadata=request.metadata,
                source_type="text",
            )
            return {
                "status": "ingested",
                "source": source,
            }
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/rag/documents/url")
    def rag_ingest_url(request: RagUrlDocumentRequest) -> dict[str, Any]:
        """Fetch and ingest a URL into the knowledge store."""

        try:
            source = require_knowledge_store().ingest_url(
                url=request.url,
                title=request.title,
                metadata=request.metadata,
            )
            return {
                "status": "ingested",
                "source": source,
            }
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/rag/documents/file")
    def rag_ingest_file(request: RagFileDocumentRequest) -> dict[str, Any]:
        """Ingest a local file into the knowledge store."""

        try:
            source = require_knowledge_store().ingest_file(
                file_path=Path(request.path),
                title=request.title,
                metadata=request.metadata,
            )
            return {
                "status": "ingested",
                "source": source,
            }
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.delete("/api/rag/sources/{source_id}")
    def rag_delete_source(source_id: str) -> dict[str, Any]:
        """Delete one knowledge source and all of its chunks."""

        deleted = require_knowledge_store().delete_source(source_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Knowledge source not found: {source_id}")

        return {
            "status": "deleted",
            "sourceId": source_id,
        }

    return app


_module_internal_api_key = str(os.getenv("AI_ENGINE_INTERNAL_API_KEY", "")).strip()
app = create_app(
    require_internal_api_key=bool(_module_internal_api_key),
    internal_api_key=_module_internal_api_key or None,
)
