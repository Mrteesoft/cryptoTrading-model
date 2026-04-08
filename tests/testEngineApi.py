"""Smoke tests for the Python AI engine API."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_signal_ml.config import TrainingConfig  # noqa: E402
from crypto_signal_ml.frontend import build_frontend_signal_snapshot  # noqa: E402
from crypto_signal_ml.modeling import RandomForestSignalModel  # noqa: E402
from crypto_signal_ml.retrieval import RagKnowledgeStore  # noqa: E402
from crypto_signal_ml.service import create_app, create_signal_api  # noqa: E402
from crypto_signal_ml.trading import TradingPortfolioStore, TradingSignalStore  # noqa: E402


def _build_trained_model(
    model_path: Path,
    config: TrainingConfig | None = None,
) -> None:
    """Train and save a tiny model artifact for engine endpoint tests."""

    feature_columns = ["return_1", "momentum_3"]
    training_df = pd.DataFrame(
        [
            {"return_1": -0.04, "momentum_3": -0.09, "target_signal": -1},
            {"return_1": -0.03, "momentum_3": -0.05, "target_signal": -1},
            {"return_1": -0.01, "momentum_3": -0.01, "target_signal": 0},
            {"return_1": 0.00, "momentum_3": 0.01, "target_signal": 0},
            {"return_1": 0.03, "momentum_3": 0.05, "target_signal": 1},
            {"return_1": 0.05, "momentum_3": 0.08, "target_signal": 1},
        ]
    )

    model = RandomForestSignalModel(
        config=config or TrainingConfig(
            coinmarketcap_use_context=False,
            n_estimators=12,
            max_depth=3,
            random_state=7,
        ),
        feature_columns=feature_columns,
    )
    model.fit(training_df)
    model.save(model_path)


def _build_chart_history_frame() -> pd.DataFrame:
    """Create a small hourly OHLCV set for chart and quote endpoint tests."""

    rows = []
    market_start = pd.Timestamp("2026-03-01T00:00:00Z")
    market_specs = [
        ("BTC-USD", "BTC", 62000.0, 250.0, 1100.0),
        ("ETH-USD", "ETH", 3200.0, 18.0, 2100.0),
    ]

    for hour_index in range(8):
        timestamp = market_start + pd.Timedelta(hours=hour_index)
        for product_id, base_currency, start_price, price_step, start_volume in market_specs:
            open_price = start_price + (hour_index * price_step)
            close_price = open_price + (price_step * 0.35)
            rows.append(
                {
                    "timestamp": timestamp,
                    "product_id": product_id,
                    "base_currency": base_currency,
                    "quote_currency": "USD",
                    "open": open_price,
                    "high": close_price + (price_step * 0.25),
                    "low": open_price - (price_step * 0.15),
                    "close": close_price,
                    "volume": start_volume + hour_index,
                    "granularity_seconds": 3600,
                    "source": "coinmarketcap",
                }
            )

    return pd.DataFrame(rows)


def _build_snapshot(snapshot_path: Path) -> None:
    """Create a small cached signal snapshot for the landing page."""

    latest_signals = [
        {
            "productId": "BTC-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.84,
            "setupScore": 4.2,
            "signalChat": "BTC-USD is a BUY setup with strong relative momentum.",
            "symbol": "BTC",
            "close": 65000.0,
            "marketState": {
                "label": "trend_up",
                "code": 2,
                "trendScore": 0.03,
                "volatilityRatio": 1.05,
                "isTrending": True,
                "isHighVolatility": False,
            },
        },
        {
            "productId": "ETH-USD",
            "signal_name": "HOLD",
            "spotAction": "wait",
            "confidence": 0.63,
            "setupScore": 0.0,
            "signalChat": "ETH-USD is a HOLD setup while momentum rebuilds.",
            "symbol": "ETH",
            "close": 3300.0,
            "marketState": {
                "label": "range_high_volatility",
                "code": 4,
                "trendScore": 0.00,
                "volatilityRatio": 1.30,
                "isTrending": False,
                "isHighVolatility": True,
            },
        },
    ]

    snapshot_payload = build_frontend_signal_snapshot(
        model_type="randomForestSignalModel",
        primary_signal=latest_signals[0],
        latest_signals=latest_signals,
        actionable_signals=[latest_signals[0]],
    )
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")


def _build_live_snapshot_payload() -> dict[str, object]:
    """Create a small live-style payload for fake realtime engine tests."""

    latest_signals = [
        {
            "productId": "BTC-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.81,
            "setupScore": 4.0,
            "signalChat": "BTC-USD is a BUY setup on the latest live candle.",
            "symbol": "BTC",
            "close": 65250.0,
            "marketState": {
                "label": "trend_up",
                "code": 2,
                "trendScore": 0.03,
                "volatilityRatio": 1.02,
                "isTrending": True,
                "isHighVolatility": False,
            },
        },
        {
            "productId": "ETH-USD",
            "signal_name": "HOLD",
            "spotAction": "wait",
            "confidence": 0.61,
            "setupScore": 0.0,
            "signalChat": "ETH-USD is a HOLD setup on the latest live candle.",
            "symbol": "ETH",
            "close": 3320.0,
            "marketState": {
                "label": "range",
                "code": 1,
                "trendScore": 0.00,
                "volatilityRatio": 0.95,
                "isTrending": False,
                "isHighVolatility": False,
            },
        },
    ]

    payload = build_frontend_signal_snapshot(
        model_type="randomForestSignalModel",
        primary_signal=latest_signals[0],
        latest_signals=latest_signals,
        actionable_signals=[latest_signals[0]],
    )
    payload.update(
        {
            "mode": "live",
            "marketDataSource": "coinbaseExchangeRest",
            "requestMode": "configured-watchlist",
            "requestedProducts": ["BTC-USD", "ETH-USD"],
            "productsCovered": 2,
            "featureRowsScored": 40,
            "granularitySeconds": 3600,
            "liveSignalCacheSeconds": 60,
            "modelPath": "models/randomForestSignalModel.pkl",
        }
    )
    return payload


def test_engine_api_exposes_health_and_signal_payloads(tmp_path: Path) -> None:
    """The engine API should expose health plus combined signal payloads."""

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "randomForestSignalModel.pkl"
    _build_trained_model(model_path)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    snapshot_path = outputs_dir / "frontendSignalSnapshot.json"
    _build_snapshot(snapshot_path)

    client = TestClient(
        create_app(
            snapshot_path=snapshot_path,
            model_dir=model_dir,
        )
    )

    status_response = client.get("/")
    health_response = client.get("/api/health")
    landing_response = client.get("/api/landing")
    signals_response = client.get("/api/signals?action=all&limit=12")

    assert status_response.status_code == 200
    assert status_response.json()["service"] == "python-ai-engine"
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert health_response.json()["modelStatus"] == "ready"
    assert landing_response.status_code == 200
    assert landing_response.json()["model"]["modelType"] == "randomForestSignalModel"
    assert landing_response.json()["modelResearch"]["tracks"][0]["title"] == "Triple-Barrier Method"
    assert landing_response.json()["snapshot"]["marketSummary"]["actionableSignals"] == 1
    assert landing_response.json()["snapshot"]["marketState"]["dominant"]["label"] == "range_high_volatility"
    assert signals_response.status_code == 200
    assert signals_response.json()["count"] == 2


def test_cached_signal_api_exposes_market_state_summary(tmp_path: Path) -> None:
    """The lightweight cached API should surface the aggregate market-state block."""

    snapshot_path = tmp_path / "frontendSignalSnapshot.json"
    _build_snapshot(snapshot_path)

    client = TestClient(create_signal_api(snapshot_path=snapshot_path))

    overview_response = client.get("/api/overview")
    market_state_response = client.get("/api/market-state")

    assert overview_response.status_code == 200
    assert overview_response.json()["marketState"]["primary"]["label"] == "trend_up"
    assert market_state_response.status_code == 200
    assert market_state_response.json()["dominant"]["label"] == "range_high_volatility"
    assert market_state_response.json()["highVolatilitySignals"] == 1


def test_engine_api_reports_missing_snapshot_cleanly(tmp_path: Path) -> None:
    """Engine endpoints should stay readable before a snapshot exists."""

    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=tmp_path / "models",
        )
    )

    landing_response = client.get("/api/landing")
    signals_response = client.get("/api/signals")

    assert landing_response.status_code == 200
    assert landing_response.json()["snapshot"]["status"] == "missing"
    assert signals_response.status_code == 404


def test_engine_api_can_require_backend_internal_api_key(tmp_path: Path) -> None:
    """Protected engine routes should reject direct callers without the backend key."""

    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=tmp_path / "models",
            require_internal_api_key=True,
            internal_api_key="test-backend-key",
        )
    )

    unauthorized_response = client.get("/api/health")
    authorized_response = client.get(
        "/api/health",
        headers={"x-ai-engine-key": "test-backend-key"},
    )

    assert unauthorized_response.status_code == 403
    assert authorized_response.status_code == 200


def test_engine_api_reads_current_signals_from_database_across_app_restarts(tmp_path: Path) -> None:
    """Current-signal endpoints should read the persisted live-signal database, not cached files."""

    signal_db_path = tmp_path / "liveSignals.sqlite3"
    initial_store = TradingSignalStore(db_path=signal_db_path)
    current_signal_rows = [
        {
            "productId": "BTC-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "actionable": True,
            "confidence": 0.84,
            "close": 65250.0,
            "timestamp": "2026-04-07T16:28:00+00:00",
            "signalSource": "live-market-refresh",
        },
        {
            "productId": "ETH-USD",
            "signal_name": "HOLD",
            "spotAction": "wait",
            "actionable": False,
            "confidence": 0.63,
            "close": 3320.0,
            "timestamp": "2026-04-07T16:28:00+00:00",
            "signalSource": "live-market-refresh",
        },
    ]
    initial_store.replace_current_signals(
        signal_summaries=current_signal_rows,
        primary_signal=current_signal_rows[0],
        generated_at="2026-04-07T16:30:00+00:00",
    )

    restarted_store = TradingSignalStore(db_path=signal_db_path)
    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=tmp_path / "models",
            signal_store=restarted_store,
        )
    )

    current_signal_response = client.get("/current-signal")
    current_signals_response = client.get("/current-signals?limit=10")
    current_buy_signals_response = client.get("/current-signals?action=buy&limit=10")
    current_actionable_signals_response = client.get("/current-signals?action=actionable&limit=10")
    signal_history_response = client.get("/signal-history?limit=10")

    assert current_signal_response.status_code == 200
    assert current_signal_response.json()["productId"] == "BTC-USD"
    assert current_signal_response.json()["isPrimary"] is True
    assert current_signal_response.json()["generatedAt"] == "2026-04-07T16:30:00+00:00"
    assert current_signals_response.status_code == 200
    assert current_signals_response.json()["count"] == 2
    assert current_signals_response.json()["primaryProductId"] == "BTC-USD"
    assert current_signals_response.json()["storageBackend"] == "sqlite"
    assert current_buy_signals_response.status_code == 200
    assert current_buy_signals_response.json()["action"] == "buy"
    assert current_buy_signals_response.json()["count"] == 1
    assert current_buy_signals_response.json()["signals"][0]["productId"] == "BTC-USD"
    assert current_actionable_signals_response.status_code == 200
    assert current_actionable_signals_response.json()["count"] == 1
    assert current_actionable_signals_response.json()["signals"][0]["productId"] == "BTC-USD"
    assert signal_history_response.status_code == 200
    assert signal_history_response.json()["count"] == 2
    assert signal_history_response.json()["signals"][0]["generationId"]


def test_engine_api_protects_current_signal_endpoint_with_internal_key(tmp_path: Path) -> None:
    """Root-level current-signal endpoints should honor the same backend-only auth gate."""

    signal_store = TradingSignalStore(db_path=tmp_path / "liveSignals.sqlite3")
    signal_row = {
        "productId": "BTC-USD",
        "signal_name": "BUY",
        "spotAction": "buy",
        "actionable": True,
        "confidence": 0.80,
        "close": 65000.0,
        "timestamp": "2026-04-07T16:28:00+00:00",
    }
    signal_store.replace_current_signals(
        signal_summaries=[signal_row],
        primary_signal=signal_row,
        generated_at="2026-04-07T16:30:00+00:00",
    )

    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=tmp_path / "models",
            signal_store=signal_store,
            require_internal_api_key=True,
            internal_api_key="test-backend-key",
        )
    )

    unauthorized_response = client.get("/current-signal")
    authorized_response = client.get(
        "/current-signal",
        headers={"x-ai-engine-key": "test-backend-key"},
    )

    assert unauthorized_response.status_code == 403
    assert authorized_response.status_code == 200
    assert authorized_response.json()["productId"] == "BTC-USD"


def test_model_summary_exposes_lifecycle_metadata_when_sidecar_exists(tmp_path: Path) -> None:
    """The model API should surface training metadata for production readiness checks."""

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "randomForestSignalModel.pkl"
    _build_trained_model(model_path)

    data_path = tmp_path / "marketPrices.csv"
    data_path.write_text("timestamp,product_id,close\n", encoding="utf-8")
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "artifactCreatedAt": "2026-03-28T12:00:00+00:00",
                "trainingDataPath": str(data_path),
                "trainingDataLastModified": "2026-03-28T11:00:00+00:00",
                "trainRows": 4,
                "testRows": 2,
                "metrics": {
                    "accuracy": 0.75,
                    "balancedAccuracy": 0.72,
                },
            }
        ),
        encoding="utf-8",
    )

    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=model_dir,
        )
    )

    model_response = client.get("/api/model")
    payload = model_response.json()

    assert model_response.status_code == 200
    assert payload["trainingMetrics"]["balancedAccuracy"] == 0.72
    assert payload["split"] == {"trainRows": 4, "testRows": 2}
    assert payload["lifecycle"]["metadataPath"] == str(metadata_path)
    assert payload["lifecycle"]["freshness"] in {"fresh", "stale"}


def test_model_summary_uses_active_market_source_settings(tmp_path: Path) -> None:
    """The model summary should describe the active configured market source, not Coinbase-only fields."""

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "randomForestSignalModel.pkl"
    _build_trained_model(
        model_path,
        config=TrainingConfig(
            market_data_source="coinmarketcap",
            coinmarketcap_use_context=False,
            coinmarketcap_fetch_all_quote_products=False,
            coinmarketcap_product_ids=("BTC-USD",),
            coinbase_fetch_all_quote_products=True,
            n_estimators=12,
            max_depth=3,
            random_state=7,
        ),
    )

    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=model_dir,
        )
    )

    model_response = client.get("/api/model")
    payload = model_response.json()

    assert model_response.status_code == 200
    assert payload["settings"]["marketDataSource"] == "coinmarketcap"
    assert payload["settings"]["quoteCurrency"] == "USD"
    assert payload["settings"]["productMode"] == "explicit product list"


def test_engine_api_exposes_live_signal_endpoints(tmp_path: Path) -> None:
    """The engine should expose live overview and per-product signal endpoints."""

    live_snapshot = _build_live_snapshot_payload()

    class FakeLiveSignalEngine:
        def get_status(self) -> dict[str, object]:
            return {
                "status": "ready",
                "modelPath": "models/randomForestSignalModel.pkl",
                "cacheAgeSeconds": 0,
                "cacheTtlSeconds": 60,
                "lastGeneratedAt": live_snapshot["generatedAt"],
            }

        def get_live_snapshot(self, force_refresh: bool = False, product_id: str | None = None) -> dict[str, object]:
            if product_id is None:
                return live_snapshot

            filtered_signal = live_snapshot["signalsByProduct"].get(product_id.upper())
            if filtered_signal is None:
                return live_snapshot

            return {
                **live_snapshot,
                "signals": [filtered_signal],
                "actionableSignals": [filtered_signal] if filtered_signal["signal_name"] != "HOLD" else [],
                "signalsByProduct": {product_id.upper(): filtered_signal},
                "primarySignal": filtered_signal,
                "productsCovered": 1,
                "requestedProducts": [product_id.upper()],
            }

        def list_signals(
            self,
            action: str = "all",
            limit: int | None = None,
            force_refresh: bool = False,
        ) -> list[dict[str, object]]:
            rows = list(live_snapshot["signals"])
            if action == "actionable":
                rows = list(live_snapshot["actionableSignals"])
            if limit is not None:
                rows = rows[:limit]
            return rows

        def get_signal_by_product(
            self,
            product_id: str,
            force_refresh: bool = False,
        ) -> dict[str, object] | None:
            return live_snapshot["signalsByProduct"].get(product_id.upper())

    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=tmp_path / "models",
            live_signal_engine=FakeLiveSignalEngine(),
        )
    )

    overview_response = client.get("/api/live/overview")
    signals_response = client.get("/api/live/signals?action=all&limit=12")
    detail_response = client.get("/api/live/signals/BTC-USD")

    assert overview_response.status_code == 200
    assert overview_response.json()["mode"] == "live"
    assert overview_response.json()["productsCovered"] == 2
    assert signals_response.status_code == 200
    assert signals_response.json()["count"] == 2
    assert detail_response.status_code == 200
    assert detail_response.json()["productId"] == "BTC-USD"
    assert detail_response.json()["signal_name"] == "BUY"


def test_engine_api_supports_trader_portfolio_and_plan_endpoints(tmp_path: Path) -> None:
    """The engine should persist portfolio state and build a portfolio-aware trader plan."""

    live_snapshot = build_frontend_signal_snapshot(
        model_type="randomForestSignalModel",
        primary_signal={
            "productId": "BTC-USD",
            "signal_name": "TAKE_PROFIT",
            "spotAction": "take_profit",
            "confidence": 0.82,
            "setupScore": 4.1,
            "signalChat": "BTC-USD is a TAKE_PROFIT setup on the latest live candle.",
            "close": 65250.0,
            "marketState": {
                "label": "trend_down_high_volatility",
                "code": 5,
                "trendScore": -0.03,
                "volatilityRatio": 1.35,
                "isTrending": True,
                "isHighVolatility": True,
            },
            "eventContext": {
                "hasEventNext7d": False,
            },
        },
        latest_signals=[
            {
                "productId": "BTC-USD",
                "signal_name": "TAKE_PROFIT",
                "spotAction": "take_profit",
                "confidence": 0.82,
                "setupScore": 4.1,
                "signalChat": "BTC-USD is a TAKE_PROFIT setup on the latest live candle.",
                "close": 65250.0,
                "marketState": {
                    "label": "trend_down_high_volatility",
                    "code": 5,
                    "trendScore": -0.03,
                    "volatilityRatio": 1.35,
                    "isTrending": True,
                    "isHighVolatility": True,
                },
                "eventContext": {
                    "hasEventNext7d": False,
                },
            },
            {
                "productId": "ETH-USD",
                "signal_name": "BUY",
                "spotAction": "buy",
                "confidence": 0.81,
                "setupScore": 4.0,
                "signalChat": "ETH-USD is a BUY setup on the latest live candle.",
                "close": 3320.0,
                "marketState": {
                    "label": "trend_up",
                    "code": 2,
                    "trendScore": 0.02,
                    "volatilityRatio": 1.02,
                    "isTrending": True,
                    "isHighVolatility": False,
                },
                "eventContext": {
                    "hasEventNext7d": False,
                },
            },
        ],
        actionable_signals=[
            {
                "productId": "BTC-USD",
                "signal_name": "TAKE_PROFIT",
                "spotAction": "take_profit",
                "confidence": 0.82,
                "setupScore": 4.1,
                "signalChat": "BTC-USD is a TAKE_PROFIT setup on the latest live candle.",
                "close": 65250.0,
                "marketState": {
                    "label": "trend_down_high_volatility",
                    "code": 5,
                    "trendScore": -0.03,
                    "volatilityRatio": 1.35,
                    "isTrending": True,
                    "isHighVolatility": True,
                },
                "eventContext": {
                    "hasEventNext7d": False,
                },
            },
            {
                "productId": "ETH-USD",
                "signal_name": "BUY",
                "spotAction": "buy",
                "confidence": 0.81,
                "setupScore": 4.0,
                "signalChat": "ETH-USD is a BUY setup on the latest live candle.",
                "close": 3320.0,
                "marketState": {
                    "label": "trend_up",
                    "code": 2,
                    "trendScore": 0.02,
                    "volatilityRatio": 1.02,
                    "isTrending": True,
                    "isHighVolatility": False,
                },
                "eventContext": {
                    "hasEventNext7d": False,
                },
            },
        ],
    )
    live_snapshot.update(
        {
            "mode": "live",
            "marketDataSource": "coinbaseExchangeRest",
            "requestMode": "configured-watchlist",
            "requestedProducts": ["BTC-USD", "ETH-USD"],
            "productsCovered": 2,
            "featureRowsScored": 40,
            "granularitySeconds": 3600,
            "liveSignalCacheSeconds": 60,
            "modelPath": "models/randomForestSignalModel.pkl",
        }
    )

    class FakeLiveSignalEngine:
        def get_status(self) -> dict[str, object]:
            return {
                "status": "ready",
                "modelPath": "models/randomForestSignalModel.pkl",
                "cacheAgeSeconds": 0,
                "cacheTtlSeconds": 60,
                "lastGeneratedAt": live_snapshot["generatedAt"],
            }

        def get_live_snapshot(self, force_refresh: bool = False, product_id: str | None = None) -> dict[str, object]:
            del force_refresh
            del product_id
            return live_snapshot

    portfolio_store = TradingPortfolioStore(
        db_path=tmp_path / "traderPortfolio.sqlite3",
        default_capital=5000.0,
    )

    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=tmp_path / "models",
            live_signal_engine=FakeLiveSignalEngine(),
            portfolio_store=portfolio_store,
        )
    )

    capital_response = client.post("/api/trader/capital", json={"capital": 12000.0})
    position_response = client.post(
        "/api/trader/positions",
        json={
            "productId": "BTC-USD",
            "quantity": 0.15,
            "entryPrice": 60000.0,
            "currentPrice": 65250.0,
            "positionFraction": 0.10,
        },
    )
    portfolio_response = client.get("/api/trader/portfolio")
    plan_response = client.get("/api/trader/plan?force_refresh=true")
    delete_response = client.delete("/api/trader/positions/BTC-USD")

    assert capital_response.status_code == 200
    assert capital_response.json()["capital"] == 12000.0
    assert position_response.status_code == 200
    assert position_response.json()["position"]["productId"] == "BTC-USD"
    assert portfolio_response.status_code == 200
    assert portfolio_response.json()["positionCount"] == 1
    assert portfolio_response.json()["storageBackend"] == "sqlite"
    assert plan_response.status_code == 200
    assert plan_response.json()["portfolio"]["positionCount"] == 1
    assert plan_response.json()["traderPlan"]["plan"]["exitCount"] == 1
    assert plan_response.json()["traderPlan"]["signals"][0]["brain"]["decision"] == "exit_position"
    assert delete_response.status_code == 200
    assert delete_response.json()["portfolio"]["positionCount"] == 0


def test_engine_api_records_executions_and_serves_trader_journal(tmp_path: Path) -> None:
    """Executed fills should update portfolio state, realized PnL, and the journal feed."""

    portfolio_store = TradingPortfolioStore(
        db_path=tmp_path / "traderPortfolio.sqlite3",
        default_capital=10000.0,
    )

    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=tmp_path / "models",
            portfolio_store=portfolio_store,
        )
    )

    capital_response = client.post("/api/trader/capital", json={"capital": 10000.0})
    buy_response = client.post(
        "/api/trader/executions",
        json={
            "productId": "BTC-USD",
            "side": "buy",
            "quantity": 0.10,
            "price": 50000.0,
            "fee": 10.0,
            "currentPrice": 51000.0,
            "metadata": {"source": "test-buy"},
        },
    )
    sell_response = client.post(
        "/api/trader/executions",
        json={
            "productId": "BTC-USD",
            "side": "sell",
            "quantity": 0.05,
            "price": 52000.0,
            "fee": 5.0,
            "currentPrice": 52000.0,
            "metadata": {"source": "test-sell"},
        },
    )
    journal_response = client.get("/api/trader/journal?limit=10")
    portfolio_response = client.get("/api/trader/portfolio")

    assert capital_response.status_code == 200
    assert buy_response.status_code == 200
    assert buy_response.json()["position"]["quantity"] == 0.10
    assert sell_response.status_code == 200
    assert sell_response.json()["execution"]["side"] == "sell"
    assert sell_response.json()["execution"]["realizedPnl"] == 90.0
    assert sell_response.json()["position"]["quantity"] == 0.05
    assert journal_response.status_code == 200
    assert journal_response.json()["count"] == 2
    assert journal_response.json()["executions"][0]["side"] == "sell"
    assert journal_response.json()["executions"][1]["side"] == "buy"
    assert portfolio_response.status_code == 200
    assert portfolio_response.json()["positionCount"] == 1
    assert portfolio_response.json()["performance"]["executionCount"] == 2
    assert portfolio_response.json()["performance"]["sellCount"] == 1
    assert portfolio_response.json()["performance"]["winningSellCount"] == 1
    assert portfolio_response.json()["performance"]["realizedPnl"] == 90.0


def test_engine_api_tracks_trade_records_and_closed_outcomes(tmp_path: Path) -> None:
    """Tracked trades should preserve entry/target prices and record losing outcomes cleanly."""

    portfolio_store = TradingPortfolioStore(
        db_path=tmp_path / "traderPortfolio.sqlite3",
        default_capital=10000.0,
    )

    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=tmp_path / "models",
            portfolio_store=portfolio_store,
        )
    )

    create_response = client.post(
        "/api/trader/trades",
        json={
            "productId": "SOL-USD",
            "entryPrice": 100.0,
            "takeProfitPrice": 112.0,
            "stopLossPrice": 95.0,
            "quantity": 2.0,
            "status": "open",
            "signalName": "BUY",
            "metadata": {"source": "manual-test"},
        },
    )
    trade_id = create_response.json()["trade"]["tradeId"]
    close_response = client.post(
        f"/api/trader/trades/{trade_id}/close",
        json={
            "exitPrice": 92.0,
            "closeReason": "stop_loss_hit",
        },
    )
    list_response = client.get("/api/trader/trades?limit=10")
    detail_response = client.get(f"/api/trader/trades/{trade_id}")
    portfolio_response = client.get("/api/trader/portfolio")

    assert create_response.status_code == 200
    assert create_response.json()["trade"]["entryPrice"] == 100.0
    assert create_response.json()["trade"]["takeProfitPrice"] == 112.0
    assert create_response.json()["trade"]["stopLossPrice"] == 95.0
    assert close_response.status_code == 200
    assert close_response.json()["trade"]["status"] == "closed"
    assert close_response.json()["trade"]["outcome"] == "loss"
    assert close_response.json()["trade"]["realizedPnl"] == -16.0
    assert round(close_response.json()["trade"]["realizedReturn"], 4) == -0.08
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1
    assert detail_response.status_code == 200
    assert detail_response.json()["closeReason"] == "stop_loss_hit"
    assert portfolio_response.status_code == 200
    assert portfolio_response.json()["trackedTrades"]["tradeCount"] == 1
    assert portfolio_response.json()["trackedTrades"]["closedTradeCount"] == 1
    assert portfolio_response.json()["trackedTrades"]["losingTradeCount"] == 1
    assert portfolio_response.json()["trackedTrades"]["closedRealizedPnl"] == -16.0


def test_engine_api_can_track_trade_directly_from_cached_signal(tmp_path: Path) -> None:
    """The engine should be able to promote a signal into a tracked trade record."""

    snapshot_path = tmp_path / "outputs" / "frontendSignalSnapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    signal_row = {
        "productId": "ADA-USD",
        "signal_name": "BUY",
        "spotAction": "buy",
        "confidence": 0.78,
        "setupScore": 4.4,
        "signalChat": "ADA-USD is a BUY setup.",
        "symbol": "ADA",
        "close": 0.50,
        "timestamp": "2026-04-05T10:00:00+00:00",
        "brain": {
            "decision": "enter_long",
            "summaryLine": "Trend and confidence support a new entry.",
            "stopLossPrice": 0.47,
            "takeProfitPrice": 0.56,
        },
        "marketState": {
            "label": "trend_up",
            "code": 2,
            "trendScore": 0.03,
            "volatilityRatio": 1.10,
            "isTrending": True,
            "isHighVolatility": False,
        },
    }
    snapshot_payload = build_frontend_signal_snapshot(
        model_type="randomForestSignalModel",
        primary_signal=signal_row,
        latest_signals=[signal_row],
        actionable_signals=[signal_row],
    )
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    portfolio_store = TradingPortfolioStore(
        db_path=tmp_path / "traderPortfolio.sqlite3",
        default_capital=10000.0,
    )
    client = TestClient(
        create_app(
            snapshot_path=snapshot_path,
            model_dir=tmp_path / "models",
            portfolio_store=portfolio_store,
        )
    )

    track_response = client.post(
        "/api/trader/trades/from-signal/ADA-USD",
        json={
            "quantity": 1000.0,
            "status": "planned",
        },
    )

    assert track_response.status_code == 200
    assert track_response.json()["trade"]["productId"] == "ADA-USD"
    assert track_response.json()["trade"]["entryPrice"] == 0.50
    assert track_response.json()["trade"]["takeProfitPrice"] == 0.56
    assert track_response.json()["trade"]["stopLossPrice"] == 0.47
    assert track_response.json()["trade"]["status"] == "planned"
    assert track_response.json()["trade"]["metadata"]["trackedFromSignal"] is True
    assert track_response.json()["trade"]["metadata"]["brainDecision"] == "enter_long"


def test_engine_api_exposes_tradingview_chart_and_event_endpoints(tmp_path: Path) -> None:
    """The engine should expose TradingView-style history plus CoinMarketCal event endpoints."""

    data_path = tmp_path / "marketPrices.csv"
    _build_chart_history_frame().to_csv(data_path, index=False)

    events_path = tmp_path / "coinMarketCalEvents.csv"
    pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "event_title": "Mainnet upgrade",
                "event_category": "protocol",
                "event_start": "2026-03-01T04:00:00Z",
                "base_currency": "BTC",
            }
        ]
    ).to_csv(events_path, index=False)

    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=tmp_path / "models",
            config=TrainingConfig(
                data_file=data_path,
                market_data_source="coinmarketcap",
                coinmarketcap_use_context=False,
                coinmarketcal_events_file=events_path,
                coinmarketcap_fetch_all_quote_products=False,
                coinmarketcap_product_ids=("BTC-USD", "ETH-USD"),
                live_product_ids=("BTC-USD", "ETH-USD"),
            ),
        )
    )

    config_response = client.get("/api/tradingview/config")
    search_response = client.get("/api/tradingview/search?query=BTC")
    symbol_response = client.get("/api/tradingview/symbols?symbol=BTC-USD")
    history_response = client.get(
        "/api/tradingview/history",
        params={
            "symbol": "BTC-USD",
            "resolution": "60",
            "from": int(pd.Timestamp("2026-03-01T00:00:00Z").timestamp()),
            "to": int(pd.Timestamp("2026-03-01T08:00:00Z").timestamp()),
        },
    )
    quotes_response = client.get("/api/tradingview/quotes?symbols=BTC-USD,ETH-USD")
    marks_response = client.get(
        "/api/tradingview/marks",
        params={
            "symbol": "BTC-USD",
            "from": int(pd.Timestamp("2026-03-01T00:00:00Z").timestamp()),
            "to": int(pd.Timestamp("2026-03-02T00:00:00Z").timestamp()),
        },
    )
    timescale_marks_response = client.get(
        "/api/tradingview/timescale_marks",
        params={
            "symbol": "BTC-USD",
            "from": int(pd.Timestamp("2026-03-01T00:00:00Z").timestamp()),
            "to": int(pd.Timestamp("2026-03-02T00:00:00Z").timestamp()),
        },
    )
    events_response = client.get("/api/events?symbol=BTC-USD&limit=10")

    assert config_response.status_code == 200
    assert "60" in config_response.json()["supported_resolutions"]
    assert search_response.status_code == 200
    assert search_response.json()[0]["symbol"] == "BTC-USD"
    assert symbol_response.status_code == 200
    assert symbol_response.json()["exchange"] == "CMC"
    assert history_response.status_code == 200
    assert history_response.json()["s"] == "ok"
    assert len(history_response.json()["t"]) == 8
    assert quotes_response.status_code == 200
    assert quotes_response.json()["s"] == "ok"
    assert len(quotes_response.json()["d"]) == 2
    assert marks_response.status_code == 200
    assert marks_response.json()["id"] == ["evt-1"]
    assert timescale_marks_response.status_code == 200
    assert timescale_marks_response.json()[0]["id"] == "evt-1"
    assert events_response.status_code == 200
    assert events_response.json()["count"] == 1
    assert events_response.json()["events"][0]["productId"] == "BTC-USD"


def test_engine_api_supports_assistant_chat_endpoints(tmp_path: Path) -> None:
    """The engine should expose assistant session and message routes."""

    class FakeLiveSignalEngine:
        def get_status(self) -> dict[str, object]:
            return {
                "status": "ready",
                "modelPath": "models/randomForestSignalModel.pkl",
                "cacheAgeSeconds": 0,
                "cacheTtlSeconds": 60,
                "lastGeneratedAt": "2026-03-29T10:00:00+00:00",
            }

    class FakeAssistantService:
        def create_session(self, title: str | None = None) -> dict[str, object]:
            return {
                "session": {
                    "sessionId": "session-1",
                    "title": title or "Crypto Signal Copilot",
                    "createdAt": "2026-03-29T10:00:00+00:00",
                    "updatedAt": "2026-03-29T10:00:00+00:00",
                    "messageCount": 1,
                },
                "messages": [
                    {
                        "messageId": 1,
                        "sessionId": "session-1",
                        "role": "assistant",
                        "content": "Crypto Signal Copilot is ready.",
                        "createdAt": "2026-03-29T10:00:00+00:00",
                        "metadata": {"type": "welcome"},
                    }
                ],
            }

        def get_session_state(self, session_id: str) -> dict[str, object]:
            return {
                "session": {
                    "sessionId": session_id,
                    "title": "Crypto Signal Copilot",
                    "createdAt": "2026-03-29T10:00:00+00:00",
                    "updatedAt": "2026-03-29T10:01:00+00:00",
                    "messageCount": 2,
                },
                "messages": [
                    {
                        "messageId": 1,
                        "sessionId": session_id,
                        "role": "assistant",
                        "content": "Crypto Signal Copilot is ready.",
                        "createdAt": "2026-03-29T10:00:00+00:00",
                        "metadata": {"type": "welcome"},
                    }
                ],
            }

        def answer_question(
            self,
            session_id: str,
            question: str,
            product_id: str | None = None,
            force_refresh: bool = False,
        ) -> dict[str, object]:
            return {
                "session": {
                    "sessionId": session_id,
                    "title": "Crypto Signal Copilot",
                    "createdAt": "2026-03-29T10:00:00+00:00",
                    "updatedAt": "2026-03-29T10:02:00+00:00",
                    "messageCount": 3,
                },
                "userMessage": {
                    "messageId": 2,
                    "sessionId": session_id,
                    "role": "user",
                    "content": question,
                    "createdAt": "2026-03-29T10:01:00+00:00",
                    "metadata": {"productId": product_id},
                },
                "assistantMessage": {
                    "messageId": 3,
                    "sessionId": session_id,
                    "role": "assistant",
                    "content": "BTC-USD is currently a BUY setup from the live signal engine.",
                    "createdAt": "2026-03-29T10:02:00+00:00",
                    "metadata": {"liveSource": "live"},
                },
                "messages": [
                    {
                        "messageId": 1,
                        "sessionId": session_id,
                        "role": "assistant",
                        "content": "Crypto Signal Copilot is ready.",
                        "createdAt": "2026-03-29T10:00:00+00:00",
                        "metadata": {"type": "welcome"},
                    },
                    {
                        "messageId": 2,
                        "sessionId": session_id,
                        "role": "user",
                        "content": question,
                        "createdAt": "2026-03-29T10:01:00+00:00",
                        "metadata": {"productId": product_id},
                    },
                    {
                        "messageId": 3,
                        "sessionId": session_id,
                        "role": "assistant",
                        "content": "BTC-USD is currently a BUY setup from the live signal engine.",
                        "createdAt": "2026-03-29T10:02:00+00:00",
                        "metadata": {"liveSource": "live"},
                    },
                ],
                "liveContext": {
                    "source": "live",
                    "error": "",
                    "snapshot": _build_live_snapshot_payload(),
                },
                "retrieval": {
                    "signals": [],
                    "messages": [],
                },
            }

    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=tmp_path / "models",
            live_signal_engine=FakeLiveSignalEngine(),
            assistant_service=FakeAssistantService(),
        )
    )

    session_response = client.post("/api/chat/sessions", json={})
    message_response = client.post(
        "/api/chat/sessions/session-1/messages",
        json={
            "message": "What is the latest BTC read?",
            "productId": "BTC-USD",
            "forceRefresh": True,
        },
    )

    assert session_response.status_code == 200
    assert session_response.json()["session"]["sessionId"] == "session-1"
    assert message_response.status_code == 200
    assert message_response.json()["assistantMessage"]["role"] == "assistant"
    assert "BUY setup" in message_response.json()["assistantMessage"]["content"]
    assert message_response.json()["liveContext"]["source"] == "live"


def test_engine_api_exposes_rag_ingest_and_search_endpoints(tmp_path: Path) -> None:
    """The engine should let users ingest and search external knowledge sources."""

    knowledge_store = RagKnowledgeStore(db_path=tmp_path / "assistantKnowledge.sqlite3")

    client = TestClient(
        create_app(
            snapshot_path=tmp_path / "outputs" / "frontendSignalSnapshot.json",
            model_dir=tmp_path / "models",
            knowledge_store=knowledge_store,
        )
    )

    ingest_response = client.post(
        "/api/rag/documents/text",
        json={
            "title": "Ethereum staking memo",
            "content": (
                "Ethereum staking yield and network activity often shape medium-term market context. "
                "Validator participation and fee burn can matter for sentiment."
            ),
            "sourceUri": "internal://eth-staking-memo",
        },
    )
    status_response = client.get("/api/rag/status")
    sources_response = client.get("/api/rag/sources?limit=10")
    search_response = client.post(
        "/api/rag/search",
        json={
            "query": "ethereum staking yield",
            "limit": 5,
        },
    )

    assert ingest_response.status_code == 200
    assert ingest_response.json()["source"]["title"] == "Ethereum staking memo"
    assert status_response.status_code == 200
    assert status_response.json()["storageBackend"] == "sqlite"
    assert status_response.json()["sourceCount"] == 1
    assert sources_response.status_code == 200
    assert sources_response.json()["count"] == 1
    assert search_response.status_code == 200
    assert search_response.json()["count"] >= 1
    assert search_response.json()["results"][0]["title"] == "Ethereum staking memo"
