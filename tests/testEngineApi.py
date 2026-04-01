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
from crypto_signal_ml.engine_api import create_app  # noqa: E402
from crypto_signal_ml.frontend import build_frontend_signal_snapshot  # noqa: E402
from crypto_signal_ml.modeling import RandomForestSignalModel  # noqa: E402
from crypto_signal_ml.rag import RagKnowledgeStore  # noqa: E402


def _build_trained_model(model_path: Path) -> None:
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
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            n_estimators=12,
            max_depth=3,
            random_state=7,
        ),
        feature_columns=feature_columns,
    )
    model.fit(training_df)
    model.save(model_path)


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
    assert signals_response.status_code == 200
    assert signals_response.json()["count"] == 2


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
    assert status_response.json()["sourceCount"] == 1
    assert sources_response.status_code == 200
    assert sources_response.json()["count"] == 1
    assert search_response.status_code == 200
    assert search_response.json()["count"] >= 1
    assert search_response.json()["results"][0]["title"] == "Ethereum staking memo"
