"""RabbitMQ worker that runs heavy jobs and publishes Kafka events."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pika

from ..application import BacktestApp, MarketDataRefreshApp, PublishedSignalViewService, SignalGenerationApp
from ..assistant import TradingAssistantService
from ..config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, TrainingConfig
from ..engine_api import ModelArtifactStore
from ..frontend import SignalSnapshotStore
from ..live import LiveSignalEngine
from ..memory import ConversationSessionStore
from ..pipeline import CryptoDatasetBuilder
from ..rag import RagKnowledgeStore
from ..trading.portfolio import TradingPortfolioStore
from ..trading.signal_store import TradingSignalStore
from .publisher import KafkaEventPublisher
from .schemas import CommandEnvelope, EventEnvelope, JobType, command_routing_key_for_job_type
from .state_store import BrokerStateStore


LOGGER = logging.getLogger(__name__)

LIVE_SNAPSHOT_PATH = OUTPUTS_DIR / "liveSignalSnapshot.json"

QUEUE_DEFINITIONS: dict[JobType, str] = {
    "generate_signal": "signal.generate.q",
    "scan_market": "market.scan.q",
    "backtest_strategy": "backtest.run.q",
    "refresh_features": "feature.refresh.q",
    "chat_analysis": "chat.analysis.q",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


class BrokerWorkerService:
    """Consume RabbitMQ commands and run the corresponding Python workflow."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.state_store = BrokerStateStore(self.config)
        self.event_publisher = KafkaEventPublisher(self.config)

    def _configure_logging(self) -> None:
        root_logger = logging.getLogger()
        if root_logger.handlers:
            return

        logging.basicConfig(
            level="INFO",
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    def _build_model_summary_provider(self) -> Callable[[], dict[str, Any]]:
        model_store = ModelArtifactStore(model_dir=MODELS_DIR, config=self.config)
        return model_store.get_summary

    def _build_assistant_service(self) -> TradingAssistantService:
        snapshot_store = SignalSnapshotStore(snapshot_path=OUTPUTS_DIR / "frontendSignalSnapshot.json")
        live_signal_engine = LiveSignalEngine(model_dir=MODELS_DIR, config=self.config)
        session_store = ConversationSessionStore(
            db_path=self.config.assistant_store_path,
            database_url=self.config.assistant_store_url,
        )
        portfolio_store = TradingPortfolioStore(
            db_path=self.config.portfolio_store_path,
            default_capital=self.config.portfolio_default_capital,
            database_url=self.config.portfolio_store_url,
        )
        signal_store = TradingSignalStore(
            db_path=self.config.signal_store_path,
            database_url=self.config.signal_store_url,
        )
        knowledge_store = (
            RagKnowledgeStore(
                db_path=self.config.rag_store_path,
                database_url=self.config.rag_store_url,
                chunk_size_chars=self.config.rag_chunk_size_chars,
                chunk_overlap_chars=self.config.rag_chunk_overlap_chars,
                fetch_timeout_seconds=self.config.rag_fetch_timeout_seconds,
                fetch_max_chars=self.config.rag_fetch_max_chars,
            )
            if self.config.rag_enabled
            else None
        )
        published_signal_service = PublishedSignalViewService(
            signal_store=signal_store,
            snapshot_store=snapshot_store,
            cached_snapshot_provider=snapshot_store.get_snapshot,
            fallback_snapshot_provider=live_signal_engine.get_live_snapshot,
        )
        return TradingAssistantService(
            live_signal_engine=live_signal_engine,
            session_store=session_store,
            model_summary_provider=self._build_model_summary_provider(),
            cached_snapshot_provider=snapshot_store.get_snapshot,
            knowledge_store=knowledge_store,
            portfolio_store=portfolio_store,
            published_signal_service=published_signal_service,
            config=self.config,
        )

    def _publish_event(
        self,
        topic_name: str,
        *,
        event_type: str,
        correlation_id: str | None,
        job_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        self.event_publisher.publish(
            topic_name,
            EventEnvelope(
                eventId=str(uuid4()),
                eventType=event_type,
                schemaVersion="1",
                source="model-service-worker",
                occurredAt=_now_iso(),
                correlationId=correlation_id,
                jobId=job_id,
                payload=payload,
            ),
        )

    def _handle_generate_signal(self, command: CommandEnvelope) -> dict[str, Any]:
        job_config = replace(
            self.config,
            signal_refresh_market_data_before_generation=bool(
                command.payload.get("refreshMarketData", True)
            ),
        )
        results = SignalGenerationApp(config=job_config).run()
        signal_store = TradingSignalStore(
            db_path=job_config.signal_store_path,
            database_url=job_config.signal_store_url,
        )
        published_view = PublishedSignalViewService(signal_store=signal_store)
        current_signals = published_view.build_current_signals_response(action="all", limit=500)
        self._publish_event(
            "signals.generated",
            event_type="signals.generated",
            correlation_id=command.correlationId,
            job_id=command.jobId,
            payload={
                "currentSignals": current_signals,
                "summary": {
                    "signalsGenerated": int(results.get("signalsGenerated", 0)),
                    "actionableSignalsGenerated": int(results.get("actionableSignalsGenerated", 0)),
                    "frontendSignalSnapshotPath": results.get("frontendSignalSnapshotPath"),
                    "latestSignalsPath": results.get("latestSignalsPath"),
                },
            },
        )
        return {
            "status": "completed",
            "jobType": command.commandType,
            "currentSignals": current_signals,
            "summary": {
                "signalsGenerated": int(results.get("signalsGenerated", 0)),
                "actionableSignalsGenerated": int(results.get("actionableSignalsGenerated", 0)),
            },
            "resultRef": str(results.get("frontendSignalSnapshotPath") or ""),
        }

    def _handle_scan_market(self, command: CommandEnvelope) -> dict[str, Any]:
        live_signal_engine = LiveSignalEngine(model_dir=MODELS_DIR, config=self.config)
        raw_product_ids = command.payload.get("productIds")
        product_ids = list(raw_product_ids) if isinstance(raw_product_ids, list) else None
        live_snapshot = live_signal_engine.get_live_snapshot(
            force_refresh=True,
            product_id=str(command.payload.get("productId") or "").strip() or None,
            product_ids=product_ids,
        )
        if bool(command.payload.get("persistLiveSnapshot", True)):
            _write_json(LIVE_SNAPSHOT_PATH, live_snapshot)
        self._publish_event(
            "signals.updated",
            event_type="signals.updated",
            correlation_id=command.correlationId,
            job_id=command.jobId,
            payload={
                "liveSnapshot": live_snapshot,
            },
        )
        return {
            "status": "completed",
            "jobType": command.commandType,
            "liveSnapshot": live_snapshot,
            "resultRef": str(LIVE_SNAPSHOT_PATH),
        }

    def _handle_backtest(self, command: CommandEnvelope) -> dict[str, Any]:
        min_confidence = command.payload.get("minConfidence")
        job_config = (
            replace(self.config, backtest_min_confidence=float(min_confidence))
            if min_confidence is not None
            else self.config
        )
        results = BacktestApp(config=job_config).run()
        return {
            "status": "completed",
            "jobType": command.commandType,
            "summary": {
                "tradeCount": int(results.get("tradeCount", 0)),
                "strategyTotalReturn": float(results.get("strategyTotalReturn", 0.0)),
                "benchmarkTotalReturn": float(results.get("benchmarkTotalReturn", 0.0)),
                "maxDrawdown": float(results.get("maxDrawdown", 0.0)),
            },
            "resultRef": str(results.get("backtestSummaryPath") or ""),
        }

    def _handle_refresh_features(self, command: CommandEnvelope) -> dict[str, Any]:
        job_config = replace(
            self.config,
            coinmarketcap_refresh_context_on_load=bool(command.payload.get("refreshContext", True)),
            coinmarketcap_refresh_market_intelligence_on_load=bool(
                command.payload.get("refreshMarketIntelligence", True)
            ),
            coinmarketcal_refresh_events_on_load=bool(command.payload.get("refreshEvents", True)),
        )
        refresh_summary: dict[str, Any] | None = None
        if bool(command.payload.get("refreshMarketData", True)):
            refresh_summary = MarketDataRefreshApp(config=job_config).run()
        dataset_builder = CryptoDatasetBuilder(job_config)
        dataset_df, feature_columns = dataset_builder.build_labeled_dataset()
        dataset_path = PROCESSED_DATA_DIR / "marketFeaturesAndLabels.csv"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_df.to_csv(dataset_path, index=False)
        result = {
            "status": "completed",
            "jobType": command.commandType,
            "datasetPath": str(dataset_path),
            "rowCount": int(len(dataset_df)),
            "featureCount": int(len(feature_columns)),
            "marketDataRefresh": refresh_summary or {},
            "resultRef": str(dataset_path),
        }
        return result

    def _handle_chat_analysis(self, command: CommandEnvelope) -> dict[str, Any]:
        assistant_service = self._build_assistant_service()
        response = assistant_service.answer_question(
            session_id=str(command.payload.get("sessionId") or "").strip(),
            question=str(command.payload.get("message") or "").strip(),
            product_id=str(command.payload.get("productId") or "").strip() or None,
            force_refresh=bool(command.payload.get("forceRefresh", False)),
        )
        return {
            "status": "completed",
            "jobType": command.commandType,
            "chatResponse": response,
            "resultRef": str(command.payload.get("sessionId") or ""),
        }

    def _run_job(self, command: CommandEnvelope) -> dict[str, Any]:
        if command.commandType == "generate_signal":
            return self._handle_generate_signal(command)
        if command.commandType == "scan_market":
            return self._handle_scan_market(command)
        if command.commandType == "backtest_strategy":
            return self._handle_backtest(command)
        if command.commandType == "refresh_features":
            return self._handle_refresh_features(command)
        if command.commandType == "chat_analysis":
            return self._handle_chat_analysis(command)

        raise ValueError(f"Unsupported command type: {command.commandType}")

    def _mark_processing(self, command: CommandEnvelope) -> None:
        self.state_store.merge_job(
            command.jobId,
            {
                "status": "processing",
                "attempt": 1,
                "progress": 10,
                "updatedAt": _now_iso(),
                "lastHeartbeatAt": _now_iso(),
            },
        )

    def _mark_completed(self, command: CommandEnvelope, result: dict[str, Any]) -> None:
        self.state_store.merge_job(
            command.jobId,
            {
                "status": "completed",
                "result": result,
                "resultRef": result.get("resultRef"),
                "progress": 100,
                "errorReason": None,
                "updatedAt": _now_iso(),
                "lastHeartbeatAt": _now_iso(),
            },
        )
        self._publish_event(
            "jobs.completed",
            event_type="jobs.completed",
            correlation_id=command.correlationId,
            job_id=command.jobId,
            payload=result,
        )

    def _mark_failed(self, command: CommandEnvelope, error: Exception) -> None:
        error_reason = str(error)
        failure_payload = {
            "status": "failed",
            "jobType": command.commandType,
            "errorReason": error_reason,
        }
        self.state_store.merge_job(
            command.jobId,
            {
                "status": "failed",
                "errorReason": error_reason,
                "result": failure_payload,
                "updatedAt": _now_iso(),
                "lastHeartbeatAt": _now_iso(),
            },
        )
        self._publish_event(
            "jobs.failed",
            event_type="jobs.failed",
            correlation_id=command.correlationId,
            job_id=command.jobId,
            payload=failure_payload,
        )

    def _handle_delivery(
        self,
        channel: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes,
    ) -> None:
        del properties
        try:
            payload = json.loads(body.decode("utf-8"))
            command = CommandEnvelope(**payload)
            self._mark_processing(command)
            result = self._run_job(command)
            self._mark_completed(command, result)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as error:
            LOGGER.exception("Broker job failed.")
            try:
                payload = json.loads(body.decode("utf-8"))
                command = CommandEnvelope(**payload)
                self._mark_failed(command, error)
            except Exception:
                LOGGER.exception("Could not persist failed job state.")
            channel.basic_ack(delivery_tag=method.delivery_tag)

    def serve(self) -> None:
        self._configure_logging()

        if not self.config.rabbitmq_url:
            raise RuntimeError("RABBITMQ_URL must be configured before starting the broker worker.")

        connection = pika.BlockingConnection(pika.URLParameters(self.config.rabbitmq_url))
        channel = connection.channel()
        channel.exchange_declare(
            exchange=self.config.rabbitmq_command_exchange,
            exchange_type="topic",
            durable=True,
        )
        channel.basic_qos(prefetch_count=max(int(self.config.worker_prefetch_count), 1))

        for job_type, queue_name in QUEUE_DEFINITIONS.items():
            routing_key = command_routing_key_for_job_type(job_type)
            channel.queue_declare(queue=queue_name, durable=True)
            channel.queue_bind(
                exchange=self.config.rabbitmq_command_exchange,
                queue=queue_name,
                routing_key=routing_key,
            )
            channel.basic_consume(queue=queue_name, on_message_callback=self._handle_delivery)
            LOGGER.info("Bound %s to %s (%s).", queue_name, routing_key, job_type)

        LOGGER.info("Broker worker started.")
        channel.start_consuming()
