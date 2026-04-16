"""Shared command/event envelope helpers for broker-backed workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


JobType = Literal[
    "generate_signal",
    "scan_market",
    "backtest_strategy",
    "refresh_features",
    "chat_analysis",
]

JobStatus = Literal[
    "queued",
    "processing",
    "completed",
    "failed",
    "retrying",
    "cancelled",
    "timed_out",
]

JobPriority = Literal["high", "normal", "low"]


@dataclass(frozen=True)
class CommandEnvelope:
    """One RabbitMQ command payload."""

    jobId: str
    commandType: JobType
    schemaVersion: str
    requestedAt: str
    correlationId: str
    priority: JobPriority
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly envelope."""

        return asdict(self)


@dataclass(frozen=True)
class EventEnvelope:
    """One Kafka event payload."""

    eventId: str
    eventType: str
    schemaVersion: str
    source: str
    occurredAt: str
    correlationId: str | None
    jobId: str | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly envelope."""

        return asdict(self)


@dataclass(frozen=True)
class JobRecord:
    """Redis-backed job status shape shared with the backend."""

    jobId: str
    type: JobType
    status: JobStatus
    priority: JobPriority
    correlationId: str
    createdAt: str
    updatedAt: str
    attempt: int
    progress: int = 0
    errorReason: str | None = None
    resultRef: str | None = None
    lastHeartbeatAt: str | None = None
    payload: dict[str, Any] | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly record."""

        return asdict(self)


def command_routing_key_for_job_type(job_type: JobType) -> str:
    """Return the RabbitMQ routing key for a job type."""

    return {
        "generate_signal": "job.generate-signal",
        "scan_market": "job.scan-market",
        "backtest_strategy": "job.backtest",
        "refresh_features": "job.refresh-features",
        "chat_analysis": "job.chat-analysis",
    }[job_type]


def kafka_topic(topic_prefix: str, topic_name: str) -> str:
    """Return the effective Kafka topic name."""

    return f"{topic_prefix}{topic_name}"
