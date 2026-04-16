"""Redis-backed shared state for async jobs."""

from __future__ import annotations

import json
from typing import Any

from redis import Redis

from ..config import TrainingConfig


def _job_state_key(job_id: str) -> str:
    return f"job:{job_id}:state"


class BrokerStateStore:
    """Persist shared job state so backend and workers can observe the same record."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.redis = Redis.from_url(config.broker_redis_url) if config.broker_redis_url else None

    def is_configured(self) -> bool:
        return self.redis is not None

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        if self.redis is None:
            return None

        raw_value = self.redis.get(_job_state_key(job_id))
        if raw_value is None:
            return None

        return json.loads(raw_value)

    def set_job(self, job_id: str, payload: dict[str, Any]) -> None:
        if self.redis is None:
            return

        self.redis.setex(
            _job_state_key(job_id),
            int(self.config.job_state_ttl_seconds),
            json.dumps(payload),
        )

    def merge_job(self, job_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        current_payload = self.get_job(job_id) or {}
        merged_payload = {**current_payload, **patch}
        self.set_job(job_id, merged_payload)
        return merged_payload
