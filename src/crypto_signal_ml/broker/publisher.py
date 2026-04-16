"""Kafka event publishing helpers."""

from __future__ import annotations

import json
import logging

from kafka import KafkaProducer

from ..config import TrainingConfig
from .schemas import EventEnvelope, kafka_topic


LOGGER = logging.getLogger(__name__)


class KafkaEventPublisher:
    """Publish broker events when Kafka is configured."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._producer: KafkaProducer | None = None

    def is_configured(self) -> bool:
        return bool(self.config.kafka_brokers)

    def _get_producer(self) -> KafkaProducer | None:
        if not self.is_configured():
            return None

        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=list(self.config.kafka_brokers),
                value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            )
        return self._producer

    def publish(self, topic_name: str, envelope: EventEnvelope) -> None:
        producer = self._get_producer()
        if producer is None:
            LOGGER.info("Kafka publish skipped because no brokers are configured: %s", topic_name)
            return

        topic = kafka_topic(self.config.kafka_topic_prefix, topic_name)
        producer.send(topic, envelope.to_dict())
        producer.flush()
