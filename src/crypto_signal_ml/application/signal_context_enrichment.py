"""Context enrichment stage for events, news, and trends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import TrainingConfig
from ..events import EventCalendar, build_event_features
from ..news import LocalNewsProvider, build_news_features, build_trend_features


@dataclass
class SignalContextEnrichmentArtifacts:
    """Contextual outputs built before portfolio enrichment."""

    signal_summaries: list[dict[str, Any]]
    event_context_by_product: dict[str, dict[str, Any]]
    news_context_by_product: dict[str, dict[str, Any]]
    trend_context_by_product: dict[str, dict[str, Any]]


class SignalContextEnrichmentStage:
    """Attach event, news, and trend context to signals."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.event_calendar = EventCalendar(config=config)
        self.news_provider = LocalNewsProvider(config=config)

    def enrich(
        self,
        signal_summaries: list[dict[str, Any]],
    ) -> SignalContextEnrichmentArtifacts:
        event_context_by_product: dict[str, dict[str, Any]] = {}
        news_context_by_product: dict[str, dict[str, Any]] = {}
        trend_context_by_product: dict[str, dict[str, Any]] = {}

        enriched_signals: list[dict[str, Any]] = []
        for signal_summary in signal_summaries:
            enriched = dict(signal_summary)
            product_id = str(enriched.get("productId", "")).strip().upper()
            base_currency = str(enriched.get("baseCurrency", "")).strip().upper()
            if not base_currency and product_id:
                base_currency = product_id.split("-")[0].upper()

            event_features = build_event_features(
                base_currency=base_currency,
                calendar=self.event_calendar,
                config=self.config,
            )
            news_features = build_news_features(
                product_id=product_id,
                base_currency=base_currency,
                provider=self.news_provider,
                config=self.config,
            )
            trend_features = build_trend_features(news_features)

            event_context = dict(enriched.get("eventContext") or {})
            event_context.update(event_features.to_dict())
            enriched["eventContext"] = event_context
            enriched["newsContext"] = news_features.to_dict()
            enriched["trendContext"] = trend_features.to_dict()

            if product_id:
                event_context_by_product[product_id] = event_context
                news_context_by_product[product_id] = news_features.to_dict()
                trend_context_by_product[product_id] = trend_features.to_dict()

            enriched_signals.append(enriched)

        return SignalContextEnrichmentArtifacts(
            signal_summaries=enriched_signals,
            event_context_by_product=event_context_by_product,
            news_context_by_product=news_context_by_product,
            trend_context_by_product=trend_context_by_product,
        )
