"""Context enrichment stage for events, news, and trends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import TrainingConfig
from ..signal_generation import (
    SignalContextEnricher,
    context_enriched_candidate_to_summary,
    gated_candidate_from_summary,
)


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
        self.enricher = SignalContextEnricher(config=config)

    def enrich(
        self,
        signal_summaries: list[dict[str, Any]],
    ) -> SignalContextEnrichmentArtifacts:
        event_context_by_product: dict[str, dict[str, Any]] = {}
        news_context_by_product: dict[str, dict[str, Any]] = {}
        trend_context_by_product: dict[str, dict[str, Any]] = {}

        enriched_candidates = self.enricher.enrich_candidates(
            [gated_candidate_from_summary(signal_summary) for signal_summary in signal_summaries]
        )
        enriched_signals: list[dict[str, Any]] = []
        for enriched_candidate in enriched_candidates:
            product_id = str(enriched_candidate.productId).strip().upper()
            if product_id:
                event_context_by_product[product_id] = dict(enriched_candidate.eventContext)
                news_context_by_product[product_id] = dict(enriched_candidate.newsContext)
                trend_context_by_product[product_id] = dict(enriched_candidate.trendContext)

            enriched_signals.append(context_enriched_candidate_to_summary(enriched_candidate))

        return SignalContextEnrichmentArtifacts(
            signal_summaries=enriched_signals,
            event_context_by_product=event_context_by_product,
            news_context_by_product=news_context_by_product,
            trend_context_by_product=trend_context_by_product,
        )
