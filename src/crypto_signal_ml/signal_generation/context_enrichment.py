"""Typed context-enrichment stage for events, news, and trends."""

from __future__ import annotations

from ..config import TrainingConfig
from ..events import EventCalendar, build_event_features
from ..news import LocalNewsProvider, build_news_features, build_trend_features
from .contracts import ContextEnrichedCandidate, GatedSignalCandidate


class SignalContextEnricher:
    """Attach event, news, and trend evidence to typed candidates."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.event_calendar = EventCalendar(config=config)
        self.news_provider = LocalNewsProvider(config=config)

    def enrich_candidate(self, candidate: GatedSignalCandidate) -> ContextEnrichedCandidate:
        """Attach fresh event, news, and trend context to one gated candidate."""

        product_id = str(candidate.productId).strip().upper()
        base_currency = str(candidate.baseCurrency or "").strip().upper()
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

        event_context = dict(candidate.eventContext)
        event_context.update(event_features.to_dict())
        news_context = news_features.to_dict()
        trend_context = trend_features.to_dict()

        candidate_payload = dict(candidate.__dict__)
        candidate_payload.update(
            {
                "eventContext": event_context,
                "newsContext": news_context,
                "trendContext": trend_context,
                "contextEvidence": {
                    "contextSource": "signal-context-enricher-v1",
                    "hasUpcomingEvent": bool(event_context.get("hasEventNext7d", False)),
                    "newsSentiment1h": float(news_context.get("newsSentiment1h", 0.0) or 0.0),
                    "topicTrendScore": float(trend_context.get("topicTrendScore", 0.0) or 0.0),
                },
            }
        )
        return ContextEnrichedCandidate(**candidate_payload)

    def enrich_candidates(
        self,
        candidates: list[GatedSignalCandidate],
    ) -> tuple[ContextEnrichedCandidate, ...]:
        """Attach context to a full batch of candidates."""

        return tuple(self.enrich_candidate(candidate) for candidate in candidates)
