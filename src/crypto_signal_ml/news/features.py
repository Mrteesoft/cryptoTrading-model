"""News feature builders for signal enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..config import TrainingConfig
from .entities import extract_entities
from .providers import LocalNewsProvider, NewsItem
from .scoring import score_news_relevance, score_sentiment


@dataclass(frozen=True)
class NewsFeaturePayload:
    """Structured news features attached to a signal."""

    news_sentiment_1h: float
    news_sentiment_4h: float
    news_sentiment_delta: float
    coin_news_count_1h: int
    market_news_count_1h: int
    coin_specific_news_score: float
    market_wide_news_score: float
    news_novelty_score: float
    news_relevance_score: float
    entity_mention_acceleration: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "newsSentiment1h": float(self.news_sentiment_1h),
            "newsSentiment4h": float(self.news_sentiment_4h),
            "newsSentimentDelta": float(self.news_sentiment_delta),
            "coinNewsCount1h": int(self.coin_news_count_1h),
            "marketNewsCount1h": int(self.market_news_count_1h),
            "coinSpecificNewsScore": float(self.coin_specific_news_score),
            "marketWideNewsScore": float(self.market_wide_news_score),
            "newsNoveltyScore": float(self.news_novelty_score),
            "newsRelevanceScore": float(self.news_relevance_score),
            "entityMentionAcceleration": float(self.entity_mention_acceleration),
        }


def _aggregate_sentiment(items: list[NewsItem]) -> float:
    if not items:
        return 0.0
    sentiment_scores = [score_sentiment(f"{item.title} {item.summary}") for item in items]
    return float(sum(sentiment_scores) / max(len(sentiment_scores), 1))


def build_news_features(
    *,
    product_id: str,
    base_currency: str,
    provider: LocalNewsProvider,
    config: TrainingConfig,
    now: datetime | None = None,
) -> NewsFeaturePayload:
    now = now or datetime.now(timezone.utc)
    known_entities = [base_currency, product_id.split("-")[0] if product_id else base_currency]

    recent_1h = list(provider.iter_recent(60.0, now=now))
    recent_4h = list(provider.iter_recent(240.0, now=now))

    def filter_coin(items: list[NewsItem]) -> list[NewsItem]:
        coin_items: list[NewsItem] = []
        for item in items:
            entities = item.entities or extract_entities(f"{item.title} {item.summary}", known_entities)
            if score_news_relevance(entities, base_currency) > 0:
                coin_items.append(item)
        return coin_items

    coin_recent_1h = filter_coin(recent_1h)
    coin_recent_4h = filter_coin(recent_4h)

    sentiment_1h = _aggregate_sentiment(coin_recent_1h or recent_1h)
    sentiment_4h = _aggregate_sentiment(coin_recent_4h or recent_4h)
    sentiment_delta = sentiment_1h - sentiment_4h

    coin_score = sentiment_1h * (1.0 + min(len(coin_recent_1h), 5) / 5.0)
    market_score = _aggregate_sentiment(recent_1h) * (1.0 + min(len(recent_1h), 8) / 8.0)

    unique_titles = {item.title for item in recent_4h if item.title}
    novelty_score = min(len(unique_titles) / 12.0, 1.0)

    relevance_score = min(len(coin_recent_4h) / max(len(recent_4h), 1), 1.0)
    acceleration = len(coin_recent_1h) / max(len(coin_recent_4h), 1)

    return NewsFeaturePayload(
        news_sentiment_1h=float(sentiment_1h),
        news_sentiment_4h=float(sentiment_4h),
        news_sentiment_delta=float(sentiment_delta),
        coin_news_count_1h=int(len(coin_recent_1h)),
        market_news_count_1h=int(len(recent_1h)),
        coin_specific_news_score=float(coin_score),
        market_wide_news_score=float(market_score),
        news_novelty_score=float(novelty_score),
        news_relevance_score=float(relevance_score),
        entity_mention_acceleration=float(acceleration),
    )
