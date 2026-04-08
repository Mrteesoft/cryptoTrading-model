"""Trend aggregation for news and entity momentum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .features import NewsFeaturePayload


@dataclass(frozen=True)
class TrendFeaturePayload:
    """Structured trend features derived from news signals."""

    topic_trend_score: float
    trend_persistence_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "topicTrendScore": float(self.topic_trend_score),
            "trendPersistenceScore": float(self.trend_persistence_score),
        }


def build_trend_features(news_features: NewsFeaturePayload) -> TrendFeaturePayload:
    acceleration = news_features.entity_mention_acceleration
    topic_trend_score = min(max((acceleration - 1.0) * 0.6, -1.0), 1.0)

    persistence = 0.5
    if news_features.coin_news_count_1h >= 2 and news_features.news_sentiment_4h > 0:
        persistence = 0.75
    elif news_features.coin_news_count_1h == 0:
        persistence = 0.25

    return TrendFeaturePayload(
        topic_trend_score=float(topic_trend_score),
        trend_persistence_score=float(persistence),
    )
