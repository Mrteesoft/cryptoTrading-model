"""News and trend enrichment helpers."""

from .entities import extract_entities
from .features import build_news_features
from .providers import LocalNewsProvider, NewsItem
from .scoring import score_sentiment, score_news_relevance
from .trends import build_trend_features

__all__ = [
    "LocalNewsProvider",
    "NewsItem",
    "build_news_features",
    "build_trend_features",
    "extract_entities",
    "score_sentiment",
    "score_news_relevance",
]
