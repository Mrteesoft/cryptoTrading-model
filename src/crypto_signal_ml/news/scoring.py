"""Scoring helpers for news sentiment and relevance."""

from __future__ import annotations


POSITIVE_TERMS = {
    "partnership",
    "launch",
    "upgrade",
    "mainnet",
    "adoption",
    "record",
    "breakout",
    "wins",
    "surge",
    "growth",
    "bullish",
}

NEGATIVE_TERMS = {
    "hack",
    "exploit",
    "lawsuit",
    "regulation",
    "down",
    "bearish",
    "collapse",
    "delay",
    "outage",
    "investigation",
}


def score_sentiment(text: str) -> float:
    """Compute a simple sentiment score in [-1, 1]."""

    normalized = str(text or "").lower()
    score = 0
    for term in POSITIVE_TERMS:
        if term in normalized:
            score += 1
    for term in NEGATIVE_TERMS:
        if term in normalized:
            score -= 1
    if score == 0:
        return 0.0
    return max(min(score / 3.0, 1.0), -1.0)


def score_news_relevance(entities: list[str], target_entity: str) -> float:
    """Return a relevance score for a target entity mention."""

    if not target_entity:
        return 0.0
    target = target_entity.upper()
    return 1.0 if target in [entity.upper() for entity in entities] else 0.0
