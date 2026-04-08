"""Event scoring helpers."""

from __future__ import annotations


IMPACT_KEYWORDS = {
    "mainnet": 0.75,
    "upgrade": 0.60,
    "hard fork": 0.70,
    "airdrop": 0.55,
    "listing": 0.50,
    "token unlock": 0.65,
    "testnet": 0.40,
    "ama": 0.25,
    "partnership": 0.45,
    "cpi": 0.80,
    "fomc": 0.90,
    "rate decision": 0.90,
}


def score_event_impact(title: str, category: str) -> float:
    """Return a 0-1 impact score for one event entry."""

    normalized_title = str(title or "").lower()
    normalized_category = str(category or "").lower()

    score = 0.35 if normalized_title else 0.0
    for keyword, weight in IMPACT_KEYWORDS.items():
        if keyword in normalized_title:
            score = max(score, weight)

    if "macro" in normalized_category or "regulation" in normalized_category:
        score = max(score, 0.70)

    return min(max(score, 0.0), 1.0)
