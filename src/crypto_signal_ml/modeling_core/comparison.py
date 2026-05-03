"""Comparison helpers built on top of candidate selection rules."""

from __future__ import annotations

from typing import Any

from .selection import rank_model_candidates


def rank_model_comparison_rows(
    candidate_rows: list[dict[str, Any]],
    *,
    minimum_trade_count: int,
) -> list[dict[str, Any]]:
    """Rank comparison rows using the production selection policy."""

    return rank_model_candidates(candidate_rows, minimum_trade_count=minimum_trade_count)
