"""Explicit candidate-discovery naming over the raw model-output stage."""

from __future__ import annotations

from typing import Any, Mapping

from .candidate_generation import build_raw_signal_candidate
from .contracts import RawSignalCandidate


def discover_raw_signal_candidate(
    *,
    signal_row: Mapping[str, Any],
    minimum_action_confidence: float,
    setup_score: float,
    symbol: str,
    pair_symbol: str,
    base_currency: str | None,
    quote_currency: str | None,
    chart_context: dict[str, Any],
    execution_context: dict[str, Any],
    market_context: dict[str, Any],
    market_state: dict[str, Any],
    event_context: dict[str, Any],
) -> RawSignalCandidate:
    """Build one discovery-stage candidate directly from model outputs."""

    return build_raw_signal_candidate(
        signal_row=signal_row,
        minimum_action_confidence=minimum_action_confidence,
        setup_score=setup_score,
        symbol=symbol,
        pair_symbol=pair_symbol,
        base_currency=base_currency,
        quote_currency=quote_currency,
        chart_context=chart_context,
        execution_context=execution_context,
        market_context=market_context,
        market_state=market_state,
        event_context=event_context,
    )


__all__ = ["discover_raw_signal_candidate"]
