"""Small orchestrator for candidate generation plus policy gating."""

from __future__ import annotations

from typing import Any, Mapping

from ..config import TrainingConfig
from .candidate_discovery import discover_raw_signal_candidate
from .chart_confirmation import apply_chart_confirmation
from .contracts import GatedSignalCandidate
from .risk_gate import apply_risk_gate


def gate_prediction_row(
    *,
    signal_row: Mapping[str, Any],
    config: TrainingConfig | None,
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
) -> GatedSignalCandidate:
    """Run discovery -> chart confirmation -> risk gating for one prediction row."""

    raw_candidate = discover_raw_signal_candidate(
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
    chart_confirmed_candidate = apply_chart_confirmation(raw_candidate, config=config)
    return apply_risk_gate(chart_confirmed_candidate, config=config)
