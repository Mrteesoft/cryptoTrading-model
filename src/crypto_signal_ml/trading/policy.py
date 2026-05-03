"""Risk-aware trading policy helpers layered on top of raw model predictions."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..config import TrainingConfig
from ..labels_core import signal_to_text
from ..signal_generation.candidate_generation import build_raw_signal_candidate
from ..signal_generation.chart_confirmation import apply_chart_confirmation
from ..signal_generation.risk_gate import apply_risk_gate


ACTIONABLE_SIGNAL_NAMES = {"BUY", "TAKE_PROFIT"}


def _safe_float(signal_row: Mapping[str, Any], column_name: str, default_value: float = 0.0) -> float:
    """Read one numeric field from a row-like object without raising."""

    raw_value = signal_row.get(column_name, default_value)
    if raw_value is None or pd.isna(raw_value):
        return default_value

    return float(raw_value)


def _safe_int(signal_row: Mapping[str, Any], column_name: str, default_value: int = 0) -> int:
    """Read one integer field from a row-like object without raising."""

    return int(round(_safe_float(signal_row, column_name, default_value=float(default_value))))


def _resolve_raw_signal_name(signal_row: Mapping[str, Any]) -> str:
    """Resolve the model's raw class label from explicit or numeric fields."""

    predicted_name = signal_row.get("predicted_name")
    if predicted_name is not None and not pd.isna(predicted_name):
        text_value = str(predicted_name).strip().upper()
        if text_value:
            return text_value

    return signal_to_text(_safe_int(signal_row, "predicted_signal"))


def _resolve_probability_column(signal_name: str) -> str:
    """Map one class name to its stable probability column."""

    return {
        "BUY": "prob_buy",
        "TAKE_PROFIT": "prob_take_profit",
        "HOLD": "prob_hold",
    }.get(signal_name, "prob_hold")


def _resolve_probability_margin(signal_row: Mapping[str, Any], raw_signal_name: str) -> tuple[float, float, bool]:
    """Return the raw class probability, its edge, and whether full probabilities exist."""

    has_probability_columns = all(
        column_name in signal_row
        for column_name in ("prob_buy", "prob_hold", "prob_take_profit")
    )
    if not has_probability_columns:
        fallback_probability = _safe_float(signal_row, "confidence")
        return fallback_probability, fallback_probability, False

    probability_map = {
        "BUY": _safe_float(signal_row, "prob_buy"),
        "TAKE_PROFIT": _safe_float(signal_row, "prob_take_profit"),
        "HOLD": _safe_float(signal_row, "prob_hold"),
    }
    primary_probability = probability_map.get(raw_signal_name, probability_map["HOLD"])
    runner_up_probability = max(
        probability
        for signal_name, probability in probability_map.items()
        if signal_name != raw_signal_name
    )

    return primary_probability, max(primary_probability - runner_up_probability, 0.0), True


def evaluate_trading_decision(
    signal_row: Mapping[str, Any],
    minimum_action_confidence: float = 0.0,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Turn a raw model output into a risk-aware trading decision."""

    config = config or TrainingConfig()
    market_state = signal_row.get("marketState") if isinstance(signal_row.get("marketState"), Mapping) else {
        "label": str(signal_row.get("market_regime_label", "unknown")).strip().lower(),
        "isHighVolatility": bool(_safe_int(signal_row, "regime_is_high_volatility")),
    }
    event_context = signal_row.get("eventContext") if isinstance(signal_row.get("eventContext"), Mapping) else {
        "hasEventNext7d": bool(_safe_int(signal_row, "cmcal_has_event_next_7d")),
    }
    raw_candidate = build_raw_signal_candidate(
        signal_row=signal_row,
        minimum_action_confidence=minimum_action_confidence,
        setup_score=_safe_float(signal_row, "setupScore"),
        symbol=str(signal_row.get("symbol") or signal_row.get("base_currency") or "").strip().upper(),
        pair_symbol=str(signal_row.get("product_id") or signal_row.get("pairSymbol") or "").strip().upper(),
        base_currency=str(signal_row.get("base_currency") or signal_row.get("baseCurrency") or "").strip().upper() or None,
        quote_currency=str(signal_row.get("quote_currency") or signal_row.get("quoteCurrency") or "").strip().upper() or None,
        chart_context=dict(signal_row.get("chartContext") or {}) if isinstance(signal_row.get("chartContext"), Mapping) else {},
        execution_context=dict(signal_row.get("executionContext") or {}) if isinstance(signal_row.get("executionContext"), Mapping) else {},
        market_context=dict(signal_row.get("marketContext") or {}) if isinstance(signal_row.get("marketContext"), Mapping) else {},
        market_state=dict(market_state),
        event_context=dict(event_context),
    )
    chart_confirmed_candidate = apply_chart_confirmation(raw_candidate, config=config)
    gated_candidate = apply_risk_gate(chart_confirmed_candidate, config=config)

    return {
        "signalName": gated_candidate.signalName,
        "predictedSignal": gated_candidate.predictedSignal,
        "spotAction": gated_candidate.spotAction,
        "actionable": gated_candidate.actionable,
        "modelSignalName": gated_candidate.rawSignalName,
        "modelPredictedSignal": gated_candidate.rawPredictedSignal,
        "confidence": float(gated_candidate.calibratedConfidence),
        "rawConfidence": float(gated_candidate.rawConfidence),
        "primaryProbability": float(gated_candidate.primaryProbability),
        "hasProbabilityColumns": gated_candidate.hasProbabilityColumns,
        "probabilityColumn": _resolve_probability_column(gated_candidate.rawSignalName),
        "probabilityMargin": float(gated_candidate.probabilityMargin),
        "minimumActionConfidence": float(gated_candidate.minimumActionConfidence),
        "requiredActionConfidence": float(gated_candidate.requiredActionConfidence),
        "confidenceGateApplied": gated_candidate.confidenceGateApplied,
        "riskGateApplied": gated_candidate.riskGateApplied,
        "tradeReadiness": str(gated_candidate.tradeReadiness),
        "policyScore": float(gated_candidate.policyScore),
        "policyNotes": list(gated_candidate.policyNotes),
        "gateReasons": list(gated_candidate.gateReasons),
        "ledger": gated_candidate.ledger.to_dict(),
        "rawProbabilities": dict(gated_candidate.rawProbabilities),
        "calibratedProbabilities": dict(gated_candidate.calibratedProbabilities),
        "chartConfirmationScore": float(gated_candidate.chartConfirmationScore),
        "chartSetupType": str(gated_candidate.chartSetupType),
        "chartConfirmationStatus": str(gated_candidate.chartConfirmationStatus),
        "chartConfirmationNotes": list(gated_candidate.chartConfirmationNotes),
        "chartPatternLabel": str(gated_candidate.chartPatternLabel),
        "chartPatternReasons": list(gated_candidate.chartPatternReasons),
        "chartDecision": str(gated_candidate.chartDecision),
    }
