"""Risk-aware trading policy helpers layered on top of raw model predictions."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..config import TrainingConfig
from ..labels import signal_to_text


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
    raw_signal_name = _resolve_raw_signal_name(signal_row)
    raw_predicted_signal = _safe_int(signal_row, "predicted_signal")
    confidence = _safe_float(signal_row, "confidence")
    minimum_action_confidence = max(float(minimum_action_confidence or 0.0), 0.0)
    primary_probability, probability_margin, has_probability_columns = _resolve_probability_margin(
        signal_row,
        raw_signal_name,
    )
    market_regime_label = str(signal_row.get("market_regime_label", "unknown")).strip().lower()
    is_high_volatility = bool(_safe_int(signal_row, "regime_is_high_volatility"))
    has_event_next_7d = bool(_safe_int(signal_row, "cmcal_has_event_next_7d"))

    confidence_gate_applied = (
        raw_signal_name in ACTIONABLE_SIGNAL_NAMES
        and confidence < minimum_action_confidence
    )
    required_action_confidence = minimum_action_confidence
    gate_reasons: list[str] = []
    policy_notes: list[str] = []

    if config.decision_policy_enabled and raw_signal_name in ACTIONABLE_SIGNAL_NAMES:
        if has_probability_columns and probability_margin < float(config.decision_min_probability_margin):
            gate_reasons.append(
                f"Probability edge over the runner-up class is only {probability_margin:.1%}, "
                "which is too thin for a fresh action."
            )

        if raw_signal_name == "BUY":
            if (
                config.decision_block_downtrend_buys
                and market_regime_label in {"trend_down", "trend_down_high_volatility"}
            ):
                gate_reasons.append(
                    "Fresh spot buys are blocked while the detected regime is still in a downtrend."
                )

            if market_regime_label in {"trend_up", "trend_up_high_volatility"}:
                policy_notes.append("Trend regime is aligned with the long setup.")

            if is_high_volatility:
                required_action_confidence = max(
                    required_action_confidence,
                    minimum_action_confidence + float(config.decision_high_volatility_confidence_buffer),
                )
                policy_notes.append(
                    f"High volatility raises the required confidence to {required_action_confidence:.1%}."
                )

            if has_event_next_7d:
                required_action_confidence = max(
                    required_action_confidence,
                    minimum_action_confidence + float(config.decision_event_risk_confidence_buffer),
                )
                policy_notes.append(
                    f"Upcoming event risk raises the required confidence to {required_action_confidence:.1%}."
                )

            if confidence >= minimum_action_confidence and confidence < required_action_confidence:
                gate_reasons.append(
                    f"Confidence is {confidence:.1%}, below the risk-adjusted entry bar of "
                    f"{required_action_confidence:.1%}."
                )

        elif raw_signal_name == "TAKE_PROFIT":
            if market_regime_label in {"trend_down", "trend_down_high_volatility"}:
                policy_notes.append("Downtrend regime supports reducing spot exposure.")
            if is_high_volatility:
                policy_notes.append("High volatility supports taking risk off faster.")

    risk_gate_applied = bool(gate_reasons)
    final_signal_name = (
        "HOLD"
        if confidence_gate_applied or risk_gate_applied
        else raw_signal_name
    )
    final_predicted_signal = {
        "TAKE_PROFIT": -1,
        "HOLD": 0,
        "BUY": 1,
    }[final_signal_name]
    spot_action = {
        "TAKE_PROFIT": "take_profit",
        "HOLD": "wait",
        "BUY": "buy",
    }[final_signal_name]

    policy_score = confidence + probability_margin
    if raw_signal_name == "BUY":
        if market_regime_label in {"trend_up", "trend_up_high_volatility"}:
            policy_score += 0.10
        if market_regime_label in {"trend_down", "trend_down_high_volatility"}:
            policy_score -= 0.15
        if is_high_volatility:
            policy_score -= 0.05
        if has_event_next_7d:
            policy_score -= 0.03
    elif raw_signal_name == "TAKE_PROFIT":
        if market_regime_label in {"trend_down", "trend_down_high_volatility"}:
            policy_score += 0.05
        if is_high_volatility:
            policy_score += 0.03

    if final_signal_name != raw_signal_name:
        policy_score -= 0.25

    if final_signal_name not in ACTIONABLE_SIGNAL_NAMES:
        trade_readiness = (
            "blocked"
            if raw_signal_name in ACTIONABLE_SIGNAL_NAMES and (confidence_gate_applied or risk_gate_applied)
            else "standby"
        )
    elif (
        confidence >= (required_action_confidence + 0.10)
        and probability_margin >= (float(config.decision_min_probability_margin) + 0.05)
        and not is_high_volatility
    ):
        trade_readiness = "high"
    else:
        trade_readiness = "medium"

    return {
        "signalName": final_signal_name,
        "predictedSignal": final_predicted_signal,
        "spotAction": spot_action,
        "actionable": final_signal_name in ACTIONABLE_SIGNAL_NAMES,
        "modelSignalName": raw_signal_name,
        "modelPredictedSignal": raw_predicted_signal,
        "confidence": confidence,
        "primaryProbability": primary_probability,
        "hasProbabilityColumns": has_probability_columns,
        "probabilityColumn": _resolve_probability_column(raw_signal_name),
        "probabilityMargin": probability_margin,
        "minimumActionConfidence": minimum_action_confidence,
        "requiredActionConfidence": required_action_confidence,
        "confidenceGateApplied": confidence_gate_applied,
        "riskGateApplied": risk_gate_applied,
        "tradeReadiness": trade_readiness,
        "policyScore": float(policy_score),
        "policyNotes": policy_notes,
        "gateReasons": gate_reasons,
    }
