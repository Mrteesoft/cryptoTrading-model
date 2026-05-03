"""Policy-gating stage for raw signal candidates."""

from __future__ import annotations

from ..config import TrainingConfig
from .audit import build_signal_contribution_ledger
from .contracts import ChartConfirmedCandidate, GatedSignalCandidate


ACTIONABLE_SIGNAL_NAMES = {"BUY", "TAKE_PROFIT"}


def apply_policy_gate(
    candidate: ChartConfirmedCandidate,
    *,
    config: TrainingConfig | None = None,
) -> GatedSignalCandidate:
    """Apply the narrow risk/execution gate to one chart-confirmed candidate."""

    config = config or TrainingConfig()
    raw_signal_name = candidate.rawSignalName
    confidence = float(candidate.calibratedConfidence)
    minimum_action_confidence = float(candidate.minimumActionConfidence)
    probability_margin = float(candidate.probabilityMargin)
    market_regime_label = str(candidate.marketState.get("label", "unknown")).strip().lower()
    is_high_volatility = bool(candidate.marketState.get("isHighVolatility", False))
    has_event_next_7d = bool(candidate.eventContext.get("hasEventNext7d", False))

    confidence_gate_applied = raw_signal_name in ACTIONABLE_SIGNAL_NAMES and confidence < minimum_action_confidence
    required_action_confidence = minimum_action_confidence
    gate_reasons: list[str] = []
    policy_notes: list[str] = []

    if config.decision_policy_enabled and raw_signal_name in ACTIONABLE_SIGNAL_NAMES:
        if candidate.hasProbabilityColumns and probability_margin < float(config.decision_min_probability_margin):
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
    final_signal_name = "HOLD" if confidence_gate_applied or risk_gate_applied else raw_signal_name
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

    policy_status = "passed"
    publication_reason = "passed_policy_gate"
    if confidence_gate_applied or risk_gate_applied:
        policy_status = "blocked"
        publication_reason = "policy_downgraded_to_hold"
    elif final_signal_name == "HOLD":
        policy_status = "standby"
        publication_reason = "standby_signal"

    ledger = build_signal_contribution_ledger(
        raw_probabilities=candidate.rawProbabilities,
        calibrated_probabilities=candidate.calibratedProbabilities,
        raw_confidence=candidate.rawConfidence,
        calibrated_confidence=candidate.calibratedConfidence,
        probability_margin=probability_margin,
        policy_status=policy_status,
        rejection_reasons=gate_reasons,
        required_action_confidence=required_action_confidence,
        final_decision_score=policy_score,
        publication_reason=publication_reason,
    )

    return GatedSignalCandidate(
        **candidate.__dict__,
        signalName=final_signal_name,
        predictedSignal=final_predicted_signal,
        spotAction=spot_action,
        actionable=final_signal_name in ACTIONABLE_SIGNAL_NAMES,
        requiredActionConfidence=float(required_action_confidence),
        confidenceGateApplied=bool(confidence_gate_applied),
        riskGateApplied=bool(risk_gate_applied),
        tradeReadiness=str(trade_readiness),
        policyScore=float(policy_score),
        policyNotes=tuple(policy_notes),
        gateReasons=tuple(gate_reasons),
        ledger=ledger,
    )
