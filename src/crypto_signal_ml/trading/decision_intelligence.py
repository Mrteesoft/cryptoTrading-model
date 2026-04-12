"""Structured evidence, decision, and critic helpers for trader-brain planning."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ..config import TrainingConfig


UPTREND_LABELS = {"trend_up", "trend_up_high_volatility"}
DOWNTREND_LABELS = {"trend_down", "trend_down_high_volatility"}
ENTRY_DECISIONS = {"enter_long_candidate", "add_to_winner_candidate"}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp one numeric value between an inclusive minimum and maximum."""

    return max(min(float(value), maximum), minimum)


def _safe_float(payload: Mapping[str, Any], key: str, default_value: float = 0.0) -> float:
    """Read one optional numeric value from a mapping."""

    raw_value = payload.get(key, default_value)
    if raw_value is None:
        return default_value

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default_value


def _safe_bool(payload: Mapping[str, Any], key: str, default_value: bool = False) -> bool:
    """Read one optional boolean-like value from a mapping."""

    raw_value = payload.get(key, default_value)
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return default_value

    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _memory_bias(trade_memory: Mapping[str, Any]) -> str:
    """Classify whether prior tracked trades support or caution the current setup."""

    if not _safe_bool(trade_memory, "available"):
        return "neutral"

    closed_trade_count = int(_safe_float(trade_memory, "closedTradeCount"))
    win_rate = trade_memory.get("winRate")
    recent_loss_streak = int(_safe_float(trade_memory, "recentLossStreak"))
    if closed_trade_count < 3 or win_rate is None:
        return "neutral"

    try:
        normalized_win_rate = float(win_rate)
    except (TypeError, ValueError):
        return "neutral"

    if recent_loss_streak >= 2 or normalized_win_rate < 0.40:
        return "cautious"
    if normalized_win_rate >= 0.60:
        return "supportive"
    return "neutral"


class TradingDecisionDeliberator:
    """Create a richer decision loop: evidence -> recommendation -> critique."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()

    def deliberate(
        self,
        *,
        signal_summary: Mapping[str, Any],
        base_decision: str,
        base_decision_score: float,
        base_reasons: list[str],
        position: Mapping[str, Any] | None,
        market_context: Mapping[str, Any],
        trade_memory: Mapping[str, Any] | None = None,
        desired_position_fraction: float = 0.0,
        suggested_reduce_fraction: float = 0.0,
        stale_position: bool = False,
        loss_cut_triggered: bool = False,
        profit_lock_triggered: bool = False,
    ) -> Dict[str, Any]:
        """Run the full deliberation loop for one signal candidate."""

        evidence = self._build_evidence(
            signal_summary=signal_summary,
            position=position,
            market_context=market_context,
            trade_memory=trade_memory or {},
            stale_position=stale_position,
            loss_cut_triggered=loss_cut_triggered,
            profit_lock_triggered=profit_lock_triggered,
        )
        decision_memo = self._build_decision_memo(
            signal_summary=signal_summary,
            evidence=evidence,
            base_decision=base_decision,
            base_decision_score=base_decision_score,
            desired_position_fraction=desired_position_fraction,
            suggested_reduce_fraction=suggested_reduce_fraction,
        )
        critic_review = self._build_critic_review(
            evidence=evidence,
            decision_memo=decision_memo,
            base_reasons=base_reasons,
        )

        return {
            "version": "decision-intelligence-v1",
            "evidence": evidence,
            "decisionMemo": decision_memo,
            "criticReview": critic_review,
        }

    def _build_evidence(
        self,
        *,
        signal_summary: Mapping[str, Any],
        position: Mapping[str, Any] | None,
        market_context: Mapping[str, Any],
        trade_memory: Mapping[str, Any],
        stale_position: bool,
        loss_cut_triggered: bool,
        profit_lock_triggered: bool,
    ) -> Dict[str, Any]:
        """Assemble one structured evidence packet for a potential trade action."""

        market_state = signal_summary.get("marketState") or {}
        event_context = signal_summary.get("eventContext") or {}
        watchlist_promotion = signal_summary.get("watchlistPromotion") or {}
        confidence_calibration = signal_summary.get("confidenceCalibration") or {}
        execution_context = signal_summary.get("executionContext") or {}
        adaptive_context = signal_summary.get("adaptiveContext") or {}
        signal_name = str(signal_summary.get("signal_name", "HOLD")).strip().upper()
        trade_readiness = str(signal_summary.get("tradeReadiness", "standby")).strip().lower()
        raw_confidence = _safe_float(signal_summary, "confidence")
        confidence = _safe_float(
            confidence_calibration,
            "calibratedConfidence",
            default_value=raw_confidence,
        )
        probability_margin = _safe_float(signal_summary, "probabilityMargin")
        setup_score = _safe_float(signal_summary, "setupScore")
        policy_score = _safe_float(signal_summary, "policyScore")
        context_alignment_score = _safe_float(confidence_calibration, "contextAlignmentScore")
        confidence_quality = str(confidence_calibration.get("confidenceQuality", "balanced"))
        regime_label = str(market_state.get("label", "unknown")).strip().lower()
        is_high_volatility = _safe_bool(market_state, "isHighVolatility")
        has_event_next_7d = _safe_bool(event_context, "hasEventNext7d")
        market_stance = str(market_context.get("marketStance", "balanced") or "balanced")
        macro_risk_mode = str(market_context.get("macroRiskMode", "neutral") or "neutral")
        execution_quality_score = _safe_float(execution_context, "executionQualityScore", default_value=0.5)
        liquidity_score = _safe_float(execution_context, "liquidityScore", default_value=0.5)
        estimated_round_trip_cost_rate = _safe_float(execution_context, "estimatedRoundTripCostRate")
        is_execution_blocked = _safe_bool(execution_context, "isExecutionBlocked")
        thin_liquidity = _safe_bool(execution_context, "isThinLiquidity")
        elevated_cost = _safe_bool(execution_context, "hasElevatedCost")
        has_position = position is not None
        position_fraction = _safe_float(position or {}, "positionFraction")
        position_age_hours = _safe_float(position or {}, "ageHours")
        position_unrealized_return = (position or {}).get("unrealizedReturn")
        if position_unrealized_return is None:
            position_unrealized_return = (position or {}).get("positionUnrealizedReturn")
        if position_unrealized_return is not None:
            try:
                position_unrealized_return = float(position_unrealized_return)
            except (TypeError, ValueError):
                position_unrealized_return = None

        normalized_setup_score = _clamp(setup_score / 6.0, 0.0, 1.0)
        normalized_policy_score = _clamp(policy_score / 1.5, 0.0, 1.0)
        normalized_probability_margin = _clamp(probability_margin / 0.20, 0.0, 1.0)
        edge_score = _clamp(
            (confidence * 0.36)
            + (normalized_probability_margin * 0.25)
            + (normalized_setup_score * 0.20)
            + (normalized_policy_score * 0.15),
            0.0,
            1.0,
        )
        edge_score = _clamp(
            edge_score
            + (max(context_alignment_score, 0.0) * 0.10)
            + (execution_quality_score * 0.08),
            0.0,
            1.0,
        )

        risk_score = 0.10
        if market_stance == "defensive":
            risk_score += 0.18
        elif market_stance == "offensive":
            risk_score -= 0.04
        if macro_risk_mode == "risk_off":
            risk_score += 0.22
        elif macro_risk_mode == "risk_on":
            risk_score -= 0.05
        if regime_label in DOWNTREND_LABELS:
            risk_score += 0.18
        elif regime_label in UPTREND_LABELS and signal_name == "BUY":
            risk_score -= 0.04
        if is_high_volatility:
            risk_score += 0.10
        if has_event_next_7d:
            risk_score += 0.08
        if trade_readiness == "blocked":
            risk_score += 0.12
        if is_execution_blocked:
            risk_score += 0.14
        elif thin_liquidity:
            risk_score += 0.08
        if elevated_cost:
            risk_score += 0.05
        if stale_position:
            risk_score += 0.08
        if loss_cut_triggered:
            risk_score += 0.16

        memory_bias = str(adaptive_context.get("bias") or _memory_bias(trade_memory))
        if memory_bias == "supportive":
            risk_score -= 0.05
        elif memory_bias == "cautious":
            risk_score += 0.10
        if int(_safe_float(trade_memory, "recentLossStreak")) >= 2:
            risk_score += 0.06
        if confidence_quality == "fragile":
            risk_score += 0.08
        elif confidence_quality == "strong":
            risk_score -= 0.03

        risk_score = _clamp(risk_score, 0.0, 1.0)

        conviction_score = edge_score - (risk_score * 0.55)
        if signal_name == "BUY" and trade_readiness == "high":
            conviction_score += 0.08
        if signal_name in {"TAKE_PROFIT", "LOSS"} and (profit_lock_triggered or loss_cut_triggered):
            conviction_score += 0.10
        if has_position and signal_name == "BUY" and position_unrealized_return is not None and position_unrealized_return > 0:
            conviction_score += 0.03
        conviction_score = _clamp(conviction_score, 0.0, 1.0)

        watchlist_stage = str(watchlist_promotion.get("stage", "") or "")
        watchlist_confirmation_strength = _safe_float(
            watchlist_promotion,
            "confirmationStrength",
        )
        watchlist_soft_risk_override = _safe_bool(watchlist_promotion, "exceptionalOverrideApplied")
        watchlist_hard_block_count = len(list(watchlist_promotion.get("hardBlocks") or []))
        watchlist_soft_penalty_count = len(list(watchlist_promotion.get("softPenalties") or []))

        supporting_factors: list[str] = []
        risk_factors: list[str] = []
        counter_arguments: list[str] = []

        if signal_name == "BUY":
            supporting_factors.append("The model is still leaning long on this coin.")
        elif signal_name == "TAKE_PROFIT":
            supporting_factors.append("The model is favoring capital preservation over fresh upside.")
        elif signal_name == "LOSS":
            supporting_factors.append("The trade has weakened enough that capital protection now takes priority.")

        if confidence >= 0.70:
            supporting_factors.append(f"Calibrated confidence is elevated at {confidence:.2f}.")
        if probability_margin >= 0.10:
            supporting_factors.append(f"Probability margin is healthy at {probability_margin:.2f}.")
        if setup_score >= 3.5:
            supporting_factors.append(f"Setup score is supportive at {setup_score:.2f}.")
        if policy_score >= 0.80:
            supporting_factors.append(f"Policy score remains constructive at {policy_score:.2f}.")
        if context_alignment_score >= 0.20:
            supporting_factors.append("Chart, news, and trend context are aligned behind the setup.")
        if execution_quality_score >= 0.50:
            supporting_factors.append("Liquidity and cost conditions look good enough for execution.")
        if regime_label in UPTREND_LABELS:
            supporting_factors.append("The active regime still favors long momentum.")
        if macro_risk_mode == "risk_on":
            supporting_factors.append("Market-wide intelligence is supportive for selective risk.")
        if watchlist_stage == "entry_ready" and watchlist_confirmation_strength >= 0.70:
            supporting_factors.append("Repeated watchlist checks and chart confirmation have materially improved the setup.")
        if profit_lock_triggered:
            supporting_factors.append("The open trade is already far enough in profit to justify active management.")
        if memory_bias == "supportive":
            supporting_factors.append("Tracked-trade memory for this coin has been supportive recently.")

        if market_stance == "defensive":
            risk_factors.append("The broader market stance is still defensive.")
        if macro_risk_mode == "risk_off":
            risk_factors.append("Market-wide intelligence is currently risk-off.")
        if regime_label in DOWNTREND_LABELS:
            risk_factors.append("The current regime is fighting fresh long exposure.")
        if is_high_volatility:
            risk_factors.append("Volatility is elevated, which increases whipsaw risk.")
        if has_event_next_7d:
            risk_factors.append("A near-term event adds catalyst risk.")
        if trade_readiness == "blocked":
            risk_factors.append("The policy layer has already blocked fresh risk.")
        elif trade_readiness == "standby":
            risk_factors.append("Trade readiness is still only on standby.")
        if confidence_quality == "fragile":
            risk_factors.append("Context-adjusted confidence is fragile after applying chart, news, and risk filters.")
        if is_execution_blocked:
            risk_factors.append("Execution quality is too weak because liquidity is thin relative to volatility.")
        elif elevated_cost:
            risk_factors.append("Estimated execution cost is elevated for a fresh entry.")
        if watchlist_hard_block_count > 0:
            risk_factors.append("Hard watchlist gating is still active against promotion.")
        elif watchlist_soft_penalty_count > 0 and not watchlist_soft_risk_override:
            risk_factors.append("Soft watchlist penalties still argue for patience.")
        if stale_position:
            risk_factors.append("The thesis has become stale.")
        if loss_cut_triggered:
            risk_factors.append("The drawdown is already beyond the configured loss budget.")
        if memory_bias == "cautious":
            risk_factors.append("Tracked-trade memory for this coin has been weak or recently losing.")

        if macro_risk_mode == "risk_off" and signal_name == "BUY":
            counter_arguments.append("This long can fail if the market keeps rotating into defense.")
        if is_high_volatility and signal_name == "BUY":
            counter_arguments.append("The setup can be right on direction but still fail on volatility noise.")
        if has_event_next_7d:
            counter_arguments.append("A catalyst can invalidate the current setup faster than the model horizon.")
        if is_execution_blocked or elevated_cost:
            counter_arguments.append("A decent directional call can still fail after costs if execution stays poor.")
        if stale_position and has_position:
            counter_arguments.append("The position may simply be old rather than still genuinely strong.")
        if memory_bias == "cautious":
            counter_arguments.append("Recent trade memory says this coin has not rewarded similar setups reliably.")

        return {
            "signalName": signal_name,
            "tradeReadiness": trade_readiness,
            "marketStance": market_stance,
            "macroRiskMode": macro_risk_mode,
            "regimeLabel": regime_label,
            "hasPosition": bool(has_position),
            "isHighVolatility": bool(is_high_volatility),
            "hasEventNext7d": bool(has_event_next_7d),
            "rawConfidence": float(raw_confidence),
            "confidence": float(confidence),
            "confidenceQuality": confidence_quality,
            "probabilityMargin": float(probability_margin),
            "setupScore": float(setup_score),
            "policyScore": float(policy_score),
            "contextAlignmentScore": float(context_alignment_score),
            "positionFraction": float(position_fraction),
            "positionAgeHours": float(position_age_hours) if position_age_hours > 0 else None,
            "positionUnrealizedReturn": position_unrealized_return,
            "thesisAgeIsStale": bool(stale_position),
            "lossCutTriggered": bool(loss_cut_triggered),
            "profitLockTriggered": bool(profit_lock_triggered),
            "edgeScore": float(edge_score),
            "riskScore": float(risk_score),
            "convictionScore": float(conviction_score),
            "executionQualityScore": float(execution_quality_score),
            "liquidityScore": float(liquidity_score),
            "estimatedRoundTripCostRate": float(estimated_round_trip_cost_rate),
            "executionBlocked": bool(is_execution_blocked),
            "supportingFactors": supporting_factors[:4],
            "riskFactors": risk_factors[:4],
            "counterArguments": counter_arguments[:3],
            "tradeMemory": {
                "available": _safe_bool(trade_memory, "available"),
                "scope": str(trade_memory.get("scope", "product")),
                "closedTradeCount": int(_safe_float(trade_memory, "closedTradeCount")),
                "winRate": trade_memory.get("winRate"),
                "averageRealizedReturn": trade_memory.get("averageRealizedReturn"),
                "recentLossStreak": int(_safe_float(trade_memory, "recentLossStreak")),
                "lastOutcome": trade_memory.get("lastOutcome"),
                "sampleAdequate": _safe_bool(trade_memory, "sampleAdequate"),
                "bias": memory_bias,
            },
            "watchlistStage": watchlist_stage or None,
            "watchlistConfirmationStrength": float(watchlist_confirmation_strength),
            "watchlistSoftRiskOverride": bool(watchlist_soft_risk_override),
            "watchlistHardBlockCount": int(watchlist_hard_block_count),
            "watchlistSoftPenaltyCount": int(watchlist_soft_penalty_count),
        }

    def _build_decision_memo(
        self,
        *,
        signal_summary: Mapping[str, Any],
        evidence: Mapping[str, Any],
        base_decision: str,
        base_decision_score: float,
        desired_position_fraction: float,
        suggested_reduce_fraction: float,
    ) -> Dict[str, Any]:
        """Turn structured evidence into one machine-readable decision memo."""

        signal_name = str(evidence.get("signalName", "HOLD"))
        conviction_score = _safe_float(evidence, "convictionScore")
        risk_score = _safe_float(evidence, "riskScore")
        edge_score = _safe_float(evidence, "edgeScore")
        trade_readiness = str(evidence.get("tradeReadiness", "standby"))
        market_stance = str(evidence.get("marketStance", "balanced"))
        has_position = _safe_bool(evidence, "hasPosition")
        memory_bias = str((evidence.get("tradeMemory") or {}).get("bias", "neutral"))

        recommended_decision = str(base_decision)
        if base_decision in ENTRY_DECISIONS:
            if trade_readiness == "blocked" or conviction_score < 0.50 or risk_score >= 0.72:
                recommended_decision = "hold_position" if has_position else "watchlist"
            elif edge_score >= 0.75 and risk_score <= 0.32 and trade_readiness == "high":
                recommended_decision = str(base_decision)
        elif (
            base_decision in {"watchlist", "avoid_long"}
            and signal_name == "BUY"
            and conviction_score >= 0.76
            and risk_score <= 0.25
            and market_stance != "defensive"
            and trade_readiness == "high"
        ):
            recommended_decision = "add_to_winner_candidate" if has_position else "enter_long_candidate"
        elif base_decision == "hold_position" and risk_score >= 0.58:
            recommended_decision = "hold_and_tighten_risk"

        size_multiplier = 1.0
        if recommended_decision in ENTRY_DECISIONS:
            if conviction_score >= 0.78 and risk_score <= 0.35:
                size_multiplier *= 1.08
            elif conviction_score < 0.58 or risk_score > 0.55:
                size_multiplier *= 0.78

            if memory_bias == "supportive":
                size_multiplier *= 1.05
            elif memory_bias == "cautious":
                size_multiplier *= 0.82
        else:
            size_multiplier = 0.0

        thesis = self._build_thesis_line(
            signal_summary=signal_summary,
            evidence=evidence,
            recommended_decision=recommended_decision,
        )
        invalidation = (
            (evidence.get("counterArguments") or [None])[0]
            or (evidence.get("riskFactors") or [None])[0]
            or "The setup loses validity if the current evidence deteriorates."
        )
        why_not_trade = None
        if recommended_decision not in ENTRY_DECISIONS:
            why_not_trade = (
                (evidence.get("riskFactors") or [None])[0]
                or "The current setup is not strong enough to justify new capital."
            )

        desired_fraction_after_review = 0.0
        if recommended_decision in ENTRY_DECISIONS:
            desired_fraction_after_review = float(desired_position_fraction) * _clamp(size_multiplier, 0.0, 1.15)

        return {
            "recommendedDecision": recommended_decision,
            "decisionConfidence": float(conviction_score),
            "baseDecisionScore": float(base_decision_score),
            "sizeMultiplier": float(_clamp(size_multiplier, 0.0, 1.15)),
            "desiredPositionFractionAfterReview": float(desired_fraction_after_review),
            "suggestedReduceFraction": float(suggested_reduce_fraction),
            "thesis": thesis,
            "invalidation": invalidation,
            "whyNotTrade": why_not_trade,
            "memoryBias": memory_bias,
            "supportingPoints": list(evidence.get("supportingFactors") or [])[:3],
            "riskPoints": list(evidence.get("riskFactors") or [])[:3],
        }

    def _build_critic_review(
        self,
        *,
        evidence: Mapping[str, Any],
        decision_memo: Mapping[str, Any],
        base_reasons: list[str],
    ) -> Dict[str, Any]:
        """Challenge the proposed action before it becomes the final brain decision."""

        recommended_decision = str(decision_memo.get("recommendedDecision", "watchlist"))
        risk_score = _safe_float(evidence, "riskScore")
        conviction_score = _safe_float(evidence, "convictionScore")
        signal_name = str(evidence.get("signalName", "HOLD"))
        watchlist_soft_risk_override = _safe_bool(evidence, "watchlistSoftRiskOverride")
        objections: list[str] = []

        if recommended_decision in ENTRY_DECISIONS:
            if str(evidence.get("macroRiskMode", "neutral")) == "risk_off" and not watchlist_soft_risk_override:
                objections.append("Macro risk mode is risk-off against a fresh long.")
            if _safe_bool(evidence, "isHighVolatility"):
                objections.append("Volatility is elevated for a new entry.")
            if _safe_bool(evidence, "hasEventNext7d"):
                objections.append("Event risk is too near for a clean entry.")
            if _safe_bool(evidence, "executionBlocked"):
                objections.append("Execution quality is too weak for a new entry.")
            if str(evidence.get("regimeLabel", "")) in DOWNTREND_LABELS:
                objections.append("The regime is still downward for fresh long risk.")
            if str(evidence.get("tradeReadiness", "standby")) == "blocked":
                objections.append("The policy layer already blocked this setup.")
            if str(evidence.get("confidenceQuality", "balanced")) == "fragile":
                objections.append("Context-adjusted confidence is too fragile for fresh capital.")
            if conviction_score < 0.58:
                objections.append("Conviction is not strong enough for fresh capital.")
            trade_memory = evidence.get("tradeMemory") or {}
            if str(trade_memory.get("bias", "neutral")) == "cautious":
                objections.append("Tracked-trade memory is cautionary on similar setups.")

        approved_decision = recommended_decision
        verdict = "approve"
        size_multiplier = float(decision_memo.get("sizeMultiplier", 1.0) or 1.0)
        score_multiplier = 1.0

        if recommended_decision in ENTRY_DECISIONS and (len(objections) >= 3 or risk_score >= 0.70):
            verdict = "block"
            approved_decision = "hold_position" if _safe_bool(evidence, "hasPosition") else "watchlist"
            size_multiplier = 0.0
            score_multiplier = 0.72
        elif len(objections) >= 1 or risk_score >= 0.52:
            verdict = "caution"
            size_multiplier *= 0.85
            score_multiplier = 0.90
            if approved_decision == "hold_position" and (
                _safe_bool(evidence, "isHighVolatility") or _safe_bool(evidence, "hasEventNext7d")
            ):
                approved_decision = "hold_and_tighten_risk"

        if signal_name in {"TAKE_PROFIT", "LOSS"} and approved_decision in ENTRY_DECISIONS:
            verdict = "block"
            approved_decision = "watchlist"
            size_multiplier = 0.0
            score_multiplier = 0.70
            objections.append("An exit-style signal should not reopen fresh long risk.")

        summary = self._build_critic_summary(
            verdict=verdict,
            approved_decision=approved_decision,
            objections=objections,
            base_reasons=base_reasons,
        )

        return {
            "verdict": verdict,
            "approvedDecision": approved_decision,
            "scoreMultiplier": float(_clamp(score_multiplier, 0.0, 1.0)),
            "sizeMultiplier": float(_clamp(size_multiplier, 0.0, 1.15)),
            "objections": objections[:4],
            "summary": summary,
        }

    @staticmethod
    def _build_thesis_line(
        *,
        signal_summary: Mapping[str, Any],
        evidence: Mapping[str, Any],
        recommended_decision: str,
    ) -> str:
        """Summarize the current trading thesis in one short line."""

        product_id = str(signal_summary.get("productId", signal_summary.get("pairSymbol", "asset"))).upper()
        signal_name = str(evidence.get("signalName", "HOLD"))
        market_stance = str(evidence.get("marketStance", "balanced"))
        confidence = _safe_float(evidence, "confidence")

        if recommended_decision in ENTRY_DECISIONS:
            return (
                f"{product_id} can justify fresh risk because the {signal_name.lower()} thesis still holds "
                f"with {confidence:.2f} confidence in a {market_stance} market."
            )
        if recommended_decision in {"exit_position", "reduce_position"}:
            return f"{product_id} should prioritize capital protection because the upside thesis is weakening."
        if recommended_decision == "hold_and_tighten_risk":
            return f"{product_id} can stay open, but only with tighter risk because the evidence is mixed."
        return f"{product_id} stays in observation mode until the evidence improves."

    @staticmethod
    def _build_critic_summary(
        *,
        verdict: str,
        approved_decision: str,
        objections: list[str],
        base_reasons: list[str],
    ) -> str:
        """Render one concise critic verdict for the final payload."""

        if verdict == "block":
            return objections[0] if objections else "The critic blocked fresh risk."
        if verdict == "caution":
            return objections[0] if objections else "The critic is allowing the trade with caution."
        if base_reasons:
            return base_reasons[0]
        return f"The critic approved the {approved_decision} plan."
