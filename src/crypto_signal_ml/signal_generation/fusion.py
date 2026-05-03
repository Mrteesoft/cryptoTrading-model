"""Bounded late-fusion stage for typed signal candidates."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..config import TrainingConfig
from ..trading.signal_quality import build_signal_quality_context
from .contracts import ContextEnrichedCandidate, ScoredSignalCandidate
from .serialization import context_enriched_candidate_to_summary


UPTREND_LABELS = {"trend_up", "trend_up_high_volatility"}
DOWNTREND_LABELS = {"trend_down", "trend_down_high_volatility"}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp one numeric value between an inclusive minimum and maximum."""

    return max(min(float(value), maximum), minimum)


def _safe_float(payload: Mapping[str, Any] | None, key: str, default_value: float = 0.0) -> float:
    """Read one optional float from a mapping."""

    if not isinstance(payload, Mapping):
        return default_value

    raw_value = payload.get(key, default_value)
    if raw_value is None:
        return default_value

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default_value


def _build_final_decision_score(
    *,
    candidate: ContextEnrichedCandidate,
    market_context: Mapping[str, Any],
    confidence_calibration: Mapping[str, Any],
    execution_context: Mapping[str, Any],
    adaptive_context: Mapping[str, Any],
) -> float:
    """Build one bounded final decision score without mutating model probabilities."""

    confidence = _safe_float(confidence_calibration, "calibratedConfidence", candidate.calibratedConfidence)
    probability_margin = float(candidate.probabilityMargin)
    normalized_setup_score = _clamp(candidate.setupScore / 6.0, 0.0, 1.0)
    normalized_policy_score = _clamp(candidate.policyScore / 1.5, 0.0, 1.0)
    trade_readiness = str(candidate.tradeReadiness).strip().lower()
    regime_label = str(candidate.marketState.get("label", "unknown")).strip().lower()
    market_stance = str(market_context.get("marketStance", "balanced") or "balanced")
    macro_risk_mode = str(market_context.get("macroRiskMode", "neutral") or "neutral")
    is_high_volatility = bool(candidate.marketState.get("isHighVolatility", False))
    has_event_next_7d = bool(candidate.eventContext.get("hasEventNext7d", False))
    context_alignment_score = _safe_float(confidence_calibration, "contextAlignmentScore")
    execution_penalty = _safe_float(execution_context, "decisionPenalty")
    adaptive_decision_adjustment = _safe_float(adaptive_context, "decisionAdjustment")

    decision_score = (
        (confidence * 0.45)
        + (probability_margin * 0.20)
        + (normalized_setup_score * 0.15)
        + (normalized_policy_score * 0.20)
    )

    if trade_readiness == "high":
        decision_score += 0.08
    elif trade_readiness == "blocked":
        decision_score -= 0.12

    if market_stance == "offensive":
        decision_score += 0.05
    elif market_stance == "defensive":
        decision_score -= 0.08
    if macro_risk_mode == "risk_on":
        decision_score += 0.03
    elif macro_risk_mode == "risk_off":
        decision_score -= 0.07

    if regime_label in UPTREND_LABELS:
        decision_score += 0.04
    elif regime_label in DOWNTREND_LABELS:
        decision_score -= 0.10

    if is_high_volatility:
        decision_score -= 0.04
    if has_event_next_7d:
        decision_score -= 0.03

    decision_score += _clamp(context_alignment_score, -1.0, 1.0) * 0.10
    decision_score -= _clamp(execution_penalty, 0.0, 0.18)
    decision_score += _clamp(adaptive_decision_adjustment, -0.08, 0.08)

    return _clamp(decision_score, 0.0, 1.25)


class SignalFusionStage:
    """Run bounded late fusion over typed signal candidates."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()

    def score_candidate(
        self,
        candidate: ContextEnrichedCandidate,
        *,
        market_context: Mapping[str, Any] | None = None,
        trade_memory: Mapping[str, Any] | None = None,
    ) -> ScoredSignalCandidate:
        """Score one context-enriched candidate without rewriting raw model outputs."""

        effective_market_context = dict(market_context or {})
        quality_context = build_signal_quality_context(
            signal_summary=context_enriched_candidate_to_summary(candidate),
            market_context=effective_market_context,
            trade_memory=trade_memory,
            config=self.config,
        )
        confidence_calibration = dict(quality_context["confidenceCalibration"])
        execution_context = dict(quality_context["executionContext"])
        adaptive_context = dict(quality_context["adaptiveContext"])
        final_decision_score = _build_final_decision_score(
            candidate=candidate,
            market_context=effective_market_context,
            confidence_calibration=confidence_calibration,
            execution_context=execution_context,
            adaptive_context=adaptive_context,
        )
        ledger = replace(
            candidate.ledger,
            confidenceAdjustments={
                **dict(candidate.ledger.confidenceAdjustments),
                "contextAdjustment": _safe_float(confidence_calibration, "confidenceAdjustment"),
                "decisionAdjustment": _safe_float(confidence_calibration, "decisionAdjustment"),
                "riskPenaltyScore": _safe_float(confidence_calibration, "riskPenaltyScore"),
                "executionPenaltyScore": _safe_float(confidence_calibration, "executionPenaltyScore"),
                "adaptiveConfidenceAdjustment": _safe_float(adaptive_context, "confidenceAdjustment"),
            },
            finalDecisionScore=float(final_decision_score),
        )
        context_evidence = dict(candidate.contextEvidence)
        context_evidence.update(
            {
                "marketStance": dict(effective_market_context),
                "chartAlignmentScore": _safe_float(confidence_calibration, "chartAlignmentScore"),
                "newsAlignmentScore": _safe_float(confidence_calibration, "newsAlignmentScore"),
                "trendAlignmentScore": _safe_float(confidence_calibration, "trendAlignmentScore"),
                "contextAlignmentScore": _safe_float(confidence_calibration, "contextAlignmentScore"),
            }
        )

        candidate_payload = dict(candidate.__dict__)
        candidate_payload.update(
            {
                "executionContext": execution_context,
                "contextEvidence": context_evidence,
                "confidenceCalibration": confidence_calibration,
                "adaptiveContext": adaptive_context,
                "finalDecisionScore": float(final_decision_score),
                "ledger": ledger,
            }
        )
        return ScoredSignalCandidate(**candidate_payload)

    def score_candidates(
        self,
        candidates: list[ContextEnrichedCandidate],
        *,
        market_context: Mapping[str, Any] | None = None,
        trade_memory_by_product: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[ScoredSignalCandidate, ...]:
        """Score a full batch of context-enriched candidates."""

        trade_memory_by_product = trade_memory_by_product or {}
        return tuple(
            self.score_candidate(
                candidate,
                market_context=market_context,
                trade_memory=trade_memory_by_product.get(str(candidate.productId).strip().upper(), {}),
            )
            for candidate in candidates
        )
