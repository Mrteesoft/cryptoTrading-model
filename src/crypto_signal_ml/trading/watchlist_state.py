"""Persistent watchlist state for setup progression and promotion logic."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from ..config import OUTPUTS_DIR, TrainingConfig
from .signal_quality import build_signal_quality_context


LOGGER = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso_timestamp(value: Any) -> datetime | None:
    normalized_value = str(value or "").strip().replace("Z", "+00:00")
    if not normalized_value:
        return None

    try:
        parsed_value = datetime.fromisoformat(normalized_value)
    except ValueError:
        return None

    if parsed_value.tzinfo is None:
        return parsed_value.replace(tzinfo=timezone.utc)

    return parsed_value.astimezone(timezone.utc)


def _bounded_history(rows: list[dict[str, Any]], max_len: int) -> list[dict[str, Any]]:
    if max_len <= 0:
        return rows
    return rows[-max_len:]


def _resolve_chart_pattern_reason(
    pattern_label: str,
    *,
    near_resistance: bool,
    weak_structure: bool,
) -> str:
    """Map one blocked chart pattern to a stable watchlist reason code."""

    normalized_label = str(pattern_label or "").strip().lower()
    if normalized_label == "near_resistance" or near_resistance:
        return "near_resistance"
    if normalized_label in {"downtrend_structure", "weak_trend_structure"} or weak_structure:
        return "weak_chart_structure"
    if normalized_label == "range_bound":
        return "range_bound_setup"
    if normalized_label == "no_clean_setup":
        return "no_clean_setup"
    if normalized_label == "reversal_attempt":
        return "reversal_attempt"
    if normalized_label:
        return f"chart_pattern_{normalized_label}"
    return "blocked_chart_pattern"


@dataclass
class WatchlistPromotionSignal:
    """Result of one watchlist progression check."""

    stage: str
    prior_stage: str
    promotion_ready: bool
    blocked_reason: str | None
    blocked_reason_detail: str | None
    hold_reason: str | None
    review_outcome: str
    review_reason: str
    primary_veto_bucket: str | None = None
    veto_buckets: tuple[str, ...] = ()
    hard_blocks: tuple[str, ...] = ()
    soft_penalties: tuple[str, ...] = ()
    confirmation_strength: float = 0.0
    exceptional_override_applied: bool = False
    strong_blocked_buy_preserved: bool = False


class WatchlistStateStore:
    """Store watchlist lifecycle state in a JSON file."""

    stage_priority = {
        "entry_ready": 0,
        "setup_confirmed": 1,
        "setup_building": 2,
        "watchlist": 3,
        "observe": 4,
    }

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.path = Path(self.config.signal_watchlist_state_path or (OUTPUTS_DIR / "watchlistState.json"))
        self._state: dict[str, Any] = {"updatedAt": None, "items": {}}
        self._cycle_events: list[dict[str, Any]] = []
        self._last_cycle_summary: dict[str, Any] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(payload, dict):
            self._state = {
                "updatedAt": payload.get("updatedAt"),
                "items": payload.get("items") if isinstance(payload.get("items"), dict) else {},
                "lastCycleSummary": (
                    dict(payload.get("lastCycleSummary"))
                    if isinstance(payload.get("lastCycleSummary"), dict)
                    else {}
                ),
            }
            self._last_cycle_summary = dict(self._state.get("lastCycleSummary") or {})

    def save(self) -> None:
        if not self._dirty:
            return
        summary = self.build_cycle_summary()
        self._state["updatedAt"] = _utc_now_iso()
        self._state["lastCycleSummary"] = summary
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        self._last_cycle_summary = dict(summary)
        self._dirty = False
        self._log_cycle_summary(summary)

    def get_state(self, product_id: str) -> dict[str, Any] | None:
        return self._state.get("items", {}).get(product_id)

    def last_cycle_summary(self) -> dict[str, Any]:
        """Return the latest saved or in-memory cycle summary."""

        if self._cycle_events:
            return self.build_cycle_summary()
        return dict(self._last_cycle_summary)

    def list_active_product_ids(self, limit: int | None = None) -> list[str]:
        """Return active watchlist products ordered by readiness and review recency."""

        item_rows = list((self._state.get("items") or {}).values())
        ranked_rows = sorted(
            [
                item_row
                for item_row in item_rows
                if str(item_row.get("productId", "")).strip()
                and str(item_row.get("stage", "observe")).strip().lower() != "invalidated"
            ],
            key=lambda item_row: (
                self.stage_priority.get(str(item_row.get("stage", "observe")).strip().lower(), 99),
                -int(item_row.get("consecutivePositiveChecks", 0) or 0),
                -int(item_row.get("positiveChecks", 0) or 0),
                -float(item_row.get("decisionScoreDelta", 0.0) or 0.0),
                -float(item_row.get("confidenceDelta", 0.0) or 0.0),
                -(
                    _parse_iso_timestamp(item_row.get("lastReviewedAt"))
                    or datetime(1970, 1, 1, tzinfo=timezone.utc)
                ).timestamp(),
                str(item_row.get("productId", "")),
            ),
        )

        product_ids = [
            str(item_row.get("productId", "")).strip().upper()
            for item_row in ranked_rows
        ]
        if limit is not None:
            product_ids = product_ids[: max(int(limit), 0)]
        return product_ids

    def build_cycle_summary(self) -> dict[str, Any]:
        """Summarize the current review cycle for observability and tuning."""

        item_rows = [
            item_row
            for item_row in (self._state.get("items") or {}).values()
            if isinstance(item_row, dict)
        ]
        stage_counts: Counter[str] = Counter()
        blocked_reason_counts: Counter[str] = Counter()
        hold_reason_counts: Counter[str] = Counter()
        for item_row in item_rows:
            stage = str(item_row.get("stage") or "observe").strip().lower() or "observe"
            stage_counts[stage] += 1

            blocked_reason = str(item_row.get("blockedReason") or "").strip()
            if blocked_reason:
                blocked_reason_counts[blocked_reason] += 1

            hold_reason = str(item_row.get("holdReason") or "").strip()
            if hold_reason:
                hold_reason_counts[hold_reason] += 1

        review_outcome_counts: Counter[str] = Counter()
        review_reason_counts: Counter[str] = Counter()
        transition_counts: Counter[str] = Counter()
        promoted_this_cycle = 0
        invalidated_this_cycle = 0
        blocked_this_cycle = 0
        advanced_this_cycle = 0
        preserved_this_cycle = 0

        for cycle_event in self._cycle_events:
            review_outcome = str(cycle_event.get("reviewOutcome") or "monitoring")
            review_reason = str(cycle_event.get("reviewReason") or "under_review")
            review_outcome_counts[review_outcome] += 1
            review_reason_counts[review_reason] += 1
            if bool(cycle_event.get("stageChanged")):
                transition_counts[
                    f"{cycle_event.get('fromStage') or 'observe'}->{cycle_event.get('toStage') or 'observe'}"
                ] += 1
            if review_outcome == "promoted":
                promoted_this_cycle += 1
            elif review_outcome == "invalidated":
                invalidated_this_cycle += 1
            elif review_outcome == "blocked":
                blocked_this_cycle += 1
            elif review_outcome == "advanced":
                advanced_this_cycle += 1
            if bool(cycle_event.get("strongBlockedBuyPreserved")):
                preserved_this_cycle += 1

        active_count = int(sum(stage != "invalidated" for stage in stage_counts.elements()))
        transition_limit = max(int(self.config.signal_watchlist_diagnostics_max_transitions), 0)
        reason_limit = max(int(self.config.signal_watchlist_diagnostics_top_reasons), 0)

        return {
            "generatedAt": _utc_now_iso(),
            "reviewedCount": len(self._cycle_events),
            "trackedCount": len(item_rows),
            "activeCount": active_count,
            "stageCounts": dict(sorted(stage_counts.items())),
            "outcomeCounts": dict(sorted(review_outcome_counts.items())),
            "transitionCounts": dict(sorted(transition_counts.items())),
            "promotedThisCycle": int(promoted_this_cycle),
            "advancedThisCycle": int(advanced_this_cycle),
            "blockedThisCycle": int(blocked_this_cycle),
            "invalidatedThisCycle": int(invalidated_this_cycle),
            "preservedStrongBlockedBuysThisCycle": int(preserved_this_cycle),
            "thresholds": {
                "promotionMinConfidence": float(self.config.signal_watchlist_promotion_min_confidence),
                "promotionMinDecisionScore": float(self.config.signal_watchlist_promotion_min_decision_score),
                "entryReadyMinConfidence": float(self.config.signal_watchlist_entry_ready_min_confidence),
                "entryReadyMinDecisionScore": float(self.config.signal_watchlist_entry_ready_min_decision_score),
                "invalidationConfidence": float(self.config.signal_watchlist_invalidation_confidence),
                "softReviewConfidenceBuffer": float(
                    getattr(self.config, "signal_watchlist_soft_review_confidence_buffer", 0.08) or 0.08
                ),
                "softReviewMinRawConfidence": float(
                    getattr(self.config, "signal_watchlist_soft_review_min_raw_confidence", 0.50) or 0.50
                ),
                "softReviewMinProbabilityMargin": float(
                    getattr(self.config, "signal_watchlist_soft_review_min_probability_margin", 0.12) or 0.12
                ),
            },
            "reasons": {
                "blocked": self._top_counts(blocked_reason_counts, reason_limit),
                "hold": self._top_counts(hold_reason_counts, reason_limit),
                "review": self._top_counts(review_reason_counts, reason_limit),
            },
            "transitions": self._build_transition_rows(limit=transition_limit),
        }

    @staticmethod
    def _top_counts(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return [
            {"reason": reason, "count": int(count)}
            for reason, count in counter.most_common(limit)
        ]

    def _build_transition_rows(self, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        ranked_events = sorted(
            self._cycle_events,
            key=lambda cycle_event: (
                0 if str(cycle_event.get("reviewOutcome")) == "promoted" else 1,
                0 if str(cycle_event.get("reviewOutcome")) == "invalidated" else 1,
                0 if bool(cycle_event.get("stageChanged")) else 1,
                0 if str(cycle_event.get("reviewOutcome")) == "blocked" else 1,
                str(cycle_event.get("productId") or ""),
            ),
        )
        return [
            {
                "productId": str(cycle_event.get("productId") or ""),
                "fromStage": str(cycle_event.get("fromStage") or "observe"),
                "toStage": str(cycle_event.get("toStage") or "observe"),
                "stageChanged": bool(cycle_event.get("stageChanged")),
                "reviewOutcome": str(cycle_event.get("reviewOutcome") or "monitoring"),
                "reviewReason": str(cycle_event.get("reviewReason") or "under_review"),
                "strongBlockedBuyPreserved": bool(cycle_event.get("strongBlockedBuyPreserved")),
                "promotionReady": bool(cycle_event.get("promotionReady")),
            }
            for cycle_event in ranked_events[:limit]
        ]

    def _log_cycle_summary(self, summary: Mapping[str, Any]) -> None:
        if not bool(getattr(self.config, "signal_watchlist_diagnostics_enabled", True)):
            return
        reviewed_count = int(summary.get("reviewedCount", 0) or 0)
        if reviewed_count <= 0:
            return

        stage_counts = summary.get("stageCounts") if isinstance(summary.get("stageCounts"), Mapping) else {}
        LOGGER.info(
            "Watchlist promotion | loaded=%s | watchlist=%s | building=%s | confirmed=%s | entry_ready=%s | invalidated=%s | promoted=%s | preserved=%s",
            reviewed_count,
            int(stage_counts.get("watchlist", 0) or 0),
            int(stage_counts.get("setup_building", 0) or 0),
            int(stage_counts.get("setup_confirmed", 0) or 0),
            int(stage_counts.get("entry_ready", 0) or 0),
            int(stage_counts.get("invalidated", 0) or 0),
            int(summary.get("promotedThisCycle", 0) or 0),
            int(summary.get("preservedStrongBlockedBuysThisCycle", 0) or 0),
        )

        reasons = summary.get("reasons") if isinstance(summary.get("reasons"), Mapping) else {}
        blocked_reasons = self._format_reason_rows(reasons.get("blocked"))
        hold_reasons = self._format_reason_rows(reasons.get("hold"))
        if blocked_reasons or hold_reasons:
            LOGGER.info(
                "Watchlist reasons | blocked=%s | hold=%s",
                blocked_reasons or "none",
                hold_reasons or "none",
            )

        transition_rows = summary.get("transitions") if isinstance(summary.get("transitions"), list) else []
        if transition_rows:
            LOGGER.info(
                "Watchlist changes | %s",
                "; ".join(
                    (
                        f"{str(row.get('productId') or '').upper()}: "
                        f"{row.get('fromStage')}->{row.get('toStage')} "
                        f"({row.get('reviewOutcome')}: {row.get('reviewReason')})"
                    ).strip()
                    for row in transition_rows
                    if isinstance(row, Mapping) and str(row.get("productId") or "").strip()
                ),
            )

    @staticmethod
    def _format_reason_rows(rows: Any) -> str:
        if not isinstance(rows, list):
            return ""
        formatted_rows = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            reason = str(row.get("reason") or "").strip()
            count = int(row.get("count", 0) or 0)
            if reason and count > 0:
                formatted_rows.append(f"{reason}:{count}")
        return ", ".join(formatted_rows)

    def update_from_signal(
        self,
        *,
        signal_summary: dict[str, Any],
        decision_score: float,
        market_context: dict[str, Any],
        trade_memory: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], WatchlistPromotionSignal]:
        product_id = str(signal_summary.get("productId", "")).strip().upper()
        now_iso = _utc_now_iso()
        if not product_id:
            empty_state = {"productId": "", "stage": "observe", "lastReviewedAt": now_iso}
            return empty_state, WatchlistPromotionSignal(
                stage="observe",
                prior_stage="observe",
                promotion_ready=False,
                blocked_reason=None,
                blocked_reason_detail=None,
                hold_reason=None,
                review_outcome="monitoring",
                review_reason="under_review",
            )

        state = dict(self._state.get("items", {}).get(product_id) or {})
        first_seen_at = state.get("firstSeenAt") or now_iso
        stage = str(state.get("stage") or "observe")

        signal_name = str(signal_summary.get("signal_name", "") or "").strip().upper()
        model_signal_name = str(
            signal_summary.get("modelSignalName")
            or signal_summary.get("rawSignalName")
            or signal_name
        ).strip().upper()
        trade_readiness = str(signal_summary.get("tradeReadiness", "standby")).strip().lower()
        raw_confidence = _safe_float(signal_summary.get("confidence")) or 0.0
        probability_margin = _safe_float(signal_summary.get("probabilityMargin")) or 0.0
        confidence = raw_confidence
        close_price = _safe_float(signal_summary.get("close"))
        market_state = signal_summary.get("marketState") or {}
        trend_score = _safe_float(market_state.get("trendScore"))
        volatility_ratio = _safe_float(market_state.get("volatilityRatio"))
        is_high_volatility = bool(market_state.get("isHighVolatility", False))
        regime_label = str(market_state.get("label", "unknown")).strip().lower()
        event_context = signal_summary.get("eventContext") or {}
        news_context = signal_summary.get("newsContext") or {}
        trend_context = signal_summary.get("trendContext") or {}
        chart_context = signal_summary.get("chartContext") or {}
        quality_context = {
            "confidenceCalibration": (
                dict(signal_summary.get("confidenceCalibration"))
                if isinstance(signal_summary.get("confidenceCalibration"), Mapping)
                else {}
            ),
            "executionContext": (
                dict(signal_summary.get("executionContext"))
                if isinstance(signal_summary.get("executionContext"), Mapping)
                else {}
            ),
            "adaptiveContext": (
                dict(signal_summary.get("adaptiveContext"))
                if isinstance(signal_summary.get("adaptiveContext"), Mapping)
                else {}
            ),
        }
        if not all(quality_context.values()):
            computed_quality_context = build_signal_quality_context(
                signal_summary=signal_summary,
                market_context=market_context,
                trade_memory=trade_memory,
                config=self.config,
            )
            for key, value in computed_quality_context.items():
                if not quality_context[key]:
                    quality_context[key] = value
        confidence_calibration = quality_context["confidenceCalibration"]
        execution_context = quality_context["executionContext"]
        adaptive_context = quality_context["adaptiveContext"]
        calibrated_confidence = _safe_float(confidence_calibration.get("calibratedConfidence"))
        if calibrated_confidence is None:
            calibrated_confidence = confidence
        chart_alignment_score = _safe_float(confidence_calibration.get("chartAlignmentScore")) or 0.0
        news_alignment_score = _safe_float(confidence_calibration.get("newsAlignmentScore")) or 0.0
        trend_alignment_score = _safe_float(confidence_calibration.get("trendAlignmentScore")) or 0.0
        context_alignment_score = _safe_float(confidence_calibration.get("contextAlignmentScore")) or 0.0
        reliability_score = _safe_float(confidence_calibration.get("reliabilityScore")) or 0.0
        risk_penalty_score = _safe_float(confidence_calibration.get("riskPenaltyScore")) or 0.0
        calibration_execution_penalty_score = _safe_float(confidence_calibration.get("executionPenaltyScore")) or 0.0
        execution_quality_score = _safe_float(execution_context.get("executionQualityScore")) or 0.0
        execution_penalty = _safe_float(execution_context.get("decisionPenalty")) or 0.0
        thin_liquidity = bool(execution_context.get("isThinLiquidity", False))
        elevated_execution_cost = bool(execution_context.get("hasElevatedCost", False))
        execution_blocked = bool(execution_context.get("isExecutionBlocked", False))
        adaptive_confirmation_adjustment = _safe_float(adaptive_context.get("confirmationAdjustment")) or 0.0
        adaptive_bias = str(adaptive_context.get("bias", "neutral") or "neutral")
        event_window_active = bool(event_context.get("eventWindowActive", False))
        post_event_cooldown_active = bool(event_context.get("postEventCooldownActive", False))
        event_risk_flag = bool(event_context.get("macroEventRiskFlag", False))
        news_sentiment = float(news_context.get("newsSentiment1h", 0.0) or 0.0)
        news_relevance = float(news_context.get("newsRelevanceScore", 0.0) or 0.0)
        negative_news = news_relevance > 0 and news_sentiment <= float(
            self.config.news_negative_sentiment_threshold
        )
        positive_news = news_relevance > 0 and news_sentiment >= float(
            self.config.news_positive_sentiment_threshold
        )
        positive_trend = float(trend_context.get("topicTrendScore", 0.0) or 0.0) >= float(
            self.config.trend_support_threshold
        )
        breakout_confirmed = bool(chart_context.get("breakoutConfirmed", False))
        retest_hold_confirmed = bool(chart_context.get("retestHoldConfirmed", False))
        near_resistance = bool(chart_context.get("nearResistance", False))
        resistance_distance_pct = _safe_float(chart_context.get("resistanceDistancePct"))
        structure_label = str(chart_context.get("structureLabel", "")).lower()
        chart_confirmation_score = _safe_float(signal_summary.get("chartConfirmationScore")) or 0.0
        chart_decision = str(
            signal_summary.get("chartDecision", signal_summary.get("chartConfirmationStatus", "early")) or "early"
        ).strip().lower()
        chart_confirmation_available = bool(
            str(signal_summary.get("chartDecision", signal_summary.get("chartConfirmationStatus", "")) or "").strip()
        )
        chart_pattern_label = str(
            signal_summary.get("chartPatternLabel", signal_summary.get("chartSetupType", "no_clean_setup"))
            or "no_clean_setup"
        ).strip().lower()
        chart_pattern_reasons = [
            str(reason).strip().lower()
            for reason in list(
                signal_summary.get("chartPatternReasons")
                or signal_summary.get("chartConfirmationNotes")
                or []
            )
            if str(reason).strip()
        ]
        weak_structure = structure_label in {"lower_highs", "lower_lows", "downtrend"}
        chart_confirmed = chart_decision == "confirmed"
        chart_early = chart_decision in {"early", "unclear"}
        chart_blocked = chart_decision in {"blocked", "invalid"}
        confirmed_chart = chart_confirmed or breakout_confirmed or retest_hold_confirmed
        chart_supportive = confirmed_chart or chart_early or chart_confirmation_score >= 0.18 or chart_alignment_score >= 0.18
        supportive_context = positive_trend or positive_news or context_alignment_score >= 0.18
        min_resistance_distance_pct = float(
            getattr(self.config, "signal_watchlist_entry_ready_min_resistance_distance_pct", 0.015) or 0.015
        )
        has_resistance_room = bool(
            chart_confirmed
            or confirmed_chart
            or (
                not near_resistance
                and (
                    resistance_distance_pct is None
                    or resistance_distance_pct >= min_resistance_distance_pct
                )
            )
        )
        prior_stage = stage
        market_stance = str(market_context.get("marketStance", "balanced"))
        macro_risk_mode = str(market_context.get("macroRiskMode", "neutral"))

        if not state:
            trigger_pct = float(self.config.signal_watchlist_breakout_pct)
            invalidation_pct = float(self.config.signal_watchlist_invalidation_pct)
            trigger_price = close_price * (1 + trigger_pct) if close_price is not None else None
            invalidation_price = close_price * (1 - invalidation_pct) if close_price is not None else None
            stage = "watchlist"
            state = {
                "productId": product_id,
                "stage": stage,
                "firstSeenAt": first_seen_at,
                "lastReviewedAt": now_iso,
                "checks": 0,
                "positiveChecks": 0,
                "consecutivePositiveChecks": 0,
                "triggerPrice": trigger_price,
                "invalidationPrice": invalidation_price,
                "confidenceHistory": [],
                "calibratedConfidenceHistory": [],
                "promotionConfidenceHistory": [],
                "decisionScoreHistory": [],
                "promotionDecisionScoreHistory": [],
                "tradeReadinessHistory": [],
                "trendScoreHistory": [],
                "volatilityRatioHistory": [],
                "executionQualityHistory": [],
                "adaptiveBiasHistory": [],
                "lastSignalName": signal_name,
                "lastModelSignalName": model_signal_name,
                "lastSpotAction": signal_summary.get("spotAction"),
            }
            prior_stage = stage

        state["lastReviewedAt"] = now_iso
        state["lastSignalName"] = signal_name
        state["lastModelSignalName"] = model_signal_name
        state["lastSpotAction"] = signal_summary.get("spotAction")

        history_max = int(getattr(self.config, "signal_watchlist_history_max", 6) or 6)
        confidence_history = list(state.get("confidenceHistory") or [])
        calibrated_confidence_history = list(state.get("calibratedConfidenceHistory") or [])
        promotion_confidence_history = list(state.get("promotionConfidenceHistory") or [])
        decision_history = list(state.get("decisionScoreHistory") or [])
        promotion_decision_history = list(state.get("promotionDecisionScoreHistory") or [])
        readiness_history = list(state.get("tradeReadinessHistory") or [])
        trend_history = list(state.get("trendScoreHistory") or [])
        volatility_history = list(state.get("volatilityRatioHistory") or [])
        execution_quality_history = list(state.get("executionQualityHistory") or [])
        adaptive_bias_history = list(state.get("adaptiveBiasHistory") or [])

        confidence_history.append({"at": now_iso, "value": confidence})
        calibrated_confidence_history.append({"at": now_iso, "value": float(calibrated_confidence)})
        decision_history.append({"at": now_iso, "value": float(decision_score)})
        promotion_confidence = float(calibrated_confidence)
        if market_stance == "defensive":
            promotion_confidence += 0.05 * 0.65
        if macro_risk_mode == "risk_off":
            promotion_confidence += 0.07 * 0.65
        promotion_confidence = min(max(promotion_confidence, 0.0), 1.0)
        promotion_decision_score = float(decision_score)
        if market_stance == "defensive":
            promotion_decision_score += 0.08
        if macro_risk_mode == "risk_off":
            promotion_decision_score += 0.07
        promotion_decision_score = min(max(promotion_decision_score, 0.0), 1.25)
        promotion_confidence_history.append({"at": now_iso, "value": promotion_confidence})
        promotion_decision_history.append({"at": now_iso, "value": promotion_decision_score})
        readiness_history.append({"at": now_iso, "value": trade_readiness})
        if trend_score is not None:
            trend_history.append({"at": now_iso, "value": float(trend_score)})
        if volatility_ratio is not None:
            volatility_history.append({"at": now_iso, "value": float(volatility_ratio)})
        execution_quality_history.append({"at": now_iso, "value": float(execution_quality_score)})
        adaptive_bias_history.append({"at": now_iso, "value": adaptive_bias})

        state["confidenceHistory"] = _bounded_history(confidence_history, history_max)
        state["calibratedConfidenceHistory"] = _bounded_history(calibrated_confidence_history, history_max)
        state["promotionConfidenceHistory"] = _bounded_history(promotion_confidence_history, history_max)
        state["decisionScoreHistory"] = _bounded_history(decision_history, history_max)
        state["promotionDecisionScoreHistory"] = _bounded_history(promotion_decision_history, history_max)
        state["tradeReadinessHistory"] = _bounded_history(readiness_history, history_max)
        state["trendScoreHistory"] = _bounded_history(trend_history, history_max)
        state["volatilityRatioHistory"] = _bounded_history(volatility_history, history_max)
        state["executionQualityHistory"] = _bounded_history(execution_quality_history, history_max)
        state["adaptiveBiasHistory"] = _bounded_history(adaptive_bias_history, history_max)
        state["calibratedConfidence"] = float(calibrated_confidence)
        state["rawConfidence"] = float(raw_confidence)
        state["probabilityMargin"] = float(probability_margin)
        state["promotionConfidence"] = float(promotion_confidence)
        state["confidenceQuality"] = str(confidence_calibration.get("confidenceQuality", "balanced"))
        state["reliabilityScore"] = float(reliability_score)
        state["riskPenaltyScore"] = float(risk_penalty_score)
        state["executionPenaltyScore"] = float(calibration_execution_penalty_score)
        state["chartAlignmentScore"] = float(chart_alignment_score)
        state["newsAlignmentScore"] = float(news_alignment_score)
        state["trendAlignmentScore"] = float(trend_alignment_score)
        state["contextAlignmentScore"] = float(context_alignment_score)
        state["chartConfirmationScore"] = float(chart_confirmation_score)
        state["chartConfirmationStatus"] = str(chart_decision)
        state["chartSetupType"] = str(chart_pattern_label)
        state["chartDecision"] = str(chart_decision)
        state["chartPatternLabel"] = str(chart_pattern_label)
        state["chartPatternReasons"] = list(chart_pattern_reasons)
        state["executionQualityScore"] = float(execution_quality_score)
        state["executionBlocked"] = bool(execution_blocked)
        state["adaptiveBias"] = adaptive_bias
        state["promotionDecisionScore"] = float(promotion_decision_score)

        state["checks"] = int(state.get("checks") or 0) + 1

        min_confidence = float(self.config.signal_watchlist_promotion_min_confidence)
        min_decision_score = float(self.config.signal_watchlist_promotion_min_decision_score)
        invalidation_confidence = float(self.config.signal_watchlist_invalidation_confidence)
        invalidation_min_probability_margin = float(
            getattr(self.config, "signal_watchlist_invalidation_min_probability_margin", 0.08) or 0.08
        )
        soft_review_confidence_buffer = float(
            getattr(self.config, "signal_watchlist_soft_review_confidence_buffer", 0.08) or 0.08
        )
        soft_review_min_raw_confidence = float(
            getattr(self.config, "signal_watchlist_soft_review_min_raw_confidence", 0.50) or 0.50
        )
        soft_review_min_probability_margin = float(
            getattr(self.config, "signal_watchlist_soft_review_min_probability_margin", 0.12) or 0.12
        )
        strong_buy_min_raw_confidence = float(
            getattr(self.config, "signal_watchlist_strong_buy_min_raw_confidence", 0.72) or 0.72
        )
        strong_buy_min_probability_margin = float(
            getattr(self.config, "signal_watchlist_strong_buy_min_probability_margin", 0.18) or 0.18
        )
        preserve_strong_blocked_buys = bool(
            getattr(self.config, "signal_watchlist_preserve_strong_blocked_buys", True)
        )

        strong_blocked_buy_candidate = bool(
            preserve_strong_blocked_buys
            and model_signal_name == "BUY"
            and raw_confidence >= strong_buy_min_raw_confidence
            and probability_margin >= strong_buy_min_probability_margin
        )
        chart_pattern_block_reason = _resolve_chart_pattern_reason(
            chart_pattern_label,
            near_resistance=near_resistance,
            weak_structure=weak_structure,
        )
        soft_review_confidence_floor = max(invalidation_confidence - soft_review_confidence_buffer, 0.0)
        soft_low_confidence_review = bool(
            model_signal_name == "BUY"
            and confirmed_chart
            and not chart_blocked
            and calibrated_confidence <= invalidation_confidence
            and calibrated_confidence > soft_review_confidence_floor
            and raw_confidence >= soft_review_min_raw_confidence
            and probability_margin >= soft_review_min_probability_margin
        )

        signal_quality_reasons: list[str] = []
        signal_quality_soft_reasons: list[str] = []
        if signal_name == "LOSS":
            signal_quality_reasons.append("loss_signal")
        if calibrated_confidence <= invalidation_confidence:
            if soft_low_confidence_review:
                signal_quality_soft_reasons.append("soft_low_calibrated_confidence")
            else:
                signal_quality_reasons.append("low_calibrated_confidence")
        if model_signal_name == "BUY" and probability_margin < invalidation_min_probability_margin:
            signal_quality_reasons.append("weak_probability_margin")
        if weak_structure:
            signal_quality_reasons.append("weak_chart_structure")
        if model_signal_name == "BUY" and chart_blocked:
            signal_quality_reasons.append(chart_pattern_block_reason)
        if not has_resistance_room:
            signal_quality_reasons.append("near_resistance")
        if trade_readiness == "blocked" and not strong_blocked_buy_candidate:
            signal_quality_reasons.append("policy_blocked_fresh_risk")

        market_regime_reasons: list[str] = []
        if event_window_active or event_risk_flag:
            market_regime_reasons.append("blocked_by_event_risk")
        if post_event_cooldown_active:
            market_regime_reasons.append("await_post_event_confirmation")
        if regime_label in {"trend_down", "trend_down_high_volatility"}:
            market_regime_reasons.append("severe_downtrend_regime")
        if market_stance == "defensive":
            market_regime_reasons.append("defensive_market")
        if macro_risk_mode == "risk_off":
            market_regime_reasons.append("macro_risk_off")
        if is_high_volatility:
            market_regime_reasons.append("high_volatility")
        if negative_news:
            market_regime_reasons.append("negative_news_conflict")

        execution_reasons: list[str] = []
        if execution_blocked:
            execution_reasons.append("execution_risk_too_high")
        if thin_liquidity:
            execution_reasons.append("thin_liquidity")
        if elevated_execution_cost:
            execution_reasons.append("elevated_execution_cost")

        signal_quality_reasons = list(dict.fromkeys(signal_quality_reasons))
        signal_quality_soft_reasons = list(dict.fromkeys(signal_quality_soft_reasons))
        market_regime_reasons = list(dict.fromkeys(market_regime_reasons))
        execution_reasons = list(dict.fromkeys(execution_reasons))

        invalidation_quality_reasons = {
            "loss_signal",
            "low_calibrated_confidence",
            "weak_probability_margin",
        }
        signal_quality_veto = bool(any(reason in invalidation_quality_reasons for reason in signal_quality_reasons))
        if model_signal_name == "BUY" and chart_blocked:
            signal_quality_veto = True
        strong_blocked_buy_preserved = bool(
            strong_blocked_buy_candidate
            and not signal_quality_veto
            and (bool(market_regime_reasons) or bool(execution_reasons))
        )
        meets_promotion_thresholds = (
            promotion_confidence >= min_confidence
            and promotion_decision_score >= min_decision_score
        )
        positive_check = (
            meets_promotion_thresholds or strong_blocked_buy_candidate
        ) and not any(
            reason in {"low_calibrated_confidence", "weak_probability_margin", "policy_blocked_fresh_risk"}
            for reason in signal_quality_reasons
        )
        if positive_check:
            state["positiveChecks"] = int(state.get("positiveChecks") or 0) + 1
            state["consecutivePositiveChecks"] = int(state.get("consecutivePositiveChecks") or 0) + 1
        else:
            state["consecutivePositiveChecks"] = 0

        first_confidence = (
            state["promotionConfidenceHistory"][0]["value"]
            if state.get("promotionConfidenceHistory")
            else promotion_confidence
        )
        first_decision_score = (
            state["promotionDecisionScoreHistory"][0]["value"]
            if state.get("promotionDecisionScoreHistory")
            else promotion_decision_score
        )
        confidence_delta = float(promotion_confidence - (first_confidence or 0.0))
        decision_delta = float(promotion_decision_score - (first_decision_score or 0.0))
        state["confidenceDelta"] = confidence_delta
        state["decisionScoreDelta"] = decision_delta

        trigger_price = _safe_float(state.get("triggerPrice"))
        invalidation_price = _safe_float(state.get("invalidationPrice"))
        trigger_distance_pct = None
        invalidation_distance_pct = None
        if close_price is not None and trigger_price:
            trigger_distance_pct = (trigger_price - close_price) / max(trigger_price, 1e-9)
        if close_price is not None and invalidation_price:
            invalidation_distance_pct = (close_price - invalidation_price) / max(close_price, 1e-9)
        state["triggerDistancePct"] = trigger_distance_pct
        state["invalidationDistancePct"] = invalidation_distance_pct

        price_invalidation_hit = bool(
            close_price is not None
            and invalidation_price is not None
            and close_price <= invalidation_price
        )
        state["priceInvalidationHit"] = price_invalidation_hit
        invalidated = False
        if signal_name == "LOSS":
            invalidated = True
        if price_invalidation_hit and not strong_blocked_buy_preserved:
            invalidated = True
        if signal_quality_veto and not strong_blocked_buy_preserved:
            invalidated = True

        recovered_from_invalidation = False
        if invalidated:
            stage = "invalidated"
            state["invalidatedAt"] = now_iso
        else:
            recoverable_buy_follow_up = bool(
                prior_stage == "invalidated"
                and model_signal_name == "BUY"
                and not signal_quality_veto
                and not price_invalidation_hit
                and not chart_blocked
                and (
                    strong_blocked_buy_preserved
                    or soft_low_confidence_review
                    or confirmed_chart
                    or chart_early
                )
            )
            if prior_stage == "invalidated" and (strong_blocked_buy_preserved or recoverable_buy_follow_up):
                stage = "watchlist"
                recovered_from_invalidation = True
                state["recoveredAt"] = now_iso
            setup_building_min_checks = int(
                getattr(self.config, "signal_watchlist_setup_building_min_checks", 2) or 2
            )
            min_positive_checks = int(self.config.signal_watchlist_promotion_min_positive_checks)
            min_conf_gain = float(self.config.signal_watchlist_promotion_min_confidence_gain)
            min_decision_gain = float(self.config.signal_watchlist_promotion_min_decision_score_gain)
            entry_ready_min_positive_checks = max(
                int(getattr(self.config, "signal_watchlist_entry_ready_min_positive_checks", 2) or 2),
                min_positive_checks,
            )
            strong_entry_setup = bool(
                trade_readiness == "high"
                and calibrated_confidence >= float(self.config.signal_watchlist_entry_ready_min_confidence)
                and decision_score >= float(self.config.signal_watchlist_entry_ready_min_decision_score)
            )
            supports_setup_confirmation = bool(
                chart_confirmed
                or chart_early
                or chart_supportive
                or supportive_context
                or strong_entry_setup
            )
            fast_track_entry_ready = bool(
                trade_readiness == "high"
                and int(state.get("checks") or 0) >= entry_ready_min_positive_checks
                and confidence_delta >= min_conf_gain
                and decision_delta >= min_decision_gain
                and has_resistance_room
                and execution_quality_score >= 0.20
                and strong_entry_setup
                and (
                    chart_confirmed
                    or (not chart_confirmation_available and chart_alignment_score >= 0.28)
                )
            )
            if (
                stage in {"observe", "watchlist"}
                and positive_check
                and int(state.get("checks") or 0) >= setup_building_min_checks
            ):
                stage = "setup_building"
            if (
                stage in {"setup_building", "watchlist"}
                and state.get("consecutivePositiveChecks", 0) >= min_positive_checks
                and confidence_delta >= min_conf_gain
                and decision_delta >= min_decision_gain
                and has_resistance_room
                and execution_quality_score >= 0.20
                and supports_setup_confirmation
            ):
                stage = "setup_confirmed"
            if (
                stage in {"watchlist", "setup_building", "setup_confirmed"}
                and positive_check
                and int(state.get("checks") or 0) >= entry_ready_min_positive_checks
                and calibrated_confidence >= float(self.config.signal_watchlist_entry_ready_min_confidence)
                and decision_score >= float(self.config.signal_watchlist_entry_ready_min_decision_score)
                and has_resistance_room
                and execution_quality_score >= 0.20
                and (
                    chart_confirmed
                    or confirmed_chart
                    or (
                        not chart_confirmation_available
                        and (
                            chart_confirmation_score >= 0.30
                            or chart_alignment_score >= 0.28
                        )
                    )
                    or fast_track_entry_ready
                )
            ):
                stage = "entry_ready"

        state["stage"] = stage
        if stage != prior_stage:
            state["lastStageChangeAt"] = now_iso
        else:
            state["lastStageChangeAt"] = state.get("lastStageChangeAt") or now_iso

        hard_blocks = list(
            dict.fromkeys(
                [
                    reason
                    for reason in signal_quality_reasons
                    if reason
                    in {
                        "loss_signal",
                        "low_calibrated_confidence",
                        "weak_probability_margin",
                        "policy_blocked_fresh_risk",
                        "weak_chart_structure",
                        "near_resistance",
                        "range_bound_setup",
                        "no_clean_setup",
                        "reversal_attempt",
                        "blocked_chart_pattern",
                    }
                    or reason.startswith("chart_pattern_")
                ]
                + [
                    reason
                    for reason in market_regime_reasons
                    if reason
                    in {
                        "blocked_by_event_risk",
                        "await_post_event_confirmation",
                        "severe_downtrend_regime",
                    }
                ]
                + [reason for reason in execution_reasons if reason == "execution_risk_too_high"]
            )
        )

        soft_penalties: list[str] = []
        if signal_quality_soft_reasons:
            soft_penalties.extend(signal_quality_soft_reasons)
        if market_stance == "defensive":
            soft_penalties.append("defensive_market")
        if macro_risk_mode == "risk_off":
            soft_penalties.append("macro_risk_off")
        if is_high_volatility:
            soft_penalties.append("high_volatility")
        if negative_news:
            soft_penalties.append("negative_news_conflict")
        if thin_liquidity:
            soft_penalties.append("thin_liquidity")
        if elevated_execution_cost:
            soft_penalties.append("elevated_execution_cost")
        if adaptive_bias == "cautious":
            soft_penalties.append("weak_recent_trade_memory")
        soft_penalties = list(dict.fromkeys(soft_penalties))

        setup_building_min_checks = int(
            getattr(self.config, "signal_watchlist_setup_building_min_checks", 2) or 2
        )
        min_positive_checks = int(self.config.signal_watchlist_promotion_min_positive_checks)
        min_conf_gain = float(self.config.signal_watchlist_promotion_min_confidence_gain)
        min_decision_gain = float(self.config.signal_watchlist_promotion_min_decision_score_gain)
        confirmation_strength = 0.0
        if positive_check:
            confirmation_strength += 0.18
        if int(state.get("checks") or 0) >= setup_building_min_checks:
            confirmation_strength += 0.12
        if int(state.get("consecutivePositiveChecks") or 0) >= min_positive_checks:
            confirmation_strength += 0.18
        if confidence_delta >= min_conf_gain:
            confirmation_strength += 0.10
        if decision_delta >= min_decision_gain:
            confirmation_strength += 0.17
        if chart_confirmed:
            confirmation_strength += 0.20
        elif chart_early:
            confirmation_strength += 0.10
        if confirmed_chart:
            confirmation_strength += 0.17
        if has_resistance_room:
            confirmation_strength += 0.05
        if supportive_context:
            confirmation_strength += 0.03
        if positive_news and positive_trend:
            confirmation_strength += 0.02
        confirmation_strength += max(chart_alignment_score, 0.0) * 0.12
        confirmation_strength += max(news_alignment_score, 0.0) * 0.05
        confirmation_strength += max(trend_alignment_score, 0.0) * 0.04
        confirmation_strength += max(adaptive_confirmation_adjustment, 0.0)
        confirmation_strength -= max(-chart_alignment_score, 0.0) * 0.08
        confirmation_strength -= max(-news_alignment_score, 0.0) * 0.05
        confirmation_strength -= max(-trend_alignment_score, 0.0) * 0.03
        confirmation_strength -= execution_penalty * 0.35
        confirmation_strength -= max(-adaptive_confirmation_adjustment, 0.0)
        confirmation_strength = min(max(confirmation_strength, 0.0), 1.0)

        soft_risk_override_min_confirmation = float(
            getattr(self.config, "signal_watchlist_soft_risk_override_min_confirmation", 0.72) or 0.72
        )
        exceptional_override_applied = bool(
            stage == "entry_ready"
            and not hard_blocks
            and bool(soft_penalties)
            and confirmation_strength >= soft_risk_override_min_confirmation
            and (
                chart_confirmed
                or confirmed_chart
                or (not chart_confirmation_available and chart_alignment_score >= 0.28)
            )
            and calibrated_confidence >= float(self.config.signal_watchlist_entry_ready_min_confidence)
            and decision_score >= float(self.config.signal_watchlist_entry_ready_min_decision_score)
        )

        veto_bucket_details: dict[str, list[str]] = {}
        if signal_quality_reasons or adaptive_bias == "cautious":
            veto_bucket_details["signal_quality_veto"] = list(
                dict.fromkeys(
                    signal_quality_reasons
                    + (["weak_recent_trade_memory"] if adaptive_bias == "cautious" else [])
                )
            )
        if market_regime_reasons:
            veto_bucket_details["market_regime_veto"] = list(dict.fromkeys(market_regime_reasons))
        if execution_reasons:
            veto_bucket_details["execution_veto"] = list(dict.fromkeys(execution_reasons))

        veto_buckets = tuple(veto_bucket_details.keys())
        primary_veto_bucket = next(iter(veto_buckets), None)
        blocked_reason_detail = (
            veto_bucket_details.get(primary_veto_bucket, [None])[0]
            if primary_veto_bucket is not None
            else None
        )
        blocking_veto_active = bool(
            hard_blocks
            or (stage == "entry_ready" and soft_penalties and not exceptional_override_applied)
            or strong_blocked_buy_preserved
        )
        blocked_reason = None
        if blocking_veto_active and primary_veto_bucket is not None:
            blocked_reason = primary_veto_bucket

        promotion_ready = (
            stage == "entry_ready"
            and trade_readiness in {"high", "medium"}
            and not hard_blocks
            and (not soft_penalties or exceptional_override_applied)
            and (
                primary_veto_bucket is None
                or exceptional_override_applied
            )
        )

        hold_reason = None
        if strong_blocked_buy_preserved:
            if "market_regime_veto" in veto_bucket_details and "execution_veto" in veto_bucket_details:
                hold_reason = "blocked_high_risk"
            elif "market_regime_veto" in veto_bucket_details:
                hold_reason = "needs_regime_improvement"
            elif "execution_veto" in veto_bucket_details:
                hold_reason = "needs_execution_improvement"
        elif stage == "watchlist":
            hold_reason = "wait_for_setup_building"
        elif stage == "setup_building":
            hold_reason = "wait_for_setup_confirmation"
        elif stage == "setup_confirmed":
            hold_reason = "wait_for_breakout_confirmation"
            if retest_hold_confirmed:
                hold_reason = "wait_for_entry_window"
        if (
            "soft_low_calibrated_confidence" in soft_penalties
            and stage != "invalidated"
            and hold_reason not in {"blocked_high_risk", "needs_regime_improvement", "needs_execution_improvement"}
        ):
            hold_reason = "needs_more_signal_confirmation"
        if (
            hold_reason in {"wait_for_setup_confirmation", "wait_for_breakout_confirmation", "wait_for_entry_window"}
            and blocked_reason == "market_regime_veto"
        ):
            hold_reason = "needs_regime_improvement"
        elif (
            hold_reason in {"wait_for_setup_confirmation", "wait_for_breakout_confirmation", "wait_for_entry_window"}
            and blocked_reason == "execution_veto"
        ):
            hold_reason = "needs_execution_improvement"
        elif stage == "entry_ready" and blocked_reason is not None:
            hold_reason = blocked_reason
        elif stage == "invalidated":
            hold_reason = "invalidated"
        elif supportive_context and hold_reason is None:
            hold_reason = "supported_by_positive_news"

        review_outcome = "monitoring"
        review_reason = hold_reason or blocked_reason or "under_review"
        if promotion_ready:
            review_outcome = "promoted"
            review_reason = "promotion_ready"
        elif stage == "invalidated":
            review_outcome = "invalidated"
            review_reason = blocked_reason_detail or blocked_reason or "invalidated"
        elif blocked_reason is not None:
            review_outcome = "blocked"
            review_reason = hold_reason or blocked_reason_detail or blocked_reason
        elif stage != prior_stage:
            review_outcome = "advanced"
            review_reason = hold_reason or f"advanced_to_{stage}"

        if bool(getattr(self.config, "signal_watchlist_diagnostics_enabled", True)) and model_signal_name == "BUY":
            if (
                calibrated_confidence <= invalidation_confidence
                or soft_low_confidence_review
                or blocked_reason is not None
                or strong_blocked_buy_candidate
            ):
                LOGGER.info(
                    "Calibration check | %s | raw=%.2f | calib=%.2f | reliability=%.2f | context=%.2f | risk=%.2f | exec=%.2f | quality=%s | cutoff=%.2f | margin=%.2f | chart=%s | pattern=%s | mode=%s",
                    product_id,
                    float(raw_confidence),
                    float(calibrated_confidence),
                    float(reliability_score),
                    float(context_alignment_score),
                    float(risk_penalty_score),
                    float(calibration_execution_penalty_score),
                    str(confidence_calibration.get("confidenceQuality", "balanced")),
                    float(invalidation_confidence),
                    float(probability_margin),
                    chart_decision,
                    chart_pattern_label,
                    (
                        "soft_review"
                        if soft_low_confidence_review
                        else "hard_veto"
                        if calibrated_confidence <= invalidation_confidence
                        else "clear"
                    ),
                )
            if strong_blocked_buy_candidate or trade_readiness == "blocked" or blocked_reason is not None:
                LOGGER.info(
                    "Preserve check | %s | buy=%s | chart=%s | pattern=%s | raw=%.2f | calib=%.2f | margin=%.2f | primary=%s | preserved=%s | reason=%s",
                    product_id,
                    "yes",
                    chart_decision,
                    chart_pattern_label,
                    float(raw_confidence),
                    float(calibrated_confidence),
                    float(probability_margin),
                    primary_veto_bucket or "none",
                    "yes" if strong_blocked_buy_preserved else "no",
                    (
                        blocked_reason_detail
                        or review_reason
                        or "none"
                    ),
                )

        state["hardBlocks"] = list(hard_blocks)
        state["signalQualityReasons"] = list(signal_quality_reasons)
        state["signalQualitySoftReasons"] = list(signal_quality_soft_reasons)
        state["softPenalties"] = list(soft_penalties)
        state["vetoBuckets"] = list(veto_buckets)
        state["vetoBucketDetails"] = dict(veto_bucket_details)
        state["primaryVetoBucket"] = primary_veto_bucket
        state["blockedReasonDetail"] = blocked_reason_detail
        state["confirmationStrength"] = float(confirmation_strength)
        state["promotionReady"] = bool(promotion_ready)
        state["blockedReason"] = blocked_reason
        state["holdReason"] = hold_reason
        state["softLowConfidenceReviewApplied"] = bool(soft_low_confidence_review)
        state["strongBlockedBuyPreserved"] = bool(strong_blocked_buy_preserved)
        state["recoveredFromInvalidation"] = bool(recovered_from_invalidation)
        state["previousStage"] = prior_stage
        state["reviewOutcome"] = review_outcome
        state["reviewReason"] = review_reason
        state["lastReview"] = {
            "at": now_iso,
            "fromStage": prior_stage,
            "toStage": stage,
            "stageChanged": bool(stage != prior_stage),
            "reviewOutcome": review_outcome,
            "reviewReason": review_reason,
            "strongBlockedBuyPreserved": bool(strong_blocked_buy_preserved),
            "promotionReady": bool(promotion_ready),
        }
        if stage != prior_stage:
            state["lastTransition"] = dict(state["lastReview"])

        self._cycle_events.append(
            {
                "productId": product_id,
                "fromStage": prior_stage,
                "toStage": stage,
                "stageChanged": bool(stage != prior_stage),
                "reviewOutcome": review_outcome,
                "reviewReason": review_reason,
                "primaryVetoBucket": primary_veto_bucket,
                "strongBlockedBuyPreserved": bool(strong_blocked_buy_preserved),
                "promotionReady": bool(promotion_ready),
            }
        )

        self._state.setdefault("items", {})[product_id] = state
        self._dirty = True

        return state, WatchlistPromotionSignal(
            stage=stage,
            prior_stage=prior_stage,
            promotion_ready=promotion_ready,
            blocked_reason=blocked_reason,
            blocked_reason_detail=blocked_reason_detail,
            hold_reason=hold_reason,
            review_outcome=review_outcome,
            review_reason=review_reason,
            primary_veto_bucket=primary_veto_bucket,
            veto_buckets=veto_buckets,
            hard_blocks=tuple(hard_blocks),
            soft_penalties=tuple(soft_penalties),
            confirmation_strength=float(confirmation_strength),
            exceptional_override_applied=exceptional_override_applied,
            strong_blocked_buy_preserved=bool(strong_blocked_buy_preserved),
        )
