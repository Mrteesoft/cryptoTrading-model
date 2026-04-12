"""Persistent watchlist state for setup progression and promotion logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ..config import OUTPUTS_DIR, TrainingConfig


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


@dataclass
class WatchlistPromotionSignal:
    """Result of one watchlist progression check."""

    stage: str
    promotion_ready: bool
    blocked_reason: str | None
    hold_reason: str | None
    hard_blocks: tuple[str, ...] = ()
    soft_penalties: tuple[str, ...] = ()
    confirmation_strength: float = 0.0
    exceptional_override_applied: bool = False


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
            }

    def save(self) -> None:
        if not self._dirty:
            return
        self._state["updatedAt"] = _utc_now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        self._dirty = False

    def get_state(self, product_id: str) -> dict[str, Any] | None:
        return self._state.get("items", {}).get(product_id)

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

    def update_from_signal(
        self,
        *,
        signal_summary: dict[str, Any],
        decision_score: float,
        market_context: dict[str, Any],
    ) -> tuple[dict[str, Any], WatchlistPromotionSignal]:
        product_id = str(signal_summary.get("productId", "")).strip().upper()
        now_iso = _utc_now_iso()
        if not product_id:
            empty_state = {"productId": "", "stage": "observe", "lastReviewedAt": now_iso}
            return empty_state, WatchlistPromotionSignal(
                stage="observe",
                promotion_ready=False,
                blocked_reason=None,
                hold_reason=None,
            )

        state = dict(self._state.get("items", {}).get(product_id) or {})
        first_seen_at = state.get("firstSeenAt") or now_iso
        stage = str(state.get("stage") or "observe")

        signal_name = str(signal_summary.get("signal_name", "") or "").strip().upper()
        trade_readiness = str(signal_summary.get("tradeReadiness", "standby")).strip().lower()
        confidence = _safe_float(signal_summary.get("confidence")) or 0.0
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
        weak_structure = structure_label in {"lower_highs", "lower_lows", "downtrend"}
        confirmed_chart = breakout_confirmed or retest_hold_confirmed
        supportive_context = positive_trend or positive_news
        min_resistance_distance_pct = float(
            getattr(self.config, "signal_watchlist_entry_ready_min_resistance_distance_pct", 0.015) or 0.015
        )
        has_resistance_room = bool(
            confirmed_chart
            or (
                not near_resistance
                and (
                    resistance_distance_pct is None
                    or resistance_distance_pct >= min_resistance_distance_pct
                )
            )
        )
        prior_stage = stage

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
                "decisionScoreHistory": [],
                "tradeReadinessHistory": [],
                "trendScoreHistory": [],
                "volatilityRatioHistory": [],
                "lastSignalName": signal_name,
                "lastSpotAction": signal_summary.get("spotAction"),
            }
            prior_stage = stage

        state["lastReviewedAt"] = now_iso
        state["lastSignalName"] = signal_name
        state["lastSpotAction"] = signal_summary.get("spotAction")

        history_max = int(getattr(self.config, "signal_watchlist_history_max", 6) or 6)
        confidence_history = list(state.get("confidenceHistory") or [])
        decision_history = list(state.get("decisionScoreHistory") or [])
        readiness_history = list(state.get("tradeReadinessHistory") or [])
        trend_history = list(state.get("trendScoreHistory") or [])
        volatility_history = list(state.get("volatilityRatioHistory") or [])

        confidence_history.append({"at": now_iso, "value": confidence})
        decision_history.append({"at": now_iso, "value": float(decision_score)})
        readiness_history.append({"at": now_iso, "value": trade_readiness})
        if trend_score is not None:
            trend_history.append({"at": now_iso, "value": float(trend_score)})
        if volatility_ratio is not None:
            volatility_history.append({"at": now_iso, "value": float(volatility_ratio)})

        state["confidenceHistory"] = _bounded_history(confidence_history, history_max)
        state["decisionScoreHistory"] = _bounded_history(decision_history, history_max)
        state["tradeReadinessHistory"] = _bounded_history(readiness_history, history_max)
        state["trendScoreHistory"] = _bounded_history(trend_history, history_max)
        state["volatilityRatioHistory"] = _bounded_history(volatility_history, history_max)

        state["checks"] = int(state.get("checks") or 0) + 1

        min_confidence = float(self.config.signal_watchlist_promotion_min_confidence)
        min_decision_score = float(self.config.signal_watchlist_promotion_min_decision_score)
        positive_check = confidence >= min_confidence and decision_score >= min_decision_score
        if positive_check:
            state["positiveChecks"] = int(state.get("positiveChecks") or 0) + 1
            state["consecutivePositiveChecks"] = int(state.get("consecutivePositiveChecks") or 0) + 1
        else:
            state["consecutivePositiveChecks"] = 0

        first_confidence = confidence_history[0]["value"] if confidence_history else confidence
        first_decision_score = decision_history[0]["value"] if decision_history else float(decision_score)
        confidence_delta = float(confidence - (first_confidence or 0.0))
        decision_delta = float(decision_score - (first_decision_score or 0.0))
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

        invalidation_confidence = float(self.config.signal_watchlist_invalidation_confidence)
        invalidated = False
        if signal_name == "LOSS":
            invalidated = True
        if close_price is not None and invalidation_price is not None and close_price <= invalidation_price:
            invalidated = True
        if confidence <= invalidation_confidence:
            invalidated = True

        if invalidated:
            stage = "invalidated"
            state["invalidatedAt"] = now_iso
        else:
            setup_building_min_checks = int(
                getattr(self.config, "signal_watchlist_setup_building_min_checks", 2) or 2
            )
            min_positive_checks = int(self.config.signal_watchlist_promotion_min_positive_checks)
            min_conf_gain = float(self.config.signal_watchlist_promotion_min_confidence_gain)
            min_decision_gain = float(self.config.signal_watchlist_promotion_min_decision_score_gain)
            entry_ready_min_positive_checks = int(
                getattr(
                    self.config,
                    "signal_watchlist_entry_ready_min_positive_checks",
                    max(min_positive_checks + 1, 3),
                )
                or max(min_positive_checks + 1, 3)
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
                and (confirmed_chart or supportive_context)
            ):
                stage = "setup_confirmed"
            if (
                stage == "setup_confirmed"
                and positive_check
                and state.get("consecutivePositiveChecks", 0) >= entry_ready_min_positive_checks
                and confidence >= float(self.config.signal_watchlist_entry_ready_min_confidence)
                and decision_score >= float(self.config.signal_watchlist_entry_ready_min_decision_score)
                and has_resistance_room
                and confirmed_chart
            ):
                stage = "entry_ready"

        state["stage"] = stage
        if stage != prior_stage:
            state["lastStageChangeAt"] = now_iso
        else:
            state["lastStageChangeAt"] = state.get("lastStageChangeAt") or now_iso

        market_stance = str(market_context.get("marketStance", "balanced"))
        macro_risk_mode = str(market_context.get("macroRiskMode", "neutral"))
        hard_blocks: list[str] = []
        if trade_readiness == "blocked":
            hard_blocks.append("blocked_by_risk")
        if event_window_active or event_risk_flag:
            hard_blocks.append("blocked_by_event_risk")
        if post_event_cooldown_active:
            hard_blocks.append("await_post_event_confirmation")
        if weak_structure:
            hard_blocks.append("weak_chart_structure")
        if regime_label in {"trend_down", "trend_down_high_volatility"}:
            hard_blocks.append("severe_downtrend_regime")
        if not has_resistance_room:
            hard_blocks.append("near_resistance")

        soft_penalties: list[str] = []
        if market_stance == "defensive":
            soft_penalties.append("defensive_market")
        if macro_risk_mode == "risk_off":
            soft_penalties.append("macro_risk_off")
        if is_high_volatility:
            soft_penalties.append("high_volatility")
        if negative_news:
            soft_penalties.append("negative_news_conflict")

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
        if confirmed_chart:
            confirmation_strength += 0.17
        if has_resistance_room:
            confirmation_strength += 0.05
        if supportive_context:
            confirmation_strength += 0.03
        if positive_news and positive_trend:
            confirmation_strength += 0.02
        confirmation_strength = min(max(confirmation_strength, 0.0), 1.0)

        soft_risk_override_min_confirmation = float(
            getattr(self.config, "signal_watchlist_soft_risk_override_min_confirmation", 0.72) or 0.72
        )
        exceptional_override_applied = bool(
            stage == "entry_ready"
            and not hard_blocks
            and bool(soft_penalties)
            and confirmation_strength >= soft_risk_override_min_confirmation
            and confirmed_chart
            and confidence >= float(self.config.signal_watchlist_entry_ready_min_confidence)
            and decision_score >= float(self.config.signal_watchlist_entry_ready_min_decision_score)
        )

        blocked_reason = None
        if hard_blocks:
            blocked_reason = hard_blocks[0]
        elif stage == "entry_ready" and soft_penalties and not exceptional_override_applied:
            blocked_reason = soft_penalties[0]

        promotion_ready = (
            stage == "entry_ready"
            and trade_readiness in {"high", "medium"}
            and not hard_blocks
            and (not soft_penalties or exceptional_override_applied)
        )

        hold_reason = None
        if stage == "watchlist":
            hold_reason = "wait_for_setup_building"
        elif stage == "setup_building":
            hold_reason = "wait_for_setup_confirmation"
        elif stage == "setup_confirmed":
            hold_reason = "wait_for_breakout_confirmation"
            if retest_hold_confirmed:
                hold_reason = "wait_for_entry_window"
        elif stage == "entry_ready" and blocked_reason is not None:
            hold_reason = blocked_reason
        elif stage == "invalidated":
            hold_reason = "invalidated"
        elif supportive_context and hold_reason is None:
            hold_reason = "supported_by_positive_news"

        state["hardBlocks"] = list(hard_blocks)
        state["softPenalties"] = list(soft_penalties)
        state["confirmationStrength"] = float(confirmation_strength)
        state["promotionReady"] = bool(promotion_ready)
        state["blockedReason"] = blocked_reason
        state["holdReason"] = hold_reason

        self._state.setdefault("items", {})[product_id] = state
        self._dirty = True

        return state, WatchlistPromotionSignal(
            stage=stage,
            promotion_ready=promotion_ready,
            blocked_reason=blocked_reason,
            hold_reason=hold_reason,
            hard_blocks=tuple(hard_blocks),
            soft_penalties=tuple(soft_penalties),
            confirmation_strength=float(confirmation_strength),
            exceptional_override_applied=exceptional_override_applied,
        )
