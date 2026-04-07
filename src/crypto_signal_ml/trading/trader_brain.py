"""Portfolio-aware trading planner layered on top of signal summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence

from ..config import TrainingConfig
from .decision_intelligence import TradingDecisionDeliberator


UPTREND_LABELS = {"trend_up", "trend_up_high_volatility"}
DOWNTREND_LABELS = {"trend_down", "trend_down_high_volatility"}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp one numeric value between an inclusive minimum and maximum."""

    return max(min(float(value), maximum), minimum)


def _safe_float(payload: Mapping[str, Any], key: str, default_value: float = 0.0) -> float:
    """Read a numeric field from a mapping without raising."""

    raw_value = payload.get(key, default_value)
    if raw_value is None:
        return default_value

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default_value


def _safe_bool(payload: Mapping[str, Any], key: str, default_value: bool = False) -> bool:
    """Read a boolean-like field from a mapping without raising."""

    raw_value = payload.get(key, default_value)
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return default_value

    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_iso_timestamp(timestamp_value: Any) -> datetime | None:
    """Parse one optional ISO timestamp into a timezone-aware datetime."""

    if timestamp_value in {None, ""}:
        return None

    normalized_value = str(timestamp_value).replace("Z", "+00:00")
    try:
        parsed_value = datetime.fromisoformat(normalized_value)
    except ValueError:
        return None

    if parsed_value.tzinfo is None:
        return parsed_value.replace(tzinfo=timezone.utc)

    return parsed_value.astimezone(timezone.utc)


@dataclass(frozen=True)
class PositionState:
    """Simple spot-position state used by the trader brain."""

    product_id: str
    quantity: float = 0.0
    entry_price: float | None = None
    current_price: float | None = None
    position_fraction: float = 0.0
    age_hours: float | None = None

    def unrealized_return(self) -> float | None:
        """Return the current unrealized return when both prices exist."""

        if self.entry_price is not None and self.current_price is not None and self.entry_price > 0:
            return (self.current_price / self.entry_price) - 1.0

        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert the normalized position into a JSON-friendly dictionary."""

        return {
            "productId": self.product_id,
            "quantity": float(self.quantity),
            "entryPrice": float(self.entry_price) if self.entry_price is not None else None,
            "currentPrice": float(self.current_price) if self.current_price is not None else None,
            "positionFraction": float(self.position_fraction),
            "ageHours": float(self.age_hours) if self.age_hours is not None else None,
            "unrealizedReturn": self.unrealized_return(),
        }


class TraderBrain:
    """Build a portfolio-minded action plan from ranked signal summaries."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.deliberator = TradingDecisionDeliberator(config=self.config)

    def build_plan(
        self,
        signal_summaries: Sequence[Dict[str, Any]],
        positions: Sequence[Mapping[str, Any]] | None = None,
        capital: float | None = None,
        trade_memory_by_product: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Return enriched signals plus a portfolio action plan."""

        copied_signals = [dict(signal_summary) for signal_summary in signal_summaries]
        if not copied_signals:
            return {
                "version": "trader-brain-v1",
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "enabled": bool(self.config.brain_enabled),
                "marketStance": "balanced",
                "summary": "No signals were available for the trader brain.",
                "portfolio": {
                    "capital": capital,
                    "openPositions": [],
                    "openPositionCount": 0,
                    "currentExposureFraction": 0.0,
                    "remainingRiskBudgetFraction": float(self.config.brain_max_portfolio_risk_fraction),
                    "maxPortfolioRiskFraction": float(self.config.brain_max_portfolio_risk_fraction),
                    "maxEntryPositions": int(self.config.brain_max_entry_positions),
                    "availableEntrySlots": int(self.config.brain_max_entry_positions),
                },
                "plan": {
                    "entries": [],
                    "addOns": [],
                    "reductions": [],
                    "exits": [],
                    "holds": [],
                    "watchlist": [],
                    "newEntryCount": 0,
                    "addOnCount": 0,
                    "reduceCount": 0,
                    "exitCount": 0,
                    "watchlistCount": 0,
                },
                "signals": copied_signals,
            }

        positions_by_product = self._normalize_positions(
            positions=positions or [],
            signals=copied_signals,
            capital=capital,
        )
        market_context = self._build_market_context(copied_signals)
        open_position_rows = list(positions_by_product.values())
        current_exposure_fraction = sum(position.position_fraction for position in open_position_rows)
        remaining_risk_budget_fraction = max(
            float(self.config.brain_max_portfolio_risk_fraction) - current_exposure_fraction,
            0.0,
        )
        available_entry_slots = max(
            int(self.config.brain_max_entry_positions) - len(open_position_rows),
            0,
        )

        preliminary_rows = []
        new_entry_candidates = []
        add_on_candidates = []
        reductions = []
        exits = []
        holds = []

        for signal_summary in copied_signals:
            preliminary_row = self._build_preliminary_signal_plan(
                signal_summary=signal_summary,
                position=positions_by_product.get(str(signal_summary.get("productId", "")).upper()),
                market_context=market_context,
                capital=capital,
                trade_memory=(
                    (trade_memory_by_product or {}).get(str(signal_summary.get("productId", "")).upper())
                    or {}
                ),
            )
            preliminary_rows.append(preliminary_row)

            proposed_decision = preliminary_row["brain"]["proposedDecision"]
            if proposed_decision == "enter_long_candidate":
                new_entry_candidates.append(preliminary_row)
            elif proposed_decision == "add_to_winner_candidate":
                add_on_candidates.append(preliminary_row)
            elif proposed_decision == "reduce_position":
                reductions.append(preliminary_row)
            elif proposed_decision == "exit_position":
                exits.append(preliminary_row)
            elif proposed_decision.startswith("hold"):
                holds.append(preliminary_row)

        selected_entries = sorted(
            new_entry_candidates,
            key=lambda row: (
                -float(row["brain"]["decisionScore"]),
                str(row.get("productId", "")),
            ),
        )[:available_entry_slots]
        selected_add_ons = sorted(
            add_on_candidates,
            key=lambda row: (
                -float(row["brain"]["decisionScore"]),
                str(row.get("productId", "")),
            ),
        )

        scaled_allocations = self._allocate_exposure(
            exposure_candidates=selected_entries + selected_add_ons,
            remaining_risk_budget_fraction=remaining_risk_budget_fraction,
            capital=capital,
        )

        entries = []
        add_ons = []
        finalized_signals = []

        for preliminary_row in preliminary_rows:
            enriched_signal = dict(preliminary_row)
            product_id = str(enriched_signal.get("productId", "")).upper()
            brain = dict(enriched_signal["brain"])

            if product_id in scaled_allocations:
                allocation_row = scaled_allocations[product_id]
                brain["decision"] = allocation_row["decision"]
                brain["allocationFraction"] = allocation_row["allocationFraction"]
                brain["capitalAllocation"] = allocation_row["capitalAllocation"]
                brain["reasonSummary"] = allocation_row["reasonSummary"]
                brain["planRank"] = allocation_row["planRank"]
                brain["summaryLine"] = allocation_row["summaryLine"]
                if allocation_row["decision"] == "enter_long":
                    entries.append(allocation_row)
                else:
                    add_ons.append(allocation_row)
            else:
                proposed_decision = str(brain["proposedDecision"])
                if proposed_decision == "enter_long_candidate":
                    brain["decision"] = "watchlist"
                    brain["allocationFraction"] = 0.0
                    brain["capitalAllocation"] = 0.0 if capital is not None else None
                    brain["summaryLine"] = (
                        "Entry candidate stayed on the watchlist because the current plan ran out of slots or risk budget."
                    )
                elif proposed_decision == "add_to_winner_candidate":
                    brain["decision"] = "hold_position"
                    brain["allocationFraction"] = 0.0
                    brain["capitalAllocation"] = 0.0 if capital is not None else None
                    brain["summaryLine"] = (
                        "Existing position remains a hold because the current plan reserved no extra risk budget for adding."
                    )

            enriched_signal["brain"] = brain
            finalized_signals.append(enriched_signal)

        plan_summary = {
            "entries": entries,
            "addOns": add_ons,
            "reductions": [row["brain"] for row in reductions],
            "exits": [row["brain"] for row in exits],
            "holds": [row["brain"] for row in holds],
            "watchlist": [
                row["brain"]
                for row in finalized_signals
                if row["brain"]["decision"] in {"watchlist", "avoid_long"}
            ],
            "newEntryCount": len(entries),
            "addOnCount": len(add_ons),
            "reduceCount": len(reductions),
            "exitCount": len(exits),
            "watchlistCount": int(
                sum(
                    row["brain"]["decision"] in {"watchlist", "avoid_long"}
                    for row in finalized_signals
                )
            ),
        }

        return {
            "version": "trader-brain-v1",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "enabled": bool(self.config.brain_enabled),
            "marketStance": market_context["marketStance"],
            "summary": self._build_summary_text(
                market_context=market_context,
                entries=entries,
                add_ons=add_ons,
                reductions=reductions,
                exits=exits,
            ),
            "portfolio": {
                "capital": float(capital) if capital is not None else None,
                "openPositions": [position.to_dict() for position in open_position_rows],
                "openPositionCount": len(open_position_rows),
                "currentExposureFraction": float(current_exposure_fraction),
                "remainingRiskBudgetFraction": float(remaining_risk_budget_fraction),
                "maxPortfolioRiskFraction": float(self.config.brain_max_portfolio_risk_fraction),
                "maxEntryPositions": int(self.config.brain_max_entry_positions),
                "availableEntrySlots": int(available_entry_slots),
            },
            "plan": plan_summary,
            "signals": finalized_signals,
        }

    def _normalize_positions(
        self,
        positions: Sequence[Mapping[str, Any]],
        signals: Sequence[Dict[str, Any]],
        capital: float | None,
    ) -> dict[str, PositionState]:
        """Normalize optional user positions into a product-keyed dictionary."""

        signal_lookup = {
            str(signal_summary.get("productId", "")).upper(): signal_summary
            for signal_summary in signals
            if signal_summary.get("productId")
        }
        normalized_positions: dict[str, PositionState] = {}

        for raw_position in positions:
            product_id = str(
                raw_position.get("productId")
                or raw_position.get("product_id")
                or ""
            ).strip().upper()
            if not product_id:
                continue

            reference_signal = signal_lookup.get(product_id, {})
            current_price = _safe_float(raw_position, "currentPrice", default_value=0.0) or _safe_float(
                raw_position,
                "current_price",
                default_value=0.0,
            )
            if current_price <= 0:
                current_price = _safe_float(reference_signal, "close", default_value=0.0)

            quantity = _safe_float(raw_position, "quantity")
            entry_price = _safe_float(raw_position, "entryPrice", default_value=0.0) or _safe_float(
                raw_position,
                "entry_price",
                default_value=0.0,
            )
            position_fraction = _safe_float(raw_position, "positionFraction", default_value=0.0) or _safe_float(
                raw_position,
                "position_fraction",
                default_value=0.0,
            )
            if position_fraction <= 0 and capital and capital > 0 and quantity > 0 and current_price > 0:
                position_fraction = (quantity * current_price) / capital

            if not (position_fraction > 0 or quantity > 0):
                continue

            age_hours = None
            raw_age_hours = raw_position.get("ageHours", raw_position.get("age_hours"))
            if raw_age_hours not in {None, ""}:
                try:
                    age_hours = float(raw_age_hours)
                except (TypeError, ValueError):
                    age_hours = None
            if age_hours is None:
                opened_at = raw_position.get("openedAt", raw_position.get("opened_at"))
                opened_timestamp = _parse_iso_timestamp(opened_at)
                if opened_timestamp is not None:
                    age_hours = max(
                        (datetime.now(timezone.utc) - opened_timestamp).total_seconds() / 3600,
                        0.0,
                    )

            normalized_positions[product_id] = PositionState(
                product_id=product_id,
                quantity=quantity,
                entry_price=entry_price if entry_price > 0 else None,
                current_price=current_price if current_price > 0 else None,
                position_fraction=max(position_fraction, 0.0),
                age_hours=age_hours,
            )

        return normalized_positions

    def _build_market_context(self, signal_summaries: Sequence[Dict[str, Any]]) -> dict[str, Any]:
        """Classify the current market posture from the signal universe."""

        total_signals = max(len(signal_summaries), 1)
        buy_count = sum(signal_summary.get("signal_name") == "BUY" for signal_summary in signal_summaries)
        take_profit_count = sum(signal_summary.get("signal_name") == "TAKE_PROFIT" for signal_summary in signal_summaries)
        loss_count = sum(signal_summary.get("signal_name") == "LOSS" for signal_summary in signal_summaries)
        exit_signal_count = take_profit_count + loss_count
        trending_count = sum(
            bool((signal_summary.get("marketState") or {}).get("isTrending", False))
            for signal_summary in signal_summaries
        )
        high_volatility_count = sum(
            bool((signal_summary.get("marketState") or {}).get("isHighVolatility", False))
            for signal_summary in signal_summaries
        )

        regime_counts: dict[str, int] = {}
        for signal_summary in signal_summaries:
            regime_label = str((signal_summary.get("marketState") or {}).get("label", "unknown")).strip().lower()
            regime_counts[regime_label] = regime_counts.get(regime_label, 0) + 1

        dominant_regime = sorted(
            regime_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0] if regime_counts else "unknown"
        trending_share = trending_count / total_signals
        high_volatility_share = high_volatility_count / total_signals
        representative_market_intelligence: Mapping[str, Any] = {}
        for signal_summary in signal_summaries:
            market_context = signal_summary.get("marketContext") or {}
            if not isinstance(market_context, Mapping):
                continue
            market_intelligence = market_context.get("marketIntelligence") or {}
            if isinstance(market_intelligence, Mapping):
                representative_market_intelligence = market_intelligence
                if _safe_bool(market_intelligence, "available"):
                    break

        market_intelligence_available = _safe_bool(representative_market_intelligence, "available")
        fear_greed_value = _safe_float(representative_market_intelligence, "fearGreedValue")
        fear_greed_classification = str(
            representative_market_intelligence.get("fearGreedClassification", "") or ""
        ).strip()
        btc_dominance = _safe_float(representative_market_intelligence, "btcDominance")
        btc_dominance_change_24h = _safe_float(representative_market_intelligence, "btcDominanceChange24h")
        macro_risk_mode = str(representative_market_intelligence.get("riskMode", "neutral") or "neutral")

        base_market_stance = "balanced"
        if (
            dominant_regime in DOWNTREND_LABELS
            or high_volatility_share >= 0.50
            or exit_signal_count > buy_count
        ):
            base_market_stance = "defensive"
        elif (
            buy_count >= max(exit_signal_count + 1, 2)
            and trending_share >= 0.40
            and high_volatility_share < 0.45
        ):
            base_market_stance = "offensive"

        market_stance = base_market_stance
        if market_intelligence_available and macro_risk_mode == "risk_off":
            market_stance = "defensive"
        elif (
            market_intelligence_available
            and macro_risk_mode == "risk_on"
            and base_market_stance == "balanced"
            and buy_count >= exit_signal_count
        ):
            market_stance = "offensive"

        return {
            "marketStance": market_stance,
            "baseMarketStance": base_market_stance,
            "dominantRegime": dominant_regime,
            "buyCount": int(buy_count),
            "takeProfitCount": int(take_profit_count),
            "lossCount": int(loss_count),
            "trendingShare": float(trending_share),
            "highVolatilityShare": float(high_volatility_share),
            "marketIntelligenceAvailable": bool(market_intelligence_available),
            "fearGreedValue": float(fear_greed_value),
            "fearGreedClassification": fear_greed_classification,
            "btcDominance": float(btc_dominance),
            "btcDominanceChange24h": float(btc_dominance_change_24h),
            "macroRiskMode": macro_risk_mode,
        }

    def _build_preliminary_signal_plan(
        self,
        signal_summary: Dict[str, Any],
        position: PositionState | None,
        market_context: Mapping[str, Any],
        capital: float | None,
        trade_memory: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Build a preliminary action for one signal before portfolio allocation."""

        enriched_signal = dict(signal_summary)
        market_state = signal_summary.get("marketState") or {}
        event_context = signal_summary.get("eventContext") or {}
        product_id = str(signal_summary.get("productId", signal_summary.get("pairSymbol", ""))).upper()
        signal_name = str(signal_summary.get("signal_name", "HOLD")).upper()
        trade_readiness = str(signal_summary.get("tradeReadiness", "standby")).lower()
        confidence = _safe_float(signal_summary, "confidence")
        probability_margin = _safe_float(signal_summary, "probabilityMargin")
        setup_score = _safe_float(signal_summary, "setupScore")
        policy_score = _safe_float(signal_summary, "policyScore")
        volatility_ratio = _safe_float(market_state, "volatilityRatio", default_value=1.0)
        regime_label = str(market_state.get("label", "unknown")).strip().lower()
        is_high_volatility = _safe_bool(market_state, "isHighVolatility")
        has_event_next_7d = _safe_bool(event_context, "hasEventNext7d")
        macro_risk_mode = str(market_context.get("macroRiskMode", "neutral"))

        decision_score = self._build_decision_score(
            confidence=confidence,
            probability_margin=probability_margin,
            setup_score=setup_score,
            policy_score=policy_score,
            trade_readiness=trade_readiness,
            market_stance=str(market_context["marketStance"]),
            macro_risk_mode=macro_risk_mode,
            regime_label=regime_label,
            is_high_volatility=is_high_volatility,
            has_event_next_7d=has_event_next_7d,
        )

        desired_position_fraction = 0.0
        allocation_fraction = 0.0
        suggested_reduce_fraction = 0.0
        proposed_decision = "watchlist"
        reasons: list[str] = []
        position_unrealized_return = position.unrealized_return() if position is not None else None
        position_age_hours = position.age_hours if position is not None else None
        stale_position = bool(
            position_age_hours is not None
            and position_age_hours >= float(self.config.brain_stale_position_age_hours)
        )
        loss_cut_triggered = bool(
            position_unrealized_return is not None
            and position_unrealized_return <= float(self.config.brain_loss_cut_threshold)
        )
        profit_lock_triggered = bool(
            position_unrealized_return is not None
            and position_unrealized_return >= float(self.config.brain_profit_lock_threshold)
        )

        if position is None:
            if signal_name == "BUY":
                if not self.config.brain_enabled:
                    proposed_decision = "watchlist"
                    reasons.append("Trader brain is disabled, so the system only publishes the model signal.")
                elif trade_readiness == "blocked":
                    proposed_decision = "watchlist"
                    reasons.append("The policy has already blocked this setup from opening fresh risk.")
                elif str(market_context["marketStance"]) == "defensive" and trade_readiness != "high":
                    proposed_decision = "watchlist"
                    reasons.append("The wider market posture is defensive, so only the strongest long setups can open.")
                else:
                    proposed_decision = "enter_long_candidate"
                    desired_position_fraction = self._build_position_fraction(
                        confidence=confidence,
                        probability_margin=probability_margin,
                        setup_score=setup_score,
                        trade_readiness=trade_readiness,
                        market_stance=str(market_context["marketStance"]),
                        macro_risk_mode=macro_risk_mode,
                        regime_label=regime_label,
                        is_high_volatility=is_high_volatility,
                        has_event_next_7d=has_event_next_7d,
                    )
                    reasons.append("The setup qualifies as a fresh long candidate under the current market posture.")
            elif signal_name == "LOSS":
                proposed_decision = "avoid_long"
                reasons.append("The lifecycle signal is in loss-cut mode, so the system should avoid fresh long risk.")
            elif signal_name == "TAKE_PROFIT":
                proposed_decision = "avoid_long"
                reasons.append("The model is in capital-preservation mode for this coin, not fresh entry mode.")
            else:
                proposed_decision = "watchlist"
                reasons.append("The setup stays on the watchlist until a stronger directional edge appears.")
        else:
            if signal_name == "LOSS":
                proposed_decision = "exit_position"
                suggested_reduce_fraction = 1.0
                reasons.append("The open position has failed the trade plan and should be exited as a loss cut.")
            elif loss_cut_triggered and (
                regime_label in DOWNTREND_LABELS
                or str(market_context["marketStance"]) == "defensive"
                or is_high_volatility
                or signal_name != "BUY"
            ):
                proposed_decision = "exit_position"
                suggested_reduce_fraction = 1.0
                reasons.append(
                    "The open position has breached the configured loss limit and should be cut before damage compounds."
                )
            elif signal_name == "TAKE_PROFIT":
                if profit_lock_triggered:
                    proposed_decision = "exit_position"
                    suggested_reduce_fraction = 1.0
                    reasons.append("The trade is already deep enough in profit to lock in gains on this take-profit signal.")
                elif stale_position and confidence >= 0.70:
                    proposed_decision = "exit_position"
                    suggested_reduce_fraction = 1.0
                    reasons.append("The thesis is aging and the model is now telling the system to distribute risk.")
                elif is_high_volatility or regime_label in DOWNTREND_LABELS or confidence >= 0.78:
                    proposed_decision = "exit_position"
                    suggested_reduce_fraction = 1.0
                    reasons.append("The open position should be exited because the model is now signaling risk reduction.")
                else:
                    proposed_decision = "reduce_position"
                    suggested_reduce_fraction = float(self.config.brain_reduce_fraction)
                    reasons.append("The open position should be trimmed while the model cools off.")
            elif signal_name == "BUY":
                if profit_lock_triggered and (is_high_volatility or regime_label in DOWNTREND_LABELS or has_event_next_7d):
                    proposed_decision = "reduce_position"
                    suggested_reduce_fraction = float(self.config.brain_reduce_fraction)
                    reasons.append("The position is already well in profit, so the brain prefers to bank some gains into fresh uncertainty.")
                elif stale_position and trade_readiness != "high" and str(market_context["marketStance"]) != "offensive":
                    proposed_decision = "hold_and_tighten_risk"
                    reasons.append("The position is aging, so it can only stay open with tighter risk control.")
                elif (
                    trade_readiness == "high"
                    and str(market_context["marketStance"]) != "defensive"
                    and position.position_fraction < float(self.config.brain_base_position_fraction)
                ):
                    proposed_decision = "add_to_winner_candidate"
                    desired_position_fraction = min(
                        float(self.config.brain_scale_in_fraction),
                        max(
                            float(self.config.brain_base_position_fraction) - position.position_fraction,
                            0.0,
                        ),
                    )
                    reasons.append("The position can be added to because the trend and confidence are still aligned.")
                else:
                    proposed_decision = "hold_position"
                    reasons.append("The open position remains a hold under the current signal state.")
            elif stale_position and str(market_context["marketStance"]) != "offensive":
                proposed_decision = "reduce_position"
                suggested_reduce_fraction = float(self.config.brain_reduce_fraction)
                reasons.append("The thesis has gone stale without a fresh BUY signal, so exposure should be reduced.")
            else:
                proposed_decision = "hold_and_tighten_risk" if (is_high_volatility or has_event_next_7d) else "hold_position"
                if proposed_decision == "hold_and_tighten_risk":
                    reasons.append("The position stays open, but the risk leash should tighten because volatility or event risk is elevated.")
                else:
                    reasons.append("The position remains a patient hold while the model waits for a clearer edge.")

        if loss_cut_triggered:
            reasons.append("Unrealized drawdown is beyond the configured loss-cut threshold.")
        elif profit_lock_triggered:
            reasons.append("Unrealized return is above the configured profit-lock threshold.")
        if stale_position:
            reasons.append("The position has outlived the configured thesis age without a strong reset signal.")

        if regime_label in UPTREND_LABELS:
            reasons.append("Trend regime is constructive for long exposure.")
        elif regime_label in DOWNTREND_LABELS:
            reasons.append("Trend regime is working against fresh long risk.")
        if macro_risk_mode == "risk_off":
            reasons.append("CoinMarketCap market intelligence is currently risk-off for fresh altcoin exposure.")
        elif macro_risk_mode == "risk_on" and signal_name == "BUY":
            reasons.append("CoinMarketCap market intelligence is supportive for taking selective risk.")
        if is_high_volatility:
            reasons.append("High volatility argues for smaller size and faster risk control.")
        if has_event_next_7d:
            reasons.append("Upcoming event risk reduces the quality of a fresh entry.")

        stop_loss_pct, take_profit_pct = self._build_exit_levels(
            signal_name=signal_name,
            trade_readiness=trade_readiness,
            volatility_ratio=volatility_ratio,
        )
        current_price = _safe_float(signal_summary, "close")
        stop_loss_price = (current_price * (1.0 - stop_loss_pct)) if current_price > 0 else None
        take_profit_price = (current_price * (1.0 + take_profit_pct)) if current_price > 0 else None

        deliberation = self.deliberator.deliberate(
            signal_summary=signal_summary,
            base_decision=proposed_decision,
            base_decision_score=decision_score,
            base_reasons=reasons[:],
            position=position.to_dict() if position is not None else None,
            market_context=market_context,
            trade_memory=trade_memory or {},
            desired_position_fraction=desired_position_fraction,
            suggested_reduce_fraction=suggested_reduce_fraction,
            stale_position=stale_position,
            loss_cut_triggered=loss_cut_triggered,
            profit_lock_triggered=profit_lock_triggered,
        )
        decision_memo = deliberation["decisionMemo"]
        critic_review = deliberation["criticReview"]
        final_proposed_decision = str(critic_review.get("approvedDecision") or proposed_decision)
        size_multiplier = _clamp(float(critic_review.get("sizeMultiplier", 1.0) or 1.0), 0.0, 1.15)
        score_multiplier = _clamp(float(critic_review.get("scoreMultiplier", 1.0) or 1.0), 0.0, 1.0)

        if final_proposed_decision in {"enter_long_candidate", "add_to_winner_candidate"}:
            desired_position_fraction *= size_multiplier
            desired_position_fraction = _clamp(
                desired_position_fraction,
                float(self.config.brain_min_position_fraction),
                float(self.config.brain_max_position_fraction),
            )
        else:
            desired_position_fraction = 0.0

        if final_proposed_decision not in {"reduce_position", "exit_position"}:
            suggested_reduce_fraction = 0.0

        decision_score = _clamp(
            (decision_score * score_multiplier)
            + (_safe_float(decision_memo, "decisionConfidence") * 0.08),
            0.0,
            1.25,
        )

        reviewed_reasons: list[str] = []
        for reason in (
            *reasons,
            str(critic_review.get("summary") or "").strip(),
            str(decision_memo.get("thesis") or "").strip(),
        ):
            if not reason or reason in reviewed_reasons:
                continue
            reviewed_reasons.append(reason)
        reasons = reviewed_reasons

        enriched_signal["brain"] = {
            "productId": product_id,
            "decision": final_proposed_decision.replace("_candidate", ""),
            "proposedDecision": final_proposed_decision,
            "decisionScore": float(decision_score),
            "marketStance": str(market_context["marketStance"]),
            "macroRiskMode": macro_risk_mode,
            "desiredPositionFraction": float(desired_position_fraction),
            "allocationFraction": float(allocation_fraction),
            "capitalAllocation": (float(allocation_fraction * capital) if capital is not None else None),
            "suggestedReduceFraction": float(suggested_reduce_fraction),
            "stopLossPct": float(stop_loss_pct),
            "takeProfitPct": float(take_profit_pct),
            "stopLossPrice": float(stop_loss_price) if stop_loss_price is not None else None,
            "takeProfitPrice": float(take_profit_price) if take_profit_price is not None else None,
            "positionAgeHours": float(position_age_hours) if position_age_hours is not None else None,
            "positionUnrealizedReturn": position_unrealized_return,
            "thesisAgeIsStale": stale_position,
            "lossCutTriggered": loss_cut_triggered,
            "profitLockTriggered": profit_lock_triggered,
            "reasons": reasons[:4],
            "reasonSummary": reasons[0],
            "summaryLine": reasons[0],
            "position": position.to_dict() if position is not None else None,
            "evidence": deliberation["evidence"],
            "decisionMemo": decision_memo,
            "criticReview": critic_review,
        }

        return enriched_signal

    def _build_decision_score(
        self,
        confidence: float,
        probability_margin: float,
        setup_score: float,
        policy_score: float,
        trade_readiness: str,
        market_stance: str,
        macro_risk_mode: str,
        regime_label: str,
        is_high_volatility: bool,
        has_event_next_7d: bool,
    ) -> float:
        """Score one candidate for portfolio planning."""

        normalized_setup_score = _clamp(setup_score / 6.0, 0.0, 1.0)
        normalized_policy_score = _clamp(policy_score / 1.5, 0.0, 1.0)
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

        return _clamp(decision_score, 0.0, 1.25)

    def _build_position_fraction(
        self,
        confidence: float,
        probability_margin: float,
        setup_score: float,
        trade_readiness: str,
        market_stance: str,
        macro_risk_mode: str,
        regime_label: str,
        is_high_volatility: bool,
        has_event_next_7d: bool,
    ) -> float:
        """Size one new position from signal quality and current market posture."""

        size_fraction = float(self.config.brain_base_position_fraction)
        size_fraction *= 0.80 + (confidence * 0.55)
        size_fraction *= 0.90 + min(probability_margin * 1.20, 0.25)
        size_fraction *= 0.90 + min((setup_score / 6.0) * 0.25, 0.25)

        if trade_readiness == "high":
            size_fraction *= 1.10
        elif trade_readiness == "medium":
            size_fraction *= 1.00
        else:
            size_fraction *= 0.75

        if market_stance == "offensive":
            size_fraction *= 1.05
        elif market_stance == "defensive":
            size_fraction *= 0.72
        if macro_risk_mode == "risk_on":
            size_fraction *= 1.04
        elif macro_risk_mode == "risk_off":
            size_fraction *= 0.80

        if regime_label in UPTREND_LABELS:
            size_fraction *= 1.05
        elif regime_label in DOWNTREND_LABELS:
            size_fraction *= 0.55

        if is_high_volatility:
            size_fraction *= 0.78
        if has_event_next_7d:
            size_fraction *= 0.85

        return _clamp(
            size_fraction,
            float(self.config.brain_min_position_fraction),
            float(self.config.brain_max_position_fraction),
        )

    def _build_exit_levels(
        self,
        signal_name: str,
        trade_readiness: str,
        volatility_ratio: float,
    ) -> tuple[float, float]:
        """Build simple stop-loss and take-profit percentages for the plan."""

        normalized_volatility_ratio = _clamp(volatility_ratio, 0.8, 2.0)
        stop_loss_pct = _clamp(0.022 * normalized_volatility_ratio, 0.02, 0.06)
        take_profit_pct = stop_loss_pct * 2.0

        if signal_name == "TAKE_PROFIT":
            take_profit_pct = stop_loss_pct * 1.1
        elif trade_readiness == "high":
            take_profit_pct *= 1.15

        return stop_loss_pct, _clamp(take_profit_pct, 0.035, 0.12)

    def _allocate_exposure(
        self,
        exposure_candidates: Sequence[Dict[str, Any]],
        remaining_risk_budget_fraction: float,
        capital: float | None,
    ) -> dict[str, dict[str, Any]]:
        """Allocate remaining risk budget across entry and add-on candidates."""

        if not exposure_candidates or remaining_risk_budget_fraction <= 0:
            return {}

        total_requested_fraction = sum(
            float(candidate["brain"]["desiredPositionFraction"])
            for candidate in exposure_candidates
        )
        if total_requested_fraction <= 0:
            return {}

        scale_factor = min(remaining_risk_budget_fraction / total_requested_fraction, 1.0)
        allocations: dict[str, dict[str, Any]] = {}

        for plan_rank, candidate in enumerate(
            sorted(
                exposure_candidates,
                key=lambda row: (
                    -float(row["brain"]["decisionScore"]),
                    str(row.get("productId", "")),
                ),
            ),
            start=1,
        ):
            product_id = str(candidate.get("productId", "")).upper()
            desired_fraction = float(candidate["brain"]["desiredPositionFraction"])
            allocated_fraction = desired_fraction * scale_factor
            decision = (
                "add_to_winner"
                if candidate["brain"]["proposedDecision"] == "add_to_winner_candidate"
                else "enter_long"
            )
            reason_summary = (
                "Budget approved for adding to the existing winner."
                if decision == "add_to_winner"
                else "Budget approved for a fresh long entry."
            )
            allocations[product_id] = {
                "productId": product_id,
                "decision": decision,
                "decisionScore": float(candidate["brain"]["decisionScore"]),
                "allocationFraction": float(allocated_fraction),
                "capitalAllocation": (float(allocated_fraction * capital) if capital is not None else None),
                "stopLossPct": float(candidate["brain"]["stopLossPct"]),
                "takeProfitPct": float(candidate["brain"]["takeProfitPct"]),
                "stopLossPrice": candidate["brain"]["stopLossPrice"],
                "takeProfitPrice": candidate["brain"]["takeProfitPrice"],
                "reasonSummary": reason_summary,
                "reasons": list(candidate["brain"]["reasons"]),
                "planRank": int(plan_rank),
                "summaryLine": (
                    f"{product_id} {decision.replace('_', ' ')} with {allocated_fraction:.1%} portfolio allocation."
                ),
            }

        return allocations

    @staticmethod
    def _build_summary_text(
        market_context: Mapping[str, Any],
        entries: Sequence[Mapping[str, Any]],
        add_ons: Sequence[Mapping[str, Any]],
        reductions: Sequence[Mapping[str, Any]],
        exits: Sequence[Mapping[str, Any]],
    ) -> str:
        """Build one operator-facing summary sentence for the current plan."""

        market_stance = str(market_context.get("marketStance", "balanced"))
        dominant_regime = str(market_context.get("dominantRegime", "unknown"))
        macro_risk_mode = str(market_context.get("macroRiskMode", "neutral"))
        return (
            f"Trader brain is {market_stance} with dominant regime {dominant_regime} "
            f"and macro risk mode {macro_risk_mode}. "
            f"Plan: {len(entries)} new entries, {len(add_ons)} add-ons, "
            f"{len(reductions)} reductions, and {len(exits)} full exits."
        )
