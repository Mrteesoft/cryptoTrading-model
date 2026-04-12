"""Helpers for turning model predictions into human-readable spot-trading signals."""

from typing import Any, Dict, List, Sequence

import pandas as pd

from ..chart import build_chart_context, render_chart_snapshot

from ..config import TrainingConfig
from .policy import evaluate_trading_decision
from .symbols import is_signal_eligible_base_currency, normalize_base_currency


ACTION_PRIORITY = {
    "LOSS": 0,
    "BUY": 1,
    "TAKE_PROFIT": 2,
    "HOLD": 3,
}

READINESS_PRIORITY = {
    "high": 0,
    "medium": 1,
    "standby": 2,
    "blocked": 3,
}

SIGNAL_TO_ACTION = {
    "LOSS": "cut_loss",
    "BUY": "buy",
    "TAKE_PROFIT": "take_profit",
    "HOLD": "wait",
}

SIGNAL_TO_NUMERIC = {
    "LOSS": -1,
    "TAKE_PROFIT": -1,
    "HOLD": 0,
    "BUY": 1,
}

ACTIONABLE_SIGNAL_NAMES = {"BUY", "TAKE_PROFIT", "LOSS"}

READINESS_HEADLINE_WEIGHT = {
    "high": 0.30,
    "medium": 0.20,
    "standby": 0.10,
    "blocked": 0.0,
}


def _resolve_excluded_signal_bases(config: TrainingConfig | None) -> set[str]:
    """Return the configured base currencies that should never surface as signals."""

    if config is None:
        return set()

    return {
        normalize_base_currency(base_currency)
        for base_currency in getattr(config, "signal_excluded_base_currencies", ())
        if normalize_base_currency(base_currency)
    }


def is_signal_product_excluded(
    product_id: str | None = None,
    base_currency: str | None = None,
    config: TrainingConfig | None = None,
) -> bool:
    """Return whether one product belongs to the excluded signal universe."""

    excluded_bases = _resolve_excluded_signal_bases(config)
    normalized_base_currency = normalize_base_currency(base_currency)
    if not normalized_base_currency and product_id:
        normalized_base_currency = normalize_base_currency(str(product_id).split("-")[0])

    if not is_signal_eligible_base_currency(normalized_base_currency):
        return True

    if not excluded_bases:
        return False

    return normalized_base_currency in excluded_bases


def _safe_mapping_float(payload: Dict[str, Any] | None, key: str, default_value: float = 0.0) -> float:
    """Read one numeric value from an optional dictionary."""

    if not isinstance(payload, dict):
        return default_value

    raw_value = payload.get(key, default_value)
    if raw_value is None:
        return default_value

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default_value


def _safe_float(signal_row: pd.Series, column_name: str, default_value: float = 0.0) -> float:
    """Read a numeric value from a row without crashing on missing data."""

    if column_name not in signal_row.index:
        return default_value

    raw_value = signal_row[column_name]
    if pd.isna(raw_value):
        return default_value

    return float(raw_value)


def _safe_text(signal_row: pd.Series, column_name: str, default_value: str = "") -> str:
    """Read a text value from a row without crashing on missing data."""

    if column_name not in signal_row.index:
        return default_value

    raw_value = signal_row[column_name]
    if raw_value is None or pd.isna(raw_value):
        return default_value

    return str(raw_value)


def _is_public_signal_candidate(signal_summary: Dict[str, Any]) -> bool:
    """Return whether one signal is suitable for public trading output."""

    base_currency = normalize_base_currency(
        signal_summary.get("baseCurrency") or signal_summary.get("symbol")
    )
    if not is_signal_eligible_base_currency(base_currency):
        return False

    if any(character.isdigit() for character in base_currency) and len(base_currency) <= 3:
        return False

    coin_category = str(signal_summary.get("coinCategory", "") or "").strip().lower()
    coin_name = str(signal_summary.get("coinName", "") or "").strip().lower()
    if "stablecoin" in coin_category or "stablecoin" in coin_name:
        return False

    close_price = _safe_mapping_float(signal_summary, "close")
    market_context = signal_summary.get("marketContext") if isinstance(signal_summary.get("marketContext"), dict) else {}
    percent_change_24h = _safe_mapping_float(market_context, "cmcPercentChange24h")
    percent_change_7d = _safe_mapping_float(market_context, "cmcPercentChange7d")
    if (
        str(signal_summary.get("quoteCurrency", "")).strip().upper() == "USD"
        and 0.97 <= close_price <= 1.03
        and abs(percent_change_24h) <= 0.02
        and abs(percent_change_7d) <= 0.05
    ):
        return False

    return True


def filter_public_signal_summaries(signal_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop public-unfriendly assets such as pegged USD tokens and odd low-quality tickers."""

    return [
        dict(signal_summary)
        for signal_summary in signal_summaries
        if _is_public_signal_candidate(signal_summary)
    ]


def _build_signal_opener(product_id: str, signal_name: str) -> str:
    """Render one consistent opener line for the current signal name."""

    if signal_name == "BUY":
        return f"{product_id} is a BUY setup."
    if signal_name == "TAKE_PROFIT":
        return f"{product_id} is a TAKE_PROFIT setup for spot trading."
    if signal_name == "LOSS":
        return f"{product_id} is a LOSS setup for spot risk control."
    return f"{product_id} is a HOLD setup."


def _build_signal_chat_from_reasons(
    product_id: str,
    signal_name: str,
    reason_items: List[str],
) -> str:
    """Build one short explanation paragraph from finalized reason bullets."""

    return _build_signal_opener(product_id, signal_name) + " " + " ".join(reason_items)


def _apply_trade_context_override(
    signal_summary: Dict[str, Any],
    *,
    signal_name: str,
    reason: str,
    extra_reason_items: List[str] | None = None,
    trade_readiness: str | None = None,
    extra_fields: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Override one published signal with a lifecycle-aware trade-context state."""

    updated_signal = dict(signal_summary)
    product_id = str(updated_signal.get("productId", "")).strip().upper() or str(
        updated_signal.get("pairSymbol", updated_signal.get("symbol", "This coin"))
    )
    existing_reason_items = [
        str(item).strip()
        for item in list(updated_signal.get("reasonItems") or [])
        if str(item).strip()
    ]
    merged_reason_items: list[str] = [reason]
    for item in list(extra_reason_items or []) + existing_reason_items:
        normalized_item = str(item).strip()
        if not normalized_item or normalized_item in merged_reason_items:
            continue
        merged_reason_items.append(normalized_item)

    updated_signal["predicted_signal"] = SIGNAL_TO_NUMERIC.get(signal_name, 0)
    updated_signal["signal_name"] = signal_name
    updated_signal["spotAction"] = SIGNAL_TO_ACTION.get(signal_name, "wait")
    updated_signal["actionable"] = signal_name in ACTIONABLE_SIGNAL_NAMES
    if trade_readiness is not None:
        updated_signal["tradeReadiness"] = str(trade_readiness)
    updated_signal["reasonItems"] = merged_reason_items[:4]
    updated_signal["reasonSummary"] = merged_reason_items[0]
    updated_signal["signalChat"] = _build_signal_chat_from_reasons(
        product_id=product_id,
        signal_name=signal_name,
        reason_items=updated_signal["reasonItems"],
    )
    updated_signal["tradeLifecycleSignalName"] = signal_name
    updated_signal["tradeLifecycleSpotAction"] = updated_signal["spotAction"]
    updated_signal["tradeLifecycleApplied"] = True
    updated_signal["tradeLifecycleReason"] = reason
    if extra_fields:
        updated_signal.update(extra_fields)

    return updated_signal


def apply_signal_trade_context(
    signal_summaries: List[Dict[str, Any]],
    active_trade_product_ids: List[str] | None = None,
    active_signal_context_by_product: Dict[str, Dict[str, Any]] | None = None,
    config: TrainingConfig | None = None,
) -> List[Dict[str, Any]]:
    """Apply lifecycle context that depends on portfolio and tracked-trade state."""

    config = config or TrainingConfig()
    filtered_signal_summaries = filter_public_signal_summaries(signal_summaries)
    active_product_id_set = {
        str(product_id).strip().upper()
        for product_id in (active_trade_product_ids or [])
        if str(product_id).strip()
    }
    active_signal_context_by_product = {
        str(product_id).strip().upper(): dict(trade_context or {})
        for product_id, trade_context in (active_signal_context_by_product or {}).items()
        if str(product_id).strip()
    }
    contextualized_signals: list[Dict[str, Any]] = []

    for raw_signal_summary in filtered_signal_summaries:
        signal_summary = dict(raw_signal_summary)
        product_id = str(signal_summary.get("productId", "")).strip().upper()
        signal_name = str(signal_summary.get("signal_name", "")).strip().upper()
        active_trade_context = active_signal_context_by_product.get(product_id, {})
        has_active_trade_context = product_id in active_product_id_set or bool(active_trade_context)

        entry_price = _safe_mapping_float(active_trade_context, "entryPrice")
        tracked_current_price = _safe_mapping_float(active_trade_context, "currentPrice")
        current_price = _safe_mapping_float(signal_summary, "close") or tracked_current_price
        stop_loss_price = _safe_mapping_float(active_trade_context, "stopLossPrice")
        take_profit_price = _safe_mapping_float(active_trade_context, "takeProfitPrice")
        unrealized_return = None
        if entry_price > 0 and current_price > 0:
            unrealized_return = (current_price / entry_price) - 1.0

        loss_cut_triggered = False
        if stop_loss_price > 0 and current_price > 0 and current_price <= stop_loss_price:
            loss_cut_triggered = True
        elif (
            unrealized_return is not None
            and unrealized_return <= float(config.brain_loss_cut_threshold)
        ):
            loss_cut_triggered = True

        profit_lock_triggered = False
        if take_profit_price > 0 and current_price > 0 and current_price >= take_profit_price:
            profit_lock_triggered = True
        elif (
            unrealized_return is not None
            and unrealized_return >= float(config.brain_profit_lock_threshold)
        ):
            profit_lock_triggered = True

        trade_context_payload = {
            "hasActiveTrade": bool(has_active_trade_context),
            "entryPrice": entry_price if entry_price > 0 else None,
            "currentPrice": current_price if current_price > 0 else None,
            "stopLossPrice": stop_loss_price if stop_loss_price > 0 else None,
            "takeProfitPrice": take_profit_price if take_profit_price > 0 else None,
            "unrealizedReturn": unrealized_return,
            "lossCutTriggered": bool(loss_cut_triggered),
            "profitLockTriggered": bool(profit_lock_triggered),
        }
        signal_summary["tradeContext"] = trade_context_payload
        signal_summary["tradeLifecycleSignalName"] = signal_name
        signal_summary["tradeLifecycleSpotAction"] = str(signal_summary.get("spotAction", "wait"))
        signal_summary["tradeLifecycleApplied"] = False

        if signal_name == "TAKE_PROFIT":
            take_profit_eligible = has_active_trade_context
            signal_summary["takeProfitEligible"] = take_profit_eligible
            signal_summary["takeProfitContext"] = "tracked-trade-or-position" if take_profit_eligible else "none"

            if not take_profit_eligible:
                suppression_reason = (
                    "The raw model leaned TAKE_PROFIT, but this was downgraded to HOLD because "
                    "there is no tracked entry or open position for this coin."
                )
                signal_summary = _apply_trade_context_override(
                    signal_summary,
                    signal_name="HOLD",
                    reason=suppression_reason,
                    trade_readiness="standby",
                    extra_fields={
                        "takeProfitSuppressed": True,
                        "takeProfitSuppressionReason": suppression_reason,
                    },
                )

        if not has_active_trade_context:
            contextualized_signals.append(signal_summary)
            continue

        if loss_cut_triggered or (signal_name == "TAKE_PROFIT" and unrealized_return is not None and unrealized_return < 0):
            loss_reason = (
                "The tracked trade is below the plan, so the public signal switches to LOSS to cut risk."
            )
            if stop_loss_price > 0 and current_price > 0 and current_price <= stop_loss_price:
                loss_reason = (
                    "Price has breached the tracked stop loss, so the signal switches to LOSS to protect capital."
                )
            elif unrealized_return is not None and unrealized_return <= float(config.brain_loss_cut_threshold):
                loss_reason = (
                    "The tracked trade is beyond the configured loss limit, so the signal switches to LOSS to protect capital."
                )
            signal_summary = _apply_trade_context_override(
                signal_summary,
                signal_name="LOSS",
                reason=loss_reason,
                extra_reason_items=[
                    "This coin is no longer behaving like the original long setup.",
                ],
                trade_readiness="high",
                extra_fields={
                    "lossSignalTriggered": True,
                    "lossSignalSource": "trade-context",
                },
            )
        elif signal_name == "TAKE_PROFIT":
            take_profit_reason = (
                "The tracked trade is in profit, so the public signal stays on TAKE_PROFIT to harvest gains."
            )
            signal_summary = _apply_trade_context_override(
                signal_summary,
                signal_name="TAKE_PROFIT",
                reason=take_profit_reason,
                trade_readiness="medium",
                extra_fields={
                    "takeProfitEligible": True,
                    "takeProfitContext": "tracked-trade-or-position",
                },
            )
        elif profit_lock_triggered:
            profit_reason = (
                "The tracked trade is in profit, so the public signal switches from entry mode to TAKE_PROFIT."
            )
            signal_summary = _apply_trade_context_override(
                signal_summary,
                signal_name="TAKE_PROFIT",
                reason=profit_reason,
                trade_readiness="medium",
                extra_fields={
                    "takeProfitEligible": True,
                    "takeProfitContext": "tracked-trade-or-position",
                },
            )
        elif signal_name == "BUY":
            hold_reason = (
                "A tracked trade already exists for this coin, so fresh BUY signals are treated as HOLD while the position develops."
            )
            signal_summary = _apply_trade_context_override(
                signal_summary,
                signal_name="HOLD",
                reason=hold_reason,
                extra_reason_items=[
                    "The system will wait for either a profit-taking setup or a loss-cut trigger before changing the exit state.",
                ],
                trade_readiness="standby",
            )
        elif signal_name == "HOLD":
            hold_reason = (
                "A tracked trade is already active, so the public signal remains HOLD while the system manages the open thesis."
            )
            signal_summary = _apply_trade_context_override(
                signal_summary,
                signal_name="HOLD",
                reason=hold_reason,
                trade_readiness="standby",
            )

        contextualized_signals.append(signal_summary)

    return contextualized_signals


def is_published_signal_summary(signal_summary: Dict[str, Any]) -> bool:
    """Return whether one lifecycle-aware signal should appear in the public feed."""

    signal_name = str(signal_summary.get("signal_name", "")).strip().upper()
    trade_context = signal_summary.get("tradeContext") if isinstance(signal_summary.get("tradeContext"), dict) else {}
    has_active_trade = bool(trade_context.get("hasActiveTrade", False))

    if signal_name == "BUY":
        return not has_active_trade and bool(signal_summary.get("actionable", False))

    if signal_name in {"HOLD", "TAKE_PROFIT", "LOSS"}:
        return has_active_trade

    return False


def filter_published_signal_summaries(signal_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only trade-ready entries and lifecycle states for already-called coins."""

    return [
        dict(signal_summary)
        for signal_summary in signal_summaries
        if is_published_signal_summary(signal_summary)
    ]


def _resolve_coin_symbol(signal_row: pd.Series) -> str:
    """Return the coin ticker in one explicit field for frontend use."""

    for column_name in ("cmc_symbol", "base_currency"):
        raw_value = signal_row.get(column_name)
        if raw_value is not None and not pd.isna(raw_value):
            text_value = str(raw_value).strip().upper()
            if text_value:
                return text_value

    product_id = signal_row.get("product_id")
    if product_id is not None and not pd.isna(product_id):
        return str(product_id).split("-")[0].upper()

    return ""


def _format_percent(decimal_value: float) -> str:
    """Format a decimal return as a readable percentage string."""

    return f"{decimal_value * 100:.2f}%"


def _collect_theme_tags(signal_row: pd.Series) -> List[str]:
    """Turn CoinMarketCap theme flags into a short readable list."""

    theme_map = {
        "cmc_has_ai_tag": "AI",
        "cmc_has_defi_tag": "DeFi",
        "cmc_has_layer1_tag": "Layer-1",
        "cmc_has_gaming_tag": "Gaming",
        "cmc_has_meme_tag": "Meme",
    }

    return [
        theme_name
        for column_name, theme_name in theme_map.items()
        if _safe_float(signal_row, column_name) >= 1
    ]


def _build_market_intelligence_context(
    signal_row: pd.Series,
    config: TrainingConfig | None = None,
) -> Dict[str, Any]:
    """Extract the market-wide CMC intelligence block from one signal row."""

    market_intelligence_available = bool(_safe_float(signal_row, "cmc_market_intelligence_available"))
    fear_threshold = float(
        getattr(config, "coinmarketcap_market_fear_threshold", 30.0)
        if config is not None
        else 30.0
    )
    greed_threshold = float(
        getattr(config, "coinmarketcap_market_greed_threshold", 65.0)
        if config is not None
        else 65.0
    )
    btc_dominance_risk_off_threshold = float(
        getattr(config, "coinmarketcap_market_btc_dominance_risk_off_threshold", 55.0)
        if config is not None
        else 55.0
    ) / 100.0

    fear_greed_value = _safe_float(signal_row, "cmc_market_fear_greed_value")
    btc_dominance = _safe_float(signal_row, "cmc_market_btc_dominance")
    btc_dominance_change_24h = _safe_float(signal_row, "cmc_market_btc_dominance_change_24h")
    risk_mode = "neutral"

    if market_intelligence_available:
        if (
            fear_greed_value <= fear_threshold
            or (btc_dominance >= btc_dominance_risk_off_threshold and btc_dominance_change_24h > 0)
        ):
            risk_mode = "risk_off"
        elif fear_greed_value >= greed_threshold and btc_dominance_change_24h <= 0:
            risk_mode = "risk_on"

    return {
        "available": market_intelligence_available,
        "lastUpdated": _safe_text(signal_row, "cmc_market_last_updated"),
        "fearGreedValue": fear_greed_value,
        "fearGreedClassification": _safe_text(signal_row, "cmc_market_fear_greed_classification"),
        "btcDominance": btc_dominance,
        "btcDominanceChange24h": btc_dominance_change_24h,
        "altcoinShare": _safe_float(signal_row, "cmc_market_altcoin_share"),
        "stablecoinShare": _safe_float(signal_row, "cmc_market_stablecoin_share"),
        "totalMarketCapChange24h": _safe_float(signal_row, "cmc_market_total_market_cap_change_24h"),
        "totalVolumeChange24h": _safe_float(signal_row, "cmc_market_total_volume_change_24h"),
        "riskMode": risk_mode,
    }


def _calculate_setup_score(signal_row: pd.Series) -> float:
    """Score how much the current features support the predicted action."""

    signal_name = str(signal_row["predicted_name"])
    score = 0.0

    breakout_up_20 = _safe_float(signal_row, "breakout_up_20")
    breakout_down_20 = _safe_float(signal_row, "breakout_down_20")
    range_position_20 = _safe_float(signal_row, "range_position_20")
    close_vs_ema_5 = _safe_float(signal_row, "close_vs_ema_5")
    relative_strength_1 = _safe_float(signal_row, "relative_strength_1")
    relative_strength_5 = _safe_float(signal_row, "relative_strength_5")
    momentum_10 = _safe_float(signal_row, "momentum_10")
    rsi_14 = _safe_float(signal_row, "rsi_14")
    cmc_percent_change_24h = _safe_float(signal_row, "cmc_percent_change_24h")
    cmc_percent_change_7d = _safe_float(signal_row, "cmc_percent_change_7d")
    theme_tags = _collect_theme_tags(signal_row)
    cmc_market_intelligence_available = _safe_float(signal_row, "cmc_market_intelligence_available")
    cmc_market_fear_greed_score = _safe_float(signal_row, "cmc_market_fear_greed_score")
    cmc_market_btc_dominance = _safe_float(signal_row, "cmc_market_btc_dominance")
    cmc_market_btc_dominance_change_24h = _safe_float(signal_row, "cmc_market_btc_dominance_change_24h")

    if signal_name == "BUY":
        score += 1.0 if breakout_up_20 > 0 else 0.0
        score += 0.5 if range_position_20 >= 0.80 else 0.0
        score += 1.0 if close_vs_ema_5 > 0 else 0.0
        score += 1.0 if relative_strength_1 > 0 else 0.0
        score += 1.0 if relative_strength_5 > 0 else 0.0
        score += 1.0 if momentum_10 > 0 else 0.0
        score += 0.5 if 50 <= rsi_14 <= 70 else 0.0
        score += 0.5 if cmc_percent_change_24h > 0 else 0.0
        score += 0.5 if cmc_percent_change_7d > 0 else 0.0
        score += 0.25 if theme_tags else 0.0
        score += 0.25 if (cmc_market_intelligence_available and cmc_market_fear_greed_score >= 0.60) else 0.0
        score += 0.25 if (
            cmc_market_intelligence_available
            and cmc_market_btc_dominance < 0.55
            and cmc_market_btc_dominance_change_24h <= 0
        ) else 0.0
    elif signal_name == "TAKE_PROFIT":
        score += 1.0 if close_vs_ema_5 < 0 else 0.0
        score += 1.0 if relative_strength_1 < 0 else 0.0
        score += 1.0 if relative_strength_5 < 0 else 0.0
        score += 1.0 if breakout_down_20 < 0 else 0.0
        score += 0.5 if range_position_20 <= 0.25 else 0.0
        score += 0.5 if cmc_percent_change_24h < 0 else 0.0
        score += 0.5 if rsi_14 >= 70 else 0.0
        score += 0.25 if (cmc_market_intelligence_available and cmc_market_fear_greed_score <= 0.35) else 0.0
        score += 0.25 if (
            cmc_market_intelligence_available
            and cmc_market_btc_dominance >= 0.55
            and cmc_market_btc_dominance_change_24h > 0
        ) else 0.0

    return score


def _build_base_reason_items(
    signal_row: pd.Series,
    signal_name: str,
) -> List[str]:
    """
    Build simple reason bullets from the current feature snapshot.

    These reasons are intentionally rule-based explanations layered on top of
    the ML output. The model decides the signal; the helper below translates
    the current feature state into plain-English support for that signal.
    """

    reasons: List[str] = []

    breakout_up_20 = _safe_float(signal_row, "breakout_up_20")
    breakout_down_20 = _safe_float(signal_row, "breakout_down_20")
    range_position_20 = _safe_float(signal_row, "range_position_20")
    close_vs_ema_5 = _safe_float(signal_row, "close_vs_ema_5")
    relative_strength_1 = _safe_float(signal_row, "relative_strength_1")
    relative_strength_5 = _safe_float(signal_row, "relative_strength_5")
    momentum_10 = _safe_float(signal_row, "momentum_10")
    rsi_14 = _safe_float(signal_row, "rsi_14")
    cmc_percent_change_24h = _safe_float(signal_row, "cmc_percent_change_24h")
    cmc_percent_change_7d = _safe_float(signal_row, "cmc_percent_change_7d")
    cmc_percent_change_30d = _safe_float(signal_row, "cmc_percent_change_30d")
    cmc_context_available = _safe_float(signal_row, "cmc_context_available")
    theme_tags = _collect_theme_tags(signal_row)
    market_intelligence_available = _safe_float(signal_row, "cmc_market_intelligence_available")
    fear_greed_value = _safe_float(signal_row, "cmc_market_fear_greed_value")
    fear_greed_classification = _safe_text(signal_row, "cmc_market_fear_greed_classification")
    btc_dominance = _safe_float(signal_row, "cmc_market_btc_dominance")
    btc_dominance_change_24h = _safe_float(signal_row, "cmc_market_btc_dominance_change_24h")

    if signal_name == "BUY":
        if breakout_up_20 > 0:
            reasons.append(
                "Price is trading above the prior 20-candle high by "
                f"{_format_percent(breakout_up_20)}, which supports a breakout entry."
            )
        elif range_position_20 >= 0.80:
            reasons.append("Price is trading near the top of its recent 20-candle range.")

        if close_vs_ema_5 > 0:
            reasons.append(
                "Price is above the 5-candle EMA by "
                f"{_format_percent(close_vs_ema_5)}, which supports short-term trend strength."
            )

        if relative_strength_1 > 0 or relative_strength_5 > 0:
            reasons.append(
                "The coin is outperforming the wider market "
                f"({ _format_percent(relative_strength_1) } over 1 candle, "
                f"{ _format_percent(relative_strength_5) } over 5 candles)."
            )

        if momentum_10 > 0:
            reasons.append(f"Ten-candle momentum is positive at {_format_percent(momentum_10)}.")

        if cmc_context_available:
            if cmc_percent_change_24h > 0:
                reasons.append(f"CoinMarketCap 24h performance is positive at {_format_percent(cmc_percent_change_24h)}.")
            if cmc_percent_change_7d > 0:
                reasons.append(f"CoinMarketCap 7d performance is positive at {_format_percent(cmc_percent_change_7d)}.")
            if theme_tags:
                reasons.append(
                    "CoinMarketCap theme tags add context for the project: "
                    + ", ".join(theme_tags) + "."
                )
        if market_intelligence_available:
            if fear_greed_value >= 65:
                reasons.append(
                    "CoinMarketCap Fear & Greed is supportive at "
                    f"{fear_greed_value:.0f} ({fear_greed_classification or 'Greed'})."
                )
            elif fear_greed_value <= 35:
                reasons.append(
                    "CoinMarketCap Fear & Greed is cautious at "
                    f"{fear_greed_value:.0f} ({fear_greed_classification or 'Fear'}), so long entries need selectivity."
                )
            if btc_dominance >= 0.55 and btc_dominance_change_24h > 0:
                reasons.append(
                    "BTC dominance is still firm at "
                    f"{_format_percent(btc_dominance)} and rising by {_format_percent(btc_dominance_change_24h)}, "
                    "which can cap broad altcoin follow-through."
                )

    elif signal_name == "TAKE_PROFIT":
        reasons.append(
            "This is a spot take-profit signal, not a short signal. "
            "The model prefers protecting gains or reducing exposure."
        )

        if close_vs_ema_5 < 0:
            reasons.append(
                "Price has slipped below the 5-candle EMA by "
                f"{_format_percent(abs(close_vs_ema_5))}, which suggests momentum is fading."
            )

        if relative_strength_1 < 0 or relative_strength_5 < 0:
            reasons.append(
                "Relative strength has weakened versus the market "
                f"({ _format_percent(relative_strength_1) } over 1 candle, "
                f"{ _format_percent(relative_strength_5) } over 5 candles)."
            )

        if breakout_down_20 < 0:
            reasons.append(
                "Price has broken below the prior 20-candle low by "
                f"{_format_percent(abs(breakout_down_20))}, which weakens the chart."
            )
        elif range_position_20 <= 0.25:
            reasons.append("Price has fallen toward the bottom of its recent 20-candle range.")

        if cmc_context_available and cmc_percent_change_24h < 0:
            reasons.append(
                "CoinMarketCap 24h performance has turned negative at "
                f"{_format_percent(cmc_percent_change_24h)}, which supports caution."
            )
        if market_intelligence_available and fear_greed_value <= 35:
            reasons.append(
                "Wider market sentiment is risk-off with CoinMarketCap Fear & Greed at "
                f"{fear_greed_value:.0f} ({fear_greed_classification or 'Fear'})."
            )

        if rsi_14 >= 70:
            reasons.append(f"RSI is elevated at {rsi_14:.1f}, so the move may be stretched for spot holders.")

    else:
        reasons.append("The model does not see a strong enough edge for a fresh spot entry right now.")

        if abs(relative_strength_1) < 0.01 and abs(relative_strength_5) < 0.02:
            reasons.append("Relative strength versus the market is close to neutral.")

        if -0.01 <= close_vs_ema_5 <= 0.01:
            reasons.append("Price is close to its 5-candle EMA, so the short-term trend is mixed.")

        if 40 <= rsi_14 <= 60:
            reasons.append(f"RSI is near the middle at {rsi_14:.1f}, which suggests no strong momentum edge.")

        if cmc_context_available and abs(cmc_percent_change_24h) < 0.02 and abs(cmc_percent_change_7d) < 0.05:
            reasons.append("CoinMarketCap performance context is relatively balanced, not strongly bullish or weak.")
        if market_intelligence_available and fear_greed_value <= 35:
            reasons.append(
                "Wider market sentiment is still cautious, with CoinMarketCap Fear & Greed at "
                f"{fear_greed_value:.0f} ({fear_greed_classification or 'Fear'})."
            )

    if cmc_context_available and cmc_percent_change_30d > 0.10 and signal_name == "BUY":
        reasons.append(f"CoinMarketCap 30d performance is also positive at {_format_percent(cmc_percent_change_30d)}.")

    if signal_name == "BUY" and len(reasons) <= 1:
        reasons.append(
            "The model still scores this as a buy, but the visible chart evidence is mixed, "
            "so this is a lower-conviction setup."
        )

    if not reasons:
        reasons.append("The signal comes from the model's combined price, market, and CoinMarketCap feature set.")

    return reasons[:4]


def _build_reason_items(
    signal_row: pd.Series,
    signal_name: str,
    raw_signal_name: str,
    confidence_gate_applied: bool,
    risk_gate_applied: bool,
    minimum_action_confidence: float,
    required_action_confidence: float,
    probability_margin: float,
    gate_reasons: List[str],
    policy_notes: List[str],
) -> List[str]:
    """Build explanation bullets, including confidence-gate context when needed."""

    base_reasons = _build_base_reason_items(signal_row=signal_row, signal_name=raw_signal_name)

    if confidence_gate_applied:
        confidence = _safe_float(signal_row, "confidence")
        gated_reasons = [
            f"The model leaned {raw_signal_name}, but confidence is {confidence:.1%}, below the "
            f"{minimum_action_confidence:.1%} action threshold.",
            "Treating this as HOLD until the edge is stronger.",
        ]
        gated_reasons.extend(gate_reasons[:2])
        gated_reasons.extend(base_reasons[:2])
        return gated_reasons[:4]

    if risk_gate_applied:
        gated_reasons = [
            f"The model leaned {raw_signal_name}, but the trading policy downgraded it to HOLD.",
        ]
        gated_reasons.extend(gate_reasons[:2])
        if raw_signal_name == "BUY":
            gated_reasons.append(
                f"This setup needs at least {required_action_confidence:.1%} confidence and a "
                f"{probability_margin:.1%} probability edge to justify fresh risk."
            )
        gated_reasons.extend(base_reasons[:1])
        return gated_reasons[:4]

    reasons = base_reasons
    if policy_notes and signal_name in {"BUY", "TAKE_PROFIT"}:
        reasons = reasons + policy_notes[:1]
    return reasons[:4]


def _build_signal_chat(
    signal_row: pd.Series,
    reason_items: List[str],
    signal_name: str,
) -> str:
    """Build one short explanation paragraph for the current signal."""

    product_id = str(signal_row.get("product_id", signal_row.get("base_currency", "This coin")))
    return _build_signal_chat_from_reasons(product_id=product_id, signal_name=signal_name, reason_items=reason_items)


def _row_to_signal_summary(
    signal_row: pd.Series,
    minimum_action_confidence: float = 0.0,
    config: TrainingConfig | None = None,
    chart_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Convert one prediction row into a JSON-friendly signal dictionary."""

    decision = evaluate_trading_decision(
        signal_row=signal_row,
        minimum_action_confidence=minimum_action_confidence,
        config=config,
    )
    raw_signal_name = str(decision["modelSignalName"])
    signal_name = str(decision["signalName"])
    model_predicted_signal = int(decision["modelPredictedSignal"])
    confidence = float(decision["confidence"])
    minimum_action_confidence = float(decision["minimumActionConfidence"])
    required_action_confidence = float(decision["requiredActionConfidence"])
    confidence_gate_applied = bool(decision["confidenceGateApplied"])
    risk_gate_applied = bool(decision["riskGateApplied"])
    probability_margin = float(decision["probabilityMargin"])
    gate_reasons = list(decision["gateReasons"])
    policy_notes = list(decision["policyNotes"])
    reason_items = _build_reason_items(
        signal_row=signal_row,
        signal_name=signal_name,
        raw_signal_name=raw_signal_name,
        confidence_gate_applied=confidence_gate_applied,
        risk_gate_applied=risk_gate_applied,
        minimum_action_confidence=minimum_action_confidence,
        required_action_confidence=required_action_confidence,
        probability_margin=probability_margin,
        gate_reasons=gate_reasons,
        policy_notes=policy_notes,
    )
    coin_symbol = _resolve_coin_symbol(signal_row)
    pair_symbol = str(signal_row.get("product_id", "")) if "product_id" in signal_row.index else ""
    spot_action = str(decision["spotAction"])
    model_spot_action = SIGNAL_TO_ACTION.get(raw_signal_name, "wait")

    base_chart_context = {
        "breakoutUp20": _safe_float(signal_row, "breakout_up_20"),
        "breakoutDown20": _safe_float(signal_row, "breakout_down_20"),
        "rangePosition20": _safe_float(signal_row, "range_position_20"),
        "closeVsEma5": _safe_float(signal_row, "close_vs_ema_5"),
        "relativeStrength1": _safe_float(signal_row, "relative_strength_1"),
        "relativeStrength5": _safe_float(signal_row, "relative_strength_5"),
        "rsi14": _safe_float(signal_row, "rsi_14"),
    }
    if chart_context:
        base_chart_context.update(chart_context)

    summary = {
        "time_step": int(signal_row["time_step"]),
        "close": float(signal_row["close"]),
        "predicted_signal": int(decision["predictedSignal"]),
        "signal_name": signal_name,
        "spotAction": spot_action,
        "actionable": bool(decision["actionable"]),
        "symbol": coin_symbol,
        "coinSymbol": coin_symbol,
        "pairSymbol": pair_symbol,
        "confidence": confidence,
        "minimumActionConfidence": minimum_action_confidence,
        "requiredActionConfidence": required_action_confidence,
        "confidenceGateApplied": confidence_gate_applied,
        "riskGateApplied": risk_gate_applied,
        "modelPredictedSignal": model_predicted_signal,
        "modelSignalName": raw_signal_name,
        "modelSpotAction": model_spot_action,
        "primaryProbability": float(decision["primaryProbability"]),
        "probabilityMargin": probability_margin,
        "tradeReadiness": str(decision["tradeReadiness"]),
        "policyScore": float(decision["policyScore"]),
        "policyNotes": policy_notes,
        "gateReasons": gate_reasons,
        "setupScore": _calculate_setup_score(signal_row),
        "probabilities": {
            "take_profit": float(signal_row["prob_take_profit"]),
            "hold": float(signal_row["prob_hold"]),
            "buy": float(signal_row["prob_buy"]),
        },
        "reasonItems": reason_items,
        "reasonSummary": reason_items[0],
        "signalChat": _build_signal_chat(signal_row, reason_items, signal_name=signal_name),
        "chartContext": base_chart_context,
        "executionContext": {
            "atrPct14": _safe_float(signal_row, "atr_pct_14"),
            "volumeVsSma20": _safe_float(signal_row, "volume_vs_sma_20"),
            "volumeZscore20": _safe_float(signal_row, "volume_zscore_20"),
            "cmcVolume24hLog": _safe_float(signal_row, "cmc_volume_24h_log"),
            "cmcNumMarketPairsLog": _safe_float(signal_row, "cmc_num_market_pairs_log"),
            "cmcRankScore": _safe_float(signal_row, "cmc_rank_score"),
        },
        "marketContext": {
            "cmcPercentChange24h": _safe_float(signal_row, "cmc_percent_change_24h"),
            "cmcPercentChange7d": _safe_float(signal_row, "cmc_percent_change_7d"),
            "cmcPercentChange30d": _safe_float(signal_row, "cmc_percent_change_30d"),
            "cmcContextAvailable": int(_safe_float(signal_row, "cmc_context_available")),
            "themeTags": _collect_theme_tags(signal_row),
            "marketIntelligence": _build_market_intelligence_context(signal_row, config=config),
        },
        "marketState": {
            "label": str(signal_row.get("market_regime_label", "unknown")),
            "code": int(_safe_float(signal_row, "market_regime_code")),
            "trendScore": _safe_float(signal_row, "regime_trend_score"),
            "volatilityRatio": _safe_float(signal_row, "regime_volatility_ratio", default_value=1.0),
            "isTrending": bool(_safe_float(signal_row, "regime_is_trending")),
            "isHighVolatility": bool(_safe_float(signal_row, "regime_is_high_volatility")),
        },
        "eventContext": {
            "eventCountNext7d": int(_safe_float(signal_row, "cmcal_event_count_next_7d")),
            "eventCountNext30d": int(_safe_float(signal_row, "cmcal_event_count_next_30d")),
            "hasEventNext7d": bool(_safe_float(signal_row, "cmcal_has_event_next_7d")),
            "daysToNextEvent": _safe_float(signal_row, "cmcal_days_to_next_event"),
        },
    }

    if "timestamp" in signal_row.index:
        summary["timestamp"] = str(signal_row["timestamp"])

    if "product_id" in signal_row.index:
        summary["productId"] = str(signal_row["product_id"])

    if "base_currency" in signal_row.index:
        summary["baseCurrency"] = str(signal_row["base_currency"])

    if "quote_currency" in signal_row.index:
        summary["quoteCurrency"] = str(signal_row["quote_currency"])

    if "cmc_name" in signal_row.index and pd.notna(signal_row["cmc_name"]):
        summary["coinName"] = str(signal_row["cmc_name"])

    if "cmc_category" in signal_row.index and pd.notna(signal_row["cmc_category"]):
        summary["coinCategory"] = str(signal_row["cmc_category"])

    return summary


def build_latest_signal_summary(
    prediction_df: pd.DataFrame,
    minimum_action_confidence: float = 0.0,
    config: TrainingConfig | None = None,
) -> Dict[str, Any]:
    """
    Convert the newest prediction row into a compact JSON-friendly summary.

    This is the file you would normally inspect first when asking:
    "What is the model saying right now?"
    """

    if prediction_df.empty:
        raise ValueError("No prediction rows were available. Check your feature generation step.")

    latest_row = prediction_df.iloc[-1]

    return _row_to_signal_summary(
        signal_row=latest_row,
        minimum_action_confidence=minimum_action_confidence,
        config=config,
    )


def build_latest_signal_summaries(
    prediction_df: pd.DataFrame,
    minimum_action_confidence: float = 0.0,
    config: TrainingConfig | None = None,
    protected_product_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Build the newest available signal for each asset in the prediction table.

    In a multi-coin dataset, one global "latest row" is not enough.
    We keep the newest prediction per `product_id` so the output can show
    the current model view across the full coin universe.
    """

    if prediction_df.empty:
        raise ValueError("No prediction rows were available. Check your feature generation step.")

    if "product_id" not in prediction_df.columns:
        return [
            build_latest_signal_summary(
                prediction_df=prediction_df,
                minimum_action_confidence=minimum_action_confidence,
                config=config,
            )
        ]

    latest_rows = (
        prediction_df
        .sort_values(["timestamp", "product_id", "time_step"])
        .groupby("product_id", as_index=False)
        .tail(1)
        .copy()
    )
    if config is not None and "timestamp" in latest_rows.columns:
        max_staleness_hours = float(getattr(config, "signal_max_staleness_hours", 0.0) or 0.0)
        if max_staleness_hours > 0:
            latest_rows["timestamp"] = pd.to_datetime(latest_rows["timestamp"], errors="coerce", utc=True)
            freshest_timestamp = latest_rows["timestamp"].max()
            if pd.notna(freshest_timestamp):
                freshness_cutoff = freshest_timestamp - pd.Timedelta(hours=max_staleness_hours)
                latest_rows = latest_rows.loc[latest_rows["timestamp"] >= freshness_cutoff].copy()

    window_map: dict[str, pd.DataFrame] = {}
    chart_context_by_product: dict[str, dict[str, Any]] = {}
    protected_product_id_set = {
        str(product_id).strip().upper()
        for product_id in (protected_product_ids or ())
        if str(product_id).strip()
    }
    if "product_id" in prediction_df.columns:
        window_size = int(getattr(config, "chart_feature_window", 60) or 60) if config is not None else 60
        for product_id, group in prediction_df.groupby("product_id"):
            window_df = group.sort_values("timestamp").tail(window_size).copy()
            window_map[str(product_id).upper()] = window_df
            chart_context_by_product[str(product_id).upper()] = build_chart_context(window_df, config=config)

    signal_summaries = []
    for _, signal_row in latest_rows.iterrows():
        product_id = str(signal_row.get("product_id", "")).strip().upper()
        chart_context = chart_context_by_product.get(product_id)
        signal_summaries.append(
            _row_to_signal_summary(
                signal_row=signal_row,
                minimum_action_confidence=minimum_action_confidence,
                config=config,
                chart_context=chart_context,
            )
        )
    signal_summaries = [
        signal_summary
        for signal_summary in signal_summaries
        if str(signal_summary.get("productId", "")).strip().upper() in protected_product_id_set
        or not is_signal_product_excluded(
            product_id=str(signal_summary.get("productId", "")),
            base_currency=str(signal_summary.get("baseCurrency", "")),
            config=config,
        )
    ]
    signal_summaries = filter_public_signal_summaries(signal_summaries)

    sorted_summaries = sorted(
        signal_summaries,
        key=lambda signal_summary: (
            ACTION_PRIORITY.get(signal_summary.get("signal_name", "HOLD"), 99),
            READINESS_PRIORITY.get(signal_summary.get("tradeReadiness", "standby"), 99),
            -float(signal_summary.get("policyScore", 0.0)),
            -float(signal_summary.get("setupScore", 0.0)),
            -float(signal_summary.get("confidence", 0.0)),
            str(signal_summary.get("productId", "")),
        ),
    )
    if config is not None and bool(getattr(config, "chart_snapshot_enabled", False)):
        _render_chart_snapshots(
            sorted_summaries,
            window_map=window_map,
            config=config,
        )

    return sorted_summaries


def _render_chart_snapshots(
    signal_summaries: List[Dict[str, Any]],
    *,
    window_map: Dict[str, pd.DataFrame],
    config: TrainingConfig,
) -> None:
    if not window_map:
        return
    max_signals = int(getattr(config, "chart_snapshot_max_signals", 6) or 6)
    snapshot_dir = getattr(config, "chart_snapshot_dir", None)
    if snapshot_dir is None:
        return
    snapshot_dir = snapshot_dir
    for signal_summary in signal_summaries[:max_signals]:
        product_id = str(signal_summary.get("productId", "")).strip().upper()
        if not product_id or product_id not in window_map:
            continue
        window_df = window_map[product_id]
        output_path = snapshot_dir / f"{product_id}.png"
        try:
            render_chart_snapshot(
                window_df,
                output_path=output_path,
                title=f"{product_id} signal snapshot",
            )
            chart_context = signal_summary.get("chartContext") or {}
            chart_context["chartSnapshotPath"] = str(output_path)
            signal_summary["chartContext"] = chart_context
        except Exception:
            continue


def build_actionable_signal_summaries(signal_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only entry and exit signals for easier spot-trading review."""

    actionable_signals = [
        signal_summary
        for signal_summary in signal_summaries
        if signal_summary["signal_name"] in ACTIONABLE_SIGNAL_NAMES
    ]

    return sorted(
        actionable_signals,
        key=lambda signal_summary: (
            ACTION_PRIORITY.get(signal_summary["signal_name"], 99),
            READINESS_PRIORITY.get(signal_summary.get("tradeReadiness", "standby"), 99),
            -float(signal_summary.get("policyScore", 0.0)),
            -float(signal_summary.get("setupScore", 0.0)),
            -float(signal_summary["confidence"]),
            str(signal_summary.get("productId", "")),
        ),
    )


def _headline_signal_score(signal_summary: Dict[str, Any]) -> float:
    """Compute one compact score for comparing featured-signal candidates."""

    signal_name = str(signal_summary.get("signal_name", "HOLD"))
    trade_readiness = str(signal_summary.get("tradeReadiness", "standby")).lower()
    action_bonus = {
        "LOSS": 2.3,
        "BUY": 2.0,
        "TAKE_PROFIT": 1.4,
        "HOLD": 0.0,
    }.get(signal_name, 0.0)

    return (
        action_bonus
        + READINESS_HEADLINE_WEIGHT.get(trade_readiness, 0.0)
        + float(signal_summary.get("policyScore", 0.0))
        + (0.12 * float(signal_summary.get("setupScore", 0.0)))
        + (0.50 * float(signal_summary.get("confidence", 0.0)))
    )


def select_primary_signal(
    signal_summaries: List[Dict[str, Any]],
    config: TrainingConfig | None = None,
    recent_primary_product_ids: List[str] | None = None,
) -> Dict[str, Any]:
    """Choose the signal to surface first in the main JSON output."""

    if not signal_summaries:
        raise ValueError("No signal summaries were available.")

    actionable_signals = build_actionable_signal_summaries(signal_summaries)
    primary_candidates = actionable_signals if actionable_signals else signal_summaries
    if not primary_candidates:
        raise ValueError("No primary-signal candidates were available.")

    if len(primary_candidates) == 1:
        return primary_candidates[0]

    rotation_enabled = bool(
        getattr(config, "signal_primary_rotation_enabled", True)
        if config is not None
        else True
    )
    if not rotation_enabled:
        return primary_candidates[0]

    recent_primary_product_ids = list(recent_primary_product_ids or [])
    if not recent_primary_product_ids:
        return primary_candidates[0]

    rotation_lookback = max(
        int(
            getattr(config, "signal_primary_rotation_lookback", 3)
            if config is not None
            else 3
        ),
        1,
    )
    candidate_window = max(
        int(
            getattr(config, "signal_primary_rotation_candidate_window", 4)
            if config is not None
            else 4
        ),
        1,
    )
    min_score_ratio = float(
        getattr(config, "signal_primary_rotation_min_score_ratio", 0.88)
        if config is not None
        else 0.88
    )

    recent_primary_set = {
        str(product_id).strip().upper()
        for product_id in recent_primary_product_ids[:rotation_lookback]
        if str(product_id).strip()
    }
    if not recent_primary_set:
        return primary_candidates[0]

    featured_candidates = primary_candidates[:candidate_window]
    strongest_candidate = featured_candidates[0]
    strongest_score = max(_headline_signal_score(strongest_candidate), 0.0001)

    for candidate in featured_candidates:
        candidate_product_id = str(candidate.get("productId", "")).strip().upper()
        if candidate_product_id in recent_primary_set:
            continue

        candidate_score = _headline_signal_score(candidate)
        if candidate_score >= strongest_score * min_score_ratio:
            return candidate

    return primary_candidates[0]
