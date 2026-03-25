"""Helpers for turning model predictions into human-readable spot-trading signals."""

from typing import Any, Dict, List

import pandas as pd


ACTION_PRIORITY = {
    "BUY": 0,
    "TAKE_PROFIT": 1,
    "HOLD": 2,
}


def _safe_float(signal_row: pd.Series, column_name: str, default_value: float = 0.0) -> float:
    """Read a numeric value from a row without crashing on missing data."""

    if column_name not in signal_row.index:
        return default_value

    raw_value = signal_row[column_name]
    if pd.isna(raw_value):
        return default_value

    return float(raw_value)


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
    elif signal_name == "TAKE_PROFIT":
        score += 1.0 if close_vs_ema_5 < 0 else 0.0
        score += 1.0 if relative_strength_1 < 0 else 0.0
        score += 1.0 if relative_strength_5 < 0 else 0.0
        score += 1.0 if breakout_down_20 < 0 else 0.0
        score += 0.5 if range_position_20 <= 0.25 else 0.0
        score += 0.5 if cmc_percent_change_24h < 0 else 0.0
        score += 0.5 if rsi_14 >= 70 else 0.0

    return score


def _build_reason_items(signal_row: pd.Series) -> List[str]:
    """
    Build simple reason bullets from the current feature snapshot.

    These reasons are intentionally rule-based explanations layered on top of
    the ML output. The model decides the signal; the helper below translates
    the current feature state into plain-English support for that signal.
    """

    signal_name = str(signal_row["predicted_name"])
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


def _build_signal_chat(signal_row: pd.Series, reason_items: List[str]) -> str:
    """Build one short explanation paragraph for the current signal."""

    product_id = str(signal_row.get("product_id", signal_row.get("base_currency", "This coin")))
    signal_name = str(signal_row["predicted_name"])

    if signal_name == "BUY":
        opener = f"{product_id} is a BUY setup."
    elif signal_name == "TAKE_PROFIT":
        opener = f"{product_id} is a TAKE_PROFIT setup for spot trading."
    else:
        opener = f"{product_id} is a HOLD setup."

    return opener + " " + " ".join(reason_items)


def _row_to_signal_summary(signal_row: pd.Series) -> Dict[str, Any]:
    """Convert one prediction row into a JSON-friendly signal dictionary."""

    reason_items = _build_reason_items(signal_row)
    signal_name = str(signal_row["predicted_name"])
    coin_symbol = _resolve_coin_symbol(signal_row)
    pair_symbol = str(signal_row.get("product_id", "")) if "product_id" in signal_row.index else ""
    spot_action = {
        "BUY": "buy",
        "TAKE_PROFIT": "take_profit",
        "HOLD": "wait",
    }.get(signal_name, "wait")

    summary = {
        "time_step": int(signal_row["time_step"]),
        "close": float(signal_row["close"]),
        "predicted_signal": int(signal_row["predicted_signal"]),
        "signal_name": signal_name,
        "spotAction": spot_action,
        "symbol": coin_symbol,
        "coinSymbol": coin_symbol,
        "pairSymbol": pair_symbol,
        "confidence": float(signal_row["confidence"]),
        "setupScore": _calculate_setup_score(signal_row),
        "probabilities": {
            "take_profit": float(signal_row["prob_take_profit"]),
            "hold": float(signal_row["prob_hold"]),
            "buy": float(signal_row["prob_buy"]),
        },
        "reasonItems": reason_items,
        "reasonSummary": reason_items[0],
        "signalChat": _build_signal_chat(signal_row, reason_items),
        "chartContext": {
            "breakoutUp20": _safe_float(signal_row, "breakout_up_20"),
            "breakoutDown20": _safe_float(signal_row, "breakout_down_20"),
            "rangePosition20": _safe_float(signal_row, "range_position_20"),
            "closeVsEma5": _safe_float(signal_row, "close_vs_ema_5"),
            "relativeStrength1": _safe_float(signal_row, "relative_strength_1"),
            "relativeStrength5": _safe_float(signal_row, "relative_strength_5"),
            "rsi14": _safe_float(signal_row, "rsi_14"),
        },
        "marketContext": {
            "cmcPercentChange24h": _safe_float(signal_row, "cmc_percent_change_24h"),
            "cmcPercentChange7d": _safe_float(signal_row, "cmc_percent_change_7d"),
            "cmcPercentChange30d": _safe_float(signal_row, "cmc_percent_change_30d"),
            "cmcContextAvailable": int(_safe_float(signal_row, "cmc_context_available")),
            "themeTags": _collect_theme_tags(signal_row),
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


def build_latest_signal_summary(prediction_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert the newest prediction row into a compact JSON-friendly summary.

    This is the file you would normally inspect first when asking:
    "What is the model saying right now?"
    """

    if prediction_df.empty:
        raise ValueError("No prediction rows were available. Check your feature generation step.")

    latest_row = prediction_df.iloc[-1]

    return _row_to_signal_summary(latest_row)


def build_latest_signal_summaries(prediction_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Build the newest available signal for each asset in the prediction table.

    In a multi-coin dataset, one global "latest row" is not enough.
    We keep the newest prediction per `product_id` so the output can show
    the current model view across the full coin universe.
    """

    if prediction_df.empty:
        raise ValueError("No prediction rows were available. Check your feature generation step.")

    if "product_id" not in prediction_df.columns:
        return [build_latest_signal_summary(prediction_df)]

    latest_rows = (
        prediction_df
        .sort_values(["timestamp", "product_id", "time_step"])
        .groupby("product_id", as_index=False)
        .tail(1)
        .copy()
    )
    latest_rows["signal_priority"] = latest_rows["predicted_name"].map(ACTION_PRIORITY).fillna(99)
    latest_rows["setup_score"] = latest_rows.apply(_calculate_setup_score, axis=1)
    latest_rows = (
        latest_rows
        .sort_values(
            ["signal_priority", "setup_score", "confidence", "product_id"],
            ascending=[True, False, False, True],
        )
        .reset_index(drop=True)
    )

    return [_row_to_signal_summary(signal_row) for _, signal_row in latest_rows.iterrows()]


def build_actionable_signal_summaries(signal_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only BUY and TAKE_PROFIT signals for easier spot-trading review."""

    actionable_signals = [
        signal_summary
        for signal_summary in signal_summaries
        if signal_summary["signal_name"] in {"BUY", "TAKE_PROFIT"}
    ]

    return sorted(
        actionable_signals,
        key=lambda signal_summary: (
            ACTION_PRIORITY.get(signal_summary["signal_name"], 99),
            -float(signal_summary.get("setupScore", 0.0)),
            -float(signal_summary["confidence"]),
            str(signal_summary.get("productId", "")),
        ),
    )


def select_primary_signal(signal_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose the signal to surface first in the main JSON output."""

    if not signal_summaries:
        raise ValueError("No signal summaries were available.")

    actionable_signals = build_actionable_signal_summaries(signal_summaries)
    if actionable_signals:
        return actionable_signals[0]

    return signal_summaries[0]
