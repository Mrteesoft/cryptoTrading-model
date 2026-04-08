"""Candlestick pattern extraction."""

from __future__ import annotations

from typing import Any

import numpy as np


def _require_talib() -> Any:
    try:
        import talib  # type: ignore
    except ImportError:
        return None
    return talib


def _manual_patterns(open_series, high_series, low_series, close_series) -> dict[str, bool]:
    open_last = float(open_series[-1])
    close_last = float(close_series[-1])
    high_last = float(high_series[-1])
    low_last = float(low_series[-1])
    open_prev = float(open_series[-2]) if len(open_series) > 1 else open_last
    close_prev = float(close_series[-2]) if len(close_series) > 1 else close_last

    body = abs(close_last - open_last)
    candle_range = max(high_last - low_last, 1e-9)
    upper_shadow = high_last - max(open_last, close_last)
    lower_shadow = min(open_last, close_last) - low_last

    doji = body <= (0.1 * candle_range)
    hammer = lower_shadow >= (2.0 * body) and upper_shadow <= body
    shooting_star = upper_shadow >= (2.0 * body) and lower_shadow <= body
    bullish_engulfing = close_prev < open_prev and close_last > open_last and close_last >= open_prev and open_last <= close_prev
    bearish_engulfing = close_prev > open_prev and close_last < open_last and open_last >= close_prev and close_last <= open_prev

    return {
        "bullishEngulfing": bullish_engulfing,
        "bearishEngulfing": bearish_engulfing,
        "hammer": hammer,
        "shootingStar": shooting_star,
        "doji": doji,
    }


def detect_candlestick_patterns(ohlcv_df) -> dict[str, float | bool]:
    """Return candlestick pattern flags for the latest candle."""

    if ohlcv_df is None or len(ohlcv_df) < 2:
        return {
            "bullishEngulfing": False,
            "bearishEngulfing": False,
            "hammer": False,
            "shootingStar": False,
            "doji": False,
            "patternScore": 0.0,
        }

    open_series = ohlcv_df["open"].to_numpy(dtype=float)
    high_series = ohlcv_df["high"].to_numpy(dtype=float)
    low_series = ohlcv_df["low"].to_numpy(dtype=float)
    close_series = ohlcv_df["close"].to_numpy(dtype=float)

    talib = _require_talib()
    if talib is not None:
        bullish_engulfing = talib.CDLENGULFING(open_series, high_series, low_series, close_series)[-1] > 0
        bearish_engulfing = talib.CDLENGULFING(open_series, high_series, low_series, close_series)[-1] < 0
        hammer = talib.CDLHAMMER(open_series, high_series, low_series, close_series)[-1] > 0
        shooting_star = talib.CDLSHOOTINGSTAR(open_series, high_series, low_series, close_series)[-1] > 0
        doji = talib.CDLDOJI(open_series, high_series, low_series, close_series)[-1] != 0
        patterns = {
            "bullishEngulfing": bullish_engulfing,
            "bearishEngulfing": bearish_engulfing,
            "hammer": hammer,
            "shootingStar": shooting_star,
            "doji": doji,
        }
    else:
        patterns = _manual_patterns(open_series, high_series, low_series, close_series)

    score = 0.0
    if patterns["bullishEngulfing"] or patterns["hammer"]:
        score += 0.6
    if patterns["bearishEngulfing"] or patterns["shootingStar"]:
        score -= 0.6
    if patterns["doji"]:
        score *= 0.6

    return {
        **patterns,
        "patternScore": float(np.clip(score, -1.0, 1.0)),
    }
