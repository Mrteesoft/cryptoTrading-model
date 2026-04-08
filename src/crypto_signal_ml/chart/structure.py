"""Trend structure analysis for chart context."""

from __future__ import annotations

import numpy as np


def _linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    return float(slope / max(np.mean(values), 1e-9))


def build_structure_features(ohlcv_df, config=None) -> dict[str, float | bool | str | None]:
    """Compute trend structure and volatility compression features."""

    if ohlcv_df is None or len(ohlcv_df) < 5:
        return {
            "trendSlope": 0.0,
            "higherHighs": False,
            "higherLows": False,
            "lowerHighs": False,
            "lowerLows": False,
            "structureLabel": "unknown",
            "rangeCompressionScore": None,
            "atrCompression": None,
        }

    close_series = ohlcv_df["close"].to_numpy(dtype=float)
    high_series = ohlcv_df["high"].to_numpy(dtype=float)
    low_series = ohlcv_df["low"].to_numpy(dtype=float)

    slope = _linear_slope(close_series[-20:])

    recent_highs = high_series[-5:]
    recent_lows = low_series[-5:]
    higher_highs = bool(len(recent_highs) >= 3 and recent_highs[-1] > recent_highs[-2] > recent_highs[-3])
    higher_lows = bool(len(recent_lows) >= 3 and recent_lows[-1] > recent_lows[-2] > recent_lows[-3])
    lower_highs = bool(len(recent_highs) >= 3 and recent_highs[-1] < recent_highs[-2] < recent_highs[-3])
    lower_lows = bool(len(recent_lows) >= 3 and recent_lows[-1] < recent_lows[-2] < recent_lows[-3])

    structure_label = "range"
    if higher_highs and higher_lows:
        structure_label = "higher_highs"
    elif lower_highs and lower_lows:
        structure_label = "lower_lows"
    elif higher_lows and not higher_highs:
        structure_label = "higher_lows"
    elif lower_highs and not lower_lows:
        structure_label = "lower_highs"

    true_range = np.maximum(high_series - low_series, 1e-9)
    atr_window = min(len(true_range), 14)
    atr = np.mean(true_range[-atr_window:])
    atr_long = np.mean(true_range[-min(len(true_range), 40):])
    atr_ratio = float(atr / max(atr_long, 1e-9))
    range_compression_score = max(min(1.0 - atr_ratio, 1.0), -1.0)

    return {
        "trendSlope": float(slope),
        "higherHighs": higher_highs,
        "higherLows": higher_lows,
        "lowerHighs": lower_highs,
        "lowerLows": lower_lows,
        "structureLabel": structure_label,
        "rangeCompressionScore": float(range_compression_score),
        "atrCompression": float(atr_ratio),
    }
