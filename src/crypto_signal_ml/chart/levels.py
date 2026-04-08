"""Support/resistance and breakout detection."""

from __future__ import annotations

import numpy as np


def build_level_features(ohlcv_df, config=None) -> dict[str, float | bool]:
    """Compute support/resistance and breakout features."""

    if ohlcv_df is None or len(ohlcv_df) < 5:
        return {
            "supportPrice": None,
            "resistancePrice": None,
            "supportDistancePct": None,
            "resistanceDistancePct": None,
            "breakoutConfirmed": False,
            "retestHoldConfirmed": False,
            "nearResistance": False,
            "channelPosition": None,
        }

    close_series = ohlcv_df["close"].to_numpy(dtype=float)
    high_series = ohlcv_df["high"].to_numpy(dtype=float)
    low_series = ohlcv_df["low"].to_numpy(dtype=float)

    window = min(len(ohlcv_df), 30)
    rolling_high = np.max(high_series[-window:-1]) if window > 1 else np.max(high_series)
    rolling_low = np.min(low_series[-window:-1]) if window > 1 else np.min(low_series)

    latest_close = close_series[-1]
    breakout_buffer = float(getattr(config, "chart_breakout_buffer_pct", 0.005)) if config is not None else 0.005
    retest_tolerance = float(getattr(config, "chart_retest_tolerance_pct", 0.003)) if config is not None else 0.003
    near_resistance_pct = float(getattr(config, "chart_near_resistance_pct", 0.01)) if config is not None else 0.01

    breakout_confirmed = latest_close >= (rolling_high * (1.0 + breakout_buffer))
    recent_lows = low_series[-3:]
    retest_hold_confirmed = breakout_confirmed and np.min(recent_lows) >= (rolling_high * (1.0 - retest_tolerance))

    resistance_distance_pct = None
    support_distance_pct = None
    if rolling_high > 0:
        resistance_distance_pct = (rolling_high - latest_close) / rolling_high
    if rolling_low > 0:
        support_distance_pct = (latest_close - rolling_low) / rolling_low

    near_resistance = False
    if resistance_distance_pct is not None:
        near_resistance = resistance_distance_pct <= near_resistance_pct

    channel_position = None
    if rolling_high > rolling_low:
        channel_position = (latest_close - rolling_low) / (rolling_high - rolling_low)

    return {
        "supportPrice": float(rolling_low),
        "resistancePrice": float(rolling_high),
        "supportDistancePct": float(support_distance_pct) if support_distance_pct is not None else None,
        "resistanceDistancePct": float(resistance_distance_pct) if resistance_distance_pct is not None else None,
        "breakoutConfirmed": bool(breakout_confirmed),
        "retestHoldConfirmed": bool(retest_hold_confirmed),
        "nearResistance": bool(near_resistance),
        "channelPosition": float(channel_position) if channel_position is not None else None,
    }
