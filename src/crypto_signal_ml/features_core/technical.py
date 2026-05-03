"""Technical and context feature family declarations."""

from __future__ import annotations


DEFAULT_CONTEXT_TIMEFRAMES = ("4h", "1d")


def build_multi_timeframe_feature_columns(timeframes: tuple[str, ...]) -> list[str]:
    """Return aligned higher-timeframe feature names for the configured aliases."""

    feature_columns: list[str] = []
    for timeframe in timeframes:
        feature_columns.extend(
            [
                f"htf_{timeframe}_return_1",
                f"htf_{timeframe}_range_pct",
                f"htf_{timeframe}_volume_change_1",
                f"htf_{timeframe}_close_vs_sma_3",
                f"htf_{timeframe}_close_vs_ema_3",
                f"htf_{timeframe}_volatility_3",
            ]
        )

    return feature_columns


MULTI_TIMEFRAME_FEATURE_COLUMNS = build_multi_timeframe_feature_columns(DEFAULT_CONTEXT_TIMEFRAMES)

RETURNS_FEATURES = (
    "return_1",
    "return_3",
    "return_5",
    "return_12",
    "return_24",
    "candle_body_pct",
    "range_pct",
)

MOMENTUM_FEATURES = (
    "momentum_10",
    "rsi_14",
    "positive_close_ratio_10",
)

VOLUME_FEATURES = (
    "volume_change_1",
    "volume_change_5",
    "volume_vs_sma_20",
    "volume_vs_sma_50",
    "volume_zscore_20",
    "volume_trend_5_20",
)

VOLATILITY_FEATURES = (
    "volatility_5",
    "volatility_20",
    "atr_pct_14",
    "volatility_compression_5_20",
)

TREND_FEATURES = (
    "close_vs_sma_5",
    "close_vs_sma_10",
    "close_vs_sma_20",
    "close_vs_sma_50",
    "close_vs_ema_5",
    "close_vs_ema_20",
    "close_vs_ema_50",
    "trend_acceleration_5_20",
    "trend_acceleration_10_50",
)

MARKET_CONTEXT_FEATURES = (
    "market_return_1",
    "relative_strength_1",
    "market_return_5",
    "relative_strength_5",
    "market_return_24",
    "relative_strength_24",
    "market_breadth_1",
    "market_breadth_5",
    "market_breadth_24",
    "market_dispersion_1",
    "market_dispersion_5",
    "relative_strength_rank_1",
    "relative_strength_rank_5",
    "relative_strength_rank_24",
    "market_volatility_5",
    "market_trend_strength_20",
    "benchmark_btc_return_1",
    "benchmark_btc_return_5",
    "benchmark_btc_return_24",
    "asset_vs_btc_return_1",
    "asset_vs_btc_return_5",
    "asset_vs_btc_return_24",
)

TIME_CONTEXT_FEATURES = (
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
)
