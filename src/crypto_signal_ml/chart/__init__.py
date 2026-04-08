"""Chart-pattern intelligence utilities."""

from .levels import build_level_features
from .patterns import detect_candlestick_patterns
from .rendering import render_chart_snapshot
from .structure import build_structure_features


def build_chart_context(ohlcv_df, config=None) -> dict:
    """Build a combined chart-context payload from OHLCV history."""

    if ohlcv_df is None or ohlcv_df.empty:
        return {}
    required_columns = {"open", "high", "low", "close"}
    if not required_columns.issubset(set(ohlcv_df.columns)):
        return {}
    context = {}
    context.update(build_level_features(ohlcv_df, config=config))
    context.update(build_structure_features(ohlcv_df, config=config))
    context.update(detect_candlestick_patterns(ohlcv_df))
    return context


__all__ = [
    "build_chart_context",
    "build_level_features",
    "build_structure_features",
    "detect_candlestick_patterns",
    "render_chart_snapshot",
]
