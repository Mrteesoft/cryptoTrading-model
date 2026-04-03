"""Helpers for classifying market regime from the engineered feature table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import TrainingConfig


REGIME_FEATURE_COLUMNS = [
    "regime_trend_score",
    "regime_volatility_ratio",
    "regime_is_trending",
    "regime_is_high_volatility",
    "market_regime_code",
]

REGIME_LABEL_TO_CODE: Dict[str, int] = {
    "range": 0,
    "range_high_volatility": 1,
    "trend_up": 2,
    "trend_down": 3,
    "trend_up_high_volatility": 4,
    "trend_down_high_volatility": 5,
}


@dataclass(frozen=True)
class TrendRegimeBuilder:
    """Estimate whether the current row belongs to an uptrend, downtrend, or range."""

    trend_strength_threshold: float = 0.0125

    @staticmethod
    def _numeric_feature(feature_df: pd.DataFrame, column_name: str) -> pd.Series:
        if column_name not in feature_df.columns:
            return pd.Series(0.0, index=feature_df.index, dtype=float)

        return pd.to_numeric(feature_df[column_name], errors="coerce").fillna(0.0)

    def build(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Build trend regime columns from current and higher-timeframe trend features."""

        trend_score = (
            self._numeric_feature(feature_df, "close_vs_ema_20") * 0.45
            + self._numeric_feature(feature_df, "trend_acceleration_5_20") * 0.20
            + self._numeric_feature(feature_df, "htf_4h_close_vs_ema_3") * 0.25
            + self._numeric_feature(feature_df, "htf_1d_close_vs_ema_3") * 0.10
        )
        is_trending = trend_score.abs() >= float(self.trend_strength_threshold)
        trend_label = np.select(
            [
                trend_score >= float(self.trend_strength_threshold),
                trend_score <= -float(self.trend_strength_threshold),
            ],
            [
                "trend_up",
                "trend_down",
            ],
            default="range",
        )

        return pd.DataFrame(
            {
                "regime_trend_score": trend_score.astype(float),
                "regime_is_trending": is_trending.astype(float),
                "trend_regime_label": trend_label,
            },
            index=feature_df.index,
        )


@dataclass(frozen=True)
class VolatilityRegimeBuilder:
    """Estimate whether volatility is expanded relative to its own recent baseline."""

    high_volatility_ratio_threshold: float = 1.20
    low_volatility_ratio_threshold: float = 0.85

    @staticmethod
    def _numeric_feature(feature_df: pd.DataFrame, column_name: str) -> pd.Series:
        if column_name not in feature_df.columns:
            return pd.Series(0.0, index=feature_df.index, dtype=float)

        return pd.to_numeric(feature_df[column_name], errors="coerce").fillna(0.0)

    def build(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Build volatility regime columns from short- and medium-horizon volatility."""

        short_volatility = self._numeric_feature(feature_df, "volatility_5")
        medium_volatility = self._numeric_feature(feature_df, "volatility_20").replace(0, np.nan)
        volatility_ratio = (short_volatility / medium_volatility).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        is_high_volatility = volatility_ratio >= float(self.high_volatility_ratio_threshold)
        volatility_label = np.select(
            [
                volatility_ratio >= float(self.high_volatility_ratio_threshold),
                volatility_ratio <= float(self.low_volatility_ratio_threshold),
            ],
            [
                "high_volatility",
                "low_volatility",
            ],
            default="normal_volatility",
        )

        return pd.DataFrame(
            {
                "regime_volatility_ratio": volatility_ratio.astype(float),
                "regime_is_high_volatility": is_high_volatility.astype(float),
                "volatility_regime_label": volatility_label,
            },
            index=feature_df.index,
        )


class MarketRegimeDetector:
    """Combine trend and volatility regime builders into one market-state classifier."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.trend_builder = TrendRegimeBuilder(
            trend_strength_threshold=float(self.config.regime_trend_strength_threshold)
        )
        self.volatility_builder = VolatilityRegimeBuilder(
            high_volatility_ratio_threshold=float(self.config.regime_high_volatility_ratio_threshold),
            low_volatility_ratio_threshold=float(self.config.regime_low_volatility_ratio_threshold),
        )

    def enrich_feature_table(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Attach market regime labels and numeric regime features to the feature table."""

        output_df = feature_df.copy()
        trend_df = self.trend_builder.build(output_df)
        volatility_df = self.volatility_builder.build(output_df)

        output_df = pd.concat([output_df, trend_df, volatility_df], axis=1)
        output_df["market_regime_label"] = output_df.apply(self._resolve_market_regime_label, axis=1)
        output_df["market_regime_code"] = (
            output_df["market_regime_label"].map(REGIME_LABEL_TO_CODE).fillna(0).astype(float)
        )

        return output_df

    @staticmethod
    def _resolve_market_regime_label(signal_row: pd.Series) -> str:
        """Map trend and volatility states into one combined regime label."""

        trend_label = str(signal_row.get("trend_regime_label", "range"))
        volatility_label = str(signal_row.get("volatility_regime_label", "normal_volatility"))

        if trend_label == "trend_up":
            if volatility_label == "high_volatility":
                return "trend_up_high_volatility"
            return "trend_up"

        if trend_label == "trend_down":
            if volatility_label == "high_volatility":
                return "trend_down_high_volatility"
            return "trend_down"

        if volatility_label == "high_volatility":
            return "range_high_volatility"

        return "range"
