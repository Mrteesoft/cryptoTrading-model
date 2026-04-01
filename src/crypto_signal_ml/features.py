"""Class-based feature engineering for the crypto signal model."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


# This list is the exact set of columns the model will learn from.
# Keeping it in one place prevents accidental training/prediction mismatch.
FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_5",
    "momentum_10",
    "candle_body_pct",
    "range_pct",
    "volume_change_1",
    "volatility_5",
    "close_vs_sma_5",
    "close_vs_sma_10",
    "close_vs_ema_5",
    "rsi_14",
    "market_return_1",
    "relative_strength_1",
    "market_return_5",
    "relative_strength_5",
    "breakout_up_20",
    "breakout_down_20",
    "range_position_20",
    "volatility_compression_5_20",
    "cmc_context_available",
    "cmc_rank_score",
    "cmc_market_cap_log",
    "cmc_volume_24h_log",
    "cmc_percent_change_24h",
    "cmc_percent_change_7d",
    "cmc_percent_change_30d",
    "cmc_circulating_supply_ratio",
    "cmc_num_market_pairs_log",
    "cmc_tags_count",
    "cmc_platform_present",
    "cmc_is_mineable",
    "cmc_has_defi_tag",
    "cmc_has_ai_tag",
    "cmc_has_layer1_tag",
    "cmc_has_gaming_tag",
    "cmc_has_meme_tag",
]


class BaseFeatureEngineer(ABC):
    """
    Base class for feature engineering.

    The base class contains reusable indicator helpers.
    Subclasses combine those helpers in their own `build` strategy instead
    of rewriting the same formulas every time.
    """

    default_feature_columns: List[str] = FEATURE_COLUMNS
    asset_key_column = "product_id"

    def __init__(self, feature_columns: List[str] = None) -> None:
        self.feature_columns = feature_columns or list(self.default_feature_columns)

    def build(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a feature table from raw OHLCV data.

        The base class controls the common workflow:
        - copy the raw data
        - let the subclass add feature groups
        - return the finished feature table
        """

        feature_df = price_df.copy()
        self._add_features(feature_df)
        return feature_df

    @abstractmethod
    def _add_features(self, feature_df: pd.DataFrame) -> None:
        """Add feature columns to the working DataFrame."""

    @staticmethod
    def calculate_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI is a popular momentum indicator.
        A high RSI can suggest strong recent buying pressure.
        A low RSI can suggest strong recent selling pressure.
        """

        price_change = close_series.diff()

        # Positive values are gains. Negative values are losses.
        gains = price_change.clip(lower=0)
        losses = -price_change.clip(upper=0)

        average_gain = gains.rolling(window=period).mean()
        average_loss = losses.rolling(window=period).mean()

        relative_strength = average_gain / average_loss
        rsi = 100 - (100 / (1 + relative_strength))

        return rsi

    def _transform_by_asset(
        self,
        feature_df: pd.DataFrame,
        column_name: str,
        transform_function,
    ) -> pd.Series:
        """
        Apply a calculation per asset when a product column exists.

        This is the key DRY helper for multi-coin support.
        Without it, ETH rows could accidentally use BTC history when we
        calculate rolling indicators on a combined dataset.
        """

        if self.asset_key_column in feature_df.columns:
            return feature_df.groupby(self.asset_key_column)[column_name].transform(transform_function)

        return transform_function(feature_df[column_name])

    def _get_numeric_column_or_default(
        self,
        feature_df: pd.DataFrame,
        column_name: str,
        default_value: float = 0.0,
    ) -> pd.Series:
        """
        Return a numeric column if it exists, otherwise a default-filled series.

        This makes optional enrichment data safe to use.
        If CoinMarketCap context is missing, the model still trains with zeros
        instead of crashing because the columns do not exist yet.
        """

        if column_name not in feature_df.columns:
            return pd.Series(default_value, index=feature_df.index, dtype=float)

        return pd.to_numeric(feature_df[column_name], errors="coerce").fillna(default_value)

    @staticmethod
    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Divide two series while preventing divide-by-zero explosions."""

        safe_denominator = denominator.replace(0, np.nan)
        return numerator / safe_denominator

    def _add_return_features(self, feature_df: pd.DataFrame) -> None:
        """Add return and momentum features."""

        feature_df["return_1"] = self._transform_by_asset(
            feature_df,
            "close",
            lambda close_series: close_series.pct_change(1),
        )
        feature_df["return_3"] = self._transform_by_asset(
            feature_df,
            "close",
            lambda close_series: close_series.pct_change(3),
        )
        feature_df["return_5"] = self._transform_by_asset(
            feature_df,
            "close",
            lambda close_series: close_series.pct_change(5),
        )
        feature_df["momentum_10"] = self._transform_by_asset(
            feature_df,
            "close",
            lambda close_series: close_series.pct_change(10),
        )

    def _add_candle_features(self, feature_df: pd.DataFrame) -> None:
        """Add candle shape features."""

        feature_df["candle_body_pct"] = (feature_df["close"] - feature_df["open"]) / feature_df["open"]
        feature_df["range_pct"] = (feature_df["high"] - feature_df["low"]) / feature_df["close"]

    def _add_volume_features(self, feature_df: pd.DataFrame) -> None:
        """Add volume-based features."""

        feature_df["volume_change_1"] = self._transform_by_asset(
            feature_df,
            "volume",
            lambda volume_series: volume_series.pct_change(1),
        )

    def _add_trend_features(self, feature_df: pd.DataFrame) -> None:
        """Add moving-average and volatility features."""

        feature_df["volatility_5"] = self._transform_by_asset(
            feature_df,
            "return_1",
            lambda return_series: return_series.rolling(window=5).std(),
        )
        feature_df["sma_5"] = self._transform_by_asset(
            feature_df,
            "close",
            lambda close_series: close_series.rolling(window=5).mean(),
        )
        feature_df["sma_10"] = self._transform_by_asset(
            feature_df,
            "close",
            lambda close_series: close_series.rolling(window=10).mean(),
        )
        feature_df["ema_5"] = self._transform_by_asset(
            feature_df,
            "close",
            lambda close_series: close_series.ewm(span=5, adjust=False).mean(),
        )
        feature_df["close_vs_sma_5"] = (feature_df["close"] / feature_df["sma_5"]) - 1
        feature_df["close_vs_sma_10"] = (feature_df["close"] / feature_df["sma_10"]) - 1
        feature_df["close_vs_ema_5"] = (feature_df["close"] / feature_df["ema_5"]) - 1

    def _add_momentum_features(self, feature_df: pd.DataFrame) -> None:
        """Add momentum indicators such as RSI."""

        feature_df["rsi_14"] = self._transform_by_asset(
            feature_df,
            "close",
            lambda close_series: self.calculate_rsi(close_series, period=14),
        )

    def _add_market_relative_features(self, feature_df: pd.DataFrame) -> None:
        """
        Add cross-sectional market context from the current timestamp.

        This helps the model answer questions like:
        - is this coin outperforming the average coin right now?
        - is its move special, or is the whole market moving together?
        """

        if "timestamp" in feature_df.columns:
            feature_df["market_return_1"] = feature_df.groupby("timestamp")["return_1"].transform("mean")
            feature_df["market_return_5"] = feature_df.groupby("timestamp")["return_5"].transform("mean")
        else:
            feature_df["market_return_1"] = feature_df["return_1"]
            feature_df["market_return_5"] = feature_df["return_5"]

        feature_df["market_return_1"] = feature_df["market_return_1"].fillna(0.0)
        feature_df["market_return_5"] = feature_df["market_return_5"].fillna(0.0)
        feature_df["relative_strength_1"] = feature_df["return_1"] - feature_df["market_return_1"]
        feature_df["relative_strength_5"] = feature_df["return_5"] - feature_df["market_return_5"]

    def _add_chart_pattern_features(self, feature_df: pd.DataFrame) -> None:
        """
        Add live chart-pattern style features from the candle structure itself.

        These are not pattern labels like "head and shoulders".
        Instead, they are reusable numeric measurements that describe patterns:
        - breakouts above recent highs
        - breakdowns below recent lows
        - where price sits inside its recent range
        - whether short volatility is compressed relative to a longer window
        """

        rolling_high_20 = self._transform_by_asset(
            feature_df,
            "high",
            lambda high_series: high_series.rolling(window=20).max(),
        )
        rolling_low_20 = self._transform_by_asset(
            feature_df,
            "low",
            lambda low_series: low_series.rolling(window=20).min(),
        )
        rolling_high_20_prev = self._transform_by_asset(
            feature_df,
            "high",
            lambda high_series: high_series.rolling(window=20).max().shift(1),
        )
        rolling_low_20_prev = self._transform_by_asset(
            feature_df,
            "low",
            lambda low_series: low_series.rolling(window=20).min().shift(1),
        )
        volatility_20 = self._transform_by_asset(
            feature_df,
            "return_1",
            lambda return_series: return_series.rolling(window=20).std(),
        )

        recent_range_width = (rolling_high_20 - rolling_low_20).replace(0, np.nan)

        feature_df["breakout_up_20"] = self._safe_ratio(
            feature_df["close"],
            rolling_high_20_prev,
        ) - 1
        feature_df["breakout_down_20"] = self._safe_ratio(
            feature_df["close"],
            rolling_low_20_prev,
        ) - 1
        feature_df["range_position_20"] = (feature_df["close"] - rolling_low_20) / recent_range_width
        feature_df["volatility_compression_5_20"] = self._safe_ratio(
            feature_df["volatility_5"],
            volatility_20,
        ) - 1

    def _add_coinmarketcap_context_features(self, feature_df: pd.DataFrame) -> None:
        """
        Add numeric features derived from CoinMarketCap market/fundamental context.

        These are designed to be robust even when the cache is missing:
        every feature falls back to zero so the training pipeline still works.
        """

        cmc_context_available = self._get_numeric_column_or_default(feature_df, "cmc_context_available")
        cmc_rank = self._get_numeric_column_or_default(feature_df, "cmc_rank")
        cmc_market_cap = self._get_numeric_column_or_default(feature_df, "cmc_market_cap")
        cmc_volume_24h = self._get_numeric_column_or_default(feature_df, "cmc_volume_24h")
        cmc_percent_change_24h = self._get_numeric_column_or_default(feature_df, "cmc_percent_change_24h")
        cmc_percent_change_7d = self._get_numeric_column_or_default(feature_df, "cmc_percent_change_7d")
        cmc_percent_change_30d = self._get_numeric_column_or_default(feature_df, "cmc_percent_change_30d")
        cmc_circulating_supply = self._get_numeric_column_or_default(feature_df, "cmc_circulating_supply")
        cmc_max_supply = self._get_numeric_column_or_default(feature_df, "cmc_max_supply")
        cmc_num_market_pairs = self._get_numeric_column_or_default(feature_df, "cmc_num_market_pairs")
        cmc_tags_count = self._get_numeric_column_or_default(feature_df, "cmc_tags_count")
        cmc_platform_present = self._get_numeric_column_or_default(feature_df, "cmc_platform_present")
        cmc_is_mineable = self._get_numeric_column_or_default(feature_df, "cmc_is_mineable")
        cmc_has_defi_tag = self._get_numeric_column_or_default(feature_df, "cmc_has_defi_tag")
        cmc_has_ai_tag = self._get_numeric_column_or_default(feature_df, "cmc_has_ai_tag")
        cmc_has_layer1_tag = self._get_numeric_column_or_default(feature_df, "cmc_has_layer1_tag")
        cmc_has_gaming_tag = self._get_numeric_column_or_default(feature_df, "cmc_has_gaming_tag")
        cmc_has_meme_tag = self._get_numeric_column_or_default(feature_df, "cmc_has_meme_tag")

        feature_df["cmc_context_available"] = cmc_context_available
        feature_df["cmc_rank_score"] = self._safe_ratio(
            pd.Series(1.0, index=feature_df.index, dtype=float),
            cmc_rank,
        ).fillna(0.0)
        feature_df["cmc_market_cap_log"] = np.log1p(cmc_market_cap.clip(lower=0.0))
        feature_df["cmc_volume_24h_log"] = np.log1p(cmc_volume_24h.clip(lower=0.0))
        feature_df["cmc_percent_change_24h"] = cmc_percent_change_24h / 100.0
        feature_df["cmc_percent_change_7d"] = cmc_percent_change_7d / 100.0
        feature_df["cmc_percent_change_30d"] = cmc_percent_change_30d / 100.0
        feature_df["cmc_circulating_supply_ratio"] = self._safe_ratio(
            cmc_circulating_supply,
            cmc_max_supply,
        ).fillna(0.0)
        feature_df["cmc_num_market_pairs_log"] = np.log1p(cmc_num_market_pairs.clip(lower=0.0))
        feature_df["cmc_tags_count"] = cmc_tags_count
        feature_df["cmc_platform_present"] = cmc_platform_present
        feature_df["cmc_is_mineable"] = cmc_is_mineable
        feature_df["cmc_has_defi_tag"] = cmc_has_defi_tag
        feature_df["cmc_has_ai_tag"] = cmc_has_ai_tag
        feature_df["cmc_has_layer1_tag"] = cmc_has_layer1_tag
        feature_df["cmc_has_gaming_tag"] = cmc_has_gaming_tag
        feature_df["cmc_has_meme_tag"] = cmc_has_meme_tag


class TechnicalFeatureEngineer(BaseFeatureEngineer):
    """
    Concrete feature engineer for the starter crypto model.

    The subclass references the base helper methods rather than rewriting
    the formulas inside one long function.
    """

    def _add_features(self, feature_df: pd.DataFrame) -> None:
        """
        Create model features from raw OHLCV data.

        Each feature group describes market behaviour from a different angle:
        - returns describe direction
        - moving averages describe trend
        - volatility describes instability
        - RSI describes momentum
        - candle range/body describe the current candle shape
        """

        self._add_return_features(feature_df)
        self._add_candle_features(feature_df)
        self._add_volume_features(feature_df)
        self._add_trend_features(feature_df)
        self._add_momentum_features(feature_df)
        self._add_market_relative_features(feature_df)
        self._add_chart_pattern_features(feature_df)
        self._add_coinmarketcap_context_features(feature_df)


def calculate_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
    """
    Backward-compatible helper that delegates to the base feature class.
    """

    return BaseFeatureEngineer.calculate_rsi(close_series, period=period)


def build_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible helper that delegates to the concrete feature class.
    """

    return TechnicalFeatureEngineer().build(price_df)
