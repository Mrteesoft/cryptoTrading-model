"""ATR-aware labelers for supervised signal training."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..config import TrainingConfig
from ..regimes import REGIME_LABEL_TO_CODE
from .contracts import LabelRecipe


SIGNAL_NAME_MAP = {
    -1: "TAKE_PROFIT",
    0: "HOLD",
    1: "BUY",
}


class BaseSignalLabeler(ABC):
    """Base class for turning future market behavior into supervised labels."""

    signal_name_map = SIGNAL_NAME_MAP
    asset_key_column = "product_id"

    @classmethod
    def signal_to_text(cls, signal_value: int) -> str:
        """Convert numeric labels into human-readable trading actions."""

        return cls.signal_name_map.get(int(signal_value), "UNKNOWN")

    def _attach_target_names(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """Add the readable text label beside the numeric target column."""

        output_df = labeled_df.copy()
        output_df["target_name"] = output_df["target_signal"].map(self.signal_name_map)
        return output_df

    def _shift_by_asset(
        self,
        labeled_df: pd.DataFrame,
        column_name: str,
        periods: int,
    ) -> pd.Series:
        """Shift a series per asset when the dataset contains multiple coins."""

        if self.asset_key_column in labeled_df.columns:
            return labeled_df.groupby(self.asset_key_column)[column_name].shift(periods)

        return labeled_df[column_name].shift(periods)

    @abstractmethod
    def add_labels(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Create model targets from a feature table."""

    @abstractmethod
    def label_recipe(self) -> LabelRecipe:
        """Describe the active label recipe in a versioned form."""


class FutureReturnSignalLabeler(BaseSignalLabeler):
    """Labeler that classifies candles from future returns for spot trading."""

    def __init__(
        self,
        prediction_horizon: int,
        buy_threshold: float,
        sell_threshold: float,
    ) -> None:
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def add_labels(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Create the target column the model will try to predict."""

        labeled_df = feature_df.copy()
        labeled_df["future_close"] = self._shift_by_asset(
            labeled_df=labeled_df,
            column_name="close",
            periods=-self.prediction_horizon,
        )
        labeled_df["future_return"] = (labeled_df["future_close"] / labeled_df["close"]) - 1
        labeled_df["target_signal"] = np.select(
            [
                labeled_df["future_return"] >= self.buy_threshold,
                labeled_df["future_return"] <= self.sell_threshold,
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return self._attach_target_names(labeled_df)

    def label_recipe(self) -> LabelRecipe:
        """Describe the active label recipe in a versioned form."""

        return LabelRecipe(
            version="future-return-v1",
            strategy="future_return",
            prediction_horizon=int(self.prediction_horizon),
            buy_threshold=float(self.buy_threshold),
            sell_threshold=float(self.sell_threshold),
            use_high_low=True,
            tie_break="na",
            use_atr_barriers=False,
            atr_period=0,
            buy_atr_multiplier=0.0,
            sell_atr_multiplier=0.0,
        )


class AtrTripleBarrierSignalLabeler(BaseSignalLabeler):
    """Triple-barrier labeler with optional ATR-scaled barriers."""

    def __init__(
        self,
        prediction_horizon: int,
        buy_threshold: float,
        sell_threshold: float,
        use_high_low: bool = True,
        tie_break: str = "stop_loss",
        use_atr_barriers: bool = True,
        atr_period: int = 14,
        buy_atr_multiplier: float = 1.25,
        sell_atr_multiplier: float = 1.00,
    ) -> None:
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1.")
        if buy_threshold <= 0:
            raise ValueError("buy_threshold must be positive for triple-barrier labeling.")
        if sell_threshold >= 0:
            raise ValueError("sell_threshold must be negative for triple-barrier labeling.")

        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.use_high_low = use_high_low
        self.tie_break = tie_break
        self.use_atr_barriers = use_atr_barriers
        self.atr_period = atr_period
        self.buy_atr_multiplier = buy_atr_multiplier
        self.sell_atr_multiplier = sell_atr_multiplier

    def add_labels(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Label rows by scanning the realized path candle by candle."""

        labeled_df = feature_df.copy()
        labeled_df["future_close"] = np.nan
        labeled_df["future_return"] = np.nan
        labeled_df["target_signal"] = 0
        labeled_df["label_barrier"] = None
        labeled_df["label_holding_period"] = np.nan
        labeled_df["label_upper_threshold"] = np.nan
        labeled_df["label_lower_threshold"] = np.nan
        labeled_df["label_recipe_version"] = self.label_recipe().version

        if self.asset_key_column in labeled_df.columns:
            grouped_indices = labeled_df.groupby(self.asset_key_column, sort=False).groups.values()
        else:
            grouped_indices = [labeled_df.index]

        for asset_index in grouped_indices:
            asset_df = labeled_df.loc[list(asset_index)]
            self._label_asset_rows(labeled_df=labeled_df, asset_df=asset_df)

        labeled_df["target_signal"] = labeled_df["target_signal"].astype(int)
        return self._attach_target_names(labeled_df)

    def label_recipe(self) -> LabelRecipe:
        """Describe the active label recipe in a versioned form."""

        return LabelRecipe(
            version="atr-triple-barrier-v1" if self.use_atr_barriers else "triple-barrier-v1",
            strategy="triple_barrier",
            prediction_horizon=int(self.prediction_horizon),
            buy_threshold=float(self.buy_threshold),
            sell_threshold=float(self.sell_threshold),
            use_high_low=bool(self.use_high_low),
            tie_break=str(self.tie_break),
            use_atr_barriers=bool(self.use_atr_barriers),
            atr_period=int(self.atr_period),
            buy_atr_multiplier=float(self.buy_atr_multiplier),
            sell_atr_multiplier=float(self.sell_atr_multiplier),
        )

    def _label_asset_rows(
        self,
        labeled_df: pd.DataFrame,
        asset_df: pd.DataFrame,
    ) -> None:
        """Apply the barrier scan to one asset history without mixing symbols."""

        close_values = asset_df["close"].astype(float).to_numpy()
        high_values = close_values
        low_values = close_values
        atr_values = pd.to_numeric(asset_df.get("atr_pct_14", self.buy_threshold), errors="coerce").to_numpy()

        if self.use_high_low and "high" in asset_df.columns:
            high_values = asset_df["high"].astype(float).to_numpy()
        if self.use_high_low and "low" in asset_df.columns:
            low_values = asset_df["low"].astype(float).to_numpy()

        row_indices = list(asset_df.index)

        for start_position, row_index in enumerate(row_indices):
            vertical_position = start_position + self.prediction_horizon
            if vertical_position >= len(row_indices):
                continue

            entry_close = close_values[start_position]
            if pd.isna(entry_close) or entry_close <= 0:
                continue

            upper_threshold, lower_threshold = self._resolve_barriers(atr_values[start_position])
            realized_return = np.nan
            realized_close = np.nan
            target_signal = 0
            barrier_name = "vertical"
            holding_period = self.prediction_horizon

            for forward_offset in range(1, self.prediction_horizon + 1):
                scan_position = start_position + forward_offset
                upper_return = (high_values[scan_position] / entry_close) - 1
                lower_return = (low_values[scan_position] / entry_close) - 1
                upper_hit = upper_return >= upper_threshold
                lower_hit = lower_return <= lower_threshold

                if not upper_hit and not lower_hit:
                    continue

                barrier_name = self._resolve_barrier_name(upper_hit=upper_hit, lower_hit=lower_hit)
                holding_period = forward_offset

                if barrier_name == "upper":
                    target_signal = 1
                    realized_return = upper_threshold
                    realized_close = entry_close * (1 + upper_threshold)
                else:
                    target_signal = -1
                    realized_return = lower_threshold
                    realized_close = entry_close * (1 + lower_threshold)
                break

            if pd.isna(realized_return):
                realized_close = close_values[vertical_position]
                realized_return = (realized_close / entry_close) - 1
                target_signal = 0
                barrier_name = "vertical"

            labeled_df.at[row_index, "future_close"] = realized_close
            labeled_df.at[row_index, "future_return"] = realized_return
            labeled_df.at[row_index, "target_signal"] = target_signal
            labeled_df.at[row_index, "label_barrier"] = barrier_name
            labeled_df.at[row_index, "label_holding_period"] = holding_period
            labeled_df.at[row_index, "label_upper_threshold"] = upper_threshold
            labeled_df.at[row_index, "label_lower_threshold"] = lower_threshold

    def _resolve_barriers(self, atr_value: float) -> tuple[float, float]:
        """Resolve the live upper and lower barriers for one labeled row."""

        if not self.use_atr_barriers or pd.isna(atr_value) or atr_value <= 0:
            return float(self.buy_threshold), float(self.sell_threshold)

        upper_threshold = max(float(self.buy_threshold), float(atr_value) * float(self.buy_atr_multiplier))
        lower_threshold = min(float(self.sell_threshold), -float(atr_value) * float(self.sell_atr_multiplier))
        return float(upper_threshold), float(lower_threshold)

    def _resolve_barrier_name(
        self,
        upper_hit: bool,
        lower_hit: bool,
    ) -> str:
        """Resolve ambiguous candles whose high and low cross both barriers."""

        if upper_hit and lower_hit:
            if self.tie_break == "take_profit":
                return "upper"
            return "both_stop_loss"

        if upper_hit:
            return "upper"

        return "lower"


TripleBarrierSignalLabeler = AtrTripleBarrierSignalLabeler


class MarketRegimeLabeler(BaseSignalLabeler):
    """Create explicit future market-state targets from the regime columns."""

    def __init__(self, prediction_horizon: int = 1) -> None:
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1 for regime labeling.")

        self.prediction_horizon = prediction_horizon

    def add_labels(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Attach current and future regime labels to the feature table."""

        if "market_regime_label" not in feature_df.columns:
            raise ValueError("market_regime_label is missing. Run regime enrichment before regime labeling.")

        labeled_df = feature_df.copy()
        labeled_df["current_market_regime_label"] = labeled_df["market_regime_label"].fillna("unknown").astype(str)
        labeled_df["current_market_regime_code"] = pd.to_numeric(
            labeled_df.get("market_regime_code", 0.0),
            errors="coerce",
        )
        labeled_df["target_market_regime_label"] = self._shift_by_asset(
            labeled_df=labeled_df,
            column_name="current_market_regime_label",
            periods=-self.prediction_horizon,
        )
        labeled_df["target_market_regime_code"] = self._shift_by_asset(
            labeled_df=labeled_df,
            column_name="current_market_regime_code",
            periods=-self.prediction_horizon,
        )

        if labeled_df["target_market_regime_code"].isna().any():
            mapped_target_codes = labeled_df["target_market_regime_label"].map(REGIME_LABEL_TO_CODE)
            labeled_df["target_market_regime_code"] = labeled_df["target_market_regime_code"].fillna(mapped_target_codes)

        labeled_df["target_market_regime_code"] = pd.to_numeric(
            labeled_df["target_market_regime_code"],
            errors="coerce",
        )
        labeled_df["market_regime_changed"] = (
            labeled_df["target_market_regime_label"].notna()
            & (labeled_df["target_market_regime_label"] != labeled_df["current_market_regime_label"])
        ).astype(float)
        labeled_df["market_regime_transition"] = np.where(
            labeled_df["target_market_regime_label"].notna(),
            labeled_df["current_market_regime_label"] + " -> " + labeled_df["target_market_regime_label"].astype(str),
            np.nan,
        )

        return labeled_df

    def label_recipe(self) -> LabelRecipe:
        """Describe the regime-label recipe in a versioned form."""

        return LabelRecipe(
            version="regime-transition-v1",
            strategy="regime_transition",
            prediction_horizon=int(self.prediction_horizon),
            buy_threshold=0.0,
            sell_threshold=0.0,
            use_high_low=True,
            tie_break="na",
            use_atr_barriers=False,
            atr_period=0,
            buy_atr_multiplier=0.0,
            sell_atr_multiplier=0.0,
        )


def create_labeler_from_config(config: TrainingConfig) -> BaseSignalLabeler:
    """Build the configured labeling strategy from one config object."""

    if config.labeling_strategy == "future_return":
        return FutureReturnSignalLabeler(
            prediction_horizon=config.prediction_horizon,
            buy_threshold=config.buy_threshold,
            sell_threshold=config.sell_threshold,
        )

    if config.labeling_strategy == "triple_barrier":
        return TripleBarrierSignalLabeler(
            prediction_horizon=config.prediction_horizon,
            buy_threshold=config.buy_threshold,
            sell_threshold=config.sell_threshold,
            use_high_low=config.triple_barrier_use_high_low,
            tie_break=config.triple_barrier_tie_break,
            use_atr_barriers=bool(getattr(config, "triple_barrier_use_atr", True)),
            atr_period=int(getattr(config, "triple_barrier_atr_period", 14)),
            buy_atr_multiplier=float(getattr(config, "triple_barrier_buy_atr_multiplier", 1.25)),
            sell_atr_multiplier=float(getattr(config, "triple_barrier_sell_atr_multiplier", 1.00)),
        )

    raise ValueError(
        "Unsupported labeling_strategy. "
        "Currently supported: future_return, triple_barrier"
    )


def create_regime_labeler_from_config(config: TrainingConfig) -> MarketRegimeLabeler:
    """Build the default regime labeler from the active training config."""

    return MarketRegimeLabeler(prediction_horizon=config.prediction_horizon)


def signal_to_text(signal_value: int) -> str:
    """Backward-compatible helper that delegates to the base labeler class."""

    return BaseSignalLabeler.signal_to_text(signal_value)


def add_signal_labels(
    feature_df: pd.DataFrame,
    prediction_horizon: int,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    """Backward-compatible helper that delegates to the concrete labeler class."""

    return FutureReturnSignalLabeler(
        prediction_horizon=prediction_horizon,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    ).add_labels(feature_df)


def add_regime_labels(
    feature_df: pd.DataFrame,
    prediction_horizon: int = 1,
) -> pd.DataFrame:
    """Backward-compatible helper for attaching explicit regime targets."""

    return MarketRegimeLabeler(
        prediction_horizon=prediction_horizon,
    ).add_labels(feature_df)
