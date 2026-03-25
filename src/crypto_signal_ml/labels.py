"""Class-based target creation for supervised learning."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


SIGNAL_NAME_MAP = {
    -1: "TAKE_PROFIT",
    0: "HOLD",
    1: "BUY",
}


class BaseSignalLabeler(ABC):
    """
    Base class for turning future market behavior into supervised labels.

    Different labeling strategies can inherit from this class and reuse
    the same signal-name mapping instead of rewriting it.
    """

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
        """
        Shift a series per asset when the dataset contains multiple coins.

        This prevents the last BTC candle from being paired with the first ETH
        candle when we create future-return targets.
        """

        if self.asset_key_column in labeled_df.columns:
            return labeled_df.groupby(self.asset_key_column)[column_name].shift(periods)

        return labeled_df[column_name].shift(periods)

    @abstractmethod
    def add_labels(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Create model targets from a feature table."""


class FutureReturnSignalLabeler(BaseSignalLabeler):
    """
    Labeler that classifies candles from future returns for spot trading.

    The rule is:
    - strong upward move becomes BUY
    - strong downward move becomes TAKE_PROFIT
    - everything in between becomes HOLD
    """

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
        """
        Create the target column the model will try to predict.

        The idea is:
        - look forward by `prediction_horizon` rows
        - calculate the future return from the current close
        - label strong upward move as BUY
        - label strong downward move as TAKE_PROFIT
        - everything between them becomes HOLD
        """

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
            [
                1,
                -1,
            ],
            default=0,
        ).astype(int)

        return self._attach_target_names(labeled_df)


def signal_to_text(signal_value: int) -> str:
    """
    Backward-compatible helper that delegates to the base labeler class.
    """

    return BaseSignalLabeler.signal_to_text(signal_value)


def add_signal_labels(
    feature_df: pd.DataFrame,
    prediction_horizon: int,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    """
    Backward-compatible helper that delegates to the concrete labeler class.
    """

    return FutureReturnSignalLabeler(
        prediction_horizon=prediction_horizon,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    ).add_labels(feature_df)
