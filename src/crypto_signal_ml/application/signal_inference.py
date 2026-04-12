"""Signal inference stage for normalized candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from ..config import TrainingConfig
from ..trading.signals import build_latest_signal_summaries


@dataclass
class SignalInferenceArtifacts:
    """Normalized outputs of one signal-inference pass."""

    prediction_df: pd.DataFrame
    signal_candidates: list[dict[str, Any]]
    summary: dict[str, Any]


class SignalInferenceStage:
    """Turn model predictions into normalized signal candidates."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    @staticmethod
    def combine_prediction_frames(prediction_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple prediction frames while keeping only the newest duplicate rows."""

        if not prediction_frames:
            return pd.DataFrame()
        if len(prediction_frames) == 1:
            return prediction_frames[0]

        combined_prediction_df = pd.concat(prediction_frames, ignore_index=True)
        duplicate_subset = [
            column_name
            for column_name in ("product_id", "timestamp", "time_step")
            if column_name in combined_prediction_df.columns
        ]
        if duplicate_subset:
            combined_prediction_df = combined_prediction_df.drop_duplicates(
                subset=duplicate_subset,
                keep="last",
            )

        return combined_prediction_df

    def build_from_prediction_frame(
        self,
        prediction_df: pd.DataFrame,
        *,
        summary: dict[str, Any] | None = None,
        mode: str = "historical-market-data",
        warning: str = "",
        max_products: int | None = None,
        products_requested: int | None = None,
        total_available_products: int | None = None,
        raise_on_empty: bool = True,
        empty_message: str | None = None,
        protected_product_ids: Sequence[str] | None = None,
    ) -> SignalInferenceArtifacts:
        """Build one inference artifact bundle from a prediction frame."""

        signal_candidates = build_latest_signal_summaries(
            prediction_df,
            minimum_action_confidence=self.config.backtest_min_confidence,
            config=self.config,
            protected_product_ids=protected_product_ids,
        )
        if raise_on_empty and not signal_candidates:
            raise ValueError(
                empty_message
                or "No signal summaries remained after applying the configured signal-universe exclusions."
            )

        if summary is None:
            product_count = (
                int(prediction_df["product_id"].nunique())
                if "product_id" in prediction_df.columns
                else int(len(prediction_df))
            )
            summary = {
                "mode": mode,
                "warning": warning,
                "maxProducts": max_products,
                "productsRequested": product_count if products_requested is None else int(products_requested),
                "totalAvailableProducts": total_available_products,
                "rowsScored": int(len(prediction_df)),
                "productsScored": product_count,
            }
        else:
            summary = dict(summary)
            summary.setdefault("mode", mode)
            summary.setdefault("warning", warning)

        return SignalInferenceArtifacts(
            prediction_df=prediction_df,
            signal_candidates=signal_candidates,
            summary=summary,
        )
