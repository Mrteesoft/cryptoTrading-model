"""LightGBM-backed signal model implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...labels_core import signal_to_text
from ...modeling import BaseSignalModel, register_model


def _require_lightgbm() -> Any:
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for LightGBM models. "
            "Install it with `pip install lightgbm`."
        ) from exc
    return lgb


@register_model
class LightGBMClassifierSignalModel(BaseSignalModel):
    """LightGBM classifier for tabular signal prediction."""

    model_type = "lightgbmClassifierSignalModel"
    default_model_filename = "lightgbmClassifierSignalModel.pkl"

    def _build_estimator(self):
        lgb = _require_lightgbm()
        return lgb.LGBMClassifier(
            n_estimators=int(self.config.lightgbm_n_estimators),
            learning_rate=float(self.config.lightgbm_learning_rate),
            num_leaves=int(self.config.lightgbm_num_leaves),
            max_depth=int(self.config.lightgbm_max_depth),
            min_child_samples=int(self.config.lightgbm_min_child_samples),
            subsample=float(self.config.lightgbm_subsample),
            colsample_bytree=float(self.config.lightgbm_colsample_bytree),
            random_state=int(self.config.random_state),
        )


@register_model
class LightGBMRankerSignalModel(BaseSignalModel):
    """LightGBM ranker for opportunity ranking (not default signal inference)."""

    model_type = "lightgbmRankerSignalModel"
    default_model_filename = "lightgbmRankerSignalModel.pkl"

    def _build_estimator(self):
        lgb = _require_lightgbm()
        return lgb.LGBMRanker(
            n_estimators=int(self.config.lightgbm_n_estimators),
            learning_rate=float(self.config.lightgbm_learning_rate),
            num_leaves=int(self.config.lightgbm_num_leaves),
            max_depth=int(self.config.lightgbm_max_depth),
            min_child_samples=int(self.config.lightgbm_min_child_samples),
            subsample=float(self.config.lightgbm_subsample),
            colsample_bytree=float(self.config.lightgbm_colsample_bytree),
            random_state=int(self.config.random_state),
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_frame: pd.DataFrame | None = None,
    ) -> "LightGBMRankerSignalModel":
        self._validate_feature_columns()
        if "timestamp" not in train_df.columns:
            raise ValueError("LightGBM ranker requires a timestamp column for group sizing.")

        self.estimator = self._build_estimator()
        feature_frame = train_df[self.feature_columns]
        if "future_return" in train_df.columns:
            target_series = train_df["future_return"]
        else:
            target_series = train_df["target_signal"]

        group_sizes = (
            train_df.groupby("timestamp", sort=True)
            .size()
            .astype(int)
            .tolist()
        )
        self.estimator.fit(feature_frame, target_series, group=group_sizes)
        return self

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        self._validate_trained_model()
        usable_rows = feature_df.dropna(subset=self.feature_columns).copy()
        scores = self.estimator.predict(usable_rows[self.feature_columns])
        usable_rows["rank_score"] = scores

        ranked = usable_rows.copy()
        ranked["predicted_signal"] = np.where(scores >= np.nanmedian(scores), 1, 0)
        ranked["predicted_name"] = ranked["predicted_signal"].map(signal_to_text)
        ranked["prob_take_profit"] = 0.0
        ranked["prob_hold"] = np.where(ranked["predicted_signal"] == 0, 1.0, 0.0)
        ranked["prob_buy"] = np.where(ranked["predicted_signal"] == 1, 1.0, 0.0)
        ranked["confidence"] = 1.0

        return ranked

    def predict_proba(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("LightGBM ranker does not provide predict_proba. Use rank() instead.")

    def rank(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        self._validate_trained_model()
        usable_rows = feature_df.dropna(subset=self.feature_columns).copy()
        usable_rows["rank_score"] = self.estimator.predict(usable_rows[self.feature_columns])
        return usable_rows

    def metadata(self) -> dict[str, Any]:
        return {
            "modelType": self.model_type,
            "featureCount": len(self.feature_columns),
            "featurePreview": list(self.feature_columns[:10]),
            "supportsRank": True,
            "supportsProba": False,
            "notes": "Ranker models require rank() for outputs; predict_proba is unavailable.",
        }
