"""River-backed online signal model implementations."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from ...modeling import BaseSignalModel, CLASS_ORDER, register_model


def _require_river() -> Any:
    try:
        from river import compose, preprocessing, linear_model, multiclass  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "river is required for River online models. "
            "Install it with `pip install river`."
        ) from exc

    return compose, preprocessing, linear_model, multiclass


class RiverClassifierAdapter:
    """A tiny sklearn-like adapter around a River classifier."""

    def __init__(self, model) -> None:
        self.model = model
        self.classes_ = np.array(CLASS_ORDER, dtype=int)

    def fit(self, feature_frame: pd.DataFrame, target_series: pd.Series) -> "RiverClassifierAdapter":
        for _, row in feature_frame.iterrows():
            features = row.to_dict()
            label = target_series.loc[row.name]
            self.model = self.model.learn_one(features, label)
        return self

    def predict(self, feature_frame: pd.DataFrame) -> np.ndarray:
        predictions = []
        for _, row in feature_frame.iterrows():
            prediction = self.model.predict_one(row.to_dict())
            if prediction is None:
                prediction = 0
            predictions.append(prediction)
        return np.asarray(predictions)

    def predict_proba(self, feature_frame: pd.DataFrame) -> np.ndarray:
        probability_rows = []
        for _, row in feature_frame.iterrows():
            proba = self.model.predict_proba_one(row.to_dict()) or {}
            probability_rows.append([proba.get(label, 0.0) for label in CLASS_ORDER])
        return np.asarray(probability_rows, dtype=float)


@register_model
class RiverOnlineSignalModel(BaseSignalModel):
    """Online signal model using River for incremental learning."""

    model_type = "riverOnlineSignalModel"
    default_model_filename = "riverOnlineSignalModel.pkl"

    def _build_estimator(self):
        compose, preprocessing, linear_model, multiclass = _require_river()
        base_model = multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
        pipeline = compose.Pipeline(
            preprocessing.StandardScaler(),
            base_model,
        )
        return RiverClassifierAdapter(pipeline)

    def get_feature_importance(self) -> Dict[str, float]:
        return {}
