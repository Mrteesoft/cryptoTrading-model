"""Post-hoc probability calibration helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .diagnostics import build_calibration_summary


class SigmoidProbabilityCalibrator:
    """Small one-vs-rest sigmoid calibrator layered on top of a fitted estimator."""

    def __init__(self, estimator: Any, class_labels: list[int]) -> None:
        self.estimator = estimator
        self.class_labels = [int(class_label) for class_label in class_labels]
        self.class_calibrators: dict[int, LogisticRegression | None] = {}

    def fit(self, feature_frame: pd.DataFrame, target_series: pd.Series) -> np.ndarray:
        """Fit per-class sigmoid calibrators on raw model probabilities."""

        raw_probabilities = self.estimator.predict_proba(feature_frame)
        for class_index, class_label in enumerate(self.class_labels):
            binary_target = (target_series == class_label).astype(int)
            if int(binary_target.nunique()) < 2:
                self.class_calibrators[class_label] = None
                continue

            calibrator = LogisticRegression(solver="lbfgs")
            calibrator.fit(raw_probabilities[:, [class_index]], binary_target)
            self.class_calibrators[class_label] = calibrator

        return raw_probabilities

    def predict_proba(self, feature_frame: pd.DataFrame) -> np.ndarray:
        """Return normalized calibrated probabilities for one feature frame."""

        raw_probabilities = self.estimator.predict_proba(feature_frame)
        calibrated_probabilities = np.zeros_like(raw_probabilities, dtype=float)

        for class_index, class_label in enumerate(self.class_labels):
            calibrator = self.class_calibrators.get(class_label)
            if calibrator is None:
                calibrated_probabilities[:, class_index] = raw_probabilities[:, class_index]
                continue

            calibrated_probabilities[:, class_index] = calibrator.predict_proba(
                raw_probabilities[:, [class_index]]
            )[:, 1]

        row_sums = calibrated_probabilities.sum(axis=1, keepdims=True)
        invalid_rows = row_sums.squeeze(axis=1) <= 0
        if np.any(invalid_rows):
            calibrated_probabilities[invalid_rows] = raw_probabilities[invalid_rows]
            row_sums = calibrated_probabilities.sum(axis=1, keepdims=True)

        return calibrated_probabilities / np.clip(row_sums, 1e-12, None)


def fit_sigmoid_calibrator(
    *,
    estimator: Any,
    calibration_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[Any | None, dict[str, object]]:
    """Fit one sigmoid calibrator on a dedicated calibration frame."""

    if estimator is None or calibration_df is None or calibration_df.empty:
        return None, {"enabled": False, "reason": "empty_calibration_frame"}
    if not hasattr(estimator, "predict_proba"):
        return None, {"enabled": False, "reason": "estimator_without_predict_proba"}
    if "target_signal" not in calibration_df.columns:
        return None, {"enabled": False, "reason": "missing_target_signal"}

    target_series = calibration_df["target_signal"]
    if int(target_series.nunique()) < 2:
        return None, {"enabled": False, "reason": "insufficient_target_classes"}

    feature_frame = calibration_df[feature_columns]
    class_labels = [int(class_label) for class_label in list(estimator.classes_)]
    calibrator = SigmoidProbabilityCalibrator(estimator=estimator, class_labels=class_labels)
    raw_probabilities = calibrator.fit(feature_frame, target_series)
    calibrated_probabilities = calibrator.predict_proba(feature_frame)
    summary = build_calibration_summary(
        targets=target_series,
        raw_probabilities=raw_probabilities,
        calibrated_probabilities=calibrated_probabilities,
        class_labels=class_labels,
        method="sigmoid",
    )
    return calibrator, summary
