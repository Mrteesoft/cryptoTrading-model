"""Diagnostics helpers for calibrated classifier outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def multiclass_brier_score(
    targets: pd.Series,
    probabilities: np.ndarray,
    class_labels: list[int],
) -> float:
    """Compute a one-vs-rest multiclass Brier score."""

    if probabilities.size == 0 or targets.empty:
        return 0.0

    target_array = pd.Series(targets).to_numpy()
    one_hot = np.zeros((len(target_array), len(class_labels)), dtype=float)
    for class_index, class_label in enumerate(class_labels):
        one_hot[:, class_index] = (target_array == class_label).astype(float)

    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def build_calibration_summary(
    *,
    targets: pd.Series,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    class_labels: list[int],
    method: str,
) -> dict[str, object]:
    """Build one compact calibration summary from raw and calibrated probabilities."""

    raw_brier = multiclass_brier_score(targets=targets, probabilities=raw_probabilities, class_labels=class_labels)
    calibrated_brier = multiclass_brier_score(
        targets=targets,
        probabilities=calibrated_probabilities,
        class_labels=class_labels,
    )
    return {
        "enabled": True,
        "method": str(method),
        "classOrder": list(class_labels),
        "rawBrierScore": float(raw_brier),
        "calibratedBrierScore": float(calibrated_brier),
        "improved": bool(calibrated_brier <= raw_brier),
    }
