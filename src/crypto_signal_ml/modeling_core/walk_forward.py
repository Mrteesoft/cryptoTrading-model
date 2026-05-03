"""Walk-forward helpers for train-calibrate-test splits."""

from __future__ import annotations

import pandas as pd


def split_calibration_tail_by_time(
    dataset: pd.DataFrame,
    holdout_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split a training frame into fit and calibration tails using whole timestamps."""

    if dataset.empty or "timestamp" not in dataset.columns:
        return dataset.copy(), None

    normalized_fraction = float(holdout_fraction or 0.0)
    if normalized_fraction <= 0.0:
        return dataset.copy(), None

    prepared_df = dataset.copy()
    prepared_df["timestamp"] = pd.to_datetime(prepared_df["timestamp"], errors="coerce", utc=True)
    unique_timestamps = pd.Index(prepared_df["timestamp"].drop_duplicates()).sort_values()
    if len(unique_timestamps) < 4:
        return prepared_df.reset_index(drop=True), None

    calibration_timestamp_count = max(1, int(len(unique_timestamps) * normalized_fraction))
    calibration_timestamp_count = min(calibration_timestamp_count, max(len(unique_timestamps) - 2, 1))
    calibration_start_timestamp = unique_timestamps[-calibration_timestamp_count]

    fit_df = prepared_df.loc[prepared_df["timestamp"] < calibration_start_timestamp].copy()
    calibration_df = prepared_df.loc[prepared_df["timestamp"] >= calibration_start_timestamp].copy()
    if fit_df.empty or calibration_df.empty:
        return prepared_df.reset_index(drop=True), None

    return fit_df.reset_index(drop=True), calibration_df.reset_index(drop=True)
