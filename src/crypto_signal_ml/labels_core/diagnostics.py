"""Diagnostics helpers for label distributions and barrier behavior."""

from __future__ import annotations

import pandas as pd


def build_label_diagnostics(labeled_df: pd.DataFrame) -> dict[str, object]:
    """Summarize label balance and barrier behavior for one labeled frame."""

    if labeled_df.empty or "target_signal" not in labeled_df.columns:
        return {"available": False}

    diagnostics: dict[str, object] = {
        "available": True,
        "rowCount": int(len(labeled_df)),
        "labelDistribution": {
            str(signal_value): int(signal_count)
            for signal_value, signal_count in labeled_df["target_signal"].value_counts(dropna=False).sort_index().items()
        },
    }

    if "label_barrier" in labeled_df.columns:
        diagnostics["barrierDistribution"] = {
            str(barrier_name): int(barrier_count)
            for barrier_name, barrier_count in labeled_df["label_barrier"].fillna("missing").value_counts().items()
        }

    if "label_holding_period" in labeled_df.columns:
        diagnostics["holdingPeriod"] = {
            "mean": float(pd.to_numeric(labeled_df["label_holding_period"], errors="coerce").mean() or 0.0),
            "median": float(pd.to_numeric(labeled_df["label_holding_period"], errors="coerce").median() or 0.0),
        }

    if "market_regime_label" in labeled_df.columns:
        diagnostics["labelDistributionByRegime"] = (
            labeled_df.groupby("market_regime_label")["target_signal"]
            .value_counts()
            .unstack(fill_value=0)
            .astype(int)
            .to_dict(orient="index")
        )

    return diagnostics
