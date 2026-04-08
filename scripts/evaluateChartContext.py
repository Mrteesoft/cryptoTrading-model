"""Evaluate chart-context filters on recent prediction data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.chart import build_chart_context  # noqa: E402
from crypto_signal_ml.config import OUTPUTS_DIR, TrainingConfig  # noqa: E402


def main() -> None:
    config = TrainingConfig()
    source_path = OUTPUTS_DIR / "testPredictions.csv"
    if not source_path.exists():
        source_path = OUTPUTS_DIR / "historicalSignals.csv"
    if not source_path.exists():
        print("No prediction CSV found for chart evaluation.")
        return

    usecols = [
        "product_id",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "predicted_name",
        "target_signal",
        "future_return",
    ]
    df = pd.read_csv(source_path, usecols=lambda col: col in usecols)
    max_rows = int(getattr(config, "chart_eval_max_rows_total", 50000))
    if len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values(["timestamp", "product_id"]).reset_index(drop=True)

    window = int(getattr(config, "chart_feature_window", 60))
    chart_flags = []
    for product_id, group in df.groupby("product_id"):
        group = group.reset_index(drop=True)
        for idx in range(len(group)):
            start = max(idx - window + 1, 0)
            window_df = group.loc[start:idx, ["timestamp", "open", "high", "low", "close", "volume"]]
            chart_context = build_chart_context(window_df, config=config)
            chart_flags.append(
                {
                    "index": group.index[idx],
                    **chart_context,
                }
            )

    chart_df = pd.DataFrame(chart_flags).set_index("index")
    df = df.join(chart_df, how="left")

    baseline_buy = df["predicted_name"] == "BUY"
    baseline_precision = float((baseline_buy & (df["target_signal"] == 1)).sum() / max(int(baseline_buy.sum()), 1))
    baseline_return = float(df.loc[baseline_buy, "future_return"].mean()) if baseline_buy.any() else 0.0

    chart_support = (
        df.get("breakoutConfirmed", False).fillna(False)
        | df.get("retestHoldConfirmed", False).fillna(False)
        | df.get("structureLabel", "").isin(["higher_highs", "higher_lows"])
    )
    chart_block = df.get("nearResistance", False).fillna(False) | df.get("structureLabel", "").isin(
        ["lower_highs", "lower_lows"]
    )
    filtered_buy = baseline_buy & chart_support & ~chart_block
    filtered_precision = float((filtered_buy & (df["target_signal"] == 1)).sum() / max(int(filtered_buy.sum()), 1))
    filtered_return = float(df.loc[filtered_buy, "future_return"].mean()) if filtered_buy.any() else 0.0

    payload = {
        "sourcePath": str(source_path),
        "rowsEvaluated": int(len(df)),
        "baseline": {
            "buyCount": int(baseline_buy.sum()),
            "buyPrecision": baseline_precision,
            "averageForwardReturn": baseline_return,
        },
        "chartFiltered": {
            "buyCount": int(filtered_buy.sum()),
            "buyPrecision": filtered_precision,
            "averageForwardReturn": filtered_return,
        },
    }

    output_path = OUTPUTS_DIR / "chartFeatureEvaluation.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Chart evaluation saved to: {output_path}")


if __name__ == "__main__":
    main()
