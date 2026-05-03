"""Selection helpers for ranking competing model candidates."""

from __future__ import annotations

from typing import Any


def rank_model_candidates(
    candidate_rows: list[dict[str, Any]],
    *,
    minimum_trade_count: int,
) -> list[dict[str, Any]]:
    """Rank model candidates by tradable outcome quality first."""

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        trade_count = int(row.get("tradeCount", 0) or 0)
        average_trade_return = float(row.get("averageTradeReturn", 0.0) or 0.0)
        calibration_error = float(row.get("calibratedBrierScore", 999.0) or 999.0)
        fold_variance = float(row.get("foldBalancedAccuracyVariance", 999.0) or 999.0)
        win_rate = float(row.get("winRate", 0.0) or 0.0)
        trade_count_penalty = 0 if trade_count >= int(minimum_trade_count) else 1
        return (
            trade_count_penalty,
            -average_trade_return,
            calibration_error,
            fold_variance,
            -win_rate,
            str(row.get("modelType", "")),
        )

    return sorted(candidate_rows, key=sort_key)
