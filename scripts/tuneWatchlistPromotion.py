"""Tune watchlist-promotion thresholds from recent signal history."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.config import OUTPUTS_DIR, TrainingConfig  # noqa: E402


def _safe_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if percentile <= 0:
        return float(min(values))
    if percentile >= 100:
        return float(max(values))

    ordered = sorted(values)
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    if lower_index == upper_index:
        return float(ordered[lower_index])

    weight = rank - lower_index
    return float(ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight)


def _scan_threshold(
    *,
    buy_values: list[float],
    other_values: list[float],
    target_precision: float,
    min_recall: float,
    min_threshold: float = 0.30,
    max_threshold: float = 0.90,
    step: float = 0.02,
) -> tuple[float | None, dict[str, float]]:
    if not buy_values:
        return None, {"precision": 0.0, "recall": 0.0}

    best_threshold = None
    best_score = -1.0
    best_metrics = {"precision": 0.0, "recall": 0.0}

    threshold = min_threshold
    buy_total = len(buy_values)
    while threshold <= max_threshold + 1e-9:
        buy_hits = sum(value >= threshold for value in buy_values)
        other_hits = sum(value >= threshold for value in other_values)
        precision = buy_hits / max(buy_hits + other_hits, 1)
        recall = buy_hits / max(buy_total, 1)
        if precision >= target_precision and recall >= min_recall:
            score = precision + recall
            if score > best_score:
                best_score = score
                best_threshold = float(round(threshold, 4))
                best_metrics = {"precision": float(precision), "recall": float(recall)}
        threshold = round(threshold + step, 6)

    if best_threshold is None:
        fallback = _percentile(buy_values, 60) or _percentile(buy_values, 50) or 0.5
        best_threshold = float(round(fallback, 4))
        buy_hits = sum(value >= best_threshold for value in buy_values)
        other_hits = sum(value >= best_threshold for value in other_values)
        best_metrics = {
            "precision": float(buy_hits / max(buy_hits + other_hits, 1)),
            "recall": float(buy_hits / max(len(buy_values), 1)),
        }

    return best_threshold, best_metrics


def _summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": _percentile(values, 50),
        "p60": _percentile(values, 60),
        "p70": _percentile(values, 70),
        "p80": _percentile(values, 80),
        "p90": _percentile(values, 90),
    }


def _iter_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row:
                yield row


def main() -> None:
    config = TrainingConfig()
    source_path = OUTPUTS_DIR / "historicalSignals.csv"
    if not source_path.exists():
        print(f"Signal history not found at {source_path}. Run signal generation first.")
        return

    buy_confidence: list[float] = []
    other_confidence: list[float] = []
    buy_prob_buy: list[float] = []
    other_prob_buy: list[float] = []
    buy_margin: list[float] = []
    other_margin: list[float] = []

    signal_counts: dict[str, int] = {}

    for row in _iter_rows(source_path):
        signal_name = str(row.get("predicted_name") or row.get("signal_name") or "HOLD").strip().upper()
        signal_counts[signal_name] = signal_counts.get(signal_name, 0) + 1

        confidence = _safe_float(row.get("confidence"))
        prob_buy = _safe_float(row.get("prob_buy"))
        prob_hold = _safe_float(row.get("prob_hold"))
        prob_take_profit = _safe_float(row.get("prob_take_profit"))

        if confidence is None or prob_buy is None:
            continue

        probability_margin = None
        if prob_hold is not None and prob_take_profit is not None:
            probability_margin = prob_buy - max(prob_hold, prob_take_profit)

        if signal_name == "BUY":
            buy_confidence.append(confidence)
            buy_prob_buy.append(prob_buy)
            if probability_margin is not None:
                buy_margin.append(probability_margin)
        else:
            other_confidence.append(confidence)
            other_prob_buy.append(prob_buy)
            if probability_margin is not None:
                other_margin.append(probability_margin)

    promotion_conf_threshold, promotion_conf_metrics = _scan_threshold(
        buy_values=buy_confidence,
        other_values=other_confidence,
        target_precision=0.55,
        min_recall=0.35,
    )
    entry_conf_threshold, entry_conf_metrics = _scan_threshold(
        buy_values=buy_confidence,
        other_values=other_confidence,
        target_precision=0.65,
        min_recall=0.20,
    )
    promotion_decision_threshold, promotion_decision_metrics = _scan_threshold(
        buy_values=buy_prob_buy,
        other_values=other_prob_buy,
        target_precision=0.55,
        min_recall=0.35,
    )
    entry_decision_threshold, entry_decision_metrics = _scan_threshold(
        buy_values=buy_prob_buy,
        other_values=other_prob_buy,
        target_precision=0.65,
        min_recall=0.20,
    )

    payload = {
        "version": "watchlist-promotion-tuning-v1",
        "sourcePath": str(source_path),
        "signalCounts": signal_counts,
        "buySamples": len(buy_confidence),
        "otherSamples": len(other_confidence),
        "confidenceStats": {
            "buy": _summarize(buy_confidence),
            "other": _summarize(other_confidence),
        },
        "decisionScoreProxyStats": {
            "buy": _summarize(buy_prob_buy),
            "other": _summarize(other_prob_buy),
        },
        "probabilityMarginStats": {
            "buy": _summarize(buy_margin),
            "other": _summarize(other_margin),
        },
        "recommendedThresholds": {
            "promotionMinConfidence": promotion_conf_threshold,
            "entryReadyMinConfidence": entry_conf_threshold,
            "promotionMinDecisionScoreProxy": promotion_decision_threshold,
            "entryReadyMinDecisionScoreProxy": entry_decision_threshold,
            "promotionMinConfidenceGain": float(config.signal_watchlist_promotion_min_confidence_gain),
            "promotionMinDecisionScoreGain": float(config.signal_watchlist_promotion_min_decision_score_gain),
        },
        "thresholdMetrics": {
            "promotionConfidence": promotion_conf_metrics,
            "entryReadyConfidence": entry_conf_metrics,
            "promotionDecisionScoreProxy": promotion_decision_metrics,
            "entryReadyDecisionScoreProxy": entry_decision_metrics,
        },
    }

    output_path = OUTPUTS_DIR / "watchlistPromotionTuning.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_rows = [
        ("metric", "value"),
        ("promotionMinConfidence", promotion_conf_threshold),
        ("entryReadyMinConfidence", entry_conf_threshold),
        ("promotionMinDecisionScoreProxy", promotion_decision_threshold),
        ("entryReadyMinDecisionScoreProxy", entry_decision_threshold),
        ("buySamples", len(buy_confidence)),
        ("otherSamples", len(other_confidence)),
    ]
    csv_path = OUTPUTS_DIR / "watchlistPromotionTuning.csv"
    csv_path.write_text(
        "\n".join(f"{metric},{value}" for metric, value in csv_rows),
        encoding="utf-8",
    )

    print("Watchlist promotion tuning complete.")
    print(f"Source data: {source_path}")
    print(f"Output JSON: {output_path}")
    print(f"Output CSV: {csv_path}")
    print("Recommended thresholds:")
    print(f"  Promotion min confidence: {promotion_conf_threshold}")
    print(f"  Entry-ready min confidence: {entry_conf_threshold}")
    print(f"  Promotion min decision score (proxy): {promotion_decision_threshold}")
    print(f"  Entry-ready min decision score (proxy): {entry_decision_threshold}")


if __name__ == "__main__":
    main()
