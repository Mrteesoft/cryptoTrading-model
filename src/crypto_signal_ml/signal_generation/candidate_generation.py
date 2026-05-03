"""Candidate-generation helpers built directly on model outputs."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ..labels_core import signal_to_text
from ..modeling_core.predict import probability_margin
from .contracts import RawSignalCandidate


SIGNAL_TO_ACTION = {
    "LOSS": "cut_loss",
    "BUY": "buy",
    "TAKE_PROFIT": "take_profit",
    "HOLD": "wait",
}


def _safe_float(signal_row: Mapping[str, Any], column_name: str, default_value: float = 0.0) -> float:
    """Read one numeric field from a row-like payload without raising."""

    raw_value = signal_row.get(column_name, default_value)
    if raw_value is None or pd.isna(raw_value):
        return default_value

    return float(raw_value)


def _safe_text(signal_row: Mapping[str, Any], column_name: str, default_value: str = "") -> str:
    """Read one text field from a row-like payload without raising."""

    raw_value = signal_row.get(column_name, default_value)
    if raw_value is None or pd.isna(raw_value):
        return default_value

    return str(raw_value)


def _resolve_probabilities(signal_row: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Resolve stable per-class probabilities from a row payload."""

    return {
        "TAKE_PROFIT": _safe_float(signal_row, f"{prefix}prob_take_profit"),
        "HOLD": _safe_float(signal_row, f"{prefix}prob_hold"),
        "BUY": _safe_float(signal_row, f"{prefix}prob_buy"),
    }


def _resolve_raw_signal_name(signal_row: Mapping[str, Any], calibrated_probabilities: dict[str, float]) -> str:
    """Resolve the raw class name from explicit or probability outputs."""

    predicted_name = _safe_text(signal_row, "predicted_name").upper()
    if predicted_name:
        return predicted_name

    if calibrated_probabilities:
        return max(
            calibrated_probabilities.items(),
            key=lambda item: (float(item[1]), item[0]),
        )[0]

    return signal_to_text(int(round(_safe_float(signal_row, "predicted_signal"))))


def build_raw_signal_candidate(
    *,
    signal_row: Mapping[str, Any],
    minimum_action_confidence: float,
    setup_score: float,
    symbol: str,
    pair_symbol: str,
    base_currency: str | None,
    quote_currency: str | None,
    chart_context: dict[str, Any],
    execution_context: dict[str, Any],
    market_context: dict[str, Any],
    market_state: dict[str, Any],
    event_context: dict[str, Any],
) -> RawSignalCandidate:
    """Build one typed raw candidate from a prediction row."""

    raw_probabilities = _resolve_probabilities(signal_row, prefix="raw_")
    calibrated_probabilities = _resolve_probabilities(signal_row)
    if all(probability == 0.0 for probability in raw_probabilities.values()):
        raw_probabilities = dict(calibrated_probabilities)
    if all(probability == 0.0 for probability in calibrated_probabilities.values()):
        calibrated_probabilities = dict(raw_probabilities)

    raw_signal_name = _resolve_raw_signal_name(signal_row, calibrated_probabilities)
    raw_predicted_signal = {
        "TAKE_PROFIT": -1,
        "HOLD": 0,
        "BUY": 1,
    }.get(raw_signal_name, int(round(_safe_float(signal_row, "predicted_signal"))))
    primary_probability = float(calibrated_probabilities.get(raw_signal_name, 0.0))
    raw_confidence = max(raw_probabilities.values()) if raw_probabilities else _safe_float(signal_row, "raw_confidence")
    calibrated_confidence = (
        max(calibrated_probabilities.values())
        if calibrated_probabilities
        else _safe_float(signal_row, "confidence")
    )

    return RawSignalCandidate(
        productId=_safe_text(signal_row, "product_id"),
        timestamp=_safe_text(signal_row, "timestamp") or None,
        close=_safe_float(signal_row, "close"),
        symbol=str(symbol).strip().upper(),
        pairSymbol=str(pair_symbol).strip().upper(),
        baseCurrency=(str(base_currency).strip().upper() or None) if base_currency is not None else None,
        quoteCurrency=(str(quote_currency).strip().upper() or None) if quote_currency is not None else None,
        coinName=_safe_text(signal_row, "cmc_name") or None,
        coinCategory=_safe_text(signal_row, "cmc_category") or None,
        timeStep=int(round(_safe_float(signal_row, "time_step"))),
        rawSignalName=raw_signal_name,
        rawPredictedSignal=int(raw_predicted_signal),
        rawSpotAction=SIGNAL_TO_ACTION.get(raw_signal_name, "wait"),
        rawProbabilities=raw_probabilities,
        calibratedProbabilities=calibrated_probabilities,
        rawConfidence=float(raw_confidence),
        calibratedConfidence=float(calibrated_confidence),
        primaryProbability=primary_probability,
        probabilityMargin=probability_margin(calibrated_probabilities, raw_signal_name),
        hasProbabilityColumns=all(
            column_name in signal_row
            for column_name in ("prob_take_profit", "prob_hold", "prob_buy")
        ),
        minimumActionConfidence=float(max(minimum_action_confidence or 0.0, 0.0)),
        setupScore=float(setup_score),
        chartContext=dict(chart_context),
        executionContext=dict(execution_context),
        marketContext=dict(market_context),
        marketState=dict(market_state),
        eventContext=dict(event_context),
        metadata={},
    )
