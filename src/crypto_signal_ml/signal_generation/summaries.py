"""Signal-summary builders that route inference through the staged contracts."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pandas as pd

from ..chart import build_chart_context
from ..config import TrainingConfig
from ..trading.signals import (
    ACTION_PRIORITY,
    READINESS_PRIORITY,
    _build_market_intelligence_context,
    _build_reason_items,
    _build_signal_chat,
    _calculate_setup_score,
    _collect_theme_tags,
    _render_chart_snapshots,
    _resolve_coin_symbol,
    _safe_float,
    filter_public_signal_summaries,
    is_signal_product_excluded,
)
from .orchestrator import gate_prediction_row
from .serialization import gated_candidate_to_summary


CHART_CONFIRMATION_PRIORITY = {
    "confirmed": 0,
    "early": 1,
    "blocked": 2,
    "unclear": 3,
    "invalid": 4,
}


def build_signal_summary_from_row(
    signal_row: pd.Series,
    minimum_action_confidence: float = 0.0,
    config: TrainingConfig | None = None,
    chart_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Convert one prediction row into a stable JSON-friendly summary via staged gating."""

    setup_score = _calculate_setup_score(signal_row)
    coin_symbol = _resolve_coin_symbol(signal_row)
    pair_symbol = str(signal_row.get("product_id", "")) if "product_id" in signal_row.index else ""

    base_chart_context = {
        "breakoutUp20": _safe_float(signal_row, "breakout_up_20"),
        "breakoutDown20": _safe_float(signal_row, "breakout_down_20"),
        "rangePosition20": _safe_float(signal_row, "range_position_20"),
        "closeVsEma5": _safe_float(signal_row, "close_vs_ema_5"),
        "relativeStrength1": _safe_float(signal_row, "relative_strength_1"),
        "relativeStrength5": _safe_float(signal_row, "relative_strength_5"),
        "rsi14": _safe_float(signal_row, "rsi_14"),
    }
    if chart_context:
        base_chart_context.update(chart_context)

    execution_context = {
        "atrPct14": _safe_float(signal_row, "atr_pct_14"),
        "volumeVsSma20": _safe_float(signal_row, "volume_vs_sma_20"),
        "volumeZscore20": _safe_float(signal_row, "volume_zscore_20"),
        "cmcVolume24hLog": _safe_float(signal_row, "cmc_volume_24h_log"),
        "cmcNumMarketPairsLog": _safe_float(signal_row, "cmc_num_market_pairs_log"),
        "cmcRankScore": _safe_float(signal_row, "cmc_rank_score"),
    }
    market_context = {
        "cmcPercentChange24h": _safe_float(signal_row, "cmc_percent_change_24h"),
        "cmcPercentChange7d": _safe_float(signal_row, "cmc_percent_change_7d"),
        "cmcPercentChange30d": _safe_float(signal_row, "cmc_percent_change_30d"),
        "cmcContextAvailable": int(_safe_float(signal_row, "cmc_context_available")),
        "themeTags": _collect_theme_tags(signal_row),
        "marketIntelligence": _build_market_intelligence_context(signal_row, config=config),
    }
    market_state = {
        "label": str(signal_row.get("market_regime_label", "unknown")),
        "code": int(_safe_float(signal_row, "market_regime_code")),
        "trendScore": _safe_float(signal_row, "regime_trend_score"),
        "volatilityRatio": _safe_float(signal_row, "regime_volatility_ratio", default_value=1.0),
        "isTrending": bool(_safe_float(signal_row, "regime_is_trending")),
        "isHighVolatility": bool(_safe_float(signal_row, "regime_is_high_volatility")),
    }
    event_context = {
        "eventCountNext7d": int(_safe_float(signal_row, "cmcal_event_count_next_7d")),
        "eventCountNext30d": int(_safe_float(signal_row, "cmcal_event_count_next_30d")),
        "hasEventNext7d": bool(_safe_float(signal_row, "cmcal_has_event_next_7d")),
        "daysToNextEvent": _safe_float(signal_row, "cmcal_days_to_next_event"),
    }

    gated_candidate = gate_prediction_row(
        signal_row=signal_row,
        config=config,
        minimum_action_confidence=minimum_action_confidence,
        setup_score=setup_score,
        symbol=coin_symbol,
        pair_symbol=pair_symbol,
        base_currency=str(signal_row.get("base_currency", "")).strip().upper() or None,
        quote_currency=str(signal_row.get("quote_currency", "")).strip().upper() or None,
        chart_context=base_chart_context,
        execution_context=execution_context,
        market_context=market_context,
        market_state=market_state,
        event_context=event_context,
    )
    summary = gated_candidate_to_summary(gated_candidate)
    reason_items = _build_reason_items(
        signal_row=signal_row,
        signal_name=str(summary["signal_name"]),
        raw_signal_name=str(summary["modelSignalName"]),
        confidence_gate_applied=bool(summary["confidenceGateApplied"]),
        risk_gate_applied=bool(summary["riskGateApplied"]),
        minimum_action_confidence=float(summary["minimumActionConfidence"]),
        required_action_confidence=float(summary["requiredActionConfidence"]),
        probability_margin=float(summary["probabilityMargin"]),
        gate_reasons=list(summary["gateReasons"]),
        policy_notes=list(summary["policyNotes"]),
    )
    summary["reasonItems"] = reason_items
    summary["reasonSummary"] = reason_items[0]
    summary["signalChat"] = _build_signal_chat(
        signal_row=signal_row,
        reason_items=reason_items,
        signal_name=str(summary["signal_name"]),
    )

    return summary


def build_latest_signal_summary(
    prediction_df: pd.DataFrame,
    minimum_action_confidence: float = 0.0,
    config: TrainingConfig | None = None,
) -> Dict[str, Any]:
    """Convert the newest prediction row into one summary via staged gating."""

    if prediction_df.empty:
        raise ValueError("No prediction rows were available. Check your feature generation step.")

    latest_row = prediction_df.iloc[-1]
    return build_signal_summary_from_row(
        signal_row=latest_row,
        minimum_action_confidence=minimum_action_confidence,
        config=config,
    )


def build_latest_signal_summaries(
    prediction_df: pd.DataFrame,
    minimum_action_confidence: float = 0.0,
    config: TrainingConfig | None = None,
    protected_product_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Build the newest available staged summary for each asset in the prediction frame."""

    if prediction_df.empty:
        raise ValueError("No prediction rows were available. Check your feature generation step.")

    if "product_id" not in prediction_df.columns:
        return [
            build_latest_signal_summary(
                prediction_df=prediction_df,
                minimum_action_confidence=minimum_action_confidence,
                config=config,
            )
        ]

    latest_rows = (
        prediction_df
        .sort_values(["timestamp", "product_id", "time_step"])
        .groupby("product_id", as_index=False)
        .tail(1)
        .copy()
    )
    if config is not None and "timestamp" in latest_rows.columns:
        max_staleness_hours = float(getattr(config, "signal_max_staleness_hours", 0.0) or 0.0)
        if max_staleness_hours > 0:
            latest_rows["timestamp"] = pd.to_datetime(latest_rows["timestamp"], errors="coerce", utc=True)
            freshest_timestamp = latest_rows["timestamp"].max()
            if pd.notna(freshest_timestamp):
                freshness_cutoff = freshest_timestamp - pd.Timedelta(hours=max_staleness_hours)
                latest_rows = latest_rows.loc[latest_rows["timestamp"] >= freshness_cutoff].copy()

    window_map: dict[str, pd.DataFrame] = {}
    chart_context_by_product: dict[str, dict[str, Any]] = {}
    protected_product_id_set = {
        str(product_id).strip().upper()
        for product_id in (protected_product_ids or ())
        if str(product_id).strip()
    }
    if "product_id" in prediction_df.columns:
        window_size = int(getattr(config, "chart_feature_window", 60) or 60) if config is not None else 60
        for product_id, group in prediction_df.groupby("product_id"):
            window_df = group.sort_values("timestamp").tail(window_size).copy()
            normalized_product_id = str(product_id).upper()
            window_map[normalized_product_id] = window_df
            chart_context_by_product[normalized_product_id] = build_chart_context(window_df, config=config)

    signal_summaries = []
    for _, signal_row in latest_rows.iterrows():
        product_id = str(signal_row.get("product_id", "")).strip().upper()
        signal_summaries.append(
            build_signal_summary_from_row(
                signal_row=signal_row,
                minimum_action_confidence=minimum_action_confidence,
                config=config,
                chart_context=chart_context_by_product.get(product_id),
            )
        )

    signal_summaries = [
        signal_summary
        for signal_summary in signal_summaries
        if str(signal_summary.get("productId", "")).strip().upper() in protected_product_id_set
        or not is_signal_product_excluded(
            product_id=str(signal_summary.get("productId", "")),
            base_currency=str(signal_summary.get("baseCurrency", "")),
            config=config,
        )
    ]
    signal_summaries = filter_public_signal_summaries(signal_summaries)

    sorted_summaries = sorted(
        signal_summaries,
        key=lambda signal_summary: (
            ACTION_PRIORITY.get(signal_summary.get("signal_name", "HOLD"), 99),
            READINESS_PRIORITY.get(signal_summary.get("tradeReadiness", "standby"), 99),
            CHART_CONFIRMATION_PRIORITY.get(
                signal_summary.get("chartDecision", signal_summary.get("chartConfirmationStatus", "early")),
                99,
            ),
            -float(signal_summary.get("policyScore", 0.0)),
            -float(signal_summary.get("setupScore", 0.0)),
            -float(signal_summary.get("confidence", 0.0)),
            str(signal_summary.get("productId", "")),
        ),
    )
    if config is not None and bool(getattr(config, "chart_snapshot_enabled", False)):
        _render_chart_snapshots(
            sorted_summaries,
            window_map=window_map,
            config=config,
        )

    return sorted_summaries
