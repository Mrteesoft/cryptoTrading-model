"""Shared signal-quality helpers for contextual confidence and execution scoring."""

from __future__ import annotations

from typing import Any, Mapping

from ..config import TrainingConfig


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp one numeric value between an inclusive minimum and maximum."""

    return max(min(float(value), maximum), minimum)


def _coerce_float(value: Any, default_value: float = 0.0) -> float:
    """Convert one optional value to float without raising."""

    if value is None:
        return default_value

    try:
        return float(value)
    except (TypeError, ValueError):
        return default_value


def _safe_float(payload: Mapping[str, Any] | None, key: str, default_value: float = 0.0) -> float:
    """Read one optional numeric field from a mapping."""

    if not isinstance(payload, Mapping):
        return default_value

    return _coerce_float(payload.get(key), default_value=default_value)


def _safe_bool(payload: Mapping[str, Any] | None, key: str, default_value: bool = False) -> bool:
    """Read one optional boolean-like field from a mapping."""

    if not isinstance(payload, Mapping):
        return default_value

    raw_value = payload.get(key, default_value)
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return default_value

    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _structure_alignment_score(structure_label: str) -> float:
    """Map one chart-structure label to an alignment score."""

    return {
        "higher_highs": 0.24,
        "higher_lows": 0.14,
        "range": 0.0,
        "lower_highs": -0.16,
        "lower_lows": -0.22,
        "downtrend": -0.26,
    }.get(str(structure_label or "").strip().lower(), 0.0)


def _confidence_quality_label(
    *,
    calibrated_confidence: float,
    reliability_score: float,
    risk_penalty_score: float,
    execution_quality_score: float,
) -> str:
    """Classify the confidence regime for operator-facing output."""

    if (
        calibrated_confidence >= 0.72
        and reliability_score >= 0.64
        and risk_penalty_score <= 0.16
        and execution_quality_score >= 0.42
    ):
        return "strong"
    if (
        calibrated_confidence <= 0.54
        or reliability_score <= 0.46
        or risk_penalty_score >= 0.28
        or execution_quality_score <= 0.22
    ):
        return "fragile"
    return "balanced"


def _build_adaptive_context(
    trade_memory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Summarize live trade-history bias into small scoring adjustments."""

    trade_memory = trade_memory or {}
    closed_trade_count = int(_safe_float(trade_memory, "closedTradeCount"))
    recent_loss_streak = int(_safe_float(trade_memory, "recentLossStreak"))
    sample_adequate = bool(_safe_bool(trade_memory, "sampleAdequate")) and closed_trade_count >= 3
    last_outcome = str(trade_memory.get("lastOutcome", "") or "").strip().lower()

    raw_win_rate = trade_memory.get("winRate")
    raw_average_return = trade_memory.get("averageRealizedReturn")
    win_rate = None
    average_realized_return = None
    if raw_win_rate is not None:
        try:
            win_rate = float(raw_win_rate)
        except (TypeError, ValueError):
            win_rate = None
    if raw_average_return is not None:
        try:
            average_realized_return = float(raw_average_return)
        except (TypeError, ValueError):
            average_realized_return = None

    performance_score = 0.0
    if sample_adequate and win_rate is not None:
        performance_score += (win_rate - 0.50) * 1.40
    if sample_adequate and average_realized_return is not None:
        performance_score += _clamp(average_realized_return / 0.05, -1.0, 1.0) * 0.18

    performance_score -= min(recent_loss_streak, 3) * 0.10
    if last_outcome == "win":
        performance_score += 0.04
    elif last_outcome == "loss":
        performance_score -= 0.04

    performance_score = _clamp(performance_score, -0.50, 0.50)
    if performance_score >= 0.10:
        bias = "supportive"
    elif performance_score <= -0.08:
        bias = "cautious"
    else:
        bias = "neutral"

    return {
        "available": bool(_safe_bool(trade_memory, "available")) or closed_trade_count > 0,
        "bias": bias,
        "sampleAdequate": bool(sample_adequate),
        "performanceScore": float(performance_score),
        "confidenceAdjustment": float(_clamp(performance_score * 0.10, -0.05, 0.05)),
        "decisionAdjustment": float(_clamp(performance_score * 0.16, -0.08, 0.08)),
        "confirmationAdjustment": float(_clamp(performance_score * 0.18, -0.08, 0.08)),
        "riskAdjustment": float(_clamp(-performance_score * 0.16, -0.05, 0.08)),
        "sizeMultiplier": float(_clamp(1.0 + (performance_score * 0.18), 0.88, 1.10)),
        "closedTradeCount": int(closed_trade_count),
        "recentLossStreak": int(recent_loss_streak),
        "winRate": win_rate,
        "averageRealizedReturn": average_realized_return,
        "lastOutcome": last_outcome or None,
    }


def _build_execution_context(
    signal_summary: Mapping[str, Any],
    config: TrainingConfig,
) -> dict[str, Any]:
    """Estimate execution quality from the signal's liquidity and volatility inputs."""

    raw_execution_context = signal_summary.get("executionContext") if isinstance(
        signal_summary.get("executionContext"),
        Mapping,
    ) else {}
    market_state = signal_summary.get("marketState") if isinstance(signal_summary.get("marketState"), Mapping) else {}
    base_round_trip_cost_rate = 2.0 * (
        float(config.backtest_trading_fee_rate)
        + float(config.backtest_slippage_rate)
    )
    volatility_ratio = _safe_float(market_state, "volatilityRatio", default_value=1.0)
    is_high_volatility = _safe_bool(market_state, "isHighVolatility")

    # Lightweight signal summaries often do not carry execution features yet. Treat
    # that absence as "unknown / neutral" instead of silently penalizing the setup.
    if not raw_execution_context:
        neutral_slippage_risk_score = _clamp(
            0.18
            + (0.10 if is_high_volatility else 0.0)
            + max(volatility_ratio - 1.0, 0.0) * 0.10,
            0.0,
            1.0,
        )
        estimated_round_trip_cost_rate = base_round_trip_cost_rate * (1.0 + (neutral_slippage_risk_score * 0.30))
        execution_quality_score = _clamp(
            0.58
            - (neutral_slippage_risk_score * 0.18),
            0.0,
            1.0,
        )
        return {
            "liquidityScore": 0.56,
            "slippageRiskScore": float(neutral_slippage_risk_score),
            "executionQualityScore": float(execution_quality_score),
            "baseRoundTripCostRate": float(base_round_trip_cost_rate),
            "estimatedRoundTripCostRate": float(estimated_round_trip_cost_rate),
            "decisionPenalty": 0.0,
            "isThinLiquidity": False,
            "hasElevatedCost": False,
            "isExecutionBlocked": False,
        }

    atr_pct_14 = _safe_float(raw_execution_context, "atrPct14", default_value=0.018)
    volume_vs_sma_20 = _safe_float(raw_execution_context, "volumeVsSma20", default_value=1.0)
    volume_zscore_20 = _safe_float(raw_execution_context, "volumeZscore20", default_value=0.0)
    cmc_volume_24h_log = _safe_float(raw_execution_context, "cmcVolume24hLog", default_value=10.5)
    cmc_num_market_pairs_log = _safe_float(raw_execution_context, "cmcNumMarketPairsLog", default_value=2.5)
    cmc_rank_score = _safe_float(raw_execution_context, "cmcRankScore", default_value=0.45)

    liquidity_components = [
        _clamp((cmc_volume_24h_log - 10.0) / 8.0, 0.0, 1.0),
        _clamp((cmc_num_market_pairs_log - 1.5) / 4.5, 0.0, 1.0),
        _clamp(cmc_rank_score, 0.0, 1.0),
        _clamp((volume_vs_sma_20 - 0.65) / 1.35, 0.0, 1.0),
        _clamp((volume_zscore_20 + 1.5) / 3.5, 0.0, 1.0),
    ]
    liquidity_score = (
        (liquidity_components[0] * 0.30)
        + (liquidity_components[1] * 0.18)
        + (liquidity_components[2] * 0.18)
        + (liquidity_components[3] * 0.20)
        + (liquidity_components[4] * 0.14)
    )

    atr_pressure = _clamp((atr_pct_14 - 0.015) / 0.05, 0.0, 1.0)
    regime_volatility_pressure = _clamp((volatility_ratio - 1.0) / 0.70, 0.0, 1.0)
    slippage_risk_score = _clamp(
        ((1.0 - liquidity_score) * 0.62)
        + (atr_pressure * 0.23)
        + (regime_volatility_pressure * 0.15)
        + (0.08 if is_high_volatility else 0.0),
        0.0,
        1.0,
    )

    estimated_round_trip_cost_rate = base_round_trip_cost_rate * (1.0 + (slippage_risk_score * 1.40))
    execution_quality_score = _clamp(
        (liquidity_score * 0.72)
        + (liquidity_components[3] * 0.10)
        + (liquidity_components[2] * 0.08)
        - (slippage_risk_score * 0.45)
        - min(estimated_round_trip_cost_rate / 0.01, 0.25),
        0.0,
        1.0,
    )
    is_thin_liquidity = liquidity_score < 0.34
    has_elevated_cost = estimated_round_trip_cost_rate >= max(base_round_trip_cost_rate * 1.40, 0.004)
    is_execution_blocked = is_thin_liquidity and slippage_risk_score >= 0.70
    decision_penalty = _clamp(
        max(0.0, 0.45 - execution_quality_score) * 0.22
        + max(0.0, estimated_round_trip_cost_rate - base_round_trip_cost_rate) * 8.0,
        0.0,
        0.18,
    )

    return {
        **dict(raw_execution_context),
        "liquidityScore": float(liquidity_score),
        "slippageRiskScore": float(slippage_risk_score),
        "executionQualityScore": float(execution_quality_score),
        "baseRoundTripCostRate": float(base_round_trip_cost_rate),
        "estimatedRoundTripCostRate": float(estimated_round_trip_cost_rate),
        "decisionPenalty": float(decision_penalty),
        "isThinLiquidity": bool(is_thin_liquidity),
        "hasElevatedCost": bool(has_elevated_cost),
        "isExecutionBlocked": bool(is_execution_blocked),
    }


def _build_confidence_calibration(
    signal_summary: Mapping[str, Any],
    market_context: Mapping[str, Any],
    execution_context: Mapping[str, Any],
    adaptive_context: Mapping[str, Any],
    config: TrainingConfig,
) -> dict[str, Any]:
    """Re-weight raw model confidence with chart/news/risk/execution context."""

    market_state = signal_summary.get("marketState") if isinstance(signal_summary.get("marketState"), Mapping) else {}
    event_context = signal_summary.get("eventContext") if isinstance(signal_summary.get("eventContext"), Mapping) else {}
    news_context = signal_summary.get("newsContext") if isinstance(signal_summary.get("newsContext"), Mapping) else {}
    trend_context = signal_summary.get("trendContext") if isinstance(signal_summary.get("trendContext"), Mapping) else {}
    chart_context = signal_summary.get("chartContext") if isinstance(signal_summary.get("chartContext"), Mapping) else {}

    raw_confidence = _coerce_float(signal_summary.get("confidence"), default_value=0.0)
    probability_margin = _coerce_float(signal_summary.get("probabilityMargin"), default_value=0.0)
    policy_score = _coerce_float(signal_summary.get("policyScore"), default_value=0.0)
    trade_readiness = str(signal_summary.get("tradeReadiness", "standby") or "standby").strip().lower()

    structure_label = str(chart_context.get("structureLabel", "") or "").strip().lower()
    pattern_score = _safe_float(chart_context, "patternScore")
    trend_slope = _safe_float(chart_context, "trendSlope")
    channel_position = _safe_float(chart_context, "channelPosition", default_value=0.50)
    range_compression_score = _safe_float(chart_context, "rangeCompressionScore")
    resistance_distance_pct = _safe_float(chart_context, "resistanceDistancePct", default_value=0.02)
    breakout_confirmed = _safe_bool(chart_context, "breakoutConfirmed")
    retest_hold_confirmed = _safe_bool(chart_context, "retestHoldConfirmed")
    near_resistance = _safe_bool(chart_context, "nearResistance")

    chart_alignment_score = _structure_alignment_score(structure_label)
    if breakout_confirmed:
        chart_alignment_score += 0.28
    if retest_hold_confirmed:
        chart_alignment_score += 0.16
    chart_alignment_score += _clamp(pattern_score, -1.0, 1.0) * 0.12
    if trend_slope >= 0.003:
        chart_alignment_score += 0.08
    elif trend_slope <= -0.003:
        chart_alignment_score -= 0.08
    if 0.55 <= channel_position <= 0.88:
        chart_alignment_score += 0.04
    elif channel_position >= 0.95:
        chart_alignment_score -= 0.05
    if range_compression_score >= 0.10 and breakout_confirmed:
        chart_alignment_score += 0.06
    elif range_compression_score <= -0.20:
        chart_alignment_score -= 0.04
    if near_resistance:
        chart_alignment_score -= 0.18
    if resistance_distance_pct <= 0.01:
        chart_alignment_score -= 0.06
    elif resistance_distance_pct >= 0.03:
        chart_alignment_score += 0.04
    chart_alignment_score = _clamp(chart_alignment_score, -1.0, 1.0)

    news_sentiment_1h = _safe_float(news_context, "newsSentiment1h")
    news_sentiment_delta = _safe_float(news_context, "newsSentimentDelta")
    coin_specific_news_score = _safe_float(news_context, "coinSpecificNewsScore")
    market_wide_news_score = _safe_float(news_context, "marketWideNewsScore")
    news_novelty_score = _safe_float(news_context, "newsNoveltyScore", default_value=0.5)
    news_relevance_score = max(_safe_float(news_context, "newsRelevanceScore"), 0.0)
    entity_mention_acceleration = _safe_float(news_context, "entityMentionAcceleration", default_value=1.0)

    news_alignment_score = (
        (news_sentiment_1h * 0.40)
        + (news_sentiment_delta * 0.20)
        + (_clamp(coin_specific_news_score / 2.0, -1.0, 1.0) * 0.20)
        + (_clamp(market_wide_news_score / 2.0, -1.0, 1.0) * 0.10)
    )
    news_alignment_score *= max(news_relevance_score, 0.30)
    news_alignment_score += (news_novelty_score - 0.5) * 0.06
    if entity_mention_acceleration > 1.0:
        news_alignment_score += min((entity_mention_acceleration - 1.0) * 0.08, 0.10)
    news_alignment_score = _clamp(news_alignment_score, -1.0, 1.0)

    topic_trend_score = _safe_float(trend_context, "topicTrendScore")
    trend_persistence_score = _safe_float(trend_context, "trendPersistenceScore", default_value=0.5)
    trend_alignment_score = _clamp(
        (topic_trend_score * 0.75)
        + ((trend_persistence_score - 0.5) * 0.50),
        -1.0,
        1.0,
    )

    context_alignment_score = _clamp(
        (chart_alignment_score * 0.50)
        + (news_alignment_score * 0.25)
        + (trend_alignment_score * 0.25),
        -1.0,
        1.0,
    )

    is_high_volatility = _safe_bool(market_state, "isHighVolatility")
    has_event_next_7d = _safe_bool(event_context, "hasEventNext7d")
    event_window_active = _safe_bool(event_context, "eventWindowActive")
    post_event_cooldown_active = _safe_bool(event_context, "postEventCooldownActive")
    macro_event_risk_flag = _safe_bool(event_context, "macroEventRiskFlag")
    market_stance = str(market_context.get("marketStance", "balanced") or "balanced")
    macro_risk_mode = str(market_context.get("macroRiskMode", "neutral") or "neutral")

    risk_penalty_score = 0.0
    if is_high_volatility:
        risk_penalty_score += 0.10
    if has_event_next_7d:
        risk_penalty_score += 0.06
    if event_window_active or macro_event_risk_flag:
        risk_penalty_score += 0.14
    if post_event_cooldown_active:
        risk_penalty_score += 0.06
    if market_stance == "defensive":
        risk_penalty_score += 0.05
    if macro_risk_mode == "risk_off":
        risk_penalty_score += 0.07
    if trade_readiness == "blocked":
        risk_penalty_score += 0.08
    risk_penalty_score += max(_coerce_float(adaptive_context.get("riskAdjustment")), 0.0)
    risk_penalty_score = _clamp(risk_penalty_score, 0.0, 0.40)

    normalized_probability_margin = _clamp(
        probability_margin / max(float(config.decision_min_probability_margin) * 2.0, 0.12),
        0.0,
        1.0,
    )
    normalized_policy_score = _clamp(policy_score / 1.25, 0.0, 1.0)
    reliability_score = _clamp(
        (raw_confidence * 0.55)
        + (normalized_probability_margin * 0.20)
        + (normalized_policy_score * 0.15)
        + (max(context_alignment_score, 0.0) * 0.10),
        0.0,
        1.0,
    )

    calibrated_confidence = raw_confidence
    calibrated_confidence += (reliability_score - raw_confidence) * 0.55
    calibrated_confidence += context_alignment_score * 0.12
    calibrated_confidence += _coerce_float(adaptive_context.get("confidenceAdjustment"))
    calibrated_confidence -= risk_penalty_score * 0.65
    calibrated_confidence -= _safe_float(execution_context, "decisionPenalty") * 0.55
    calibrated_confidence = _clamp(calibrated_confidence, 0.0, 1.0)

    decision_adjustment = _clamp(
        (context_alignment_score * 0.10)
        + _coerce_float(adaptive_context.get("decisionAdjustment"))
        - (risk_penalty_score * 0.10)
        - _safe_float(execution_context, "decisionPenalty"),
        -0.16,
        0.16,
    )

    return {
        "rawConfidence": float(raw_confidence),
        "calibratedConfidence": float(calibrated_confidence),
        "confidenceAdjustment": float(calibrated_confidence - raw_confidence),
        "reliabilityScore": float(reliability_score),
        "normalizedProbabilityMargin": float(normalized_probability_margin),
        "chartAlignmentScore": float(chart_alignment_score),
        "newsAlignmentScore": float(news_alignment_score),
        "trendAlignmentScore": float(trend_alignment_score),
        "contextAlignmentScore": float(context_alignment_score),
        "riskPenaltyScore": float(risk_penalty_score),
        "executionPenaltyScore": float(_safe_float(execution_context, "decisionPenalty")),
        "decisionAdjustment": float(decision_adjustment),
        "confidenceQuality": _confidence_quality_label(
            calibrated_confidence=calibrated_confidence,
            reliability_score=reliability_score,
            risk_penalty_score=risk_penalty_score,
            execution_quality_score=_safe_float(execution_context, "executionQualityScore"),
        ),
    }


def build_signal_quality_context(
    *,
    signal_summary: Mapping[str, Any],
    market_context: Mapping[str, Any] | None = None,
    trade_memory: Mapping[str, Any] | None = None,
    config: TrainingConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Build contextual quality outputs shared by the watchlist and trader brain."""

    config = config or TrainingConfig()
    effective_market_context = market_context or {}
    adaptive_context = _build_adaptive_context(trade_memory)
    execution_context = _build_execution_context(signal_summary, config)
    confidence_calibration = _build_confidence_calibration(
        signal_summary=signal_summary,
        market_context=effective_market_context,
        execution_context=execution_context,
        adaptive_context=adaptive_context,
        config=config,
    )

    return {
        "confidenceCalibration": confidence_calibration,
        "executionContext": execution_context,
        "adaptiveContext": adaptive_context,
    }
