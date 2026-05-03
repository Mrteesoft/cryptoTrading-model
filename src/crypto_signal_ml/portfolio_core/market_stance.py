"""Market-stance helpers shared by fusion and portfolio action mapping."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


UPTREND_LABELS = {"trend_up", "trend_up_high_volatility"}
DOWNTREND_LABELS = {"trend_down", "trend_down_high_volatility"}


def _safe_float(payload: Mapping[str, Any], key: str, default_value: float = 0.0) -> float:
    """Read one numeric field from a mapping without raising."""

    raw_value = payload.get(key, default_value)
    if raw_value is None:
        return default_value

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default_value


def _safe_bool(payload: Mapping[str, Any], key: str, default_value: bool = False) -> bool:
    """Read one boolean-like field from a mapping without raising."""

    raw_value = payload.get(key, default_value)
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return default_value

    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def build_market_context(signal_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify the current market posture from the current signal universe."""

    total_signals = max(len(signal_summaries), 1)
    buy_count = sum(
        str(signal_summary.get("signal_name", "")).strip().upper() == "BUY"
        for signal_summary in signal_summaries
    )
    take_profit_count = sum(
        str(signal_summary.get("signal_name", "")).strip().upper() == "TAKE_PROFIT"
        for signal_summary in signal_summaries
    )
    loss_count = sum(
        str(signal_summary.get("signal_name", "")).strip().upper() == "LOSS"
        for signal_summary in signal_summaries
    )
    exit_signal_count = take_profit_count + loss_count
    trending_count = sum(
        bool((signal_summary.get("marketState") or {}).get("isTrending", False))
        for signal_summary in signal_summaries
    )
    high_volatility_count = sum(
        bool((signal_summary.get("marketState") or {}).get("isHighVolatility", False))
        for signal_summary in signal_summaries
    )

    regime_counts: dict[str, int] = {}
    for signal_summary in signal_summaries:
        market_state = signal_summary.get("marketState") if isinstance(signal_summary.get("marketState"), Mapping) else {}
        regime_label = str(market_state.get("label", "unknown")).strip().lower()
        regime_counts[regime_label] = regime_counts.get(regime_label, 0) + 1

    dominant_regime = sorted(
        regime_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0][0] if regime_counts else "unknown"
    trending_share = trending_count / total_signals
    high_volatility_share = high_volatility_count / total_signals

    representative_market_intelligence: Mapping[str, Any] = {}
    for signal_summary in signal_summaries:
        market_context = signal_summary.get("marketContext") if isinstance(signal_summary.get("marketContext"), Mapping) else {}
        market_intelligence = market_context.get("marketIntelligence") if isinstance(market_context.get("marketIntelligence"), Mapping) else {}
        if market_intelligence:
            representative_market_intelligence = market_intelligence
            if _safe_bool(market_intelligence, "available"):
                break

    market_intelligence_available = _safe_bool(representative_market_intelligence, "available")
    fear_greed_value = _safe_float(representative_market_intelligence, "fearGreedValue")
    fear_greed_classification = str(
        representative_market_intelligence.get("fearGreedClassification", "") or ""
    ).strip()
    btc_dominance = _safe_float(representative_market_intelligence, "btcDominance")
    btc_dominance_change_24h = _safe_float(representative_market_intelligence, "btcDominanceChange24h")
    macro_risk_mode = str(representative_market_intelligence.get("riskMode", "neutral") or "neutral")

    base_market_stance = "balanced"
    if (
        dominant_regime in DOWNTREND_LABELS
        or high_volatility_share >= 0.50
        or exit_signal_count > buy_count
    ):
        base_market_stance = "defensive"
    elif (
        buy_count >= max(exit_signal_count + 1, 2)
        and trending_share >= 0.40
        and high_volatility_share < 0.45
    ):
        base_market_stance = "offensive"

    market_stance = base_market_stance
    if market_intelligence_available and macro_risk_mode == "risk_off":
        market_stance = "defensive"
    elif (
        market_intelligence_available
        and macro_risk_mode == "risk_on"
        and base_market_stance == "balanced"
        and buy_count >= exit_signal_count
    ):
        market_stance = "offensive"

    return {
        "marketStance": market_stance,
        "baseMarketStance": base_market_stance,
        "dominantRegime": dominant_regime,
        "buyCount": int(buy_count),
        "takeProfitCount": int(take_profit_count),
        "lossCount": int(loss_count),
        "trendingShare": float(trending_share),
        "highVolatilityShare": float(high_volatility_share),
        "marketIntelligenceAvailable": bool(market_intelligence_available),
        "fearGreedValue": float(fear_greed_value),
        "fearGreedClassification": fear_greed_classification,
        "btcDominance": float(btc_dominance),
        "btcDominanceChange24h": float(btc_dominance_change_24h),
        "macroRiskMode": macro_risk_mode,
    }
