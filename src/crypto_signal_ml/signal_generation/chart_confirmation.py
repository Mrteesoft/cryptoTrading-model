"""Chart-confirmation stage for raw signal candidates."""

from __future__ import annotations

from ..config import TrainingConfig
from .contracts import ChartConfirmedCandidate, RawSignalCandidate


_STRUCTURE_SCORES = {
    "higher_highs": 0.24,
    "higher_lows": 0.14,
    "range": 0.0,
    "lower_highs": -0.14,
    "lower_lows": -0.20,
    "downtrend": -0.26,
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp one numeric value between inclusive bounds."""

    return max(min(float(value), maximum), minimum)


def _safe_float(payload: dict[str, object], key: str, default_value: float = 0.0) -> float:
    """Read one float field from a chart-context payload."""

    raw_value = payload.get(key, default_value)
    if raw_value is None:
        return default_value

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default_value


def _resolve_pattern_label(
    *,
    breakout_confirmed: bool,
    retest_hold_confirmed: bool,
    near_resistance: bool,
    structure_label: str,
    support_distance_pct: float,
    resistance_distance_pct: float,
    range_compression_score: float,
    channel_position: float,
    trend_slope: float,
) -> str:
    """Map raw chart features into one operator-facing pattern label."""

    if retest_hold_confirmed:
        return "retest_confirmed"
    if breakout_confirmed:
        return "breakout_confirmed"
    if near_resistance:
        return "near_resistance"
    if structure_label == "downtrend":
        return "downtrend_structure"
    if structure_label in {"lower_highs", "lower_lows"}:
        return "weak_trend_structure"
    if structure_label == "range":
        return "range_bound"
    if (
        structure_label in {"higher_highs", "higher_lows"}
        and range_compression_score >= 0.10
        and 0.45 <= channel_position <= 0.88
    ):
        return "breakout_attempt"
    if (
        structure_label in {"higher_highs", "higher_lows", "range"}
        and support_distance_pct <= 0.025
        and resistance_distance_pct > 0.01
        and trend_slope >= 0.0
    ):
        return "reversal_attempt"

    return "no_clean_setup"


def apply_chart_confirmation(
    candidate: RawSignalCandidate,
    *,
    config: TrainingConfig | None = None,
) -> ChartConfirmedCandidate:
    """Score one raw candidate for chart cleanliness before risk gating."""

    config = config or TrainingConfig()
    chart_context = dict(candidate.chartContext)

    breakout_confirmed = bool(chart_context.get("breakoutConfirmed", False))
    retest_hold_confirmed = bool(chart_context.get("retestHoldConfirmed", False))
    near_resistance = bool(chart_context.get("nearResistance", False))
    structure_label = str(chart_context.get("structureLabel", "") or "").strip().lower()
    pattern_score = _safe_float(chart_context, "patternScore")
    trend_slope = _safe_float(chart_context, "trendSlope")
    channel_position = _safe_float(chart_context, "channelPosition", default_value=0.50)
    range_compression_score = _safe_float(chart_context, "rangeCompressionScore")
    resistance_distance_pct = _safe_float(chart_context, "resistanceDistancePct", default_value=0.02)
    support_distance_pct = _safe_float(chart_context, "supportDistancePct", default_value=0.02)

    pattern_label = _resolve_pattern_label(
        breakout_confirmed=breakout_confirmed,
        retest_hold_confirmed=retest_hold_confirmed,
        near_resistance=near_resistance,
        structure_label=structure_label,
        support_distance_pct=support_distance_pct,
        resistance_distance_pct=resistance_distance_pct,
        range_compression_score=range_compression_score,
        channel_position=channel_position,
        trend_slope=trend_slope,
    )

    pattern_reasons: list[str] = []
    score = 0.0
    score += _STRUCTURE_SCORES.get(structure_label, 0.0)
    score += _clamp(pattern_score, -1.0, 1.0) * 0.10

    if breakout_confirmed:
        score += 0.28
        pattern_reasons.append("breakout_confirmed")
    if retest_hold_confirmed:
        score += 0.16
        pattern_reasons.append("retest_hold_confirmed")

    if trend_slope >= 0.003:
        score += 0.08
        pattern_reasons.append("positive_trend_slope")
    elif trend_slope <= -0.003:
        score -= 0.08
        pattern_reasons.append("negative_trend_slope")

    if 0.55 <= channel_position <= 0.88:
        score += 0.04
        pattern_reasons.append("healthy_channel_position")
    elif channel_position >= 0.95:
        score -= 0.08
        pattern_reasons.append("overextended_channel_position")

    if range_compression_score >= 0.10:
        score += 0.06 if breakout_confirmed else 0.04
        pattern_reasons.append("range_compression_support")
    elif range_compression_score <= -0.20:
        score -= 0.05
        pattern_reasons.append("expanding_range_noise")

    if near_resistance:
        score -= 0.18
        pattern_reasons.append("price_near_local_resistance")
    if resistance_distance_pct <= float(getattr(config, "chart_near_resistance_pct", 0.01) or 0.01):
        score -= 0.06
        pattern_reasons.append("tight_resistance_room")
    elif resistance_distance_pct >= 0.03:
        score += 0.04
        pattern_reasons.append("has_resistance_room")

    if support_distance_pct <= 0.025 and structure_label in {"higher_lows", "range"}:
        score += 0.03
        pattern_reasons.append("support_nearby")

    score = _clamp(score, -1.0, 1.0)

    confirmed_min_score = float(
        getattr(config, "chart_confirmation_confirmed_min_score", 0.30) or 0.30
    )
    early_min_score = float(getattr(config, "chart_confirmation_early_min_score", 0.12) or 0.12)
    blocked_max_score = float(
        getattr(config, "chart_confirmation_invalid_max_score", -0.10) or -0.10
    )

    if pattern_label in {"breakout_confirmed", "retest_confirmed"} or (
        score >= confirmed_min_score and pattern_label not in {"near_resistance", "weak_trend_structure", "downtrend_structure"}
    ):
        chart_decision = "confirmed"
    elif pattern_label in {"breakout_attempt", "range_bound", "reversal_attempt"} or score >= early_min_score:
        chart_decision = "early"
    else:
        chart_decision = "blocked"

    if score <= blocked_max_score and chart_decision != "confirmed":
        chart_decision = "blocked"

    deduped_pattern_reasons: list[str] = []
    for reason in pattern_reasons:
        if reason and reason not in deduped_pattern_reasons:
            deduped_pattern_reasons.append(reason)

    return ChartConfirmedCandidate(
        **candidate.__dict__,
        chartConfirmationScore=float(score),
        chartSetupType=str(pattern_label),
        chartConfirmationStatus=str(chart_decision),
        chartPatternLabel=str(pattern_label),
        chartDecision=str(chart_decision),
        chartConfirmationNotes=tuple(deduped_pattern_reasons),
        chartPatternReasons=tuple(deduped_pattern_reasons),
    )


__all__ = ["apply_chart_confirmation"]
