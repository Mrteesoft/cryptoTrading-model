"""Typed chart-confirmation review for portfolio planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


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


@dataclass(frozen=True)
class ChartConfirmationResult:
    """Normalized chart-confirmation result for trader-brain planning."""

    decision: str
    score: float
    pattern_label: str
    near_resistance: bool
    weak_structure: bool
    breakout_confirmed: bool
    retest_hold_confirmed: bool
    supportive: bool
    confirmed: bool
    early: bool
    blocked: bool
    needs_confirmation: bool
    structure_label: str = "unknown"
    alignment_score: float = 0.0
    pattern_reasons: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def status(self) -> str:
        """Compatibility alias for older consumers."""

        return self.decision

    @property
    def setup_type(self) -> str:
        """Compatibility alias for older consumers."""

        return self.pattern_label

    @property
    def invalid(self) -> bool:
        """Compatibility alias for older consumers."""

        return self.blocked


def review_chart_confirmation(
    *,
    signal_summary: Mapping[str, Any],
    confidence_calibration: Mapping[str, Any] | None = None,
) -> ChartConfirmationResult:
    """Build one typed chart-confirmation result from the signal payload."""

    chart_context = (
        signal_summary.get("chartContext")
        if isinstance(signal_summary.get("chartContext"), Mapping)
        else {}
    )
    confidence_calibration = confidence_calibration or (
        signal_summary.get("confidenceCalibration")
        if isinstance(signal_summary.get("confidenceCalibration"), Mapping)
        else {}
    )

    breakout_confirmed = _safe_bool(chart_context, "breakoutConfirmed")
    retest_hold_confirmed = _safe_bool(chart_context, "retestHoldConfirmed")
    near_resistance = _safe_bool(chart_context, "nearResistance")
    structure_label = str(chart_context.get("structureLabel", "") or "unknown").strip().lower()
    weak_structure = structure_label in {"lower_highs", "lower_lows", "downtrend"}
    score = _safe_float(signal_summary, "chartConfirmationScore")
    raw_status = str(
        signal_summary.get("chartDecision", signal_summary.get("chartConfirmationStatus", "")) or ""
    ).strip().lower()
    pattern_label = str(
        signal_summary.get("chartPatternLabel", signal_summary.get("chartSetupType", "no_clean_setup"))
        or "no_clean_setup"
    ).strip().lower()
    alignment_score = _safe_float(confidence_calibration, "chartAlignmentScore")
    raw_pattern_reasons = signal_summary.get("chartPatternReasons")
    if not isinstance(raw_pattern_reasons, (list, tuple)):
        raw_pattern_reasons = signal_summary.get("chartConfirmationNotes") or []

    if raw_status in {"blocked", "invalid"}:
        decision = "blocked"
    elif raw_status == "confirmed" or breakout_confirmed or retest_hold_confirmed:
        decision = "confirmed"
    elif raw_status == "early":
        decision = "early"
    elif raw_status == "unclear":
        decision = "early"
    elif weak_structure and near_resistance:
        decision = "blocked"
    elif score >= 0.18 or alignment_score >= 0.18:
        decision = "early"
    elif pattern_label in {"breakout_attempt", "range_bound", "reversal_attempt"}:
        decision = "early"
    else:
        decision = "blocked"

    confirmed = decision == "confirmed"
    early = decision == "early"
    blocked = decision == "blocked"
    needs_confirmation = early
    supportive = confirmed or early or score >= 0.18 or alignment_score >= 0.18

    pattern_reasons: list[str] = []
    for raw_reason in raw_pattern_reasons:
        normalized_reason = str(raw_reason).strip().lower()
        if normalized_reason:
            pattern_reasons.append(normalized_reason)

    reasons: list[str] = []
    if confirmed:
        reasons.append("chart_confirmed")
    elif blocked:
        reasons.append("chart_blocked")
    elif needs_confirmation:
        reasons.append("needs_chart_confirmation")

    if breakout_confirmed:
        reasons.append("breakout_confirmed")
    if retest_hold_confirmed:
        reasons.append("retest_hold_confirmed")
    if near_resistance:
        reasons.append("near_resistance")
    if weak_structure:
        reasons.append("weak_structure")
    if pattern_label and pattern_label != "no_clean_setup":
        reasons.append(f"pattern_{pattern_label}")
    reasons.extend(pattern_reasons)

    deduped_reasons: list[str] = []
    for reason in reasons:
        if reason and reason not in deduped_reasons:
            deduped_reasons.append(reason)

    deduped_pattern_reasons: list[str] = []
    for reason in pattern_reasons:
        if reason and reason not in deduped_pattern_reasons:
            deduped_pattern_reasons.append(reason)

    return ChartConfirmationResult(
        decision=decision,
        score=float(score),
        pattern_label=pattern_label,
        near_resistance=bool(near_resistance),
        weak_structure=bool(weak_structure),
        breakout_confirmed=bool(breakout_confirmed),
        retest_hold_confirmed=bool(retest_hold_confirmed),
        supportive=bool(supportive),
        confirmed=bool(confirmed),
        early=bool(early),
        blocked=bool(blocked),
        needs_confirmation=bool(needs_confirmation),
        structure_label=structure_label,
        alignment_score=float(alignment_score),
        pattern_reasons=tuple(deduped_pattern_reasons),
        reasons=tuple(deduped_reasons),
    )


__all__ = ["ChartConfirmationResult", "review_chart_confirmation"]
