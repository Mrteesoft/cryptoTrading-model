"""Audit helpers for contribution-ledger construction."""

from __future__ import annotations

from .contracts import SignalContributionLedger


def build_signal_contribution_ledger(
    *,
    raw_probabilities: dict[str, float],
    calibrated_probabilities: dict[str, float],
    raw_confidence: float,
    calibrated_confidence: float,
    probability_margin: float,
    policy_status: str,
    rejection_reasons: list[str] | tuple[str, ...],
    required_action_confidence: float,
    final_decision_score: float,
    publication_reason: str,
) -> SignalContributionLedger:
    """Build the canonical ledger object for one gated signal."""

    return SignalContributionLedger(
        rawProbabilities={signal_name: float(probability) for signal_name, probability in raw_probabilities.items()},
        calibratedProbabilities={
            signal_name: float(probability)
            for signal_name, probability in calibrated_probabilities.items()
        },
        rawConfidence=float(raw_confidence),
        calibratedConfidence=float(calibrated_confidence),
        probabilityMargin=float(probability_margin),
        policyStatus=str(policy_status),
        rejectionReasons=tuple(str(reason) for reason in rejection_reasons),
        confidenceAdjustments={
            "calibrationDelta": float(calibrated_confidence - raw_confidence),
            "requiredActionConfidence": float(required_action_confidence),
        },
        finalDecisionScore=float(final_decision_score),
        publicationReason=str(publication_reason),
    )
