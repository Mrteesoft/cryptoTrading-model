"""Explicit risk-gate naming over the policy-gating stage."""

from __future__ import annotations

from ..config import TrainingConfig
from .contracts import ChartConfirmedCandidate, GatedSignalCandidate
from .policy_gating import apply_policy_gate


def apply_risk_gate(
    candidate: ChartConfirmedCandidate,
    *,
    config: TrainingConfig | None = None,
) -> GatedSignalCandidate:
    """Run the narrow risk and execution review after chart confirmation."""

    return apply_policy_gate(candidate, config=config)


__all__ = ["apply_risk_gate"]
