"""Core staged signal-generation contracts and stage helpers."""

from .audit import build_signal_contribution_ledger
from .candidate_discovery import discover_raw_signal_candidate
from .candidate_generation import build_raw_signal_candidate
from .chart_confirmation import apply_chart_confirmation
from .context_enrichment import SignalContextEnricher
from .contracts import (
    ActionableSignal,
    ChartConfirmedCandidate,
    ContextEnrichedCandidate,
    GatedSignalCandidate,
    PublishedSignal,
    RawSignalCandidate,
    ScoredSignalCandidate,
    SignalContributionLedger,
)
from .fusion import SignalFusionStage
from .orchestrator import gate_prediction_row
from .policy_gating import apply_policy_gate
from .risk_gate import apply_risk_gate
from .serialization import (
    actionable_signal_from_summary,
    actionable_signal_to_summary,
    context_enriched_candidate_from_summary,
    context_enriched_candidate_to_summary,
    gated_candidate_from_summary,
    gated_candidate_to_summary,
    published_signal_from_summary,
    published_signal_to_summary,
    scored_candidate_to_summary,
)

__all__ = [
    "ActionableSignal",
    "ChartConfirmedCandidate",
    "ContextEnrichedCandidate",
    "GatedSignalCandidate",
    "PublishedSignal",
    "RawSignalCandidate",
    "ScoredSignalCandidate",
    "SignalContextEnricher",
    "SignalContributionLedger",
    "SignalFusionStage",
    "apply_chart_confirmation",
    "apply_risk_gate",
    "actionable_signal_from_summary",
    "actionable_signal_to_summary",
    "apply_policy_gate",
    "build_raw_signal_candidate",
    "build_signal_contribution_ledger",
    "discover_raw_signal_candidate",
    "context_enriched_candidate_from_summary",
    "context_enriched_candidate_to_summary",
    "gate_prediction_row",
    "gated_candidate_from_summary",
    "gated_candidate_to_summary",
    "published_signal_from_summary",
    "published_signal_to_summary",
    "scored_candidate_to_summary",
]
