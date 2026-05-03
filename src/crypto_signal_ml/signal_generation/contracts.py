"""Frozen contracts for the staged signal lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SignalContributionLedger:
    """Auditable contribution trace for one final signal."""

    rawProbabilities: dict[str, float]
    calibratedProbabilities: dict[str, float]
    rawConfidence: float
    calibratedConfidence: float
    probabilityMargin: float
    policyStatus: str
    rejectionReasons: tuple[str, ...]
    confidenceAdjustments: dict[str, float]
    finalDecisionScore: float
    publicationReason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ledger into a JSON-friendly dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class RawSignalCandidate:
    """Base signal candidate directly derived from model outputs."""

    productId: str
    timestamp: str | None
    close: float
    symbol: str
    pairSymbol: str
    baseCurrency: str | None
    quoteCurrency: str | None
    coinName: str | None
    coinCategory: str | None
    timeStep: int
    rawSignalName: str
    rawPredictedSignal: int
    rawSpotAction: str
    rawProbabilities: dict[str, float]
    calibratedProbabilities: dict[str, float]
    rawConfidence: float
    calibratedConfidence: float
    primaryProbability: float
    probabilityMargin: float
    hasProbabilityColumns: bool
    minimumActionConfidence: float
    setupScore: float
    chartContext: dict[str, Any]
    executionContext: dict[str, Any]
    marketContext: dict[str, Any]
    marketState: dict[str, Any]
    eventContext: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict, kw_only=True)


@dataclass(frozen=True)
class ChartConfirmedCandidate(RawSignalCandidate):
    """Raw candidate after chart-structure confirmation has been evaluated."""

    chartConfirmationScore: float
    chartSetupType: str
    chartConfirmationStatus: str
    chartPatternLabel: str
    chartDecision: str
    chartConfirmationNotes: tuple[str, ...] = field(default_factory=tuple, kw_only=True)
    chartPatternReasons: tuple[str, ...] = field(default_factory=tuple, kw_only=True)


@dataclass(frozen=True)
class GatedSignalCandidate(ChartConfirmedCandidate):
    """Chart-confirmed candidate after risk and execution gating has been applied."""

    signalName: str
    predictedSignal: int
    spotAction: str
    actionable: bool
    requiredActionConfidence: float
    confidenceGateApplied: bool
    riskGateApplied: bool
    tradeReadiness: str
    policyScore: float
    policyNotes: tuple[str, ...]
    gateReasons: tuple[str, ...]
    ledger: SignalContributionLedger
    reasonItems: tuple[str, ...] = ()
    reasonSummary: str = ""
    signalChat: str = ""


@dataclass(frozen=True)
class ContextEnrichedCandidate(GatedSignalCandidate):
    """Placeholder contract for future context enrichment."""

    newsContext: dict[str, Any] = field(default_factory=dict)
    trendContext: dict[str, Any] = field(default_factory=dict)
    contextEvidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredSignalCandidate(ContextEnrichedCandidate):
    """Placeholder contract for future fusion and scoring."""

    confidenceCalibration: dict[str, Any] = field(default_factory=dict)
    adaptiveContext: dict[str, Any] = field(default_factory=dict)
    finalDecisionScore: float = 0.0


@dataclass(frozen=True)
class ActionableSignal(ScoredSignalCandidate):
    """Placeholder contract for future portfolio action mapping."""

    portfolioDecision: str = "watchlist"
    tradeContext: dict[str, Any] = field(default_factory=dict)
    brain: dict[str, Any] = field(default_factory=dict)
    watchlistState: dict[str, Any] = field(default_factory=dict)
    watchlistPromotion: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublishedSignal(ActionableSignal):
    """Placeholder contract for future publication output."""

    publicationReason: str = "candidate_only"
    publicSignalType: str = "candidate"
    watchlistFallback: bool = False
    published: bool = True
