"""Application-layer facade for workflows and signal-lifecycle stages."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BacktestApp",
    "BaseSignalApp",
    "MarketDataRefreshApp",
    "MarketEventsRefreshApp",
    "MarketUniverseRefreshApp",
    "ModelComparisonApp",
    "PrimarySignalHistoryStore",
    "ProductionCycleApp",
    "PublishedSignalViewService",
    "RegimeTrainingApp",
    "RegimeWalkForwardValidationApp",
    "SignalDecisionArtifacts",
    "SignalDecisionStage",
    "SignalEnrichmentArtifacts",
    "SignalEnrichmentStage",
    "SignalGenerationApp",
    "SignalGenerationCoordinator",
    "SignalInferenceArtifacts",
    "SignalInferenceStage",
    "SignalParameterTuningApp",
    "SignalPipelineArtifacts",
    "SignalPublicationArtifacts",
    "SignalPublicationStage",
    "TrainingApp",
    "WalkForwardValidationApp",
]


def __getattr__(name: str) -> Any:
    """Lazily expose workflow and stage classes without creating import cycles."""

    if name in {
        "BacktestApp",
        "BaseSignalApp",
        "MarketDataRefreshApp",
        "MarketEventsRefreshApp",
        "MarketUniverseRefreshApp",
        "ModelComparisonApp",
        "ProductionCycleApp",
        "RegimeTrainingApp",
        "RegimeWalkForwardValidationApp",
        "SignalGenerationApp",
        "SignalParameterTuningApp",
        "TrainingApp",
        "WalkForwardValidationApp",
    }:
        from ..app import (
            BacktestApp,
            BaseSignalApp,
            MarketDataRefreshApp,
            MarketEventsRefreshApp,
            MarketUniverseRefreshApp,
            ModelComparisonApp,
            ProductionCycleApp,
            RegimeTrainingApp,
            RegimeWalkForwardValidationApp,
            SignalGenerationApp,
            SignalParameterTuningApp,
            TrainingApp,
            WalkForwardValidationApp,
        )

        exported_values = {
            "BacktestApp": BacktestApp,
            "BaseSignalApp": BaseSignalApp,
            "MarketDataRefreshApp": MarketDataRefreshApp,
            "MarketEventsRefreshApp": MarketEventsRefreshApp,
            "MarketUniverseRefreshApp": MarketUniverseRefreshApp,
            "ModelComparisonApp": ModelComparisonApp,
            "ProductionCycleApp": ProductionCycleApp,
            "RegimeTrainingApp": RegimeTrainingApp,
            "RegimeWalkForwardValidationApp": RegimeWalkForwardValidationApp,
            "SignalGenerationApp": SignalGenerationApp,
            "SignalParameterTuningApp": SignalParameterTuningApp,
            "TrainingApp": TrainingApp,
            "WalkForwardValidationApp": WalkForwardValidationApp,
        }
        return exported_values[name]

    if name in {"SignalInferenceArtifacts", "SignalInferenceStage"}:
        from .signal_inference import SignalInferenceArtifacts, SignalInferenceStage

        exported_values = {
            "SignalInferenceArtifacts": SignalInferenceArtifacts,
            "SignalInferenceStage": SignalInferenceStage,
        }
        return exported_values[name]

    if name in {"SignalEnrichmentArtifacts", "SignalEnrichmentStage"}:
        from .signal_enrichment import SignalEnrichmentArtifacts, SignalEnrichmentStage

        exported_values = {
            "SignalEnrichmentArtifacts": SignalEnrichmentArtifacts,
            "SignalEnrichmentStage": SignalEnrichmentStage,
        }
        return exported_values[name]

    if name in {
        "PrimarySignalHistoryStore",
        "SignalDecisionArtifacts",
        "SignalDecisionStage",
    }:
        from .signal_decision import (
            PrimarySignalHistoryStore,
            SignalDecisionArtifacts,
            SignalDecisionStage,
        )

        exported_values = {
            "PrimarySignalHistoryStore": PrimarySignalHistoryStore,
            "SignalDecisionArtifacts": SignalDecisionArtifacts,
            "SignalDecisionStage": SignalDecisionStage,
        }
        return exported_values[name]

    if name in {
        "PublishedSignalViewService",
        "SignalPublicationArtifacts",
        "SignalPublicationStage",
    }:
        from .signal_publication import (
            PublishedSignalViewService,
            SignalPublicationArtifacts,
            SignalPublicationStage,
        )

        exported_values = {
            "PublishedSignalViewService": PublishedSignalViewService,
            "SignalPublicationArtifacts": SignalPublicationArtifacts,
            "SignalPublicationStage": SignalPublicationStage,
        }
        return exported_values[name]

    if name in {"SignalGenerationCoordinator", "SignalPipelineArtifacts"}:
        from .signal_generation import SignalGenerationCoordinator, SignalPipelineArtifacts

        exported_values = {
            "SignalGenerationCoordinator": SignalGenerationCoordinator,
            "SignalPipelineArtifacts": SignalPipelineArtifacts,
        }
        return exported_values[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
