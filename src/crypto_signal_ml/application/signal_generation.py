"""Signal-generation coordinator over inference, enrichment, decision, and publication stages."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from ..frontend import build_frontend_signal_snapshot
from ..logging_utils import format_bool_for_log
from ..trading.portfolio import TradingPortfolioStore
from .signal_context_enrichment import SignalContextEnrichmentArtifacts, SignalContextEnrichmentStage
from .signal_decision import SignalDecisionArtifacts
from .signal_enrichment import SignalEnrichmentArtifacts, SignalEnrichmentStage
from .signal_inference import SignalInferenceArtifacts, SignalInferenceStage
from .signal_publication import SignalPublicationArtifacts, SignalPublicationStage


LOGGER = logging.getLogger(__name__)


@dataclass
class SignalPipelineArtifacts:
    """End-to-end signal lifecycle outputs before publication."""

    inference: SignalInferenceArtifacts
    context: SignalContextEnrichmentArtifacts | None
    enrichment: SignalEnrichmentArtifacts
    decision: SignalDecisionArtifacts


class SignalGenerationCoordinator:
    """Coordinate the explicit signal lifecycle stages."""

    def __init__(
        self,
        *,
        inference_stage: SignalInferenceStage,
        context_stage: SignalContextEnrichmentStage | None = None,
        enrichment_stage: SignalEnrichmentStage,
        decision_stage,
        publication_stage: SignalPublicationStage | None = None,
    ) -> None:
        self.inference_stage = inference_stage
        self.context_stage = context_stage
        self.enrichment_stage = enrichment_stage
        self.decision_stage = decision_stage
        self.publication_stage = publication_stage

    def run_pipeline(
        self,
        *,
        inference_artifacts: SignalInferenceArtifacts,
        portfolio_store: TradingPortfolioStore,
        save_watchlist_pool_snapshot: bool = False,
    ) -> SignalPipelineArtifacts:
        """Run enrichment and decision stages for one inference result."""

        LOGGER.info(
            "Stages start | candidates=%s | context=%s | watchlist_snapshot=%s",
            len(inference_artifacts.signal_candidates),
            format_bool_for_log(self.context_stage is not None),
            format_bool_for_log(bool(save_watchlist_pool_snapshot and self.publication_stage is not None)),
        )
        context_artifacts = None
        enriched_candidates = inference_artifacts.signal_candidates
        if self.context_stage is not None:
            LOGGER.info(
                "Context | start | candidates=%s",
                len(inference_artifacts.signal_candidates),
            )
            context_artifacts = self.context_stage.enrich(inference_artifacts.signal_candidates)
            enriched_candidates = context_artifacts.signal_summaries
            LOGGER.info(
                "Context | done | rows=%s | products=%s",
                len(enriched_candidates),
                len(context_artifacts.event_context_by_product),
            )

        LOGGER.info(
            "Portfolio | start | rows=%s",
            len(enriched_candidates),
        )
        enrichment_artifacts = self.enrichment_stage.enrich(enriched_candidates, portfolio_store)
        LOGGER.info(
            "Portfolio | done | rows=%s | products=%s",
            len(enrichment_artifacts.signal_summaries),
            len(enrichment_artifacts.active_signal_context_by_product),
        )
        if save_watchlist_pool_snapshot and self.publication_stage is not None:
            LOGGER.info(
                "Watchlist pool | save from %s row(s)",
                len(enrichment_artifacts.signal_summaries),
            )
            self.publication_stage.save_watchlist_pool_snapshot(enrichment_artifacts.signal_summaries)

        LOGGER.info(
            "Decision | start | rows=%s",
            len(enrichment_artifacts.signal_summaries),
        )
        decision_artifacts = self.decision_stage.decide(enrichment_artifacts.signal_summaries)
        primary_signal_reference = (
            (
                f"{str(decision_artifacts.primary_signal.get('productId', '')).strip().upper() or 'UNKNOWN'}:"
                f"{str(decision_artifacts.primary_signal.get('signal_name', '')).strip().upper() or 'UNKNOWN'}"
            )
            if decision_artifacts.primary_signal is not None
            else "none"
        )
        LOGGER.info(
            "Decision | done | published=%s | actionable=%s | primary=%s",
            len(decision_artifacts.published_signals),
            len(decision_artifacts.actionable_signals),
            primary_signal_reference,
        )
        return SignalPipelineArtifacts(
            inference=inference_artifacts,
            context=context_artifacts,
            enrichment=enrichment_artifacts,
            decision=decision_artifacts,
        )

    def build_live_snapshot(
        self,
        *,
        model_type: str,
        pipeline_artifacts: SignalPipelineArtifacts,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build one non-persisted live snapshot from the shared pipeline outputs."""

        live_snapshot = build_frontend_signal_snapshot(
            model_type=model_type,
            primary_signal=pipeline_artifacts.decision.primary_signal,
            latest_signals=pipeline_artifacts.decision.published_signals,
            actionable_signals=pipeline_artifacts.decision.actionable_signals,
            trader_brain=pipeline_artifacts.enrichment.trader_brain_snapshot,
        )
        if extra_fields:
            live_snapshot.update(dict(extra_fields))
        return live_snapshot

    def publish_signal_generation(
        self,
        *,
        model_type: str,
        historical_prediction_df,
        pipeline_artifacts: SignalPipelineArtifacts,
        signal_source: str,
        signal_metadata: dict[str, Any],
        market_data_refresh: dict[str, Any] | None,
        market_data_refreshed_at: str | None,
        portfolio_store: TradingPortfolioStore,
    ) -> SignalPublicationArtifacts:
        """Run the publication stage over the shared pipeline outputs."""

        if self.publication_stage is None:
            raise ValueError("Signal publication is not configured for this coordinator.")

        primary_signal_reference = (
            (
                f"{str(pipeline_artifacts.decision.primary_signal.get('productId', '')).strip().upper() or 'UNKNOWN'}:"
                f"{str(pipeline_artifacts.decision.primary_signal.get('signal_name', '')).strip().upper() or 'UNKNOWN'}"
            )
            if pipeline_artifacts.decision.primary_signal is not None
            else "none"
        )
        LOGGER.info(
            "Publish | start | model=%s | source=%s | latest=%s | actionable=%s | primary=%s",
            model_type,
            signal_source,
            len(pipeline_artifacts.decision.published_signals),
            len(pipeline_artifacts.decision.actionable_signals),
            primary_signal_reference,
        )
        publication_artifacts = self.publication_stage.publish(
            model_type=model_type,
            historical_prediction_df=historical_prediction_df,
            latest_signals=pipeline_artifacts.decision.published_signals,
            actionable_signals=pipeline_artifacts.decision.actionable_signals,
            primary_signal=pipeline_artifacts.decision.primary_signal,
            trader_brain_snapshot=pipeline_artifacts.enrichment.trader_brain_snapshot,
            signal_source=signal_source,
            signal_metadata=signal_metadata,
            market_data_refresh=market_data_refresh,
            market_data_refreshed_at=market_data_refreshed_at,
            signal_inference_summary=pipeline_artifacts.inference.summary,
            portfolio_store=portfolio_store,
        )
        LOGGER.info(
            "Publish | done | store=%s | tracked_created=%s | tracked_refreshed=%s",
            int(publication_artifacts.signal_store_summary.get("signalCount", 0)),
            int(publication_artifacts.tracked_trade_sync.get("createdCount", 0)),
            int(publication_artifacts.tracked_trade_sync.get("refreshedCount", 0)),
        )
        return publication_artifacts
