"""Signal enrichment stage for portfolio and trader context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import TrainingConfig
from ..portfolio_core import PortfolioActionMapper, build_market_context
from ..signal_generation import (
    SignalFusionStage,
    actionable_signal_to_summary,
    context_enriched_candidate_from_summary,
)
from ..trading.portfolio import TradingPortfolioStore


@dataclass
class SignalEnrichmentArtifacts:
    """Portfolio-aware outputs built on top of raw signal candidates."""

    signal_summaries: list[dict[str, Any]]
    portfolio: dict[str, Any]
    trader_brain_plan: dict[str, Any]
    trader_brain_snapshot: dict[str, Any]
    active_signal_context_by_product: dict[str, dict[str, Any]]


class SignalEnrichmentStage:
    """Attach portfolio, trade, and trader-brain context to signals."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.fusion_stage = SignalFusionStage(config=config)
        self.portfolio_action_mapper = PortfolioActionMapper(config=config)

    def enrich(
        self,
        signal_summaries: list[dict[str, Any]],
        portfolio_store: TradingPortfolioStore,
    ) -> SignalEnrichmentArtifacts:
        """Build one enriched signal set from normalized candidates."""

        market_context = build_market_context(signal_summaries)
        trade_memory_by_product = portfolio_store.build_trade_learning_map(signal_summaries)
        scored_candidates = self.fusion_stage.score_candidates(
            [context_enriched_candidate_from_summary(signal_summary) for signal_summary in signal_summaries],
            market_context=market_context,
            trade_memory_by_product=trade_memory_by_product,
        )
        portfolio_actions = self.portfolio_action_mapper.plan_actions(
            list(scored_candidates),
            portfolio_store=portfolio_store,
        )

        return SignalEnrichmentArtifacts(
            signal_summaries=[
                actionable_signal_to_summary(signal)
                for signal in portfolio_actions.actionable_signals
            ],
            portfolio=portfolio_actions.portfolio,
            trader_brain_plan=portfolio_actions.trader_brain_plan,
            trader_brain_snapshot=portfolio_actions.trader_brain_snapshot,
            active_signal_context_by_product=portfolio_actions.active_signal_context_by_product,
        )
