"""Portfolio action mapping over scored signal candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import TrainingConfig
from ..signal_generation import ActionableSignal, ScoredSignalCandidate
from ..signal_generation.serialization import actionable_signal_from_summary, scored_candidate_to_summary
from ..trading.portfolio import TradingPortfolioStore
from ..trading.signals import apply_signal_trade_context
from .trader_brain import TraderBrain


@dataclass(frozen=True)
class PortfolioActionArtifacts:
    """Portfolio-aware action outputs for one scored signal batch."""

    actionable_signals: tuple[ActionableSignal, ...]
    portfolio: dict[str, Any]
    trader_brain_plan: dict[str, Any]
    trader_brain_snapshot: dict[str, Any]
    active_signal_context_by_product: dict[str, dict[str, Any]]


class PortfolioActionMapper:
    """Bridge scored candidates into portfolio-aware action planning."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()

    def plan_actions(
        self,
        scored_candidates: list[ScoredSignalCandidate],
        *,
        portfolio_store: TradingPortfolioStore,
    ) -> PortfolioActionArtifacts:
        """Map scored candidates into portfolio-aware actions."""

        portfolio = portfolio_store.get_portfolio()
        positions_by_product = {
            str(position.get("productId", "")).strip().upper(): position
            for position in list(portfolio.get("positions", []))
            if str(position.get("productId", "")).strip()
        }
        active_signal_context_by_product: dict[str, dict[str, Any]] = {}
        for candidate in scored_candidates:
            product_id = str(candidate.productId).strip().upper()
            if not product_id:
                continue

            active_trade = portfolio_store.get_active_trade_for_product(product_id)
            position = positions_by_product.get(product_id)
            if active_trade is None and position is None:
                continue

            active_signal_context_by_product[product_id] = {
                "entryPrice": (
                    position.get("entryPrice")
                    if position is not None and position.get("entryPrice") is not None
                    else (active_trade.get("entryPrice") if active_trade is not None else None)
                ),
                "currentPrice": (
                    position.get("currentPrice")
                    if position is not None and position.get("currentPrice") is not None
                    else (active_trade.get("currentPrice") if active_trade is not None else None)
                ),
                "stopLossPrice": active_trade.get("stopLossPrice") if active_trade is not None else None,
                "takeProfitPrice": active_trade.get("takeProfitPrice") if active_trade is not None else None,
                "positionFraction": position.get("positionFraction") if position is not None else None,
                "quantity": position.get("quantity") if position is not None else None,
                "openedAt": (
                    position.get("openedAt")
                    if position is not None and position.get("openedAt") is not None
                    else (active_trade.get("openedAt") if active_trade is not None else None)
                ),
                "status": active_trade.get("status") if active_trade is not None else None,
            }

        scored_signal_summaries = [
            scored_candidate_to_summary(candidate)
            for candidate in scored_candidates
        ]
        trade_enriched_signals = apply_signal_trade_context(
            scored_signal_summaries,
            active_trade_product_ids=portfolio_store.get_active_signal_product_ids(),
            active_signal_context_by_product=active_signal_context_by_product,
            config=self.config,
        )
        trader_brain_plan = TraderBrain(config=self.config).build_plan(
            signal_summaries=trade_enriched_signals,
            positions=list(portfolio.get("positions", [])),
            capital=float(portfolio.get("capital") or 0.0),
            trade_memory_by_product=portfolio_store.build_trade_learning_map(trade_enriched_signals),
        )
        trader_brain_snapshot = {
            key: value
            for key, value in trader_brain_plan.items()
            if key != "signals"
        }

        return PortfolioActionArtifacts(
            actionable_signals=tuple(
                actionable_signal_from_summary(signal_summary)
                for signal_summary in list(trader_brain_plan.get("signals", []))
            ),
            portfolio=portfolio,
            trader_brain_plan=trader_brain_plan,
            trader_brain_snapshot=trader_brain_snapshot,
            active_signal_context_by_product=active_signal_context_by_product,
        )
