"""Structured trader tools backed by the portfolio store and trader brain."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..config import TrainingConfig
from ..application import PublishedSignalViewService
from ..portfolio_core import TraderBrain
from ..trading.portfolio import TradingPortfolioStore
from .signals import SignalToolService


TRADER_PLAN_TOOL_SCHEMA = {
    "name": "get_trader_plan",
    "description": "Return the authoritative portfolio-aware trader plan from the trading engine.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "capital": {
                "type": "number",
                "description": "Optional capital override used only for this plan request.",
            },
            "force_refresh": {
                "type": "boolean",
                "default": False,
                "description": "When true, bypass the current live-signal cache before building the plan.",
            },
        },
        "additionalProperties": False,
    },
}

TOOL_SCHEMAS = [TRADER_PLAN_TOOL_SCHEMA]


def _utc_now_iso() -> str:
    """Return one UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


class TraderToolService:
    """Expose stable trader-plan tooling over the live engine and portfolio state."""

    tool_schemas = TOOL_SCHEMAS

    def __init__(
        self,
        signal_tools: SignalToolService,
        portfolio_store: TradingPortfolioStore,
        config: TrainingConfig | None = None,
        published_signal_service: PublishedSignalViewService | None = None,
    ) -> None:
        self.signal_tools = signal_tools
        self.portfolio_store = portfolio_store
        self.config = config or TrainingConfig()
        self.published_signal_service = published_signal_service

    def get_trader_plan(
        self,
        *,
        capital: float | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Return one stable trader-plan tool payload."""

        if self.published_signal_service is not None:
            state = self.published_signal_service.load_tool_signal_state(force_refresh=force_refresh)
            snapshot = {
                "signals": list(state.get("signals", [])),
                "generatedAt": state.get("overview", {}).get("generatedAt"),
                "marketDataSource": state.get("overview", {}).get("marketDataSource"),
                "requestMode": state.get("overview", {}).get("requestMode"),
                "productsCovered": state.get("overview", {}).get("productsCovered"),
                "primarySignal": state.get("overview", {}).get("primarySignal"),
                "marketSummary": state.get("overview", {}).get("marketSummary", {}),
            }
            snapshot_source = state.get("source")
            snapshot_warning = state.get("warning", "")
            snapshot_error = state.get("error", "")
            if state.get("status") != "ok":
                return {
                    "toolName": "get_trader_plan",
                    "requestedAt": _utc_now_iso(),
                    "status": "error",
                    "source": snapshot_source,
                    "forceRefresh": bool(force_refresh),
                    "warning": snapshot_warning,
                    "error": snapshot_error or "Trader plan is unavailable.",
                    "capital": capital,
                    "portfolio": {},
                    "liveSnapshot": {},
                    "traderPlan": {},
                }
        else:
            snapshot_result = self.signal_tools.load_market_snapshot(force_refresh=force_refresh)
            snapshot = snapshot_result.get("snapshot")
            snapshot_source = snapshot_result.get("source")
            snapshot_warning = snapshot_result.get("warning", "")
            snapshot_error = snapshot_result.get("error", "")
            if snapshot is None:
                return {
                    "toolName": "get_trader_plan",
                    "requestedAt": _utc_now_iso(),
                    "status": "error",
                    "source": snapshot_source,
                    "forceRefresh": bool(force_refresh),
                    "warning": snapshot_warning,
                    "error": snapshot_error or "Trader plan is unavailable.",
                    "capital": capital,
                    "portfolio": {},
                    "liveSnapshot": {},
                    "traderPlan": {},
                }

        if snapshot is None:
            return {
                "toolName": "get_trader_plan",
                "requestedAt": _utc_now_iso(),
                "status": "error",
                "source": snapshot_source,
                "forceRefresh": bool(force_refresh),
                "warning": snapshot_warning,
                "error": snapshot_error or "Trader plan is unavailable.",
                "capital": capital,
                "portfolio": {},
                "liveSnapshot": {},
                "traderPlan": {},
            }

        portfolio = self.portfolio_store.get_portfolio()
        resolved_capital = float(capital) if capital is not None else float(portfolio["capital"])

        trader_plan = TraderBrain(config=self.config).build_plan(
            signal_summaries=list(snapshot.get("signals", [])),
            positions=list(portfolio.get("positions", [])),
            capital=resolved_capital,
            trade_memory_by_product=self.portfolio_store.build_trade_learning_map(
                list(snapshot.get("signals", []))
            ),
        )

        return {
            "toolName": "get_trader_plan",
            "requestedAt": _utc_now_iso(),
            "status": "ok",
            "source": snapshot_source,
            "forceRefresh": bool(force_refresh),
            "warning": snapshot_warning,
            "error": "",
            "capital": resolved_capital,
            "portfolio": {
                **portfolio,
                "capital": resolved_capital,
            },
            "liveSnapshot": {
                "generatedAt": snapshot.get("generatedAt"),
                "marketDataSource": snapshot.get("marketDataSource"),
                "requestMode": snapshot.get("requestMode"),
                "productsCovered": snapshot.get("productsCovered"),
                "primarySignal": snapshot.get("primarySignal"),
                "marketSummary": snapshot.get("marketSummary", {}),
            },
            "traderPlan": trader_plan,
        }
