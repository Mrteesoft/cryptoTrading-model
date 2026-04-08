"""Structured tool layer for LLM-callable access to the authoritative engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .model import MODEL_STATUS_TOOL_SCHEMA, ModelToolService
from .retrieval import RETRIEVAL_SEARCH_TOOL_SCHEMA, RetrievalToolService
from .signals import (
    MARKET_OVERVIEW_TOOL_SCHEMA,
    SIGNAL_DETAIL_TOOL_SCHEMA,
    SignalToolService,
)
from .trader import TRADER_PLAN_TOOL_SCHEMA, TraderToolService


TOOL_SCHEMAS = [
    MARKET_OVERVIEW_TOOL_SCHEMA,
    SIGNAL_DETAIL_TOOL_SCHEMA,
    TRADER_PLAN_TOOL_SCHEMA,
    MODEL_STATUS_TOOL_SCHEMA,
    RETRIEVAL_SEARCH_TOOL_SCHEMA,
]


def _utc_now_iso() -> str:
    """Return one UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


class ToolRegistry:
    """List and execute the structured tools exposed by the engine."""

    def __init__(
        self,
        *,
        signal_tools: SignalToolService,
        trader_tools: TraderToolService,
        model_tools: ModelToolService,
        retrieval_tools: RetrievalToolService,
    ) -> None:
        self.signal_tools = signal_tools
        self.trader_tools = trader_tools
        self.model_tools = model_tools
        self.retrieval_tools = retrieval_tools
        self._tool_schemas = list(TOOL_SCHEMAS)
        self._handlers = {
            "get_market_overview": self.signal_tools.get_market_overview,
            "get_signal": self.signal_tools.get_signal,
            "get_trader_plan": self.trader_tools.get_trader_plan,
            "get_model_status": self.model_tools.get_model_status,
            "search_knowledge": self.retrieval_tools.search_knowledge,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the stable tool catalog."""

        return list(self._tool_schemas)

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one named tool with explicit JSON-style arguments."""

        normalized_tool_name = str(tool_name).strip()
        handler = self._handlers.get(normalized_tool_name)
        if handler is None:
            return {
                "toolName": normalized_tool_name,
                "requestedAt": _utc_now_iso(),
                "status": "unsupported",
                "error": f"Unknown tool: {normalized_tool_name}",
            }

        normalized_arguments = dict(arguments or {})
        return handler(**normalized_arguments)


__all__ = [
    "MARKET_OVERVIEW_TOOL_SCHEMA",
    "MODEL_STATUS_TOOL_SCHEMA",
    "ModelToolService",
    "RETRIEVAL_SEARCH_TOOL_SCHEMA",
    "RetrievalToolService",
    "SIGNAL_DETAIL_TOOL_SCHEMA",
    "SignalToolService",
    "TOOL_SCHEMAS",
    "TRADER_PLAN_TOOL_SCHEMA",
    "ToolRegistry",
    "TraderToolService",
]
