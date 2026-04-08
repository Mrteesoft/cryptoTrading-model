"""Chat-layer facade for assistant sessions and response orchestration."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AssistantIntentRouter",
    "ConversationSessionStore",
    "PlannedToolCall",
    "ToolDrivenChatFlow",
    "TradingAssistantService",
]


def __getattr__(name: str) -> Any:
    """Lazily expose chat-layer classes without creating import cycles."""

    if name in {"ConversationSessionStore", "TradingAssistantService"}:
        from ..assistant import ConversationSessionStore, TradingAssistantService

        exported_values = {
            "ConversationSessionStore": ConversationSessionStore,
            "TradingAssistantService": TradingAssistantService,
        }
        return exported_values[name]

    if name in {"AssistantIntentRouter", "PlannedToolCall", "ToolDrivenChatFlow"}:
        from .flow import AssistantIntentRouter, PlannedToolCall, ToolDrivenChatFlow

        exported_values = {
            "AssistantIntentRouter": AssistantIntentRouter,
            "PlannedToolCall": PlannedToolCall,
            "ToolDrivenChatFlow": ToolDrivenChatFlow,
        }
        return exported_values[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
