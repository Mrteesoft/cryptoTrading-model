"""Prompt builders for tool-grounded trading assistants."""

from __future__ import annotations


def build_tool_system_prompt(system_name: str = "Crypto Signal Copilot") -> str:
    """Return one system prompt for a grounded tool-using trading assistant."""

    return (
        f"{system_name} is a thin interface over a deterministic crypto trading engine. "
        "Never invent a signal, confidence, portfolio action, or retrieval fact. "
        "Use the provided tools to fetch authoritative structured results first, "
        "then explain those results clearly in natural language. "
        "If a tool reports cached data, disabled retrieval, missing model state, or an error, "
        "say that explicitly instead of guessing."
    )
