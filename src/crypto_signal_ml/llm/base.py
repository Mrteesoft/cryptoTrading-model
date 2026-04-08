"""Provider-agnostic LLM adapter contracts for tool-calling chat layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class LlmToolSpec:
    """Stable tool schema passed into provider-specific LLM adapters."""

    name: str
    description: str
    input_schema: dict[str, Any]
    strict: bool = True


@dataclass(frozen=True)
class LlmMessage:
    """One normalized chat message sent into an LLM adapter."""

    role: str
    content: str


@dataclass(frozen=True)
class LlmToolCall:
    """One structured tool call emitted by an LLM adapter."""

    name: str
    arguments: dict[str, Any]


@dataclass
class LlmCompletionResponse:
    """One normalized model response from an LLM adapter."""

    message: str
    tool_calls: list[LlmToolCall] = field(default_factory=list)
    raw_response: Any = None


class ChatModelAdapter(ABC):
    """Abstract base class for provider-specific chat adapters."""

    provider_name = "base"

    @abstractmethod
    def is_configured(self) -> bool:
        """Return whether the adapter is configured well enough to make requests."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> LlmCompletionResponse:
        """Generate one response plus any structured tool calls."""
