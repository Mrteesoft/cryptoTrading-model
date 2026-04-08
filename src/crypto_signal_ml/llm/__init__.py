"""LLM adapter package for provider-specific integration code."""

from .base import ChatModelAdapter, LlmCompletionResponse, LlmMessage, LlmToolCall, LlmToolSpec
from .openai_adapter import OpenAIChatModelAdapter
from .prompting import build_tool_system_prompt

__all__ = [
    "ChatModelAdapter",
    "LlmCompletionResponse",
    "LlmMessage",
    "LlmToolCall",
    "LlmToolSpec",
    "OpenAIChatModelAdapter",
    "build_tool_system_prompt",
]
