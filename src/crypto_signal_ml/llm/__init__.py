"""LLM adapter package for provider-specific integration code."""

from .base import ChatModelAdapter, LlmCompletionResponse, LlmMessage, LlmToolCall, LlmToolSpec
from .lunatrix_adapter import LunatrixChatModelAdapter
from .ollama_adapter import OllamaChatModelAdapter
from .openai_adapter import OpenAIChatModelAdapter
from .prompting import build_tool_system_prompt

__all__ = [
    "ChatModelAdapter",
    "LunatrixChatModelAdapter",
    "LlmCompletionResponse",
    "LlmMessage",
    "LlmToolCall",
    "LlmToolSpec",
    "OllamaChatModelAdapter",
    "OpenAIChatModelAdapter",
    "build_tool_system_prompt",
]
