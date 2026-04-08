"""OpenAI-specific adapter shell that isolates provider request wiring."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .base import ChatModelAdapter, LlmCompletionResponse, LlmMessage, LlmToolCall, LlmToolSpec


class OpenAIChatModelAdapter(ChatModelAdapter):
    """
    Isolate OpenAI-specific tool-calling integration behind one adapter.

    The concrete SDK or HTTP invocation is intentionally injected through
    ``request_executor`` so the rest of the codebase does not depend on a
    specific OpenAI client version.
    """

    provider_name = "openai"

    def __init__(
        self,
        *,
        model: str,
        request_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.model = str(model).strip()
        self.request_executor = request_executor

    def is_configured(self) -> bool:
        """Return whether the adapter has both a model id and an executor."""

        return bool(self.model and self.request_executor is not None)

    def build_request_payload(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Normalize one provider request payload for the injected executor."""

        normalized_messages = []
        if system_prompt:
            normalized_messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        normalized_messages.extend(
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        )

        normalized_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "strict": bool(tool.strict),
                },
            }
            for tool in tools
        ]

        return {
            "provider": self.provider_name,
            "model": self.model,
            "messages": normalized_messages,
            "tools": normalized_tools,
        }

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> LlmCompletionResponse:
        """Call the injected executor and normalize the response."""

        if self.request_executor is None:
            raise RuntimeError(
                "OpenAIChatModelAdapter requires a request_executor. "
                "Inject the concrete SDK or HTTP client from the host application."
            )

        payload = self.build_request_payload(
            messages=messages,
            tools=tools,
            system_prompt=system_prompt,
        )
        raw_response = self.request_executor(payload)
        return self._parse_response(raw_response)

    @staticmethod
    def _parse_response(raw_response: dict[str, Any]) -> LlmCompletionResponse:
        """Normalize one executor response into the shared completion structure."""

        message = str(raw_response.get("message") or "")
        raw_tool_calls = list(raw_response.get("toolCalls", []))
        tool_calls = [
            LlmToolCall(
                name=str(tool_call.get("name") or ""),
                arguments=dict(tool_call.get("arguments") or {}),
            )
            for tool_call in raw_tool_calls
            if str(tool_call.get("name") or "").strip()
        ]
        return LlmCompletionResponse(
            message=message,
            tool_calls=tool_calls,
            raw_response=raw_response,
        )
