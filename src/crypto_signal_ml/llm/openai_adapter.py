"""OpenAI-specific adapter shell that isolates provider request wiring."""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

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
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        request_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.model = str(model).strip()
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url or "https://api.openai.com/v1").strip().rstrip("/")
        self.request_executor = request_executor

    def is_configured(self) -> bool:
        """Return whether the adapter has enough configuration to make requests."""

        return bool(self.model and (self.request_executor is not None or self.api_key))

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

        payload: dict[str, Any] = {
            "provider": self.provider_name,
            "model": self.model,
            "messages": normalized_messages,
        }
        if normalized_tools:
            payload["tools"] = normalized_tools
        return payload

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> LlmCompletionResponse:
        """Call the injected executor and normalize the response."""

        payload = self.build_request_payload(
            messages=messages,
            tools=tools,
            system_prompt=system_prompt,
        )
        raw_response = (
            self.request_executor(payload)
            if self.request_executor is not None
            else self._execute_http_request(payload)
        )
        return self._parse_response(raw_response)

    def _execute_http_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute one Chat Completions request using the standard library."""

        if not self.api_key:
            raise RuntimeError("OpenAI API key is not configured.")

        request_payload = {
            key: value
            for key, value in payload.items()
            if key != "provider" and value is not None and value != [] and value != {}
        }
        encoded_payload = json.dumps(request_payload, ensure_ascii=True).encode("utf-8")
        request = urllib_request.Request(
            f"{self.base_url}/chat/completions",
            data=encoded_payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI chat request failed with HTTP {error.code}: {error_body}") from error

    @staticmethod
    def _parse_response(raw_response: dict[str, Any]) -> LlmCompletionResponse:
        """Normalize one executor response into the shared completion structure."""

        choices = raw_response.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                provider_message = first_choice.get("message")
                if isinstance(provider_message, dict):
                    raw_response = {
                        "message": provider_message.get("content") or "",
                        "toolCalls": [
                            {
                                "name": ((tool_call.get("function") or {}).get("name") if isinstance(tool_call, dict) else ""),
                                "arguments": OpenAIChatModelAdapter._parse_tool_arguments(
                                    (tool_call.get("function") or {}).get("arguments")
                                    if isinstance(tool_call, dict)
                                    else None
                                ),
                            }
                            for tool_call in list(provider_message.get("tool_calls") or [])
                        ],
                        "raw": raw_response,
                    }

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

    @staticmethod
    def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
        """Parse provider tool arguments without raising on malformed JSON."""

        if isinstance(raw_arguments, dict):
            return dict(raw_arguments)
        if not isinstance(raw_arguments, str) or not raw_arguments.strip():
            return {}
        try:
            parsed_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}
        return dict(parsed_arguments) if isinstance(parsed_arguments, dict) else {}
