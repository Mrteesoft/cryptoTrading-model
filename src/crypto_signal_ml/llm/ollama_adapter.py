"""Ollama adapter for self-hosted local chat models."""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from .base import ChatModelAdapter, LlmCompletionResponse, LlmMessage, LlmToolSpec


class OllamaChatModelAdapter(ChatModelAdapter):
    """Call a locally hosted Ollama chat model through the HTTP API."""

    provider_name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        request_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.model = str(model or "").strip()
        self.base_url = str(base_url or "http://127.0.0.1:11434").strip().rstrip("/")
        self.request_executor = request_executor

    def is_configured(self) -> bool:
        """Return whether the adapter has a model id and endpoint."""

        return bool(self.model and self.base_url)

    def build_request_payload(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Normalize one Ollama chat request payload."""

        del tools
        normalized_messages = []
        if system_prompt:
            normalized_messages.append({"role": "system", "content": system_prompt})

        normalized_messages.extend(
            {"role": message.role, "content": message.content}
            for message in messages
        )

        return {
            "model": self.model,
            "messages": normalized_messages,
            "stream": False,
            "options": {
                "temperature": 0.35,
                "top_p": 0.9,
            },
        }

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> LlmCompletionResponse:
        """Call Ollama and normalize the response."""

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
        """Execute one Ollama /api/chat request using the standard library."""

        encoded_payload = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        request = urllib_request.Request(
            f"{self.base_url}/api/chat",
            data=encoded_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama chat request failed with HTTP {error.code}: {error_body}") from error

    @staticmethod
    def _parse_response(raw_response: dict[str, Any]) -> LlmCompletionResponse:
        """Normalize one Ollama response into the shared completion structure."""

        message_payload = raw_response.get("message")
        message = ""
        if isinstance(message_payload, dict):
            message = str(message_payload.get("content") or "")
        if not message:
            message = str(raw_response.get("response") or "")

        return LlmCompletionResponse(
            message=message,
            tool_calls=[],
            raw_response=raw_response,
        )
