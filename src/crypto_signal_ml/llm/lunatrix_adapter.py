"""First-party local chat adapter for Lunatrix final responses."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .base import ChatModelAdapter, LlmCompletionResponse, LlmMessage, LlmToolSpec


class LunatrixChatModelAdapter(ChatModelAdapter):
    """
    Local response model for the assistant's final wording.

    This adapter does not call an external LLM. It reads the grounded context
    prepared by the chat response layer and rewrites it into a more interactive
    answer while preserving the deterministic router/tool facts.
    """

    provider_name = "lunatrix"

    def __init__(self, *, model: str = "lunatrix-grounded-chat-v1") -> None:
        self.model = str(model).strip() or "lunatrix-grounded-chat-v1"

    def is_configured(self) -> bool:
        """The local response model is always available."""

        return True

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> LlmCompletionResponse:
        """Generate a local final response from grounded context."""

        del tools, system_prompt
        context = self._extract_context(messages)
        return LlmCompletionResponse(
            message=self._compose_from_context(context),
            raw_response={"provider": self.provider_name, "model": self.model},
        )

    def _compose_from_context(self, context: dict[str, Any]) -> str:
        """Compose one final answer from the context payload."""

        question = str(context.get("userQuestion") or "").strip()
        route_plan = dict(context.get("routePlan") or {})
        intents = {str(intent) for intent in list(route_plan.get("intents") or [])}
        response_style = str(route_plan.get("responseStyle") or "")
        draft = self._clean_text(context.get("deterministicDraft"))

        if "conversation" in intents or response_style == "conversation":
            return self._compose_conversation(question)

        if not draft:
            return "I could not build a grounded answer from the current model context."

        if "general_knowledge" in intents:
            return self._with_short_follow_up(
                draft,
                "I can also connect that back to the current market if you ask about a specific coin.",
            )

        if response_style == "advice":
            return self._with_short_follow_up(
                draft,
                "For entries, I will only treat an explicit BUY signal as a fresh long setup.",
            )

        if response_style == "compare":
            return self._with_short_follow_up(
                draft,
                "Ask for an entry plan if you want the strongest one translated into risk and sizing.",
            )

        if "market_overview" in intents:
            if "no active buy" in draft.lower() or "does not show a fresh buy" in draft.lower():
                return self._with_short_follow_up(
                    draft,
                    "That means watch-only until a BUY clears the gate.",
                )
            return self._with_short_follow_up(
                draft,
                "I can narrow that to one coin if you want a pair-level read.",
            )

        return draft

    @staticmethod
    def _compose_conversation(question: str) -> str:
        """Return local small-talk/help replies without touching market tools."""

        normalized = re.sub(r"[^a-z0-9\s]", " ", question.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if any(keyword in normalized for keyword in ("thank", "thanks", "appreciate")):
            return "You are welcome. I am here for coin checks, market reads, model status, and knowledge-base questions."
        if any(keyword in normalized for keyword in ("bye", "goodbye", "see you")):
            return "Got it. I will be here when you want another market or model read."
        if any(keyword in normalized for keyword in ("help", "what can you do", "commands", "how do i use")):
            return (
                "I can check a coin signal, compare coins, summarize the live market, explain model freshness, "
                "build a risk-aware trade plan, or search the RAG knowledge base."
            )
        return "Hi. I am here. Ask me about a coin, the live market, model freshness, or your knowledge base."

    @classmethod
    def _extract_context(cls, messages: Sequence[LlmMessage]) -> dict[str, Any]:
        """Parse the response-layer JSON context from the last user message."""

        for message in reversed(list(messages)):
            content = str(message.content or "")
            json_start = content.find("{")
            if json_start < 0:
                continue
            try:
                parsed = json.loads(content[json_start:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    @classmethod
    def _with_short_follow_up(cls, body: str, follow_up: str) -> str:
        """Append one short interactive next-step line without over-talking."""

        clean_body = cls._clean_text(body)
        clean_follow_up = cls._clean_text(follow_up)
        if not clean_body:
            return clean_follow_up
        if not clean_follow_up or clean_follow_up.lower() in clean_body.lower():
            return clean_body
        return f"{clean_body}\n\n{clean_follow_up}"

    @staticmethod
    def _clean_text(value: Any) -> str:
        """Normalize whitespace while preserving paragraph breaks."""

        raw_text = str(value or "").strip()
        if not raw_text:
            return ""
        paragraphs = [
            " ".join(paragraph.split())
            for paragraph in re.split(r"\n\s*\n", raw_text)
            if paragraph.strip()
        ]
        return "\n\n".join(paragraphs)
