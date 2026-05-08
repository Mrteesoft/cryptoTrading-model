"""Final response layer for model-side chat answers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Protocol, Sequence

from ..config import TrainingConfig
from ..llm import ChatModelAdapter, LlmMessage, LunatrixChatModelAdapter


class DeterministicComposerProtocol(Protocol):
    """Minimal composer surface used as the safe fallback response path."""

    def compose(
        self,
        *,
        question: str,
        tool_results: Sequence[dict[str, Any]],
        recalled_messages: Sequence[dict[str, Any]],
        route_plan: Any,
    ) -> str:
        """Return a grounded deterministic reply."""


@dataclass(frozen=True)
class ChatResponseLayerResult:
    """One final response plus metadata about how it was produced."""

    text: str
    mode: str
    provider: str
    model: str | None = None
    fallback_used: bool = False
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-safe metadata payload for assistant messages."""

        payload: dict[str, Any] = {
            "mode": self.mode,
            "provider": self.provider,
            "fallbackUsed": bool(self.fallback_used),
        }
        if self.model:
            payload["model"] = self.model
        if self.error:
            payload["error"] = self.error
        if self.metadata:
            payload.update(self.metadata)
        return payload


class AssistantChatResponseLayer:
    """
    Convert routed tool outputs into the final chat answer.

    The deterministic composer remains the fallback. When configured, an LLM is
    only used after routing and tool execution, so it writes from facts that the
    model-service already fetched.
    """

    def __init__(
        self,
        *,
        config: TrainingConfig | None = None,
        deterministic_composer: DeterministicComposerProtocol,
        llm_adapter: ChatModelAdapter | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.deterministic_composer = deterministic_composer
        self.llm_adapter = llm_adapter or self._build_llm_adapter()

    def compose(
        self,
        *,
        question: str,
        tool_results: Sequence[dict[str, Any]],
        recalled_messages: Sequence[dict[str, Any]],
        route_plan: Any,
    ) -> ChatResponseLayerResult:
        """Return the final assistant reply using the configured response path."""

        deterministic_reply = self.deterministic_composer.compose(
            question=question,
            tool_results=tool_results,
            recalled_messages=recalled_messages,
            route_plan=route_plan,
        )

        if not bool(getattr(self.config, "assistant_use_llm", False)):
            return ChatResponseLayerResult(
                text=deterministic_reply,
                mode="deterministic",
                provider="deterministic",
            )

        llm_adapter = self.llm_adapter
        if llm_adapter is None or not llm_adapter.is_configured():
            return ChatResponseLayerResult(
                text=deterministic_reply,
                mode="deterministic",
                provider=self._configured_provider_name(),
                model=self._configured_model_name() or None,
                fallback_used=True,
                error="LLM response layer is enabled but not configured.",
            )

        try:
            completion = llm_adapter.complete(
                messages=self._build_messages(
                    question=question,
                    tool_results=tool_results,
                    recalled_messages=recalled_messages,
                    route_plan=route_plan,
                    deterministic_reply=deterministic_reply,
                ),
                tools=(),
                system_prompt=self._build_system_prompt(),
            )
        except Exception as error:  # pragma: no cover - provider failures are environment dependent.
            return ChatResponseLayerResult(
                text=deterministic_reply,
                mode="deterministic",
                provider=getattr(llm_adapter, "provider_name", self._configured_provider_name()),
                model=self._configured_model_name() or None,
                fallback_used=True,
                error=f"LLM response generation failed: {error}",
            )

        response_text = str(completion.message or "").strip()
        if not response_text:
            return ChatResponseLayerResult(
                text=deterministic_reply,
                mode="deterministic",
                provider=getattr(llm_adapter, "provider_name", self._configured_provider_name()),
                model=self._configured_model_name() or None,
                fallback_used=True,
                error="LLM response generation returned an empty message.",
            )

        return ChatResponseLayerResult(
            text=response_text,
            mode="local_chat" if getattr(llm_adapter, "provider_name", "") == "lunatrix" else "llm",
            provider=getattr(llm_adapter, "provider_name", self._configured_provider_name()),
            model=self._configured_model_name() or None,
            metadata={"toolCallCount": len(tool_results)},
        )

    def _build_llm_adapter(self) -> ChatModelAdapter | None:
        """Build the configured provider adapter when available."""

        provider_name = self._configured_provider_name()
        if provider_name in {"lunatrix", "local", "internal"}:
            return LunatrixChatModelAdapter(model=self._configured_model_name())
        return None

    def _configured_provider_name(self) -> str:
        """Return the normalized provider configured for final responses."""

        return str(getattr(self.config, "llm_provider", "lunatrix") or "lunatrix").strip().lower()

    def _configured_model_name(self) -> str:
        """Return the configured model id, if any."""

        configured_name = str(getattr(self.config, "assistant_response_model", "") or "").strip()
        return configured_name or "lunatrix-grounded-chat-v1"

    def _build_system_prompt(self) -> str:
        """Return instructions for final-answer generation."""

        return (
            "You are the Lunatrix AI chat response layer. The model-service router and tools have already run; "
            "do not request or invent additional tool data. Answer conversationally and directly from the provided "
            "route plan, tool results, memory, and deterministic draft. If the user is greeting or chatting casually, "
            "reply naturally without mentioning market data. For trading answers, never call LOSS, TAKE_PROFIT, or HOLD "
            "a buy entry. Only call something a BUY when a provided tool result explicitly says BUY. If no BUY is active, "
            "say no active BUY and mention a watch-only spotlight only as observation. Keep answers concise, practical, "
            "and clear that this is not financial advice."
        )

    def _build_messages(
        self,
        *,
        question: str,
        tool_results: Sequence[dict[str, Any]],
        recalled_messages: Sequence[dict[str, Any]],
        route_plan: Any,
        deterministic_reply: str,
    ) -> list[LlmMessage]:
        """Build the compact final-response prompt."""

        route_payload = route_plan.to_dict() if hasattr(route_plan, "to_dict") else dict(route_plan or {})
        context_payload = {
            "userQuestion": str(question),
            "routePlan": route_payload,
            "toolResults": [self._compact_tool_result(tool_result) for tool_result in tool_results],
            "sessionMemory": [self._compact_message(message) for message in recalled_messages[-4:]],
            "deterministicDraft": self._trim_text(deterministic_reply, 1800),
            "finalAnswerRequirements": [
                "Answer the user, not the tool log.",
                "Do not mention tools unless an error or cached-data caveat matters.",
                "Do not invent prices, signals, confidence, or sources.",
                "For greetings, answer as a normal chat assistant.",
            ],
        }
        return [
            LlmMessage(
                role="user",
                content="Write the final assistant response from this grounded context:\n"
                + self._json_dumps(context_payload, max_length=9000),
            )
        ]

    def _compact_tool_result(self, tool_result: dict[str, Any]) -> dict[str, Any]:
        """Keep only the fields needed for final answer generation."""

        tool_name = str(tool_result.get("name") or "")
        result = dict(tool_result.get("result") or {})
        compact: dict[str, Any] = {
            "name": tool_name,
            "arguments": self._compact_value(tool_result.get("arguments") or {}),
            "status": result.get("status"),
            "source": result.get("source"),
            "warning": result.get("warning"),
            "error": result.get("error"),
        }

        if tool_name == "get_signal":
            compact["productId"] = result.get("productId")
            compact["signal"] = self._compact_signal(result.get("signal"))
        elif tool_name == "get_market_overview":
            compact["overview"] = self._compact_overview(result.get("overview"))
        elif tool_name == "get_trader_plan":
            compact["capital"] = result.get("capital")
            compact["liveSnapshot"] = {
                "primarySignal": self._compact_signal(
                    (dict(result.get("liveSnapshot") or {})).get("primarySignal")
                )
            }
            compact["traderPlan"] = self._compact_trader_plan(result.get("traderPlan"))
        elif tool_name == "get_model_status":
            compact["model"] = self._compact_model_status(result.get("model"))
        elif tool_name == "search_knowledge":
            compact["query"] = result.get("query")
            compact["results"] = [
                {
                    "title": item.get("title"),
                    "sourceUri": item.get("sourceUri"),
                    "snippet": self._trim_text(item.get("snippet"), 500),
                }
                for item in list(result.get("results") or [])[:3]
                if isinstance(item, dict)
            ]

        return self._drop_empty(compact)

    def _compact_overview(self, overview: Any) -> dict[str, Any]:
        """Compact a market overview payload."""

        if not isinstance(overview, dict):
            return {}
        market_summary = dict(overview.get("marketSummary") or {})
        return self._drop_empty(
            {
                "marketSummary": {
                    "actionableSignals": market_summary.get("actionableSignals"),
                    "totalSignals": market_summary.get("totalSignals"),
                    "signalCounts": market_summary.get("signalCounts"),
                    "marketStance": market_summary.get("marketStance"),
                },
                "primarySignal": self._compact_signal(overview.get("primarySignal")),
                "topBuys": [self._compact_signal(item) for item in list(overview.get("topBuys") or [])[:5]],
                "topSignals": [self._compact_signal(item) for item in list(overview.get("topSignals") or [])[:5]],
                "coinOfTheDay": self._compact_signal(overview.get("coinOfTheDay")),
                "spotlightCandidates": [
                    self._compact_signal(item)
                    for item in list(overview.get("spotlightCandidates") or [])[:3]
                ],
            }
        )

    def _compact_signal(self, signal: Any) -> dict[str, Any]:
        """Compact one signal-like payload."""

        if not isinstance(signal, dict):
            return {}
        fields = (
            "productId",
            "pairSymbol",
            "symbol",
            "signal_name",
            "signalName",
            "confidence",
            "close",
            "finalDecisionScore",
            "coinOfDayScore",
            "spotlightScore",
            "tradeReadiness",
            "spotAction",
            "marketStance",
            "signalChat",
            "spotlightReason",
            "reasonSummary",
        )
        compact = {field: self._compact_value(signal.get(field)) for field in fields if field in signal}
        brain = signal.get("brain")
        if isinstance(brain, dict):
            compact["brain"] = {
                "decision": brain.get("decision"),
                "summaryLine": self._trim_text(brain.get("summaryLine"), 400),
                "reasonSummary": self._trim_text(brain.get("reasonSummary"), 400),
            }
        return self._drop_empty(compact)

    def _compact_trader_plan(self, trader_plan: Any) -> dict[str, Any]:
        """Compact trader-plan output."""

        if not isinstance(trader_plan, dict):
            return {}
        plan = dict(trader_plan.get("plan") or {})
        return self._drop_empty(
            {
                "marketStance": trader_plan.get("marketStance"),
                "summary": self._trim_text(trader_plan.get("summary"), 700),
                "plan": {
                    "newEntryCount": plan.get("newEntryCount"),
                    "addOnCount": plan.get("addOnCount"),
                    "reduceCount": plan.get("reduceCount"),
                    "exitCount": plan.get("exitCount"),
                    "watchlistCount": plan.get("watchlistCount"),
                    "entries": plan.get("entries"),
                    "reductions": plan.get("reductions"),
                    "exits": plan.get("exits"),
                },
            }
        )

    def _compact_model_status(self, model_status: Any) -> dict[str, Any]:
        """Compact model-status output."""

        if not isinstance(model_status, dict):
            return {}
        return self._drop_empty(
            {
                "status": model_status.get("status"),
                "modelType": model_status.get("modelType"),
                "lifecycle": model_status.get("lifecycle"),
                "trainingMetrics": model_status.get("trainingMetrics"),
            }
        )

    def _compact_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Compact one recalled session-memory message."""

        return self._drop_empty(
            {
                "role": message.get("role"),
                "content": self._trim_text(message.get("content"), 500),
                "createdAt": message.get("createdAt"),
            }
        )

    def _compact_value(self, value: Any) -> Any:
        """Trim nested strings/lists/dicts for prompt safety."""

        if isinstance(value, str):
            return self._trim_text(value, 700)
        if isinstance(value, dict):
            return self._drop_empty({str(key): self._compact_value(raw_value) for key, raw_value in value.items()})
        if isinstance(value, list):
            return [self._compact_value(item) for item in value[:8]]
        return value

    @staticmethod
    def _drop_empty(payload: dict[str, Any]) -> dict[str, Any]:
        """Remove empty values while preserving explicit zero/false fields."""

        return {
            key: value
            for key, value in payload.items()
            if value is not None and value != "" and value != [] and value != {}
        }

    @staticmethod
    def _trim_text(value: Any, max_length: int = 1000) -> str:
        """Normalize and trim text for prompt payloads."""

        text = " ".join(str(value or "").split())
        if len(text) <= max_length:
            return text
        return text[: max_length - 3].rstrip() + "..."

    @classmethod
    def _json_dumps(cls, payload: Any, *, max_length: int) -> str:
        """Serialize JSON and cap the final prompt context."""

        serialized = json.dumps(payload, ensure_ascii=True, default=str, indent=2)
        if len(serialized) <= max_length:
            return serialized
        return serialized[: max_length - 3].rstrip() + "..."
