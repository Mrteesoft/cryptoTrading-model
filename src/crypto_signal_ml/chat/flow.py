"""Tool-driven chat flow for grounded assistant responses."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol, Sequence

from ..config import TrainingConfig
from ..tools import ToolRegistry


COMMON_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "should",
    "the",
    "this",
    "to",
    "what",
    "with",
}


class SessionStoreProtocol(Protocol):
    """Minimal session-store surface needed by the tool-driven chat flow."""

    def list_messages(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent messages for one session."""


@dataclass(frozen=True)
class PlannedToolCall:
    """One intended tool call chosen by the chat router."""

    name: str
    arguments: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return the planned call in JSON-friendly form."""

        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "reason": self.reason,
        }


class AssistantIntentRouter:
    """Pick the smallest useful set of tools for one user request."""

    def __init__(
        self,
        *,
        known_product_ids: Sequence[str],
        retrieval_enabled: bool = True,
    ) -> None:
        self.known_product_ids = tuple(str(product_id).upper() for product_id in known_product_ids)
        self.retrieval_enabled = bool(retrieval_enabled)

    def resolve_product_id(
        self,
        question: str,
        explicit_product_id: str | None = None,
    ) -> str | None:
        """Resolve one product id from the explicit request or the question text."""

        if explicit_product_id:
            normalized_product_id = str(explicit_product_id).strip().upper()
            return normalized_product_id or None

        explicit_match = re.search(r"\b([A-Z]{2,10}-[A-Z]{2,10})\b", str(question).upper())
        if explicit_match is not None:
            return explicit_match.group(1).upper()

        upper_question = str(question).upper()
        for candidate in self.known_product_ids:
            base_currency = str(candidate).split("-")[0].upper()
            if re.search(rf"\b{re.escape(base_currency)}\b", upper_question):
                return str(candidate).upper()

        return None

    def resolve_capital_override(self, question: str) -> float | None:
        """Parse a simple capital override from the question when present."""

        capital_match = re.search(
            r"\b(?:capital|with)\s+\$?([0-9][0-9,]*(?:\.[0-9]+)?)\b",
            str(question).lower(),
        )
        if capital_match is None:
            return None

        try:
            return float(capital_match.group(1).replace(",", ""))
        except ValueError:
            return None

    def route(
        self,
        question: str,
        *,
        explicit_product_id: str | None = None,
        force_refresh: bool = False,
    ) -> tuple[str | None, list[PlannedToolCall]]:
        """Resolve one product id and the tool calls needed for the request."""

        normalized_question = str(question).strip()
        resolved_product_id = self.resolve_product_id(
            question=normalized_question,
            explicit_product_id=explicit_product_id,
        )
        capital_override = self.resolve_capital_override(normalized_question)
        lowered_question = normalized_question.lower()

        planned_calls: list[PlannedToolCall] = []
        if self._is_model_request(lowered_question):
            planned_calls.append(
                PlannedToolCall(
                    name="get_model_status",
                    arguments={},
                    reason="The question asks about model readiness, freshness, or training quality.",
                )
            )

        if self._is_trader_request(lowered_question):
            trader_arguments: dict[str, Any] = {
                "force_refresh": bool(force_refresh),
            }
            if capital_override is not None:
                trader_arguments["capital"] = capital_override

            planned_calls.append(
                PlannedToolCall(
                    name="get_trader_plan",
                    arguments=trader_arguments,
                    reason="The question asks for a portfolio-aware trading plan or risk framing.",
                )
            )
        elif resolved_product_id is not None:
            planned_calls.append(
                PlannedToolCall(
                    name="get_signal",
                    arguments={
                        "product_id": resolved_product_id,
                        "force_refresh": bool(force_refresh),
                    },
                    reason="A specific product id was requested, so the assistant should fetch that exact signal.",
                )
            )
        else:
            planned_calls.append(
                PlannedToolCall(
                    name="get_market_overview",
                    arguments={"force_refresh": bool(force_refresh)},
                    reason="No specific product id was resolved, so the assistant should start from the market overview.",
                )
            )

        if self.retrieval_enabled and self._should_use_knowledge(lowered_question):
            planned_calls.append(
                PlannedToolCall(
                    name="search_knowledge",
                    arguments={
                        "query": normalized_question,
                        "limit": 3,
                    },
                    reason="The question asks for supporting context or external knowledge.",
                )
            )

        return resolved_product_id, self._dedupe_calls(planned_calls)

    @staticmethod
    def _is_model_request(lowered_question: str) -> bool:
        return any(
            keyword in lowered_question
            for keyword in ("model", "accuracy", "artifact", "train", "retrain", "feature importance")
        )

    @staticmethod
    def _is_trader_request(lowered_question: str) -> bool:
        return any(
            keyword in lowered_question
            for keyword in (
                "portfolio",
                "position",
                "risk",
                "plan",
                "allocation",
                "sizing",
                "capital",
                "exit",
                "entry plan",
            )
        )

    @staticmethod
    def _should_use_knowledge(lowered_question: str) -> bool:
        return any(
            keyword in lowered_question
            for keyword in ("why", "research", "source", "context", "memo", "document", "knowledge", "explain")
        )

    @staticmethod
    def _dedupe_calls(planned_calls: Sequence[PlannedToolCall]) -> list[PlannedToolCall]:
        """Keep tool order stable while removing duplicate tool names."""

        deduped_calls = []
        seen_tool_names = set()
        for planned_call in planned_calls:
            if planned_call.name in seen_tool_names:
                continue
            seen_tool_names.add(planned_call.name)
            deduped_calls.append(planned_call)

        return deduped_calls


class AssistantToolExecutor:
    """Execute the tool plan chosen by the router."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry

    def execute(self, planned_calls: Sequence[PlannedToolCall]) -> list[dict[str, Any]]:
        """Execute each planned tool call in order."""

        executed_calls = []
        for planned_call in planned_calls:
            result = self.tool_registry.execute(
                tool_name=planned_call.name,
                arguments=planned_call.arguments,
            )
            executed_calls.append(
                {
                    "name": planned_call.name,
                    "arguments": dict(planned_call.arguments),
                    "reason": planned_call.reason,
                    "result": result,
                }
            )

        return executed_calls


class AssistantResponseComposer:
    """Compose a grounded natural-language reply from structured tool outputs."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()

    def compose(
        self,
        *,
        question: str,
        tool_results: Sequence[dict[str, Any]],
        recalled_messages: Sequence[dict[str, Any]],
    ) -> str:
        """Turn tool results into one concise answer."""

        del question
        results_by_name = {
            str(tool_result.get("name")): dict(tool_result.get("result") or {})
            for tool_result in tool_results
        }
        paragraphs = []

        signal_result = results_by_name.get("get_signal", {})
        if signal_result:
            paragraphs.extend(self._compose_signal_paragraphs(signal_result))

        overview_result = results_by_name.get("get_market_overview", {})
        if overview_result and not signal_result:
            paragraphs.extend(self._compose_overview_paragraphs(overview_result))

        trader_result = results_by_name.get("get_trader_plan", {})
        if trader_result:
            paragraphs.extend(self._compose_trader_paragraphs(trader_result))

        model_result = results_by_name.get("get_model_status", {})
        if model_result:
            paragraphs.extend(self._compose_model_paragraphs(model_result))

        retrieval_result = results_by_name.get("search_knowledge", {})
        if retrieval_result:
            paragraphs.extend(self._compose_retrieval_paragraphs(retrieval_result))

        if recalled_messages:
            paragraphs.append(
                "Relevant session memory: "
                + str(recalled_messages[-1].get("content") or "").strip()
            )

        if not paragraphs:
            paragraphs.append(
                "I could not produce a grounded answer because none of the trading tools returned usable data."
            )

        return "\n\n".join(paragraph for paragraph in paragraphs if paragraph.strip())

    def _compose_signal_paragraphs(self, signal_result: dict[str, Any]) -> list[str]:
        """Explain one single-signal tool result."""

        status = str(signal_result.get("status") or "")
        product_id = str(signal_result.get("productId") or "")
        source = str(signal_result.get("source") or "")
        warning = str(signal_result.get("warning") or "").strip()
        signal_summary = signal_result.get("signal")
        if status == "error":
            return [f"I could not load an authoritative signal for {product_id or 'that product'}: {signal_result.get('error')}"]

        if status == "not_found" or not isinstance(signal_summary, dict):
            paragraphs = [f"No authoritative signal is currently available for {product_id}."]
            if warning:
                paragraphs.append(f"The tool fell back to cached data because the live refresh failed: {warning}")
            return paragraphs

        signal_name = str(signal_summary.get("signal_name") or "UNKNOWN")
        confidence = self._format_percent(signal_summary.get("confidence"))
        close_price = self._format_price(signal_summary.get("close"))
        explanation = str(signal_summary.get("signalChat") or "").strip()

        paragraphs = [
            f"{product_id} is currently a {signal_name} setup at {close_price} with {confidence} confidence from the {source} engine path."
        ]
        if explanation:
            paragraphs.append(explanation)
        if warning:
            paragraphs.append(f"Live refresh was unavailable, so this answer used cached signal data: {warning}")
        return paragraphs

    def _compose_overview_paragraphs(self, overview_result: dict[str, Any]) -> list[str]:
        """Explain one market-overview tool result."""

        if str(overview_result.get("status")) == "error":
            return [f"I could not load the market overview: {overview_result.get('error')}"]

        overview = dict(overview_result.get("overview") or {})
        market_summary = dict(overview.get("marketSummary") or {})
        primary_signal = dict(overview.get("primarySignal") or {})
        actionable_count = int(market_summary.get("actionableSignals") or 0)
        total_signals = int(market_summary.get("totalSignals") or 0)
        lead_product = str(primary_signal.get("productId") or "the lead pair")
        lead_signal = str(primary_signal.get("signal_name") or "HOLD")
        lead_confidence = self._format_percent(primary_signal.get("confidence"))

        paragraphs = [
            f"Market overview: {actionable_count} actionable setups across {total_signals} tracked pairs. "
            f"The lead signal is {lead_product} with a {lead_signal} call at {lead_confidence} confidence."
        ]
        top_signals = list(overview.get("topSignals", []))
        if top_signals:
            top_lines = [
                f"{signal_summary.get('productId')} {signal_summary.get('signal_name')} "
                f"({self._format_percent(signal_summary.get('confidence'))})"
                for signal_summary in top_signals[:3]
            ]
            paragraphs.append("Top setups: " + ", ".join(top_lines) + ".")

        warning = str(overview_result.get("warning") or "").strip()
        if warning:
            paragraphs.append(f"Live refresh was unavailable, so the overview used cached data: {warning}")
        return paragraphs

    def _compose_trader_paragraphs(self, trader_result: dict[str, Any]) -> list[str]:
        """Explain one trader-plan tool result."""

        if str(trader_result.get("status")) == "error":
            return [f"I could not build the trader plan: {trader_result.get('error')}"]

        trader_plan = dict(trader_result.get("traderPlan") or {})
        plan_summary = dict(trader_plan.get("plan") or {})
        market_stance = str(trader_plan.get("marketStance") or "neutral")
        entry_count = int(plan_summary.get("entryCount") or 0)
        reduce_count = int(plan_summary.get("reduceCount") or 0)
        exit_count = int(plan_summary.get("exitCount") or 0)

        paragraphs = [
            f"Trader plan: the current market stance is {market_stance}, with {entry_count} entries, "
            f"{reduce_count} reductions, and {exit_count} exits suggested."
        ]
        warning = str(trader_result.get("warning") or "").strip()
        if warning:
            paragraphs.append(f"The plan used cached market data because the live refresh failed: {warning}")
        return paragraphs

    def _compose_model_paragraphs(self, model_result: dict[str, Any]) -> list[str]:
        """Explain one model-status tool result."""

        model_summary = dict(model_result.get("model") or {})
        if not model_summary:
            return []

        model_status = str(model_summary.get("status") or "")
        if model_status == "missing":
            return [str(model_summary.get("message") or "No trained model artifact is available yet.")]
        if model_status == "error":
            return [str(model_summary.get("message") or "The saved model artifact could not be loaded.")]

        lifecycle = dict(model_summary.get("lifecycle") or {})
        training_metrics = dict(model_summary.get("trainingMetrics") or {})
        balanced_accuracy = self._format_percent(training_metrics.get("balancedAccuracy"))
        freshness = str(lifecycle.get("freshness") or "unknown")
        retraining_due = bool(lifecycle.get("retrainingDue"))

        recommended_action = str(lifecycle.get("recommendedAction") or "").strip()
        summary_line = (
            f"Model status: {model_summary.get('modelType')} is {freshness}."
            f" Last recorded balanced accuracy was {balanced_accuracy}."
        )
        if retraining_due:
            summary_line += " Retraining is currently due."

        paragraphs = [summary_line]
        if recommended_action:
            paragraphs.append(recommended_action)
        return paragraphs

    def _compose_retrieval_paragraphs(self, retrieval_result: dict[str, Any]) -> list[str]:
        """Explain one knowledge-search tool result."""

        status = str(retrieval_result.get("status") or "")
        if status == "disabled":
            return ["External knowledge retrieval is disabled in the current configuration."]
        if status == "error":
            return [f"I could not search the knowledge store: {retrieval_result.get('error')}"]

        results = list(retrieval_result.get("results", []))
        if not results:
            return []

        knowledge_lines = []
        for result in results[:2]:
            source_title = str(result.get("title") or "External source")
            source_uri = str(result.get("sourceUri") or "").strip()
            snippet = self._trim_text(str(result.get("snippet") or ""), max_length=220)
            source_label = source_title if not source_uri else f"{source_title} ({source_uri})"
            if snippet:
                knowledge_lines.append(f"{source_label}: {snippet}")
            else:
                knowledge_lines.append(source_label)

        if not knowledge_lines:
            return []

        return ["External knowledge context: " + " ".join(knowledge_lines)]

    @staticmethod
    def _format_percent(value: Any) -> str:
        """Format a decimal probability into a readable percentage."""

        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return "-"

    @staticmethod
    def _format_price(value: Any) -> str:
        """Format numeric prices consistently."""

        try:
            return f"${float(value):,.2f}"
        except (TypeError, ValueError):
            return "-"

    @staticmethod
    def _trim_text(text: str, max_length: int = 220) -> str:
        """Trim long retrieval snippets into concise assistant-ready text."""

        normalized_text = str(text).strip()
        if len(normalized_text) <= max_length:
            return normalized_text

        return normalized_text[: max_length - 3].rstrip() + "..."


class ToolDrivenChatFlow:
    """Run one grounded chat turn through tool selection, execution, and composition."""

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        session_store: SessionStoreProtocol,
        config: TrainingConfig | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.tool_registry = tool_registry
        self.session_store = session_store
        self.router = AssistantIntentRouter(
            known_product_ids=self.config.live_product_ids,
            retrieval_enabled=self.config.assistant_enable_retrieval,
        )
        self.executor = AssistantToolExecutor(tool_registry)
        self.composer = AssistantResponseComposer(config=self.config)

    def run(
        self,
        *,
        session_id: str,
        question: str,
        explicit_product_id: str | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Run one assistant turn through routing, tools, and response composition."""

        recalled_messages = self._select_relevant_messages(session_id=session_id, question=question)
        resolved_product_id, planned_calls = self.router.route(
            question=question,
            explicit_product_id=explicit_product_id,
            force_refresh=force_refresh,
        )
        executed_calls = self.executor.execute(planned_calls)
        reply_text = self.composer.compose(
            question=question,
            tool_results=executed_calls,
            recalled_messages=recalled_messages,
        )

        return {
            "resolvedProductId": resolved_product_id,
            "toolCalls": [planned_call.to_dict() for planned_call in planned_calls],
            "toolResults": executed_calls,
            "replyText": reply_text,
            "retrieval": {
                "signals": self._collect_signal_context(executed_calls),
                "messages": recalled_messages,
                "knowledge": self._collect_knowledge_context(executed_calls),
            },
            "liveContext": self._build_live_context(executed_calls),
        }

    def _select_relevant_messages(
        self,
        *,
        session_id: str,
        question: str,
    ) -> list[dict[str, Any]]:
        """Return the most relevant recent messages for the current question."""

        question_tokens = self._tokenize(question)
        if not question_tokens:
            return []

        message_rows = self.session_store.list_messages(
            session_id=session_id,
            limit=self.config.assistant_memory_message_limit,
        )
        scored_messages = []
        for message_row in message_rows[:-1]:
            message_tokens = self._tokenize(str(message_row.get("content") or ""))
            score = len(question_tokens & message_tokens)
            if score > 0:
                scored_messages.append((score, message_row))

        return [
            {
                "role": message_row["role"],
                "content": message_row["content"],
                "createdAt": message_row["createdAt"],
            }
            for _, message_row in sorted(scored_messages, key=lambda item: item[0], reverse=True)[
                : self.config.assistant_retrieval_item_limit
            ]
        ]

    @staticmethod
    def _collect_signal_context(tool_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collect signal-context rows from tool execution results."""

        signal_rows = []
        for tool_result in tool_results:
            tool_name = str(tool_result.get("name") or "")
            result_payload = dict(tool_result.get("result") or {})
            if tool_name == "get_signal" and isinstance(result_payload.get("signal"), dict):
                signal_rows.append(dict(result_payload["signal"]))
            elif tool_name == "get_market_overview":
                overview = dict(result_payload.get("overview") or {})
                signal_rows.extend(list(overview.get("topSignals", [])))
            elif tool_name == "get_trader_plan":
                live_snapshot = dict(result_payload.get("liveSnapshot") or {})
                primary_signal = live_snapshot.get("primarySignal")
                if isinstance(primary_signal, dict):
                    signal_rows.append(dict(primary_signal))

        return signal_rows[:4]

    @staticmethod
    def _collect_knowledge_context(tool_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collect knowledge results from tool execution results."""

        for tool_result in tool_results:
            if str(tool_result.get("name") or "") != "search_knowledge":
                continue
            result_payload = dict(tool_result.get("result") or {})
            return list(result_payload.get("results", []))

        return []

    @staticmethod
    def _build_live_context(tool_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Build the legacy liveContext shape from tool execution results."""

        for preferred_tool_name in ("get_signal", "get_market_overview", "get_trader_plan"):
            for tool_result in tool_results:
                if str(tool_result.get("name") or "") != preferred_tool_name:
                    continue
                result_payload = dict(tool_result.get("result") or {})
                if preferred_tool_name == "get_signal":
                    snapshot_payload = {
                        "productId": result_payload.get("productId"),
                        "signal": result_payload.get("signal"),
                        "overview": result_payload.get("overview"),
                    }
                elif preferred_tool_name == "get_market_overview":
                    snapshot_payload = result_payload.get("overview")
                else:
                    snapshot_payload = result_payload.get("liveSnapshot")

                return {
                    "source": result_payload.get("source", "unavailable"),
                    "error": result_payload.get("error") or result_payload.get("warning", ""),
                    "snapshot": snapshot_payload,
                }

        return {
            "source": "unavailable",
            "error": "",
            "snapshot": None,
        }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Turn free text into a normalized token set for lightweight scoring."""

        return {
            token
            for token in re.findall(r"[a-z0-9\-]+", str(text).lower())
            if len(token) > 1 and token not in COMMON_STOP_WORDS
        }
