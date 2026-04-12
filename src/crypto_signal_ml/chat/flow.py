"""Tool-driven chat flow for grounded assistant responses."""

from __future__ import annotations

from dataclasses import dataclass
import json
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

COMMON_PRODUCT_ALIASES = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "ether": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "ripple": "XRP",
    "xrp": "XRP",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "cardano": "ADA",
    "ada": "ADA",
    "chainlink": "LINK",
    "link": "LINK",
    "litecoin": "LTC",
    "ltc": "LTC",
    "avalanche": "AVAX",
    "avax": "AVAX",
    "binance coin": "BNB",
    "binancecoin": "BNB",
    "bnb": "BNB",
    "polkadot": "DOT",
    "dot": "DOT",
    "tron": "TRX",
    "trx": "TRX",
    "aptos": "APT",
    "apt": "APT",
    "arbitrum": "ARB",
    "arb": "ARB",
    "sui": "SUI",
    "pepe": "PEPE",
    "shiba inu": "SHIB",
    "shib": "SHIB",
    "uniswap": "UNI",
    "uni": "UNI",
}

MODEL_REQUEST_KEYWORDS = (
    "model",
    "accuracy",
    "artifact",
    "train",
    "trained",
    "training",
    "retrain",
    "fresh",
    "freshness",
    "feature importance",
)

TRADER_REQUEST_KEYWORDS = (
    "portfolio",
    "position",
    "risk",
    "plan",
    "allocation",
    "sizing",
    "size",
    "capital",
    "stop",
    "target",
    "take profit",
    "take-profit",
    "enter",
    "entry",
    "exit",
    "entry plan",
    "exit plan",
)

KNOWLEDGE_REQUEST_KEYWORDS = (
    "why",
    "research",
    "source",
    "context",
    "memo",
    "document",
    "knowledge",
    "explain",
)

MARKET_OVERVIEW_KEYWORDS = (
    "market",
    "overview",
    "broad",
    "overall",
    "top setup",
    "top setups",
    "leaders",
    "best setup",
    "best setups",
    "market read",
    "market overview",
)

ADVISORY_KEYWORDS = (
    "should i",
    "should we",
    "buy",
    "sell",
    "enter",
    "exit",
    "trim",
    "add",
    "safe",
    "risk",
    "allocate",
)

COMPARISON_KEYWORDS = ("compare", "vs", "versus", "between")
FOLLOW_UP_TRADER_KEYWORDS = ("should i", "buy", "sell", "enter", "exit", "trim", "add")
FORCE_REFRESH_KEYWORDS = (
    "right now",
    "latest",
    "current price",
    "current",
    "live",
    "now",
    "today",
    "just now",
    "currently",
    "enter now",
)


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


@dataclass(frozen=True)
class AssistantRoutePlan:
    """Structured routing decision for one assistant turn."""

    primary_product_id: str | None
    resolved_product_ids: tuple[str, ...]
    intents: tuple[str, ...]
    response_style: str
    force_refresh: bool
    capital_override: float | None
    planned_calls: tuple[PlannedToolCall, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the route plan in JSON-friendly form."""

        return {
            "primaryProductId": self.primary_product_id,
            "resolvedProductIds": list(self.resolved_product_ids),
            "intents": list(self.intents),
            "responseStyle": self.response_style,
            "forceRefresh": bool(self.force_refresh),
            "capitalOverride": self.capital_override,
            "plannedCalls": [planned_call.to_dict() for planned_call in self.planned_calls],
        }


class AssistantIntentRouter:
    """Pick the smallest useful set of tools for one user request."""

    def __init__(
        self,
        *,
        known_product_ids: Sequence[str],
        retrieval_enabled: bool = True,
        default_quote_currency: str = "USD",
        max_direct_signals: int = 3,
    ) -> None:
        self.known_product_ids = tuple(str(product_id).upper() for product_id in known_product_ids)
        self.retrieval_enabled = bool(retrieval_enabled)
        self.default_quote_currency = str(default_quote_currency).strip().upper() or "USD"
        self.max_direct_signals = max(int(max_direct_signals), 1)
        self._default_product_by_symbol = self._build_default_product_map(self.known_product_ids)
        self._product_aliases = self._build_product_aliases(self.known_product_ids)

    def resolve_product_id(
        self,
        question: str,
        explicit_product_id: str | None = None,
    ) -> str | None:
        """Resolve the primary product id from the explicit request or question text."""

        resolved_product_ids = self.resolve_product_ids(
            question=question,
            explicit_product_id=explicit_product_id,
        )
        return resolved_product_ids[0] if resolved_product_ids else None

    def resolve_product_ids(
        self,
        question: str,
        explicit_product_id: str | None = None,
    ) -> list[str]:
        """Resolve one or more product ids from the explicit request or question text."""

        normalized_question = str(question).strip()
        if not normalized_question and explicit_product_id is None:
            return []

        resolved_product_ids: list[str] = []
        if explicit_product_id is not None:
            for raw_product_id in re.split(r"[\s,]+", str(explicit_product_id)):
                normalized_product_id = self._normalize_product_id(raw_product_id)
                if normalized_product_id is not None:
                    resolved_product_ids.append(normalized_product_id)

        upper_question = normalized_question.upper()
        lower_question = normalized_question.lower()

        for base_currency, quote_currency in re.findall(r"\b([A-Z0-9]{2,10})[-/]([A-Z0-9]{2,10})\b", upper_question):
            normalized_product_id = self._normalize_product_id(f"{base_currency}-{quote_currency}")
            if normalized_product_id is not None:
                resolved_product_ids.append(normalized_product_id)

        for compact_symbol, quote_currency in re.findall(r"\b([A-Z0-9]{2,10})(USD|USDT|USDC)\b", upper_question):
            normalized_product_id = self._normalize_product_id(f"{compact_symbol}-{quote_currency}")
            if normalized_product_id is not None:
                resolved_product_ids.append(normalized_product_id)

        for alias, product_id in sorted(self._product_aliases.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", lower_question):
                resolved_product_ids.append(product_id)

        for raw_token in re.findall(r"\$?[A-Za-z0-9]{2,12}", normalized_question):
            normalized_product_id = self._resolve_symbol_candidate(raw_token)
            if normalized_product_id is not None:
                resolved_product_ids.append(normalized_product_id)

        return self._stable_unique(resolved_product_ids)

    def resolve_capital_override(self, question: str) -> float | None:
        """Parse a simple capital override from the question when present."""

        capital_match = re.search(
            r"\b(?:capital|with|using)\s+\$?([0-9][0-9,]*(?:\.[0-9]+)?)\b",
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
    ) -> AssistantRoutePlan:
        """Resolve the tool plan needed for the request."""

        normalized_question = str(question).strip()
        resolved_product_ids = tuple(
            self.resolve_product_ids(
                question=normalized_question,
                explicit_product_id=explicit_product_id,
            )
        )
        capital_override = self.resolve_capital_override(normalized_question)
        lowered_question = normalized_question.lower()
        response_style = self._classify_response_style(lowered_question, resolved_product_ids)
        resolved_force_refresh = bool(force_refresh or self._should_force_refresh(lowered_question))
        intents = self._classify_intents(
            lowered_question,
            resolved_product_ids=resolved_product_ids,
        )

        planned_calls: list[PlannedToolCall] = []
        for product_id in resolved_product_ids[: self.max_direct_signals]:
            planned_calls.append(
                PlannedToolCall(
                    name="get_signal",
                    arguments={
                        "product_id": product_id,
                        "force_refresh": bool(resolved_force_refresh),
                    },
                    reason=(
                        f"The question names {product_id}, so the assistant should fetch the authoritative "
                        "single-asset signal first."
                    ),
                )
            )

        if "market_overview" in intents and (not resolved_product_ids or "market" in lowered_question):
            planned_calls.append(
                PlannedToolCall(
                    name="get_market_overview",
                    arguments={"force_refresh": bool(resolved_force_refresh)},
                    reason="The question asks for broader market context, so the assistant should fetch the market overview.",
                )
            )

        if "trader_plan" in intents:
            trader_arguments: dict[str, Any] = {
                "force_refresh": bool(resolved_force_refresh),
            }
            if capital_override is not None:
                trader_arguments["capital"] = capital_override

            planned_calls.append(
                PlannedToolCall(
                    name="get_trader_plan",
                    arguments=trader_arguments,
                    reason="The question asks for risk, sizing, or portfolio-aware action framing.",
                )
            )

        if "model_status" in intents:
            planned_calls.append(
                PlannedToolCall(
                    name="get_model_status",
                    arguments={},
                    reason="The question asks about model freshness, readiness, or training quality.",
                )
            )

        if self.retrieval_enabled and "research_context" in intents:
            planned_calls.append(
                PlannedToolCall(
                    name="search_knowledge",
                    arguments={
                        "query": self._build_knowledge_query(normalized_question, resolved_product_ids),
                        "limit": 3,
                    },
                    reason="The question asks for supporting context, sources, or explanatory background.",
                )
            )

        if not planned_calls:
            planned_calls.append(
                PlannedToolCall(
                    name="get_market_overview",
                    arguments={"force_refresh": bool(resolved_force_refresh)},
                    reason="No specific asset or specialized request was resolved, so the assistant should start from the market overview.",
                )
            )

        deduped_calls = tuple(self._dedupe_calls(planned_calls))
        return AssistantRoutePlan(
            primary_product_id=resolved_product_ids[0] if resolved_product_ids else None,
            resolved_product_ids=resolved_product_ids,
            intents=intents,
            response_style=response_style,
            force_refresh=bool(resolved_force_refresh),
            capital_override=capital_override,
            planned_calls=deduped_calls,
        )

    def should_follow_up_with_trader_plan(self, question: str) -> bool:
        """Return whether an advisory question should upgrade into a trader-plan follow-up."""

        lowered_question = str(question).lower()
        return any(keyword in lowered_question for keyword in FOLLOW_UP_TRADER_KEYWORDS)

    def _classify_intents(
        self,
        lowered_question: str,
        *,
        resolved_product_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Map one free-form question into routing intents."""

        intents: list[str] = []
        asks_for_model_status = self._contains_any_keyword(lowered_question, MODEL_REQUEST_KEYWORDS)
        asks_for_trader_plan = self._contains_any_keyword(lowered_question, TRADER_REQUEST_KEYWORDS)
        asks_for_research = self._contains_any_keyword(lowered_question, KNOWLEDGE_REQUEST_KEYWORDS)
        asks_for_market_overview = self._contains_any_keyword(lowered_question, MARKET_OVERVIEW_KEYWORDS)
        asks_for_comparison = self._contains_any_keyword(lowered_question, COMPARISON_KEYWORDS)

        if resolved_product_ids:
            intents.append("multi_asset_signal" if len(resolved_product_ids) > 1 else "single_asset_signal")
        if asks_for_market_overview and (not resolved_product_ids or "market" in lowered_question):
            intents.append("market_overview")
        if asks_for_trader_plan:
            intents.append("trader_plan")
        if asks_for_model_status:
            intents.append("model_status")
        if self.retrieval_enabled and asks_for_research:
            intents.append("research_context")
        if not intents:
            intents.append("market_overview")
        if asks_for_comparison and "market_overview" not in intents and len(resolved_product_ids) >= 2:
            intents.append("market_overview")

        material_intents = [
            intent
            for intent in intents
            if intent not in {"single_asset_signal", "multi_asset_signal"}
        ]
        if len(material_intents) + int(bool(resolved_product_ids)) > 1:
            intents.append("mixed_request")

        return tuple(self._stable_unique(intents))

    @staticmethod
    def _classify_response_style(
        lowered_question: str,
        resolved_product_ids: Sequence[str],
    ) -> str:
        """Classify the answer style needed for the question."""

        if len(resolved_product_ids) >= 2 and any(keyword in lowered_question for keyword in COMPARISON_KEYWORDS):
            return "compare"
        if any(keyword in lowered_question for keyword in KNOWLEDGE_REQUEST_KEYWORDS):
            return "explain"
        if any(keyword in lowered_question for keyword in ADVISORY_KEYWORDS):
            return "advice"
        return "direct"

    @staticmethod
    def _contains_any_keyword(text: str, keywords: Sequence[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _should_force_refresh(lowered_question: str) -> bool:
        return any(keyword in lowered_question for keyword in FORCE_REFRESH_KEYWORDS)

    def _build_knowledge_query(
        self,
        question: str,
        resolved_product_ids: Sequence[str],
    ) -> str:
        """Append product context to knowledge lookups when that sharpens retrieval."""

        normalized_question = str(question).strip()
        if not resolved_product_ids:
            return normalized_question

        query_terms = [normalized_question]
        for product_id in resolved_product_ids[:2]:
            if product_id not in normalized_question.upper():
                query_terms.append(product_id)
        return " ".join(query_terms)

    def _resolve_symbol_candidate(self, raw_token: str) -> str | None:
        """Resolve one token like BTC, $ETH, or sol into a product id when safe."""

        normalized_token = str(raw_token).strip()
        if not normalized_token:
            return None

        stripped_token = normalized_token.lstrip("$")
        lower_token = stripped_token.lower()
        if lower_token in COMMON_STOP_WORDS:
            return None

        alias_product_id = self._product_aliases.get(lower_token)
        if alias_product_id is not None:
            return alias_product_id

        if not re.fullmatch(r"[A-Za-z0-9]{2,10}", stripped_token):
            return None
        if stripped_token.isdigit():
            return None

        if normalized_token.startswith("$") or stripped_token.isupper():
            return self._default_product_for_symbol(stripped_token.upper())

        return None

    def _default_product_for_symbol(self, symbol: str) -> str:
        """Resolve the canonical product id for one symbol, defaulting to the configured quote currency."""

        normalized_symbol = str(symbol).strip().upper()
        return self._default_product_by_symbol.get(
            normalized_symbol,
            f"{normalized_symbol}-{self.default_quote_currency}",
        )

    @staticmethod
    def _build_default_product_map(known_product_ids: Sequence[str]) -> dict[str, str]:
        """Prefer known USD pairs when mapping base symbols back to products."""

        default_product_by_symbol: dict[str, str] = {}
        for product_id in known_product_ids:
            normalized_product_id = str(product_id).strip().upper()
            if "-" not in normalized_product_id:
                continue
            base_currency, quote_currency = normalized_product_id.split("-", 1)
            existing_product_id = default_product_by_symbol.get(base_currency)
            if existing_product_id is None or quote_currency == "USD":
                default_product_by_symbol[base_currency] = normalized_product_id
        return default_product_by_symbol

    def _build_product_aliases(self, known_product_ids: Sequence[str]) -> dict[str, str]:
        """Build a product alias map from configured products plus common coin names."""

        aliases: dict[str, str] = {}
        for product_id in known_product_ids:
            normalized_product_id = str(product_id).strip().upper()
            if "-" not in normalized_product_id:
                continue
            base_currency, quote_currency = normalized_product_id.split("-", 1)
            aliases[base_currency.lower()] = normalized_product_id
            aliases[f"{base_currency.lower()}-{quote_currency.lower()}"] = normalized_product_id
            aliases[f"{base_currency.lower()}/{quote_currency.lower()}"] = normalized_product_id

        for alias, symbol in COMMON_PRODUCT_ALIASES.items():
            aliases.setdefault(alias, self._default_product_for_symbol(symbol))

        return aliases

    @staticmethod
    def _normalize_product_id(raw_product_id: str) -> str | None:
        """Normalize one raw pair into a Coinbase-style product id."""

        normalized_product_id = str(raw_product_id).strip().upper().replace("/", "-")
        if not normalized_product_id:
            return None
        if "-" not in normalized_product_id:
            return None
        if not re.fullmatch(r"[A-Z0-9]{2,10}-[A-Z0-9]{2,10}", normalized_product_id):
            return None
        return normalized_product_id

    @staticmethod
    def _stable_unique(values: Sequence[str]) -> list[str]:
        """Keep input order stable while removing duplicates."""

        deduped_values = []
        seen_values = set()
        for value in values:
            normalized_value = str(value).strip()
            if not normalized_value or normalized_value in seen_values:
                continue
            seen_values.add(normalized_value)
            deduped_values.append(normalized_value)
        return deduped_values

    @staticmethod
    def _dedupe_calls(planned_calls: Sequence[PlannedToolCall]) -> list[PlannedToolCall]:
        """Keep tool order stable while removing duplicate tool+argument combinations."""

        deduped_calls = []
        seen_call_keys = set()
        for planned_call in planned_calls:
            normalized_key = (
                planned_call.name,
                json.dumps(planned_call.arguments, sort_keys=True, default=str),
            )
            if normalized_key in seen_call_keys:
                continue
            seen_call_keys.add(normalized_key)
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
        route_plan: AssistantRoutePlan,
    ) -> str:
        """Turn tool results into one concise answer."""

        del question
        results_by_name: dict[str, list[dict[str, Any]]] = {}
        for tool_result in tool_results:
            tool_name = str(tool_result.get("name") or "")
            if not tool_name:
                continue
            results_by_name.setdefault(tool_name, []).append(dict(tool_result.get("result") or {}))

        signal_results = results_by_name.get("get_signal", [])
        overview_results = results_by_name.get("get_market_overview", [])
        trader_results = results_by_name.get("get_trader_plan", [])
        model_results = results_by_name.get("get_model_status", [])
        retrieval_results = results_by_name.get("search_knowledge", [])

        opening_paragraphs = self._compose_opening_paragraphs(
            route_plan=route_plan,
            signal_results=signal_results,
            overview_results=overview_results,
        )

        signal_paragraphs = []
        if route_plan.response_style == "compare" and len(signal_results) > 1:
            signal_paragraphs.extend(self._compose_signal_comparison_paragraphs(signal_results))
        for signal_result in signal_results:
            signal_paragraphs.extend(self._compose_signal_paragraphs(signal_result))

        overview_paragraphs = []
        if overview_results and (not signal_results or "market_overview" in route_plan.intents):
            overview_paragraphs.extend(self._compose_overview_paragraphs(overview_results[0]))

        trader_paragraphs = []
        if trader_results:
            trader_paragraphs.extend(self._compose_trader_paragraphs(trader_results[0]))

        model_paragraphs = []
        if model_results:
            model_paragraphs.extend(self._compose_model_paragraphs(model_results[0]))

        retrieval_paragraphs = []
        if retrieval_results:
            retrieval_paragraphs.extend(self._compose_retrieval_paragraphs(retrieval_results[0]))

        conflict_paragraphs = self._compose_conflict_paragraphs(
            signal_results=signal_results,
            trader_results=trader_results,
        )

        memory_paragraphs = []
        if recalled_messages:
            memory_paragraphs.append(
                "Relevant session memory: "
                + str(recalled_messages[-1].get("content") or "").strip()
            )

        ordered_section_names = {
            "advice": (
                "opening",
                "signals",
                "trader",
                "conflicts",
                "model",
                "retrieval",
                "overview",
                "memory",
            ),
            "explain": (
                "opening",
                "signals",
                "retrieval",
                "overview",
                "model",
                "trader",
                "conflicts",
                "memory",
            ),
            "compare": (
                "opening",
                "signals",
                "overview",
                "trader",
                "model",
                "retrieval",
                "conflicts",
                "memory",
            ),
        }.get(
            route_plan.response_style,
            (
                "opening",
                "signals",
                "overview",
                "trader",
                "model",
                "retrieval",
                "conflicts",
                "memory",
            ),
        )

        section_paragraphs = {
            "opening": opening_paragraphs,
            "signals": signal_paragraphs,
            "overview": overview_paragraphs,
            "trader": trader_paragraphs,
            "model": model_paragraphs,
            "retrieval": retrieval_paragraphs,
            "conflicts": conflict_paragraphs,
            "memory": memory_paragraphs,
        }

        paragraphs = [
            paragraph
            for section_name in ordered_section_names
            for paragraph in section_paragraphs.get(section_name, [])
            if str(paragraph).strip()
        ]

        if not paragraphs:
            paragraphs.append(
                "I could not produce a grounded answer because none of the trading tools returned usable data."
            )

        return "\n\n".join(paragraph for paragraph in paragraphs if paragraph.strip())

    def _compose_opening_paragraphs(
        self,
        *,
        route_plan: AssistantRoutePlan,
        signal_results: Sequence[dict[str, Any]],
        overview_results: Sequence[dict[str, Any]],
    ) -> list[str]:
        """Tailor the opening line to the user's question style."""

        if route_plan.response_style == "advice":
            primary_signal_result = self._select_best_signal_result(
                signal_results,
                route_plan.primary_product_id,
            )
            if primary_signal_result is not None:
                opening = self._build_advice_opening(primary_signal_result)
                return [opening] if opening else []

            if overview_results:
                overview = dict(overview_results[0].get("overview") or {})
                primary_signal = dict(overview.get("primarySignal") or {})
                lead_product = str(primary_signal.get("productId") or "the lead pair")
                lead_signal = str(primary_signal.get("signal_name") or "HOLD")
                return [
                    f"The broad market read currently points to {lead_product} as the lead setup, with a {lead_signal} signal."
                ]

        return []

    def _build_advice_opening(self, signal_result: dict[str, Any]) -> str:
        """Turn one signal result into a direct yes-or-no style opening."""

        product_id = str(signal_result.get("productId") or "that asset")
        source_descriptor = self._describe_source(signal_result.get("source"))
        signal_summary = signal_result.get("signal")
        if not isinstance(signal_summary, dict):
            if str(signal_result.get("status") or "") == "not_found":
                return f"I do not have an authoritative signal for {product_id} yet."
            return ""

        signal_name = str(signal_summary.get("signal_name") or "HOLD").upper()
        if signal_name == "BUY":
            return f"The {source_descriptor} tool read supports a cautious long entry setup for {product_id}."
        if signal_name == "TAKE_PROFIT":
            return f"The {source_descriptor} tool read favors trimming {product_id} rather than adding exposure."
        if signal_name == "LOSS":
            return f"The {source_descriptor} tool read is in risk-reduction mode for {product_id}, not a fresh entry."
        return f"The {source_descriptor} tool read does not show a strong new entry edge for {product_id} right now."

    def _compose_signal_comparison_paragraphs(
        self,
        signal_results: Sequence[dict[str, Any]],
    ) -> list[str]:
        """Summarize multi-asset comparisons before per-asset detail."""

        comparison_lines = []
        for signal_result in signal_results[:3]:
            signal_summary = signal_result.get("signal")
            if not isinstance(signal_summary, dict):
                continue
            comparison_lines.append(
                f"{signal_result.get('productId')} {signal_summary.get('signal_name')} "
                f"({self._format_percent(signal_summary.get('confidence'))})"
            )

        if len(comparison_lines) < 2:
            return []

        return ["Comparison: " + ", ".join(comparison_lines) + "."]

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
        entry_count = int(plan_summary.get("newEntryCount") or plan_summary.get("entryCount") or 0)
        add_on_count = int(plan_summary.get("addOnCount") or 0)
        reduce_count = int(plan_summary.get("reduceCount") or 0)
        exit_count = int(plan_summary.get("exitCount") or 0)
        watchlist_count = int(plan_summary.get("watchlistCount") or 0)
        capital_basis = self._format_price(trader_result.get("capital"))

        paragraphs = [
            f"Trader plan: capital basis is {capital_basis}. The market stance is {market_stance}, with {entry_count} fresh entries, "
            f"{add_on_count} add-ons, {reduce_count} reductions, {exit_count} exits, and {watchlist_count} watchlist candidates."
        ]

        summary_text = str(trader_plan.get("summary") or "").strip()
        if summary_text:
            paragraphs.append(summary_text)

        entries = list(plan_summary.get("entries", []))
        if entries:
            entry_products = [
                str(entry_row.get("productId") or "")
                for entry_row in entries[:3]
                if str(entry_row.get("productId") or "").strip()
            ]
            if entry_products:
                paragraphs.append("Planned fresh entries: " + ", ".join(entry_products) + ".")

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

    def _compose_conflict_paragraphs(
        self,
        *,
        signal_results: Sequence[dict[str, Any]],
        trader_results: Sequence[dict[str, Any]],
    ) -> list[str]:
        """Call out material disagreements between tools instead of papering them over."""

        primary_signal_result = self._select_best_signal_result(signal_results, None)
        if primary_signal_result is None or not trader_results:
            return []

        signal_summary = primary_signal_result.get("signal")
        if not isinstance(signal_summary, dict):
            return []

        trader_plan = dict(trader_results[0].get("traderPlan") or {})
        plan_summary = dict(trader_plan.get("plan") or {})
        market_stance = str(trader_plan.get("marketStance") or "balanced")
        entry_count = int(plan_summary.get("newEntryCount") or plan_summary.get("entryCount") or 0)
        signal_name = str(signal_summary.get("signal_name") or "HOLD").upper()

        if signal_name == "BUY" and market_stance == "defensive":
            return [
                "The asset signal is constructive, but the broader portfolio plan is still defensive, so sizing should stay conservative."
            ]
        if signal_name == "BUY" and entry_count <= 0:
            return [
                "The asset signal is positive, but the portfolio plan is not opening fresh entries yet."
            ]
        if signal_name in {"TAKE_PROFIT", "LOSS"} and entry_count > 0:
            return [
                "This asset is in de-risking mode even though the portfolio plan still sees entries elsewhere."
            ]

        return []

    @staticmethod
    def _select_best_signal_result(
        signal_results: Sequence[dict[str, Any]],
        preferred_product_id: str | None,
    ) -> dict[str, Any] | None:
        """Select one signal result, preferring the explicitly requested product when available."""

        if preferred_product_id is not None:
            for signal_result in signal_results:
                if str(signal_result.get("productId") or "").upper() == preferred_product_id.upper():
                    return signal_result

        return signal_results[0] if signal_results else None

    @staticmethod
    def _describe_source(source: Any) -> str:
        """Translate raw tool sources into answer-safe wording."""

        normalized_source = str(source or "").strip().lower()
        if normalized_source == "live":
            return "current live"
        if normalized_source == "published":
            return "latest published"
        if normalized_source in {"cached", "snapshot"}:
            return "latest cached"
        return "current authoritative"

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
        route_plan = self.router.route(
            question=question,
            explicit_product_id=explicit_product_id,
            force_refresh=force_refresh,
        )
        initial_calls = list(route_plan.planned_calls)
        executed_calls = self.executor.execute(initial_calls)
        follow_up_calls = self._plan_follow_up_calls(
            question=question,
            route_plan=route_plan,
            executed_calls=executed_calls,
        )
        if follow_up_calls:
            executed_calls.extend(self.executor.execute(follow_up_calls))

        all_planned_calls = [*initial_calls, *follow_up_calls]
        reply_text = self.composer.compose(
            question=question,
            tool_results=executed_calls,
            recalled_messages=recalled_messages,
            route_plan=route_plan,
        )

        return {
            "resolvedProductId": route_plan.primary_product_id,
            "resolvedProductIds": list(route_plan.resolved_product_ids),
            "toolCalls": [planned_call.to_dict() for planned_call in all_planned_calls],
            "initialToolCalls": [planned_call.to_dict() for planned_call in initial_calls],
            "followUpToolCalls": [planned_call.to_dict() for planned_call in follow_up_calls],
            "toolResults": executed_calls,
            "replyText": reply_text,
            "routing": {
                "intents": list(route_plan.intents),
                "responseStyle": route_plan.response_style,
                "forceRefresh": bool(route_plan.force_refresh),
                "capitalOverride": route_plan.capital_override,
            },
            "toolTelemetry": self._build_tool_telemetry(executed_calls),
            "retrieval": {
                "signals": self._collect_signal_context(executed_calls),
                "messages": recalled_messages,
                "knowledge": self._collect_knowledge_context(executed_calls),
            },
            "liveContext": self._build_live_context(executed_calls),
        }

    def _plan_follow_up_calls(
        self,
        *,
        question: str,
        route_plan: AssistantRoutePlan,
        executed_calls: Sequence[dict[str, Any]],
    ) -> list[PlannedToolCall]:
        """Add a trader-plan follow-up when an advisory signal turns actionable."""

        if any(planned_call.name == "get_trader_plan" for planned_call in route_plan.planned_calls):
            return []
        if route_plan.primary_product_id is None:
            return []
        if not self.router.should_follow_up_with_trader_plan(question):
            return []

        signal_result = self._find_signal_result(
            executed_calls=executed_calls,
            product_id=route_plan.primary_product_id,
        )
        if signal_result is None or not self._is_actionable_signal(signal_result):
            return []

        trader_arguments: dict[str, Any] = {
            "force_refresh": bool(route_plan.force_refresh),
        }
        if route_plan.capital_override is not None:
            trader_arguments["capital"] = route_plan.capital_override

        return [
            PlannedToolCall(
                name="get_trader_plan",
                arguments=trader_arguments,
                reason=(
                    f"The advisory question about {route_plan.primary_product_id} produced an actionable signal, "
                    "so the assistant should add portfolio-aware sizing guidance."
                ),
            )
        ]

    @staticmethod
    def _find_signal_result(
        *,
        executed_calls: Sequence[dict[str, Any]],
        product_id: str,
    ) -> dict[str, Any] | None:
        """Find the executed signal result for one product id."""

        for executed_call in executed_calls:
            if str(executed_call.get("name") or "") != "get_signal":
                continue
            result_payload = dict(executed_call.get("result") or {})
            if str(result_payload.get("productId") or "").upper() == product_id.upper():
                return result_payload
        return None

    def _is_actionable_signal(self, signal_result: dict[str, Any]) -> bool:
        """Use the signal payload to decide whether a trader-plan follow-up is worth the cost."""

        signal_summary = signal_result.get("signal")
        if not isinstance(signal_summary, dict):
            return False

        signal_name = str(signal_summary.get("signal_name") or "HOLD").upper()
        confidence = self._safe_float(signal_summary.get("confidence"))
        if confidence is None:
            return False

        return signal_name in {"BUY", "TAKE_PROFIT", "LOSS"} and confidence >= float(self.config.backtest_min_confidence)

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
    def _build_tool_telemetry(tool_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Summarize executed tools for later quality review."""

        executed_tools = []
        for tool_result in tool_results:
            result_payload = dict(tool_result.get("result") or {})
            source = str(result_payload.get("source") or "").strip()
            executed_tools.append(
                {
                    "name": str(tool_result.get("name") or ""),
                    "status": str(result_payload.get("status") or ""),
                    "source": source,
                    "usedCache": source in {"cached", "snapshot", "published"},
                    "forceRefresh": bool(
                        result_payload.get("forceRefresh")
                        if "forceRefresh" in result_payload
                        else tool_result.get("arguments", {}).get("force_refresh", False)
                    ),
                }
            )

        return {
            "executedToolCount": len(executed_tools),
            "executedTools": executed_tools,
        }

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

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Parse one float without raising."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return None
