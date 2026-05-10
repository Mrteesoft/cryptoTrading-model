"""Focused tests for the tool-driven assistant chat flow."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_signal_ml.chat.flow import AssistantIntentRouter, AssistantResponseComposer, ToolDrivenChatFlow  # noqa: E402
from crypto_signal_ml.config import TrainingConfig  # noqa: E402
from crypto_signal_ml.llm import (  # noqa: E402
    ChatModelAdapter,
    LlmCompletionResponse,
    LlmMessage,
    LlmToolSpec,
    OpenAIChatModelAdapter,
)


class StubSessionStore:
    """Simple in-memory session store for chat-flow tests."""

    def __init__(self, messages: list[dict[str, object]] | None = None) -> None:
        self.messages = list(messages or [])

    def list_messages(self, session_id: str, limit: int = 50) -> list[dict[str, object]]:
        del session_id
        return list(self.messages[:limit])


class StubToolRegistry:
    """Tool registry stub that records execution order."""

    def __init__(self, knowledge_results: list[dict[str, object]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.knowledge_results = (
            list(knowledge_results)
            if knowledge_results is not None
            else [
                {
                    "title": "BTC research memo",
                    "sourceUri": "internal://btc-memo",
                    "snippet": "Liquidity and trend support remain constructive.",
                }
            ]
        )

    def execute(self, tool_name: str, arguments: dict[str, object] | None = None) -> dict[str, object]:
        normalized_arguments = dict(arguments or {})
        self.calls.append((tool_name, normalized_arguments))

        if tool_name == "get_signal":
            product_id = str(normalized_arguments["product_id"])
            return {
                "toolName": "get_signal",
                "status": "ok",
                "source": "live",
                "productId": product_id,
                "forceRefresh": bool(normalized_arguments.get("force_refresh", False)),
                "warning": "",
                "error": "",
                "signal": {
                    "productId": product_id,
                    "signal_name": "BUY",
                    "confidence": 0.84,
                    "close": 150.0,
                    "signalChat": f"{product_id} has constructive momentum.",
                },
                "overview": {},
            }

        if tool_name == "get_trader_plan":
            return {
                "toolName": "get_trader_plan",
                "status": "ok",
                "source": "live",
                "forceRefresh": bool(normalized_arguments.get("force_refresh", False)),
                "warning": "",
                "error": "",
                "capital": normalized_arguments.get("capital", 10000.0),
                "liveSnapshot": {
                    "primarySignal": {
                        "productId": "BTC-USD",
                        "signal_name": "BUY",
                        "confidence": 0.84,
                    }
                },
                "traderPlan": {
                    "marketStance": "offensive",
                    "summary": "Fresh entries are open while portfolio risk stays within budget.",
                    "plan": {
                        "entries": [{"productId": "BTC-USD"}],
                        "newEntryCount": 1,
                        "addOnCount": 0,
                        "reduceCount": 0,
                        "exitCount": 0,
                        "watchlistCount": 1,
                    },
                },
            }

        if tool_name == "get_model_status":
            return {
                "toolName": "get_model_status",
                "status": "ok",
                "error": "",
                "model": {
                    "status": "ready",
                    "modelType": "xgboost",
                    "lifecycle": {
                        "freshness": "fresh",
                        "retrainingDue": False,
                        "recommendedAction": "",
                    },
                    "trainingMetrics": {"balancedAccuracy": 0.73},
                },
            }

        if tool_name == "search_knowledge":
            return {
                "toolName": "search_knowledge",
                "status": "ok",
                "query": str(normalized_arguments.get("query") or ""),
                "count": len(self.knowledge_results),
                "results": list(self.knowledge_results),
                "error": "",
            }

        if tool_name == "get_market_overview":
            return {
                "toolName": "get_market_overview",
                "status": "ok",
                "source": "live",
                "forceRefresh": bool(normalized_arguments.get("force_refresh", False)),
                "warning": "",
                "error": "",
                "overview": {
                    "marketSummary": {
                        "actionableSignals": 2,
                        "totalSignals": 5,
                    },
                    "primarySignal": {
                        "productId": "BTC-USD",
                        "signal_name": "BUY",
                        "confidence": 0.84,
                    },
                    "topSignals": [
                        {
                            "productId": "BTC-USD",
                            "signal_name": "BUY",
                            "confidence": 0.84,
                        },
                        {
                            "productId": "ETH-USD",
                            "signal_name": "HOLD",
                            "confidence": 0.61,
                        },
                    ],
                },
            }

        raise AssertionError(f"Unexpected tool call: {tool_name}")


class StubChatModelAdapter(ChatModelAdapter):
    """Fake chat model adapter for response-layer tests."""

    provider_name = "stub"

    def __init__(self, response_text: str = "LLM response") -> None:
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    def is_configured(self) -> bool:
        return True

    def complete(
        self,
        messages: list[LlmMessage] | tuple[LlmMessage, ...],
        tools: list[LlmToolSpec] | tuple[LlmToolSpec, ...],
        *,
        system_prompt: str | None = None,
    ) -> LlmCompletionResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "systemPrompt": system_prompt,
            }
        )
        return LlmCompletionResponse(message=self.response_text)


def test_router_builds_mixed_intent_plan_for_entry_size_and_model_freshness() -> None:
    """Asset, sizing, and model-freshness questions should chain the right tools."""

    router = AssistantIntentRouter(
        known_product_ids=("BTC-USD", "ETH-USD", "SOL-USD"),
    )

    route_plan = router.route("Should I enter SOL now with $500 if the model is still fresh?")

    assert route_plan.primary_product_id == "SOL-USD"
    assert route_plan.force_refresh is True
    assert route_plan.response_style == "advice"
    assert set(route_plan.intents) >= {"single_asset_signal", "trader_plan", "model_status", "mixed_request"}
    assert [planned_call.name for planned_call in route_plan.planned_calls] == [
        "get_signal",
        "get_trader_plan",
        "get_model_status",
    ]
    assert route_plan.planned_calls[1].arguments == {
        "force_refresh": True,
        "capital": 500.0,
    }


def test_router_keeps_multi_asset_signal_calls_distinct() -> None:
    """Signal calls for different assets should not be collapsed by dedupe."""

    router = AssistantIntentRouter(
        known_product_ids=("BTC-USD", "ETH-USD", "SOL-USD"),
    )

    route_plan = router.route("Compare BTC and ETH right now.")
    signal_calls = [planned_call for planned_call in route_plan.planned_calls if planned_call.name == "get_signal"]

    assert route_plan.response_style == "compare"
    assert len(signal_calls) == 2
    assert [planned_call.arguments["product_id"] for planned_call in signal_calls] == ["BTC-USD", "ETH-USD"]


def test_router_resolves_aliases_and_generic_symbols() -> None:
    """Common names and explicit ticker mentions should map cleanly into product ids."""

    router = AssistantIntentRouter(
        known_product_ids=("BTC-USD", "ETH-USD"),
    )

    assert router.resolve_product_id("Why is bitcoin bullish?") == "BTC-USD"
    assert router.resolve_product_id("Should I buy $BONK now?") == "BONK-USD"


def test_router_routes_asset_definitions_to_general_knowledge() -> None:
    """Basic coin definition prompts should not be forced through signal lookup."""

    router = AssistantIntentRouter(
        known_product_ids=("BTC-USD", "ETH-USD"),
    )

    route_plan = router.route("What is bitcoin?")
    clarification_route = router.route("I am asking what bitcoin is")

    assert route_plan.primary_product_id == "BTC-USD"
    assert route_plan.response_style == "general"
    assert route_plan.intents == ("general_knowledge",)
    assert [planned_call.name for planned_call in route_plan.planned_calls] == ["search_knowledge"]

    assert clarification_route.primary_product_id == "BTC-USD"
    assert clarification_route.response_style == "general"
    assert clarification_route.intents == ("general_knowledge",)
    assert [planned_call.name for planned_call in clarification_route.planned_calls] == ["search_knowledge"]


def test_router_keeps_market_context_for_asset_research_questions() -> None:
    """Market explainers about a named asset should still fetch the signal first."""

    router = AssistantIntentRouter(
        known_product_ids=("BTC-USD", "ETH-USD"),
    )

    route_plan = router.route("Why is bitcoin bullish?")

    assert route_plan.primary_product_id == "BTC-USD"
    assert route_plan.response_style == "explain"
    assert set(route_plan.intents) >= {"single_asset_signal", "research_context", "mixed_request"}
    assert [planned_call.name for planned_call in route_plan.planned_calls] == [
        "get_signal",
        "search_knowledge",
    ]


def test_router_keeps_greetings_conversational() -> None:
    """A greeting should not be upgraded into a market-overview tool call."""

    router = AssistantIntentRouter(
        known_product_ids=("BTC-USD", "ETH-USD"),
    )

    route_plan = router.route("hii")

    assert route_plan.response_style == "conversation"
    assert route_plan.intents == ("conversation",)
    assert route_plan.planned_calls == ()

    plain_route_plan = router.route("hi")

    assert plain_route_plan.response_style == "conversation"
    assert plain_route_plan.intents == ("conversation",)
    assert plain_route_plan.planned_calls == ()


def test_router_ignores_selected_pair_for_plain_greetings() -> None:
    """UI-selected pairs should not force asset analysis for a small-talk prompt."""

    router = AssistantIntentRouter(
        known_product_ids=("BTC-USD", "ETH-USD"),
    )

    route_plan = router.route("hi", explicit_product_id="BTC-USD")

    assert route_plan.primary_product_id is None
    assert route_plan.response_style == "conversation"
    assert route_plan.intents == ("conversation",)
    assert route_plan.planned_calls == ()


def test_router_keeps_market_intent_when_social_words_name_an_asset() -> None:
    """Social words should not hide a real asset or trading request."""

    router = AssistantIntentRouter(
        known_product_ids=("BTC-USD", "ETH-USD"),
    )

    asset_route_plan = router.route("hi BTC")
    advice_route_plan = router.route("thanks, should I buy BTC now?")

    assert asset_route_plan.primary_product_id == "BTC-USD"
    assert asset_route_plan.response_style == "direct"
    assert [planned_call.name for planned_call in asset_route_plan.planned_calls] == ["get_signal"]

    assert advice_route_plan.primary_product_id == "BTC-USD"
    assert advice_route_plan.response_style == "advice"
    assert [planned_call.name for planned_call in advice_route_plan.planned_calls] == ["get_signal"]


def test_router_routes_buy_discovery_to_market_overview() -> None:
    """A broad buy request should ask for BUY candidates, not RAG."""

    router = AssistantIntentRouter(
        known_product_ids=("BTC-USD", "ETH-USD"),
    )

    route_plan = router.route("what should I buy?")

    assert route_plan.response_style == "advice"
    assert route_plan.intents == ("market_overview",)
    assert [planned_call.name for planned_call in route_plan.planned_calls] == ["get_market_overview"]

    broad_advice_route_plan = router.route("should I buy now?")

    assert broad_advice_route_plan.response_style == "advice"
    assert broad_advice_route_plan.intents == ("market_overview",)
    assert [planned_call.name for planned_call in broad_advice_route_plan.planned_calls] == [
        "get_market_overview"
    ]


def test_chat_flow_answers_greeting_without_market_tools() -> None:
    """Small talk should produce an interactive reply without hitting model tools."""

    tool_registry = StubToolRegistry()
    flow = ToolDrivenChatFlow(
        tool_registry=tool_registry,
        session_store=StubSessionStore(),
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_product_ids=("BTC-USD", "ETH-USD"),
        ),
    )

    result = flow.run(
        session_id="session-1",
        question="hii",
    )

    assert result["toolCalls"] == []
    assert result["routing"]["responseStyle"] == "conversation"
    assert "Hi. I am ready." in result["replyText"]
    assert tool_registry.calls == []


def test_chat_flow_can_use_separate_llm_response_layer() -> None:
    """The final response can be LLM-written after routing without changing tool selection."""

    tool_registry = StubToolRegistry()
    llm_adapter = StubChatModelAdapter("Hi - I am here. Ask me about a coin or the market.")
    flow = ToolDrivenChatFlow(
        tool_registry=tool_registry,
        session_store=StubSessionStore(),
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_product_ids=("BTC-USD", "ETH-USD"),
            assistant_use_llm=True,
            llm_provider="lunatrix",
            assistant_response_model="stub-chat-model",
        ),
        llm_adapter=llm_adapter,
    )

    result = flow.run(
        session_id="session-1",
        question="hi",
    )

    assert result["replyText"] == "Hi - I am here. Ask me about a coin or the market."
    assert result["toolCalls"] == []
    assert result["responseLayer"]["mode"] == "llm"
    assert result["responseLayer"]["provider"] == "stub"
    assert tool_registry.calls == []
    assert len(llm_adapter.calls) == 1


def test_chat_flow_can_use_first_party_lunatrix_response_layer() -> None:
    """The built-in chat response layer should work without an external provider."""

    tool_registry = StubToolRegistry()
    flow = ToolDrivenChatFlow(
        tool_registry=tool_registry,
        session_store=StubSessionStore(),
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_product_ids=("BTC-USD", "ETH-USD"),
            assistant_use_llm=True,
            llm_provider="lunatrix",
            assistant_response_model="lunatrix-grounded-chat-v1",
        ),
    )

    result = flow.run(
        session_id="session-1",
        question="hi",
    )

    assert result["toolCalls"] == []
    assert result["replyText"].startswith("Hi. I am here.")
    assert result["responseLayer"]["mode"] == "local_chat"
    assert result["responseLayer"]["provider"] == "lunatrix"
    assert result["responseLayer"]["model"] == "lunatrix-grounded-chat-v1"
    assert tool_registry.calls == []


def test_chat_flow_falls_back_when_llm_layer_is_not_configured() -> None:
    """LLM mode should not break chat when provider credentials are missing."""

    tool_registry = StubToolRegistry()
    flow = ToolDrivenChatFlow(
        tool_registry=tool_registry,
        session_store=StubSessionStore(),
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_product_ids=("BTC-USD", "ETH-USD"),
            assistant_use_llm=True,
            llm_provider="unsupported",
            assistant_response_model="unsupported-chat-model",
        ),
    )

    result = flow.run(
        session_id="session-1",
        question="hi",
    )

    assert "Hi. I am ready." in result["replyText"]
    assert result["responseLayer"]["mode"] == "deterministic"
    assert result["responseLayer"]["fallbackUsed"] is True
    assert result["toolCalls"] == []
    assert tool_registry.calls == []


def test_overview_reply_prioritizes_buy_candidates_over_loss_lead() -> None:
    """Broad market replies should not present a LOSS call as the lead buy idea."""

    router = AssistantIntentRouter(known_product_ids=("BTC-USD", "ETH-USD", "SOL-USD"))
    route_plan = router.route("what should I buy?")
    composer = AssistantResponseComposer(
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_product_ids=("BTC-USD", "ETH-USD", "SOL-USD"),
        )
    )

    reply = composer.compose(
        question="what should I buy?",
        tool_results=[
            {
                "name": "get_market_overview",
                "arguments": {"force_refresh": False},
                "reason": "",
                "result": {
                    "status": "ok",
                    "source": "published",
                    "overview": {
                        "marketSummary": {
                            "actionableSignals": 3,
                            "totalSignals": 88,
                            "signalCounts": {
                                "buy": 1,
                                "loss": 1,
                                "take_profit": 1,
                                "wait": 85,
                            },
                        },
                        "primarySignal": {
                            "productId": "BIO-USD",
                            "signal_name": "LOSS",
                            "confidence": 0.702,
                        },
                        "topBuys": [
                            {
                                "productId": "SOL-USD",
                                "signal_name": "BUY",
                                "confidence": 0.81,
                            }
                        ],
                        "topSignals": [
                            {
                                "productId": "BIO-USD",
                                "signal_name": "LOSS",
                                "confidence": 0.702,
                            }
                        ],
                    },
                },
            }
        ],
        recalled_messages=[],
        route_plan=route_plan,
    )

    assert "lead BUY is SOL-USD" in reply
    assert "BIO-USD with a LOSS call" not in reply


def test_overview_reply_uses_coin_of_day_when_no_buy_is_active() -> None:
    """No-BUY overview replies should surface a watch-only spotlight without calling it an entry."""

    router = AssistantIntentRouter(known_product_ids=("BTC-USD", "ETH-USD", "SOL-USD"))
    route_plan = router.route("what should I buy?")
    composer = AssistantResponseComposer(
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_product_ids=("BTC-USD", "ETH-USD", "SOL-USD"),
        )
    )

    reply = composer.compose(
        question="what should I buy?",
        tool_results=[
            {
                "name": "get_market_overview",
                "arguments": {"force_refresh": False},
                "reason": "",
                "result": {
                    "status": "ok",
                    "source": "published",
                    "overview": {
                        "marketSummary": {
                            "actionableSignals": 2,
                            "totalSignals": 95,
                            "signalCounts": {
                                "buy": 0,
                                "loss": 1,
                                "hold": 1,
                                "wait": 93,
                            },
                        },
                        "primarySignal": {
                            "productId": "ICP-USD",
                            "signal_name": "LOSS",
                            "confidence": 0.744,
                        },
                        "topSignals": [
                            {
                                "productId": "ICP-USD",
                                "signal_name": "LOSS",
                                "confidence": 0.744,
                            }
                        ],
                        "coinOfTheDay": {
                            "productId": "SOL-USD",
                            "signal_name": "HOLD",
                            "confidence": 0.62,
                            "coinOfDayScore": 78.0,
                            "spotlightReason": "Momentum is improving, but the BUY gate is still closed.",
                        },
                    },
                },
            }
        ],
        recalled_messages=[],
        route_plan=route_plan,
    )

    assert "does not show a fresh BUY right now" in reply
    assert "no active BUY setup" in reply
    assert "Watch-only coin of the day: SOL-USD" in reply
    assert "Spotlight score: 78/100" in reply
    assert "ICP-USD with a LOSS call" not in reply


def test_chat_flow_routes_off_topic_questions_to_rag() -> None:
    """General questions should use retrieval instead of crypto market overview."""

    tool_registry = StubToolRegistry()
    flow = ToolDrivenChatFlow(
        tool_registry=tool_registry,
        session_store=StubSessionStore(),
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_product_ids=("BTC-USD", "ETH-USD"),
        ),
    )

    result = flow.run(
        session_id="session-1",
        question="Tell me about liquidity fragmentation.",
    )

    assert [tool_call["name"] for tool_call in result["toolCalls"]] == ["search_knowledge"]
    assert result["routing"]["responseStyle"] == "general"
    assert "Based on the indexed knowledge base" in result["replyText"]
    assert "Liquidity and trend support remain constructive." in result["replyText"]
    assert [tool_name for tool_name, _ in tool_registry.calls] == ["search_knowledge"]


def test_chat_flow_lets_llm_answer_general_questions_after_rag_lookup() -> None:
    """A configured response model should handle broad questions after RAG is checked."""

    tool_registry = StubToolRegistry(knowledge_results=[])
    llm_adapter = StubChatModelAdapter("Hamlet was written by William Shakespeare.")
    flow = ToolDrivenChatFlow(
        tool_registry=tool_registry,
        session_store=StubSessionStore(),
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_product_ids=("BTC-USD", "ETH-USD"),
            assistant_use_llm=True,
            llm_provider="stub",
            assistant_response_model="stub-chat-model",
        ),
        llm_adapter=llm_adapter,
    )

    result = flow.run(
        session_id="session-1",
        question="Who wrote Hamlet?",
    )

    assert [tool_call["name"] for tool_call in result["toolCalls"]] == ["search_knowledge"]
    assert result["routing"]["responseStyle"] == "general"
    assert result["replyText"] == "Hamlet was written by William Shakespeare."
    assert result["responseLayer"]["mode"] == "llm"
    assert [tool_name for tool_name, _ in tool_registry.calls] == ["search_knowledge"]
    assert len(llm_adapter.calls) == 1


def test_chat_flow_answers_asset_definition_without_signal_lookup() -> None:
    """Definition questions about a coin should answer the concept, not report missing signal data."""

    tool_registry = StubToolRegistry()
    flow = ToolDrivenChatFlow(
        tool_registry=tool_registry,
        session_store=StubSessionStore(),
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_product_ids=("BTC-USD", "ETH-USD"),
        ),
    )

    result = flow.run(
        session_id="session-1",
        question="What is bitcoin?",
    )

    assert [tool_call["name"] for tool_call in result["toolCalls"]] == ["search_knowledge"]
    assert result["routing"]["responseStyle"] == "general"
    assert "Bitcoin is a decentralized digital currency" in result["replyText"]
    assert "No authoritative signal" not in result["replyText"]
    assert [tool_name for tool_name, _ in tool_registry.calls] == ["search_knowledge"]


def test_openai_adapter_normalizes_chat_completion_responses() -> None:
    """OpenAI responses should parse into the shared chat adapter contract."""

    captured_payloads = []

    def request_executor(payload: dict[str, object]) -> dict[str, object]:
        captured_payloads.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": "General answer.",
                    }
                }
            ]
        }

    adapter = OpenAIChatModelAdapter(
        model="test-model",
        api_key="test-key",
        request_executor=request_executor,
    )

    response = adapter.complete(
        messages=[LlmMessage(role="user", content="Who wrote Hamlet?")],
        tools=(),
        system_prompt="Answer normally.",
    )

    assert adapter.is_configured() is True
    assert response.message == "General answer."
    assert captured_payloads[0]["model"] == "test-model"


def test_chat_flow_adds_trader_plan_follow_up_for_actionable_advice_questions() -> None:
    """Advisory asset questions should chain into trader planning when the signal is actionable."""

    tool_registry = StubToolRegistry()
    flow = ToolDrivenChatFlow(
        tool_registry=tool_registry,
        session_store=StubSessionStore(),
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_product_ids=("BTC-USD", "ETH-USD"),
        ),
    )

    result = flow.run(
        session_id="session-1",
        question="Should I buy BTC now?",
    )

    assert [tool_call["name"] for tool_call in result["toolCalls"]] == ["get_signal", "get_trader_plan"]
    assert [tool_call["name"] for tool_call in result["followUpToolCalls"]] == ["get_trader_plan"]
    assert result["routing"]["responseStyle"] == "advice"
    assert "Trader plan:" in result["replyText"]
    assert [tool_name for tool_name, _ in tool_registry.calls] == ["get_signal", "get_trader_plan"]


def test_response_composer_surfaces_multi_asset_comparisons() -> None:
    """Multi-asset answers should preserve both signals and add a comparison summary."""

    router = AssistantIntentRouter(known_product_ids=("BTC-USD", "ETH-USD"))
    route_plan = router.route("Compare BTC and ETH.")
    composer = AssistantResponseComposer(
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_product_ids=("BTC-USD", "ETH-USD"),
        )
    )

    reply = composer.compose(
        question="Compare BTC and ETH.",
        tool_results=[
            {
                "name": "get_signal",
                "arguments": {"product_id": "BTC-USD", "force_refresh": False},
                "reason": "",
                "result": {
                    "status": "ok",
                    "source": "live",
                    "productId": "BTC-USD",
                    "signal": {
                        "signal_name": "BUY",
                        "confidence": 0.82,
                        "close": 65000.0,
                        "signalChat": "BTC momentum remains constructive.",
                    },
                },
            },
            {
                "name": "get_signal",
                "arguments": {"product_id": "ETH-USD", "force_refresh": False},
                "reason": "",
                "result": {
                    "status": "ok",
                    "source": "live",
                    "productId": "ETH-USD",
                    "signal": {
                        "signal_name": "HOLD",
                        "confidence": 0.61,
                        "close": 3200.0,
                        "signalChat": "ETH is still waiting for clearer follow-through.",
                    },
                },
            },
        ],
        recalled_messages=[],
        route_plan=route_plan,
    )

    assert "Comparison:" in reply
    assert "BTC-USD is currently a BUY setup" in reply
    assert "ETH-USD is currently a HOLD setup" in reply
