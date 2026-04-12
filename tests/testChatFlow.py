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


class StubSessionStore:
    """Simple in-memory session store for chat-flow tests."""

    def __init__(self, messages: list[dict[str, object]] | None = None) -> None:
        self.messages = list(messages or [])

    def list_messages(self, session_id: str, limit: int = 50) -> list[dict[str, object]]:
        del session_id
        return list(self.messages[:limit])


class StubToolRegistry:
    """Tool registry stub that records execution order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

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
                "results": [
                    {
                        "title": "BTC research memo",
                        "sourceUri": "internal://btc-memo",
                        "snippet": "Liquidity and trend support remain constructive.",
                    }
                ],
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
