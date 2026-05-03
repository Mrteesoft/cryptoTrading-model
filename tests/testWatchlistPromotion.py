"""Tests for watchlist progression and promotion behavior."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_signal_ml.config import TrainingConfig  # noqa: E402
from crypto_signal_ml.portfolio_core import TraderBrain  # noqa: E402


def _build_signal(
    *,
    product_id: str,
    signal_name: str,
    confidence: float,
    close: float,
    trade_readiness: str,
    setup_score: float,
    policy_score: float,
    probability_margin: float = 0.08,
    trend_score: float = 0.03,
    volatility_ratio: float = 1.0,
    market_risk_mode: str = "neutral",
) -> dict[str, object]:
    return {
        "productId": product_id,
        "signal_name": signal_name,
        "spotAction": "buy" if signal_name == "BUY" else "wait",
        "confidence": confidence,
        "close": close,
        "tradeReadiness": trade_readiness,
        "setupScore": setup_score,
        "policyScore": policy_score,
        "probabilityMargin": probability_margin,
        "marketState": {
            "label": "trend_up",
            "trendScore": trend_score,
            "volatilityRatio": volatility_ratio,
            "isTrending": True,
            "isHighVolatility": False,
        },
        "marketContext": {
            "marketIntelligence": {
                "available": True,
                "riskMode": market_risk_mode,
            }
        },
    }


def test_watchlist_setup_promotes_to_entry_ready(tmp_path: Path) -> None:
    config = TrainingConfig(
        coinmarketcap_use_context=False,
        signal_watchlist_state_path=tmp_path / "watchlistState.json",
        signal_watchlist_breakout_pct=0.01,
        signal_watchlist_promotion_min_confidence=0.55,
        signal_watchlist_promotion_min_decision_score=0.50,
        signal_watchlist_promotion_min_confidence_gain=0.05,
        signal_watchlist_promotion_min_decision_score_gain=0.05,
        signal_watchlist_promotion_min_positive_checks=1,
        signal_watchlist_entry_ready_min_confidence=0.60,
        signal_watchlist_entry_ready_min_decision_score=0.60,
    )
    brain = TraderBrain(config=config)

    first_signal = _build_signal(
        product_id="BTC-USD",
        signal_name="HOLD",
        confidence=0.56,
        close=100.0,
        trade_readiness="standby",
        setup_score=3.8,
        policy_score=0.8,
    )
    first_plan = brain.build_plan(
        signal_summaries=[first_signal],
        positions=[],
        capital=10000.0,
    )
    first_brain = first_plan["signals"][0]["brain"]
    assert first_brain["decision"] in {"watchlist", "avoid_long"}
    assert first_brain["watchlistStage"] in {"watchlist", "setup_building", "setup_confirmed"}

    second_signal = _build_signal(
        product_id="BTC-USD",
        signal_name="HOLD",
        confidence=0.67,
        close=101.5,
        trade_readiness="high",
        setup_score=4.6,
        policy_score=0.95,
    )
    second_plan = brain.build_plan(
        signal_summaries=[second_signal],
        positions=[],
        capital=10000.0,
    )
    second_brain = second_plan["signals"][0]["brain"]
    assert second_brain["watchlistStage"] == "entry_ready"
    assert second_brain["decision"] in {"enter_long", "watchlist"} or second_brain["proposedDecision"] == "enter_long_candidate"
    assert second_plan["watchlistPromotion"]["reviewedCount"] == 1
    assert second_plan["watchlistPromotion"]["promotedThisCycle"] == 1
    assert second_plan["watchlistPromotion"]["stageCounts"]["entry_ready"] == 1
    assert second_plan["watchlistPromotion"]["transitions"][0]["toStage"] == "entry_ready"


def test_watchlist_promotion_blocked_by_regime(tmp_path: Path) -> None:
    config = TrainingConfig(
        coinmarketcap_use_context=False,
        signal_watchlist_state_path=tmp_path / "watchlistState.json",
        signal_watchlist_breakout_pct=0.01,
        signal_watchlist_promotion_min_confidence=0.55,
        signal_watchlist_promotion_min_decision_score=0.50,
        signal_watchlist_promotion_min_confidence_gain=0.05,
        signal_watchlist_promotion_min_decision_score_gain=0.05,
        signal_watchlist_promotion_min_positive_checks=1,
        signal_watchlist_entry_ready_min_confidence=0.60,
        signal_watchlist_entry_ready_min_decision_score=0.60,
    )
    brain = TraderBrain(config=config)

    seed_signal = _build_signal(
        product_id="SOL-USD",
        signal_name="HOLD",
        confidence=0.56,
        close=100.0,
        trade_readiness="standby",
        setup_score=3.6,
        policy_score=0.7,
    )
    brain.build_plan(signal_summaries=[seed_signal], positions=[], capital=10000.0)

    blocked_signal = _build_signal(
        product_id="SOL-USD",
        signal_name="HOLD",
        confidence=0.66,
        close=101.5,
        trade_readiness="high",
        setup_score=4.5,
        policy_score=0.9,
        market_risk_mode="risk_off",
    )
    blocked_plan = brain.build_plan(signal_summaries=[blocked_signal], positions=[], capital=10000.0)
    blocked_brain = blocked_plan["signals"][0]["brain"]
    assert blocked_brain["watchlistStage"] in {"setup_building", "setup_confirmed", "entry_ready"}
    assert blocked_brain["holdReason"] in {
        "wait_for_setup_building",
        "wait_for_setup_confirmation",
        "wait_for_breakout_confirmation",
        "needs_regime_improvement",
        "market_regime_veto",
    }
    assert blocked_plan["watchlistPromotion"]["reviewedCount"] == 1
    assert blocked_plan["watchlistPromotion"]["transitions"][0]["reviewReason"] in {
        "wait_for_setup_building",
        "wait_for_setup_confirmation",
        "wait_for_breakout_confirmation",
        "needs_regime_improvement",
        "market_regime_veto",
    }
