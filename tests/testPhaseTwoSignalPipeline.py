"""Focused regression tests for the Phase 2 staged signal pipeline."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_signal_ml.application.signal_context_enrichment import SignalContextEnrichmentStage  # noqa: E402
from crypto_signal_ml.application.signal_decision import SignalDecisionStage  # noqa: E402
from crypto_signal_ml.application.signal_enrichment import SignalEnrichmentStage  # noqa: E402
from crypto_signal_ml.application.signal_generation import SignalGenerationCoordinator  # noqa: E402
from crypto_signal_ml.application.signal_inference import SignalInferenceStage  # noqa: E402
from crypto_signal_ml.application.signal_publication import SignalPublicationStage  # noqa: E402
from crypto_signal_ml.config import TrainingConfig  # noqa: E402
from crypto_signal_ml.portfolio_core import PortfolioActionMapper, TraderBrain, build_market_context  # noqa: E402
from crypto_signal_ml.portfolio_core.chart_confirmation import review_chart_confirmation  # noqa: E402
from crypto_signal_ml.signal_generation import (  # noqa: E402
    SignalFusionStage,
    actionable_signal_to_summary,
    context_enriched_candidate_from_summary,
    scored_candidate_to_summary,
)
from crypto_signal_ml.signal_generation.publication import decide_publication  # noqa: E402
from crypto_signal_ml.trading.portfolio import TradingPortfolioStore  # noqa: E402
from crypto_signal_ml.trading.signal_store import TradingSignalStore  # noqa: E402
from crypto_signal_ml.trading.signals import apply_signal_trade_context  # noqa: E402
from crypto_signal_ml.trading.signals import build_latest_signal_summaries as legacy_build_latest_signal_summaries  # noqa: E402


def _build_base_signal_summary() -> dict[str, object]:
    """Build one representative signal summary for staged-pipeline tests."""

    return {
        "time_step": 1,
        "timestamp": "2026-01-01T00:00:00Z",
        "close": 100.0,
        "predicted_signal": 1,
        "signal_name": "BUY",
        "spotAction": "buy",
        "actionable": True,
        "symbol": "BTC",
        "coinSymbol": "BTC",
        "pairSymbol": "BTC-USD",
        "productId": "BTC-USD",
        "baseCurrency": "BTC",
        "quoteCurrency": "USD",
        "coinName": "Bitcoin",
        "coinCategory": "Layer-1",
        "confidence": 0.68,
        "rawConfidence": 0.72,
        "minimumActionConfidence": 0.55,
        "requiredActionConfidence": 0.55,
        "confidenceGateApplied": False,
        "riskGateApplied": False,
        "modelSignalName": "BUY",
        "modelPredictedSignal": 1,
        "modelSpotAction": "buy",
        "primaryProbability": 0.68,
        "probabilityMargin": 0.22,
        "tradeReadiness": "high",
        "policyScore": 0.92,
        "policyNotes": ["Trend regime is aligned with the long setup."],
        "gateReasons": [],
        "setupScore": 4.4,
        "probabilities": {
            "buy": 0.68,
            "hold": 0.20,
            "take_profit": 0.12,
        },
        "rawProbabilities": {
            "buy": 0.72,
            "hold": 0.18,
            "take_profit": 0.10,
        },
        "reasonItems": ["Base breakout support."],
        "reasonSummary": "Base breakout support.",
        "signalChat": "BTC-USD is a BUY setup. Base breakout support.",
        "chartContext": {
            "structureLabel": "higher_highs",
            "breakoutConfirmed": True,
            "retestHoldConfirmed": True,
            "nearResistance": False,
            "trendSlope": 0.004,
            "patternScore": 0.50,
            "channelPosition": 0.70,
            "rangeCompressionScore": 0.15,
            "resistanceDistancePct": 0.04,
        },
        "chartConfirmationScore": 0.62,
        "chartSetupType": "retest_confirmed",
        "chartConfirmationStatus": "confirmed",
        "chartDecision": "confirmed",
        "chartPatternLabel": "retest_confirmed",
        "chartConfirmationNotes": [
            "breakout_confirmed",
            "retest_hold_confirmed",
        ],
        "chartPatternReasons": [
            "breakout_confirmed",
            "retest_hold_confirmed",
        ],
        "executionContext": {
            "atrPct14": 0.02,
            "volumeVsSma20": 1.4,
            "volumeZscore20": 1.2,
            "cmcVolume24hLog": 12.0,
            "cmcNumMarketPairsLog": 3.0,
            "cmcRankScore": 0.80,
        },
        "marketContext": {
            "cmcPercentChange24h": 0.04,
            "cmcPercentChange7d": 0.12,
            "cmcPercentChange30d": 0.20,
            "cmcContextAvailable": 1,
            "themeTags": ["Layer-1"],
            "marketIntelligence": {
                "available": True,
                "fearGreedValue": 72.0,
                "fearGreedClassification": "Greed",
                "btcDominance": 0.52,
                "btcDominanceChange24h": -0.01,
                "riskMode": "risk_on",
            },
        },
        "marketState": {
            "label": "trend_up",
            "code": 1,
            "trendScore": 0.60,
            "volatilityRatio": 1.10,
            "isTrending": True,
            "isHighVolatility": False,
        },
        "eventContext": {
            "eventCountNext7d": 0,
            "eventCountNext30d": 1,
            "hasEventNext7d": False,
            "daysToNextEvent": 14.0,
        },
    }


def _build_prediction_frame() -> pd.DataFrame:
    """Build one small prediction frame for inference-stage regression tests."""

    return pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "BTC-USD",
                "base_currency": "BTC",
                "quote_currency": "USD",
                "time_step": 1,
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
                "close": 100.0,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.68,
                "raw_confidence": 0.72,
                "prob_buy": 0.68,
                "prob_hold": 0.20,
                "prob_take_profit": 0.12,
                "raw_prob_buy": 0.72,
                "raw_prob_hold": 0.18,
                "raw_prob_take_profit": 0.10,
                "breakout_up_20": 0.03,
                "breakout_down_20": 0.00,
                "range_position_20": 0.84,
                "close_vs_ema_5": 0.02,
                "relative_strength_1": 0.01,
                "relative_strength_5": 0.03,
                "momentum_10": 0.04,
                "rsi_14": 61.0,
                "atr_pct_14": 0.02,
                "volume_vs_sma_20": 1.3,
                "volume_zscore_20": 1.1,
                "cmc_volume_24h_log": 12.1,
                "cmc_num_market_pairs_log": 3.0,
                "cmc_rank_score": 0.82,
                "cmc_percent_change_24h": 0.04,
                "cmc_percent_change_7d": 0.12,
                "cmc_percent_change_30d": 0.22,
                "cmc_context_available": 1,
                "cmc_name": "Bitcoin",
                "cmc_category": "Layer-1",
                "cmc_market_intelligence_available": 1,
                "cmc_market_fear_greed_value": 70.0,
                "cmc_market_fear_greed_classification": "Greed",
                "cmc_market_fear_greed_score": 0.70,
                "cmc_market_btc_dominance": 0.52,
                "cmc_market_btc_dominance_change_24h": -0.01,
                "market_regime_label": "trend_up",
                "market_regime_code": 1,
                "regime_trend_score": 0.60,
                "regime_volatility_ratio": 1.05,
                "regime_is_trending": 1,
                "regime_is_high_volatility": 0,
                "cmcal_event_count_next_7d": 0,
                "cmcal_event_count_next_30d": 1,
                "cmcal_has_event_next_7d": 0,
                "cmcal_days_to_next_event": 10.0,
            },
            {
                "timestamp": "2026-01-01T01:00:00Z",
                "product_id": "BTC-USD",
                "base_currency": "BTC",
                "quote_currency": "USD",
                "time_step": 2,
                "open": 100.0,
                "high": 103.0,
                "low": 99.0,
                "close": 102.0,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.71,
                "raw_confidence": 0.75,
                "prob_buy": 0.71,
                "prob_hold": 0.18,
                "prob_take_profit": 0.11,
                "raw_prob_buy": 0.75,
                "raw_prob_hold": 0.16,
                "raw_prob_take_profit": 0.09,
                "breakout_up_20": 0.04,
                "breakout_down_20": 0.00,
                "range_position_20": 0.88,
                "close_vs_ema_5": 0.025,
                "relative_strength_1": 0.015,
                "relative_strength_5": 0.035,
                "momentum_10": 0.05,
                "rsi_14": 63.0,
                "atr_pct_14": 0.021,
                "volume_vs_sma_20": 1.4,
                "volume_zscore_20": 1.3,
                "cmc_volume_24h_log": 12.2,
                "cmc_num_market_pairs_log": 3.0,
                "cmc_rank_score": 0.84,
                "cmc_percent_change_24h": 0.05,
                "cmc_percent_change_7d": 0.13,
                "cmc_percent_change_30d": 0.25,
                "cmc_context_available": 1,
                "cmc_name": "Bitcoin",
                "cmc_category": "Layer-1",
                "cmc_market_intelligence_available": 1,
                "cmc_market_fear_greed_value": 72.0,
                "cmc_market_fear_greed_classification": "Greed",
                "cmc_market_fear_greed_score": 0.72,
                "cmc_market_btc_dominance": 0.51,
                "cmc_market_btc_dominance_change_24h": -0.015,
                "market_regime_label": "trend_up",
                "market_regime_code": 1,
                "regime_trend_score": 0.64,
                "regime_volatility_ratio": 1.06,
                "regime_is_trending": 1,
                "regime_is_high_volatility": 0,
                "cmcal_event_count_next_7d": 0,
                "cmcal_event_count_next_30d": 1,
                "cmcal_has_event_next_7d": 0,
                "cmcal_days_to_next_event": 9.0,
            },
        ]
    )


def _build_real_pipeline_prediction_frame() -> pd.DataFrame:
    """Build one multi-asset historical frame for coordinator-level integration tests."""

    rows: list[dict[str, object]] = []

    def add_product_history(
        *,
        product_id: str,
        base_currency: str,
        coin_name: str,
        coin_category: str,
        closes: list[float],
        highs: list[float],
        lows: list[float],
        signal_names: list[str],
        confidences: list[float],
        buy_probabilities: list[float],
        regime_label: str = "trend_up",
    ) -> None:
        for time_step, close in enumerate(closes, start=1):
            signal_name = signal_names[time_step - 1]
            confidence = confidences[time_step - 1]
            prob_buy = buy_probabilities[time_step - 1]
            predicted_signal = {"BUY": 1, "HOLD": 0, "TAKE_PROFIT": 2}.get(signal_name, 0)
            prob_hold = 0.68 if signal_name == "HOLD" else max(0.10, 1.0 - prob_buy - 0.08)
            prob_take_profit = (
                max(0.0, 1.0 - prob_hold - prob_buy)
                if signal_name == "HOLD"
                else 0.08
            )
            timestamp = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(hours=time_step - 1)

            rows.append(
                {
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                    "product_id": product_id,
                    "base_currency": base_currency,
                    "quote_currency": "USD",
                    "time_step": time_step,
                    "open": close - 1.0,
                    "high": highs[time_step - 1],
                    "low": lows[time_step - 1],
                    "close": close,
                    "predicted_signal": predicted_signal,
                    "predicted_name": signal_name,
                    "confidence": confidence,
                    "raw_confidence": min(confidence + 0.04, 0.99),
                    "prob_buy": prob_buy,
                    "prob_hold": prob_hold,
                    "prob_take_profit": prob_take_profit,
                    "raw_prob_buy": min(prob_buy + 0.04, 0.99),
                    "raw_prob_hold": max(prob_hold - 0.02, 0.0),
                    "raw_prob_take_profit": prob_take_profit,
                    "breakout_up_20": 0.05 if signal_name == "BUY" else 0.01,
                    "breakout_down_20": 0.0,
                    "range_position_20": 0.88 if signal_name == "BUY" else 0.58,
                    "close_vs_ema_5": 0.03 if signal_name == "BUY" else 0.01,
                    "relative_strength_1": 0.02 if signal_name == "BUY" else 0.004,
                    "relative_strength_5": 0.05 if signal_name == "BUY" else 0.012,
                    "momentum_10": 0.06 if signal_name == "BUY" else 0.015,
                    "rsi_14": 64.0 if signal_name == "BUY" else 53.0,
                    "atr_pct_14": 0.019 if signal_name == "BUY" else 0.021,
                    "volume_vs_sma_20": 1.5 if signal_name == "BUY" else 1.0,
                    "volume_zscore_20": 1.4 if signal_name == "BUY" else 0.2,
                    "cmc_volume_24h_log": 12.4 if signal_name == "BUY" else 11.8,
                    "cmc_num_market_pairs_log": 3.2 if signal_name == "BUY" else 3.0,
                    "cmc_rank_score": 0.88 if signal_name == "BUY" else 0.76,
                    "cmc_percent_change_24h": 0.05 if signal_name == "BUY" else 0.02,
                    "cmc_percent_change_7d": 0.14 if signal_name == "BUY" else 0.07,
                    "cmc_percent_change_30d": 0.28 if signal_name == "BUY" else 0.10,
                    "cmc_context_available": 1,
                    "cmc_name": coin_name,
                    "cmc_category": coin_category,
                    "cmc_market_intelligence_available": 1,
                    "cmc_market_fear_greed_value": 72.0,
                    "cmc_market_fear_greed_classification": "Greed",
                    "cmc_market_fear_greed_score": 0.72,
                    "cmc_market_btc_dominance": 0.51,
                    "cmc_market_btc_dominance_change_24h": -0.015,
                    "market_regime_label": regime_label,
                    "market_regime_code": 1,
                    "regime_trend_score": 0.64 if signal_name == "BUY" else 0.52,
                    "regime_volatility_ratio": 1.04 if signal_name == "BUY" else 1.02,
                    "regime_is_trending": 1,
                    "regime_is_high_volatility": 0,
                    "cmcal_event_count_next_7d": 0,
                    "cmcal_event_count_next_30d": 1,
                    "cmcal_has_event_next_7d": 0,
                    "cmcal_days_to_next_event": 9.0,
                }
            )

    add_product_history(
        product_id="BTC-USD",
        base_currency="BTC",
        coin_name="Bitcoin",
        coin_category="Layer-1",
        closes=[96.0, 97.4, 98.6, 100.1, 102.8],
        highs=[97.0, 98.2, 99.2, 101.3, 104.0],
        lows=[95.0, 96.8, 97.9, 99.0, 101.0],
        signal_names=["HOLD", "HOLD", "BUY", "BUY", "BUY"],
        confidences=[0.54, 0.58, 0.66, 0.73, 0.79],
        buy_probabilities=[0.22, 0.28, 0.66, 0.73, 0.79],
    )
    add_product_history(
        product_id="ETH-USD",
        base_currency="ETH",
        coin_name="Ethereum",
        coin_category="Layer-1",
        closes=[3200.0, 3212.0, 3208.0, 3215.0, 3220.0],
        highs=[3218.0, 3220.0, 3216.0, 3222.0, 3224.0],
        lows=[3190.0, 3202.0, 3198.0, 3206.0, 3210.0],
        signal_names=["HOLD", "HOLD", "HOLD", "HOLD", "HOLD"],
        confidences=[0.48, 0.50, 0.54, 0.57, 0.58],
        buy_probabilities=[0.18, 0.20, 0.24, 0.26, 0.28],
    )
    return pd.DataFrame(rows)


def test_context_and_fusion_stage_keep_model_probabilities_stable() -> None:
    """Context enrichment and late fusion should add evidence without overwriting model probabilities."""

    config = TrainingConfig()
    signal_summary = _build_base_signal_summary()
    context_stage = SignalContextEnrichmentStage(config=config)
    context_artifacts = context_stage.enrich([signal_summary])

    assert len(context_artifacts.signal_summaries) == 1
    enriched_summary = context_artifacts.signal_summaries[0]
    assert "newsContext" in enriched_summary
    assert "trendContext" in enriched_summary

    market_context = build_market_context(context_artifacts.signal_summaries)
    scored_candidate = SignalFusionStage(config=config).score_candidate(
        context_enriched_candidate_from_summary(enriched_summary),
        market_context=market_context,
        trade_memory={"available": False},
    )

    assert scored_candidate.calibratedProbabilities["BUY"] == pytest.approx(0.68)
    assert scored_candidate.rawProbabilities["BUY"] == pytest.approx(0.72)
    assert "contextAlignmentScore" in scored_candidate.confidenceCalibration
    assert scored_candidate.ledger.finalDecisionScore == pytest.approx(scored_candidate.finalDecisionScore)
    assert "contextAdjustment" in scored_candidate.ledger.confidenceAdjustments


def test_portfolio_chart_confirmation_review_returns_typed_result() -> None:
    """Portfolio planning should consume one typed chart-confirmation result instead of raw chart flags."""

    signal_summary = _build_base_signal_summary()
    result = review_chart_confirmation(
        signal_summary=signal_summary,
        confidence_calibration={"chartAlignmentScore": 0.32},
    )

    assert result.decision == "confirmed"
    assert result.confirmed is True
    assert result.early is False
    assert result.blocked is False
    assert result.breakout_confirmed is True
    assert result.retest_hold_confirmed is True
    assert result.pattern_label == "retest_confirmed"
    assert "chart_confirmed" in result.reasons


def test_portfolio_action_mapper_preserves_fusion_outputs(tmp_path: Path) -> None:
    """Portfolio action mapping should keep the fusion outputs available to the trader-brain layer."""

    def normalize_plan(payload: object) -> object:
        if isinstance(payload, dict):
            return {
                key: normalize_plan(value)
                for key, value in payload.items()
                if key
                not in {
                    "at",
                    "firstSeenAt",
                    "generatedAt",
                    "lastReviewedAt",
                    "lastStageChangeAt",
                    "updatedAt",
                }
            }
        if isinstance(payload, list):
            return [normalize_plan(item) for item in payload]
        return payload

    config = TrainingConfig(signal_watchlist_state_path=tmp_path / "mapper.watchlist.json")
    direct_config = TrainingConfig(signal_watchlist_state_path=tmp_path / "direct.watchlist.json")
    signal_summary = _build_base_signal_summary()
    context_stage = SignalContextEnrichmentStage(config=config)
    context_summary = context_stage.enrich([signal_summary]).signal_summaries[0]
    market_context = build_market_context([context_summary])
    scored_candidate = SignalFusionStage(config=config).score_candidate(
        context_enriched_candidate_from_summary(context_summary),
        market_context=market_context,
        trade_memory={"available": False},
    )

    portfolio_store = TradingPortfolioStore(
        db_path=tmp_path / "portfolio.sqlite3",
        default_capital=10000.0,
    )
    action_artifacts = PortfolioActionMapper(config=config).plan_actions(
        [scored_candidate],
        portfolio_store=portfolio_store,
    )

    assert len(action_artifacts.actionable_signals) == 1
    actionable_summary = actionable_signal_to_summary(action_artifacts.actionable_signals[0])
    assert "brain" in actionable_summary
    assert "confidenceCalibration" in actionable_summary
    assert "adaptiveContext" in actionable_summary
    assert "decisionScore" in actionable_summary["brain"]

    scored_summary = scored_candidate_to_summary(scored_candidate)
    trade_enriched_signals = apply_signal_trade_context(
        [scored_summary],
        active_trade_product_ids=portfolio_store.get_active_signal_product_ids(),
        active_signal_context_by_product={},
        config=direct_config,
    )
    portfolio = portfolio_store.get_portfolio()
    direct_plan = TraderBrain(config=direct_config).build_plan(
        signal_summaries=trade_enriched_signals,
        positions=list(portfolio.get("positions", [])),
        capital=float(portfolio.get("capital") or 0.0),
        trade_memory_by_product=portfolio_store.build_trade_learning_map(trade_enriched_signals),
    )

    assert normalize_plan(action_artifacts.trader_brain_plan) == normalize_plan(direct_plan)


def test_legacy_trader_brain_import_paths_are_removed() -> None:
    """Trader-brain planning should only be importable from portfolio_core."""

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("crypto_signal_ml.trader_brain")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("crypto_signal_ml.trading.trader_brain")


def test_trader_brain_uses_shared_market_context_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trader brain should source market stance from the shared portfolio-core builder."""

    import crypto_signal_ml.portfolio_core.trader_brain as trader_brain_module  # noqa: WPS433

    original_builder = trader_brain_module.build_market_context
    recorded_product_ids: list[str] = []

    def instrumented_builder(signal_summaries: list[dict[str, object]]) -> dict[str, object]:
        recorded_product_ids.extend(
            str(signal_summary.get("productId", "")).strip().upper()
            for signal_summary in signal_summaries
        )
        market_context = original_builder(signal_summaries)
        market_context["marketStance"] = "defensive"
        return market_context

    monkeypatch.setattr(trader_brain_module, "build_market_context", instrumented_builder)

    plan = TraderBrain(config=TrainingConfig()).build_plan(
        signal_summaries=[_build_base_signal_summary()],
        capital=10000.0,
    )

    assert recorded_product_ids == ["BTC-USD"]
    assert plan["marketStance"] == "defensive"


def test_publication_can_emit_watchlist_fallback_signal() -> None:
    """The new publication helper should surface one watchlist fallback when the public feed is empty."""

    signal_summary = _build_base_signal_summary()
    signal_summary.update(
        {
            "signal_name": "HOLD",
            "spotAction": "wait",
            "actionable": False,
            "brain": {
                "decision": "watchlist",
                "decisionScore": 0.71,
                "summaryLine": "Watchlist candidate stayed strong enough for a fallback.",
            },
            "tradeContext": {
                "hasActiveTrade": False,
            },
        }
    )

    selection = decide_publication(
        signal_summaries=[signal_summary],
        config=TrainingConfig(),
        allow_watchlist_fallback=True,
        last_primary_signal_at=None,
    )

    assert len(selection.published_signals) == 1
    published_signal = selection.published_signals[0]
    assert published_signal["watchlistFallback"] is True
    assert published_signal["publicationReason"] == "watchlist_fallback"
    assert published_signal["publicSignalType"] == "watchlist"


def test_signal_inference_stage_matches_legacy_summary_shape() -> None:
    """The application inference stage should keep the legacy summary shape while using staged gating."""

    prediction_df = _build_prediction_frame()
    config = TrainingConfig(chart_snapshot_enabled=False)

    staged_artifacts = SignalInferenceStage(config).build_from_prediction_frame(
        prediction_df,
        protected_product_ids=["BTC-USD"],
    )
    legacy_summaries = legacy_build_latest_signal_summaries(
        prediction_df,
        minimum_action_confidence=config.backtest_min_confidence,
        config=config,
        protected_product_ids=["BTC-USD"],
    )

    assert len(staged_artifacts.signal_candidates) == len(legacy_summaries) == 1
    staged_summary = staged_artifacts.signal_candidates[0]
    legacy_summary = legacy_summaries[0]

    for field_name in (
        "productId",
        "signal_name",
        "modelSignalName",
        "spotAction",
        "confidence",
        "rawConfidence",
        "probabilityMargin",
        "publicationReason",
        "tradeReadiness",
    ):
        if isinstance(staged_summary[field_name], float):
            assert staged_summary[field_name] == pytest.approx(legacy_summary[field_name])
        else:
            assert staged_summary[field_name] == legacy_summary[field_name]

    assert staged_summary["reasonItems"] == legacy_summary["reasonItems"]
    assert staged_summary["reasonSummary"] == legacy_summary["reasonSummary"]
    assert staged_summary["chartContext"]["breakoutUp20"] == pytest.approx(legacy_summary["chartContext"]["breakoutUp20"])


def test_signal_inference_stage_emits_chart_confirmation_fields() -> None:
    """Inference should expose explicit chart-confirmation outputs before later stages run."""

    prediction_df = _build_prediction_frame()
    config = TrainingConfig(chart_snapshot_enabled=False)

    staged_artifacts = SignalInferenceStage(config).build_from_prediction_frame(
        prediction_df,
        protected_product_ids=["BTC-USD"],
    )

    signal_summary = staged_artifacts.signal_candidates[0]
    assert signal_summary["chartDecision"] in {"confirmed", "early", "blocked"}
    assert signal_summary["chartConfirmationStatus"] in {"confirmed", "early", "blocked"}
    assert isinstance(signal_summary["chartPatternLabel"], str)
    assert isinstance(signal_summary["chartPatternReasons"], list)
    assert isinstance(signal_summary["chartSetupType"], str)
    assert isinstance(signal_summary["chartConfirmationNotes"], list)
    assert -1.0 <= signal_summary["chartConfirmationScore"] <= 1.0


def test_trader_brain_keeps_early_chart_confirmation_on_watchlist(tmp_path: Path) -> None:
    """Fresh BUY candidates should stay on watchlist until chart confirmation is explicit."""

    signal_summary = _build_base_signal_summary()
    signal_summary["chartDecision"] = "early"
    signal_summary["chartConfirmationStatus"] = "early"
    signal_summary["chartPatternLabel"] = "breakout_attempt"
    signal_summary["chartSetupType"] = "breakout_attempt"
    signal_summary["chartConfirmationScore"] = 0.18
    signal_summary["chartConfirmationNotes"] = ["range_compression_support"]
    signal_summary["chartPatternReasons"] = ["range_compression_support"]
    signal_summary["chartContext"]["breakoutConfirmed"] = False
    signal_summary["chartContext"]["retestHoldConfirmed"] = False
    signal_summary["chartContext"]["structureLabel"] = "higher_lows"

    plan = TraderBrain(
        config=TrainingConfig(signal_watchlist_state_path=tmp_path / "watchlist.json")
    ).build_plan(
        signal_summaries=[signal_summary],
        capital=10000.0,
    )

    first_signal = plan["signals"][0]
    assert first_signal["brain"]["decision"] == "watchlist"
    assert first_signal["brain"]["chartDecision"] == "early"
    assert first_signal["brain"]["chartConfirmationStatus"] == "early"
    assert first_signal["brain"]["chartPatternLabel"] == "breakout_attempt"
    assert first_signal["brain"]["holdReason"] in {
        "needs_chart_confirmation",
        "wait_for_setup_building",
        "wait_for_setup_confirmation",
    }


def test_signal_generation_coordinator_runs_real_pipeline_and_publication_cycle(
    tmp_path: Path,
) -> None:
    """Run the explicit multi-stage pipeline end-to-end on a realistic multi-asset frame."""

    prediction_df = _build_real_pipeline_prediction_frame()
    config = TrainingConfig(
        chart_snapshot_enabled=False,
        coinmarketcap_use_context=False,
        signal_excluded_base_currencies=(),
        signal_watchlist_state_path=tmp_path / "watchlistState.json",
        signal_watchlist_pool_path=tmp_path / "watchlistPool.json",
        signal_store_path=tmp_path / "liveSignals.sqlite3",
        portfolio_store_path=tmp_path / "portfolio.sqlite3",
        signal_watchlist_min_published_signals=2,
    )

    portfolio_store = TradingPortfolioStore(
        db_path=config.portfolio_store_path,
        default_capital=10000.0,
    )
    inference_stage = SignalInferenceStage(config)
    inference_artifacts = inference_stage.build_from_prediction_frame(
        prediction_df,
        protected_product_ids=["BTC-USD", "ETH-USD"],
    )

    saved_json_payloads: dict[str, dict[str, object]] = {}
    saved_dataframes: dict[str, pd.DataFrame] = {}
    publication_stage = SignalPublicationStage(
        config=config,
        save_json=lambda payload, file_path: saved_json_payloads.__setitem__(str(file_path.name), dict(payload)),
        save_dataframe=lambda dataframe, file_path: saved_dataframes.__setitem__(str(file_path.name), dataframe.copy()),
        signal_store_factory=lambda: TradingSignalStore(db_path=config.signal_store_path),
    )
    coordinator = SignalGenerationCoordinator(
        inference_stage=inference_stage,
        context_stage=SignalContextEnrichmentStage(config),
        enrichment_stage=SignalEnrichmentStage(config),
        decision_stage=SignalDecisionStage(config, allow_watchlist_supplement=True),
        publication_stage=publication_stage,
    )

    pipeline_artifacts = coordinator.run_pipeline(
        inference_artifacts=inference_artifacts,
        portfolio_store=portfolio_store,
        save_watchlist_pool_snapshot=True,
    )

    assert len(pipeline_artifacts.inference.signal_candidates) == 2
    assert pipeline_artifacts.context is not None
    assert len(pipeline_artifacts.context.signal_summaries) == 2
    assert len(pipeline_artifacts.enrichment.signal_summaries) == 2
    assert len(pipeline_artifacts.decision.published_signals) == 1
    assert len(pipeline_artifacts.decision.actionable_signals) == 1
    assert pipeline_artifacts.decision.primary_signal is not None
    assert pipeline_artifacts.decision.primary_signal["productId"] == "BTC-USD"

    signal_summaries_by_product = {
        str(signal_summary.get("productId")): signal_summary
        for signal_summary in pipeline_artifacts.decision.signal_summaries
    }
    assert signal_summaries_by_product["BTC-USD"]["signal_name"] == "BUY"
    assert signal_summaries_by_product["BTC-USD"]["brain"]["decision"] == "enter_long"
    assert signal_summaries_by_product["ETH-USD"]["brain"]["decision"] == "watchlist"
    assert signal_summaries_by_product["ETH-USD"]["publicationReason"] == "standby_signal"

    published_signals_by_product = {
        str(signal_summary.get("productId")): signal_summary
        for signal_summary in pipeline_artifacts.decision.published_signals
    }
    assert published_signals_by_product["BTC-USD"]["publicationReason"] == "passed_policy_gate"
    assert "ETH-USD" not in published_signals_by_product

    publication_artifacts = coordinator.publish_signal_generation(
        model_type="integrationTestModel",
        historical_prediction_df=prediction_df,
        pipeline_artifacts=pipeline_artifacts,
        signal_source="integration-test",
        signal_metadata={
            "marketDataFirstTimestamp": "2026-01-01T00:00:00Z",
            "marketDataLastTimestamp": "2026-01-01T04:00:00Z",
        },
        market_data_refresh={"sourceStatus": "test"},
        market_data_refreshed_at="2026-01-01T04:00:00Z",
        portfolio_store=portfolio_store,
    )

    assert publication_artifacts.primary_signal is not None
    assert publication_artifacts.primary_signal["productId"] == "BTC-USD"
    assert publication_artifacts.signal_store_summary["signalCount"] == 1
    assert publication_artifacts.signal_store_summary["actionableCount"] == 1
    assert publication_artifacts.frontend_snapshot["primarySignal"]["productId"] == "BTC-USD"
    assert publication_artifacts.frontend_snapshot["marketSummary"]["totalSignals"] == 1
    assert publication_artifacts.frontend_snapshot["marketSummary"]["actionableSignals"] == 1

    assert "historicalSignals.csv" in saved_dataframes
    assert "watchlistPool.json" in saved_json_payloads
    assert "latestSignals.json" in saved_json_payloads
    assert "frontendSignalSnapshot.json" in saved_json_payloads
    assert saved_json_payloads["watchlistPool.json"]["productIds"] == ["ETH-USD"]
    assert saved_json_payloads["latestSignals.json"]["signals"][0]["productId"] == "BTC-USD"
    assert len(saved_json_payloads["latestSignals.json"]["signals"]) == 1
    assert saved_json_payloads["frontendSignalSnapshot.json"]["signalInference"]["rowsScored"] == len(prediction_df)

    persisted_signal_store = TradingSignalStore(db_path=config.signal_store_path)
    persisted_current_signals = persisted_signal_store.list_current_signals(limit=5)
    assert len(persisted_current_signals) == 1
    assert persisted_current_signals[0]["productId"] == "BTC-USD"
