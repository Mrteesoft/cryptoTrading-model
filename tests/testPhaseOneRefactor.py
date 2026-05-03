"""Focused regression tests for the Phase 1 model-platform refactor."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_signal_ml.config import TrainingConfig  # noqa: E402
from crypto_signal_ml.features_core import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_PACKS,
    audit_feature_registry_coverage,
    build_feature_registry_snapshot,
    get_feature_pack_columns,
    resolve_feature_group,
)
from crypto_signal_ml.labels_core import FutureReturnSignalLabeler, TripleBarrierSignalLabeler  # noqa: E402
from crypto_signal_ml.modeling import LogisticRegressionSignalModel  # noqa: E402
from crypto_signal_ml.modeling_core.artifacts import save_model_artifact_bundle  # noqa: E402
from crypto_signal_ml.pipeline import CryptoDatasetBuilder  # noqa: E402
from crypto_signal_ml.signal_generation import SignalContributionLedger  # noqa: E402
from crypto_signal_ml.trading.policy import evaluate_trading_decision  # noqa: E402


def test_feature_registry_covers_all_configured_features() -> None:
    """Every configured feature should be represented in the new registry."""

    coverage = audit_feature_registry_coverage(FEATURE_COLUMNS)
    core_feature_pack = get_feature_pack_columns("core")
    snapshot = build_feature_registry_snapshot(core_feature_pack)

    assert coverage["isComplete"] is True
    assert coverage["missingFeatures"] == []
    assert snapshot["featureCount"] == len(core_feature_pack)
    assert snapshot["features"][0]["name"] == core_feature_pack[0]


def test_feature_pack_columns_support_family_overrides() -> None:
    """Family includes and excludes should adjust the selected feature pack deterministically."""

    selected_columns = get_feature_pack_columns(
        "core",
        include_groups=("market_context",),
        exclude_groups=("regime",),
    )
    selected_groups = {resolve_feature_group(feature_name) for feature_name in selected_columns}

    assert "market_context" in selected_groups
    assert "regime" not in selected_groups
    assert "market_return_1" in selected_columns
    assert "regime_trend_score" not in selected_columns


def test_feature_pack_columns_reject_unknown_override_groups() -> None:
    """Unknown family overrides should fail fast with a clear error."""

    with pytest.raises(ValueError, match="Unknown feature include groups"):
        get_feature_pack_columns("all", include_groups=("not_a_group",))

    with pytest.raises(ValueError, match="Unknown feature exclude groups"):
        get_feature_pack_columns("all", exclude_groups=("not_a_group",))


def test_dataset_builder_rejects_empty_feature_family_selection() -> None:
    """Ablation config should fail loudly instead of silently reverting to all features."""

    with pytest.raises(ValueError, match="resolved to zero columns"):
        CryptoDatasetBuilder(
            TrainingConfig(
                feature_pack="core",
                feature_exclude_groups=tuple(FEATURE_PACKS["core"]),
            )
        )


def test_atr_triple_barrier_labeler_uses_dynamic_barriers() -> None:
    """ATR-aware triple barriers should widen the label thresholds when ATR is larger than the static rule."""

    feature_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "BTC-USD",
                "close": 100.0,
                "high": 101.0,
                "low": 99.0,
                "atr_pct_14": 0.03,
            },
            {
                "timestamp": "2026-01-01T01:00:00Z",
                "product_id": "BTC-USD",
                "close": 106.0,
                "high": 106.0,
                "low": 100.0,
                "atr_pct_14": 0.03,
            },
            {
                "timestamp": "2026-01-01T02:00:00Z",
                "product_id": "BTC-USD",
                "close": 104.0,
                "high": 105.0,
                "low": 103.0,
                "atr_pct_14": 0.03,
            },
        ]
    )

    labeler = TripleBarrierSignalLabeler(
        prediction_horizon=1,
        buy_threshold=0.01,
        sell_threshold=-0.015,
        use_atr_barriers=True,
        buy_atr_multiplier=1.5,
        sell_atr_multiplier=1.0,
    )
    labeled_df = labeler.add_labels(feature_df)

    assert labeled_df.loc[0, "label_recipe_version"] == "atr-triple-barrier-v1"
    assert labeled_df.loc[0, "label_upper_threshold"] == pytest.approx(0.045)
    assert labeled_df.loc[0, "target_signal"] == 1


def test_logistic_model_outputs_calibrated_and_raw_probabilities(tmp_path: Path) -> None:
    """The refactored model path should fit a calibrator and persist a full artifact bundle."""

    config = TrainingConfig(
        model_type="logisticRegressionSignalModel",
        calibration_enabled=True,
        calibration_holdout_fraction=0.20,
        comparison_min_trade_count=1,
    )
    model = LogisticRegressionSignalModel(config=config, feature_columns=["return_1"])

    train_df = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "return_1": -0.04, "target_signal": -1},
            {"timestamp": "2026-01-01T01:00:00Z", "return_1": -0.03, "target_signal": -1},
            {"timestamp": "2026-01-01T02:00:00Z", "return_1": 0.00, "target_signal": 0},
            {"timestamp": "2026-01-01T03:00:00Z", "return_1": 0.01, "target_signal": 0},
            {"timestamp": "2026-01-01T04:00:00Z", "return_1": 0.04, "target_signal": 1},
            {"timestamp": "2026-01-01T05:00:00Z", "return_1": 0.05, "target_signal": 1},
        ]
    )
    calibration_df = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T06:00:00Z", "return_1": -0.02, "target_signal": -1},
            {"timestamp": "2026-01-01T07:00:00Z", "return_1": 0.00, "target_signal": 0},
            {"timestamp": "2026-01-01T08:00:00Z", "return_1": 0.03, "target_signal": 1},
        ]
    )
    test_df = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T09:00:00Z", "return_1": -0.01, "target_signal": -1},
            {"timestamp": "2026-01-01T10:00:00Z", "return_1": 0.02, "target_signal": 1},
        ]
    )

    model.fit(train_df, calibration_df)
    prediction_df, metrics = model.evaluate(train_df, test_df)

    assert model.calibration_summary["enabled"] is True
    assert "raw_prob_buy" in prediction_df.columns
    assert "prob_buy" in prediction_df.columns
    assert "raw_confidence" in prediction_df.columns
    assert metrics["calibration"]["enabled"] is True

    artifact_dir = tmp_path / "logistic.artifact"
    manifest = save_model_artifact_bundle(
        artifact_dir=artifact_dir,
        model=model,
        metrics=metrics,
        prediction_df=prediction_df,
        label_recipe=FutureReturnSignalLabeler(
            prediction_horizon=2,
            buy_threshold=0.01,
            sell_threshold=-0.015,
        ).label_recipe(),
        dataset_path=tmp_path / "dataset.csv",
        train_df=train_df,
        test_df=test_df,
    )

    assert manifest.calibrationVersion == "sigmoid-v1"
    assert (artifact_dir / "manifest.json").exists()
    assert (artifact_dir / "feature_registry_snapshot.json").exists()
    assert (artifact_dir / "label_recipe.json").exists()


def test_signal_contribution_ledger_is_frozen_and_exposed() -> None:
    """Final signal decisions should expose a ledger and the ledger contract should be immutable."""

    ledger = SignalContributionLedger(
        rawProbabilities={"BUY": 0.7, "HOLD": 0.2, "TAKE_PROFIT": 0.1},
        calibratedProbabilities={"BUY": 0.6, "HOLD": 0.3, "TAKE_PROFIT": 0.1},
        rawConfidence=0.7,
        calibratedConfidence=0.6,
        probabilityMargin=0.3,
        policyStatus="passed",
        rejectionReasons=(),
        confidenceAdjustments={"calibrationDelta": -0.1, "requiredActionConfidence": 0.55},
        finalDecisionScore=0.9,
        publicationReason="passed_policy_gate",
    )
    with pytest.raises(FrozenInstanceError):
        ledger.policyStatus = "changed"  # type: ignore[misc]

    decision = evaluate_trading_decision(
        signal_row={
            "predicted_signal": 1,
            "predicted_name": "BUY",
            "confidence": 0.61,
            "prob_buy": 0.61,
            "prob_hold": 0.24,
            "prob_take_profit": 0.15,
            "raw_prob_buy": 0.66,
            "raw_prob_hold": 0.20,
            "raw_prob_take_profit": 0.14,
            "market_regime_label": "trend_down",
            "regime_is_high_volatility": 0,
            "cmcal_has_event_next_7d": 0,
        },
        minimum_action_confidence=0.55,
        config=TrainingConfig(decision_block_downtrend_buys=True),
    )

    assert decision["signalName"] == "HOLD"
    assert decision["ledger"]["policyStatus"] == "blocked"
    assert decision["ledger"]["publicationReason"] == "policy_downgraded_to_hold"
