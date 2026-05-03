"""Focused tests for multi-timeframe context and regime enrichment."""

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_signal_ml.config import TrainingConfig  # noqa: E402
from crypto_signal_ml.app import RegimeTrainingApp, RegimeWalkForwardValidationApp  # noqa: E402
from crypto_signal_ml.data import align_multi_timeframe_context  # noqa: E402
from crypto_signal_ml.labels import MarketRegimeLabeler  # noqa: E402
from crypto_signal_ml.pipeline import CryptoDatasetBuilder  # noqa: E402
from crypto_signal_ml.regimes import MarketRegimeDetector  # noqa: E402


def _build_hourly_market_frame(total_hours: int = 72) -> pd.DataFrame:
    """Create a small hourly multi-coin dataset for multi-timeframe tests."""

    rows = []
    market_start = pd.Timestamp("2026-02-01T00:00:00Z")
    market_specs = [
        ("BTC-USD", 100.0, 1.0, 10.0),
        ("ETH-USD", 200.0, 2.0, 20.0),
    ]

    for hour_index in range(total_hours):
        timestamp = market_start + pd.Timedelta(hours=hour_index)
        for product_id, start_price, price_step, start_volume in market_specs:
            open_price = start_price + (hour_index * price_step)
            close_price = open_price + (price_step * 0.5)
            rows.append(
                {
                    "timestamp": timestamp,
                    "product_id": product_id,
                    "base_currency": product_id.split("-")[0],
                    "quote_currency": "USD",
                    "open": open_price,
                    "high": close_price + 1.0,
                    "low": open_price - 1.0,
                    "close": close_price,
                    "volume": start_volume + hour_index,
                    "granularity_seconds": 3600,
                }
            )

    return pd.DataFrame(rows)


def test_multi_timeframe_alignment_uses_completed_four_hour_bars() -> None:
    """A base row should only see the most recent completed 4h candle."""

    base_df = _build_hourly_market_frame(total_hours=8)
    btc_df = base_df[base_df["product_id"] == "BTC-USD"].reset_index(drop=True)

    aligned_df = align_multi_timeframe_context(
        price_df=btc_df,
        timeframes=("4h",),
        base_granularity_seconds=3600,
    )

    assert aligned_df.loc[0:2, "htf_4h_close"].isna().all()
    assert aligned_df.loc[3, "htf_4h_close"] == btc_df.loc[3, "close"]
    assert aligned_df.loc[4, "htf_4h_close"] == btc_df.loc[3, "close"]
    assert aligned_df.loc[7, "htf_4h_close"] == btc_df.loc[7, "close"]


def test_market_regime_detector_adds_combined_market_state_columns() -> None:
    """The regime detector should classify both trend and volatility state."""

    feature_df = pd.DataFrame(
        [
            {
                "close_vs_ema_20": 0.030,
                "trend_acceleration_5_20": 0.015,
                "htf_4h_close_vs_ema_3": 0.020,
                "htf_1d_close_vs_ema_3": 0.010,
                "volatility_5": 0.060,
                "volatility_20": 0.030,
            },
            {
                "close_vs_ema_20": 0.001,
                "trend_acceleration_5_20": 0.000,
                "htf_4h_close_vs_ema_3": 0.002,
                "htf_1d_close_vs_ema_3": 0.001,
                "volatility_5": 0.020,
                "volatility_20": 0.030,
            },
        ]
    )

    enriched_df = MarketRegimeDetector(TrainingConfig()).enrich_feature_table(feature_df)

    assert "market_regime_label" in enriched_df.columns
    assert "market_regime_code" in enriched_df.columns
    assert enriched_df.loc[0, "market_regime_label"] == "trend_up_high_volatility"
    assert enriched_df.loc[0, "regime_is_trending"] == 1.0
    assert enriched_df.loc[0, "regime_is_high_volatility"] == 1.0
    assert enriched_df.loc[1, "market_regime_label"] == "range"


def test_dataset_builder_exposes_multi_timeframe_and_regime_columns(tmp_path: Path) -> None:
    """The dataset builder should wire aligned timeframe context into the feature table."""

    raw_data_path = tmp_path / "marketPrices.csv"
    _build_hourly_market_frame(total_hours=72).to_csv(raw_data_path, index=False)

    feature_df = CryptoDatasetBuilder(
        TrainingConfig(
            data_file=raw_data_path,
            coinmarketcap_use_context=False,
        )
    ).build_feature_table()

    assert "htf_4h_close_vs_ema_3" in feature_df.columns
    assert "htf_1d_close_vs_sma_3" in feature_df.columns
    assert "market_regime_label" in feature_df.columns
    assert "market_regime_code" in feature_df.columns


def test_market_regime_labeler_creates_future_targets_without_asset_leakage() -> None:
    """Future regime labels should shift within each asset instead of across assets."""

    feature_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-02-01T00:00:00Z",
                "product_id": "BTC-USD",
                "market_regime_label": "range",
                "market_regime_code": 1.0,
            },
            {
                "timestamp": "2026-02-01T01:00:00Z",
                "product_id": "BTC-USD",
                "market_regime_label": "trend_up",
                "market_regime_code": 2.0,
            },
            {
                "timestamp": "2026-02-01T00:00:00Z",
                "product_id": "ETH-USD",
                "market_regime_label": "trend_down",
                "market_regime_code": 3.0,
            },
            {
                "timestamp": "2026-02-01T01:00:00Z",
                "product_id": "ETH-USD",
                "market_regime_label": "range_high_volatility",
                "market_regime_code": 4.0,
            },
        ]
    )

    labeled_df = MarketRegimeLabeler(prediction_horizon=1).add_labels(feature_df)

    assert labeled_df.loc[0, "target_market_regime_label"] == "trend_up"
    assert labeled_df.loc[0, "target_market_regime_code"] == 2.0
    assert labeled_df.loc[0, "market_regime_changed"] == 1.0
    assert labeled_df.loc[2, "target_market_regime_label"] == "range_high_volatility"
    assert labeled_df.loc[2, "target_market_regime_code"] == 4.0
    assert pd.isna(labeled_df.loc[1, "target_market_regime_label"])
    assert pd.isna(labeled_df.loc[3, "target_market_regime_label"])


def test_regime_training_app_writes_dedicated_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regime training flow should save its own artifact bundle and metadata."""

    raw_data_path = tmp_path / "marketPrices.csv"
    _build_hourly_market_frame(total_hours=12).to_csv(raw_data_path, index=False)

    import crypto_signal_ml.app as app_module  # noqa: WPS433

    monkeypatch.setattr(app_module, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(app_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(app_module, "OUTPUTS_DIR", tmp_path / "outputs")

    feature_df = pd.DataFrame(
        [
            {
                "timestamp": f"2026-02-01T0{index}:00:00Z",
                "product_id": "BTC-USD",
                "return_1": float(index) * 0.01,
                "momentum_3": float(index) * 0.015,
                "market_regime_label": regime_label,
                "market_regime_code": regime_code,
            }
            for index, (regime_label, regime_code) in enumerate(
                [
                    ("range", 1.0),
                    ("trend_up", 2.0),
                    ("trend_up", 2.0),
                    ("range_high_volatility", 4.0),
                    ("trend_down", 3.0),
                    ("trend_down_high_volatility", 5.0),
                    ("range", 1.0),
                    ("trend_up_high_volatility", 6.0),
                ]
            )
        ]
    )

    class StubDatasetBuilder:
        feature_columns = ["return_1", "momentum_3"]

        def build_feature_table(self) -> pd.DataFrame:
            return feature_df

    app = RegimeTrainingApp(
        config=TrainingConfig(
            data_file=raw_data_path,
            coinmarketcap_use_context=False,
            model_type="randomForestSignalModel",
            n_estimators=12,
            max_depth=3,
            random_state=7,
        ),
        dataset_builder=StubDatasetBuilder(),
    )

    result = app.run()
    metadata_path = Path(result["metadataPath"])
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert result["modelType"] == "marketRegimeModel"
    assert result["estimatorType"] == "randomForestSignalModel"
    assert metadata_payload["artifactType"] == "market_regime_model"
    assert metadata_payload["target"]["targetColumn"] == "target_market_regime_code"
    assert metadata_payload["target"]["predictionHorizon"] == 2


def test_regime_walk_forward_validation_app_builds_out_of_sample_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regime walk-forward app should export fold results and a summary."""

    raw_data_path = tmp_path / "marketPrices.csv"
    _build_hourly_market_frame(total_hours=12).to_csv(raw_data_path, index=False)

    feature_df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-02-01T00:00:00Z") + pd.Timedelta(hours=index),
                "product_id": "BTC-USD",
                "return_1": [0.00, 0.02, 0.03, -0.01, -0.02, 0.01, 0.04, -0.03, 0.02, -0.01, 0.01, 0.03][index],
                "momentum_3": [0.01, 0.03, 0.04, -0.02, -0.03, 0.02, 0.05, -0.04, 0.03, -0.02, 0.02, 0.04][index],
                "market_regime_label": regime_label,
                "market_regime_code": regime_code,
            }
            for index, (regime_label, regime_code) in enumerate(
                [
                    ("range", 1.0),
                    ("trend_up", 2.0),
                    ("trend_up", 2.0),
                    ("range_high_volatility", 4.0),
                    ("trend_down", 3.0),
                    ("trend_down_high_volatility", 5.0),
                    ("range", 1.0),
                    ("trend_up_high_volatility", 6.0),
                    ("range", 1.0),
                    ("trend_down", 3.0),
                    ("trend_up", 2.0),
                    ("range_high_volatility", 4.0),
                ]
            )
        ]
    )

    class StubDatasetBuilder:
        feature_columns = ["return_1", "momentum_3"]

        def build_feature_table(self) -> pd.DataFrame:
            return feature_df

    validation_app = RegimeWalkForwardValidationApp(
        config=TrainingConfig(
            data_file=raw_data_path,
            coinmarketcap_use_context=False,
            model_type="logisticRegressionSignalModel",
            walkforward_min_train_size=0.50,
            walkforward_test_size=0.25,
            walkforward_step_size=0.25,
            logistic_max_iter=200,
        ),
        dataset_builder=StubDatasetBuilder(),
    )

    monkeypatch.setattr(validation_app, "save_dataframe", lambda dataframe, file_path: None)
    monkeypatch.setattr(validation_app, "save_json", lambda payload, file_path: None)

    result = validation_app.run()

    assert result["modelType"] == "marketRegimeModel"
    assert result["foldCount"] == 2
    assert result["outOfSampleRows"] > 0
    assert Path(result["runDirectory"]).parent.name == "regimeWalkForwardRuns"
    assert result["walkForwardSummaryPath"].endswith("regimeWalkForwardSummary.json")
