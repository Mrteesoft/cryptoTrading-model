"""Very small smoke test for the dataset-building pipeline."""

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
from pandas import DataFrame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_signal_ml.config import TrainingConfig  # noqa: E402
from crypto_signal_ml.app import (  # noqa: E402
    MarketDataRefreshApp,
    MarketUniverseRefreshApp,
    SignalParameterTuningApp,
    WalkForwardValidationApp,
)
from crypto_signal_ml.backtesting import EqualWeightSignalBacktester  # noqa: E402
from crypto_signal_ml.data import CoinbaseExchangePriceDataLoader, CoinMarketCapContextEnricher, CsvPriceDataLoader  # noqa: E402
from crypto_signal_ml.environment import load_env_file  # noqa: E402
from crypto_signal_ml.features import TechnicalFeatureEngineer  # noqa: E402
from crypto_signal_ml.frontend import SignalSnapshotStore, build_frontend_signal_snapshot  # noqa: E402
from crypto_signal_ml.labels import FutureReturnSignalLabeler  # noqa: E402
from crypto_signal_ml.modeling import (  # noqa: E402
    BaseSignalModel,
    LogisticRegressionSignalModel,
    RandomForestSignalModel,
    create_model_from_config,
)
from crypto_signal_ml.pipeline import CryptoDatasetBuilder  # noqa: E402
from crypto_signal_ml.signals import build_actionable_signal_summaries, build_latest_signal_summaries  # noqa: E402


def _build_sample_market_frame(total_hours: int = 24) -> DataFrame:
    """
    Create a small but realistic multi-coin market dataset for tests.

    We generate enough candles per asset for rolling indicators like RSI(14)
    and for the 3-step future-return labels used by the default config.
    """

    rows = []
    market_start = pd.Timestamp("2026-01-01T00:00:00Z")
    market_specs = [
        ("BTC-USD", 100.0, 1.5, 10.0),
        ("ETH-USD", 200.0, 3.0, 20.0),
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
                    "open": open_price,
                    "high": close_price + 1.0,
                    "low": open_price - 1.0,
                    "close": close_price,
                    "volume": start_volume + hour_index,
                }
            )

    return pd.DataFrame(rows)


def _build_mixed_market_frame(total_hours: int = 48) -> DataFrame:
    """
    Create a multi-coin dataset with alternating up and down moves.

    The walk-forward tests need more than one target class, so this helper
    generates swings large enough to produce BUY, HOLD, and TAKE_PROFIT labels.
    """

    rows = []
    market_start = pd.Timestamp("2026-01-01T00:00:00Z")
    market_specs = [
        ("BTC-USD", 100.0, [4.0, -3.0, 5.0, -4.0, 2.0, -2.0], 10.0),
        ("ETH-USD", 200.0, [6.0, -5.0, 7.0, -6.0, 3.0, -3.0], 20.0),
    ]

    latest_close_by_product = {
        product_id: start_price
        for product_id, start_price, _, _ in market_specs
    }

    for hour_index in range(total_hours):
        timestamp = market_start + pd.Timedelta(hours=hour_index)

        for product_id, _, move_pattern, start_volume in market_specs:
            open_price = latest_close_by_product[product_id]
            close_price = max(1.0, open_price + move_pattern[hour_index % len(move_pattern)])

            rows.append(
                {
                    "timestamp": timestamp,
                    "product_id": product_id,
                    "open": open_price,
                    "high": max(open_price, close_price) + 1.0,
                    "low": min(open_price, close_price) - 1.0,
                    "close": close_price,
                    "volume": start_volume + hour_index,
                }
            )

            latest_close_by_product[product_id] = close_price

    return pd.DataFrame(rows)


def test_pipeline_builds_non_empty_dataset(tmp_path: Path) -> None:
    """A small raw CSV should produce rows and the expected feature columns."""

    raw_data_path = tmp_path / "marketPrices.csv"
    _build_sample_market_frame().to_csv(raw_data_path, index=False)

    dataset_builder = CryptoDatasetBuilder(TrainingConfig(data_file=raw_data_path))
    dataset, feature_columns = dataset_builder.build_labeled_dataset()

    assert isinstance(dataset, DataFrame)
    assert not dataset.empty
    assert len(feature_columns) > 0
    assert "target_signal" in dataset.columns


def test_config_can_create_logistic_regression_model() -> None:
    """The config model_type should map to the correct model subclass."""

    config = TrainingConfig(model_type="logisticRegressionSignalModel")
    model = create_model_from_config(config=config, feature_columns=["return_1"])

    assert isinstance(model, LogisticRegressionSignalModel)


def test_comparison_model_types_can_create_registered_models() -> None:
    """Every configured comparison model type should map to a concrete subclass."""

    config = TrainingConfig()
    created_models = [
        create_model_from_config(
            config=TrainingConfig(model_type=model_type),
            feature_columns=["return_1"],
        )
        for model_type in config.comparison_model_types
    ]

    assert any(isinstance(model, RandomForestSignalModel) for model in created_models)
    assert any(isinstance(model, LogisticRegressionSignalModel) for model in created_models)


def test_base_model_can_build_walk_forward_splits_without_timestamp_leakage() -> None:
    """Walk-forward folds should keep whole timestamps together in either train or test."""

    dataset = _build_sample_market_frame(total_hours=24)
    walk_forward_splits = BaseSignalModel.split_walk_forward_by_time(
        dataset=dataset,
        min_train_size=0.50,
        test_size=0.25,
        step_size=0.25,
    )

    assert len(walk_forward_splits) == 2
    assert walk_forward_splits[0]["train_end_timestamp"] < walk_forward_splits[0]["test_start_timestamp"]
    assert walk_forward_splits[0]["test_df"]["timestamp"].nunique() == 6
    assert len(walk_forward_splits[0]["test_df"]) == 12


def test_coinbase_loader_can_normalize_candle_rows() -> None:
    """The Coinbase loader should convert raw candle arrays into the project schema."""

    loader = CoinbaseExchangePriceDataLoader(
        data_path=Path("btcPrices.csv"),
        product_id="BTC-USD",
        granularity_seconds=3600,
        total_candles=2,
        should_save_downloaded_data=False,
    )

    normalized_rows = loader._normalize_candle_rows(
        product_id="BTC-USD",
        base_currency="BTC",
        quote_currency="USD",
        candle_rows=[
            [1700003600, 99.0, 105.0, 100.0, 104.0, 1250.0],
            [1700000000, 95.0, 101.0, 96.0, 100.0, 1180.0],
        ],
    )
    price_df = loader._build_price_frame(normalized_rows)

    assert list(price_df.columns[:6]) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(price_df) == 2
    assert price_df.iloc[0]["timestamp"] < price_df.iloc[1]["timestamp"]


def test_coinbase_loader_merges_batches_into_existing_market_file(tmp_path: Path) -> None:
    """Batch refresh mode should merge new products into the saved raw-data file."""

    data_path = tmp_path / "marketPrices.csv"
    existing_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "product_id": "BTC-USD",
            }
        ]
    )
    existing_df.to_csv(data_path, index=False)

    loader = CoinbaseExchangePriceDataLoader(
        data_path=data_path,
        product_id="ETH-USD",
        granularity_seconds=3600,
        total_candles=2,
        product_batch_size=25,
        should_save_downloaded_data=True,
    )

    new_batch_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.0,
                "volume": 20.0,
                "product_id": "ETH-USD",
            }
        ]
    )

    loader._save_downloaded_data(new_batch_df)
    merged_df = pd.read_csv(data_path)

    assert len(merged_df) == 2
    assert set(merged_df["product_id"]) == {"BTC-USD", "ETH-USD"}


def test_coinbase_loader_dedupes_existing_batch_rows_by_coin_and_timestamp(tmp_path: Path) -> None:
    """Refreshing the same batch twice should not duplicate saved candle rows."""

    data_path = tmp_path / "marketPrices.csv"
    existing_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "product_id": "BTC-USD",
            }
        ]
    )
    existing_df.to_csv(data_path, index=False)

    loader = CoinbaseExchangePriceDataLoader(
        data_path=data_path,
        product_id="BTC-USD",
        granularity_seconds=3600,
        total_candles=2,
        product_batch_size=25,
        should_save_downloaded_data=True,
    )

    new_batch_df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "product_id": "BTC-USD",
            }
        ]
    )

    loader._save_downloaded_data(new_batch_df)
    merged_df = pd.read_csv(data_path)

    assert len(merged_df) == 1


def test_coinbase_loader_reports_total_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader should report how many product batches the universe needs."""

    loader = CoinbaseExchangePriceDataLoader(
        data_path=Path("marketPrices.csv"),
        product_id="BTC-USD",
        granularity_seconds=3600,
        total_candles=2,
        product_batch_size=25,
        should_save_downloaded_data=False,
    )

    monkeypatch.setattr(
        loader,
        "_resolve_products_to_download",
        lambda: [{"product_id": f"COIN{i}-USD", "base_currency": f"COIN{i}", "quote_currency": "USD"} for i in range(52)],
    )

    assert loader.get_total_batches() == 3


def test_csv_loader_rebuilds_time_step_from_saved_raw_file(tmp_path: Path) -> None:
    """The loader should always rebuild time_step instead of trusting the raw CSV."""

    data_path = tmp_path / "marketPrices.csv"
    pd.DataFrame(
        [
            {
                "time_step": 99,
                "timestamp": "2026-01-01T01:00:00Z",
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 11.0,
                "product_id": "BTC-USD",
            },
            {
                "time_step": 42,
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
                "product_id": "BTC-USD",
            },
        ]
    ).to_csv(data_path, index=False)

    loaded_df = CsvPriceDataLoader(data_path).load()

    assert list(loaded_df["time_step"]) == [1, 2]


def test_feature_engineer_keeps_each_coin_history_separate() -> None:
    """Feature calculations should use each coin's own past candles only."""

    price_df = pd.DataFrame(
        [
            {"time_step": 1, "timestamp": "2026-01-01T00:00:00Z", "product_id": "BTC-USD", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0},
            {"time_step": 2, "timestamp": "2026-01-01T00:00:00Z", "product_id": "ETH-USD", "open": 200.0, "high": 202.0, "low": 198.0, "close": 200.0, "volume": 20.0},
            {"time_step": 3, "timestamp": "2026-01-01T01:00:00Z", "product_id": "BTC-USD", "open": 100.0, "high": 111.0, "low": 99.0, "close": 110.0, "volume": 12.0},
            {"time_step": 4, "timestamp": "2026-01-01T01:00:00Z", "product_id": "ETH-USD", "open": 200.0, "high": 222.0, "low": 199.0, "close": 220.0, "volume": 24.0},
        ]
    )

    feature_df = TechnicalFeatureEngineer().build(price_df)

    assert pd.isna(feature_df.iloc[0]["return_1"])
    assert pd.isna(feature_df.iloc[1]["return_1"])
    assert feature_df.iloc[2]["return_1"] == pytest.approx(0.10)
    assert feature_df.iloc[3]["return_1"] == pytest.approx(0.10)


def test_feature_engineer_adds_context_and_chart_pattern_columns() -> None:
    """The richer feature engineer should expose CMC and chart-pattern signals."""

    price_df = _build_sample_market_frame()
    price_df["base_currency"] = price_df["product_id"].str.split("-").str[0]
    price_df["cmc_context_available"] = 1
    price_df["cmc_rank"] = price_df["base_currency"].map({"BTC": 1, "ETH": 2})
    price_df["cmc_market_cap"] = price_df["base_currency"].map({"BTC": 2_000_000_000_000, "ETH": 400_000_000_000})
    price_df["cmc_volume_24h"] = price_df["base_currency"].map({"BTC": 50_000_000_000, "ETH": 30_000_000_000})
    price_df["cmc_percent_change_24h"] = 4.0
    price_df["cmc_percent_change_7d"] = 8.0
    price_df["cmc_percent_change_30d"] = 12.0
    price_df["cmc_circulating_supply"] = price_df["base_currency"].map({"BTC": 19_800_000, "ETH": 120_000_000})
    price_df["cmc_max_supply"] = price_df["base_currency"].map({"BTC": 21_000_000, "ETH": 0})
    price_df["cmc_num_market_pairs"] = 500
    price_df["cmc_tags_count"] = 6
    price_df["cmc_platform_present"] = price_df["base_currency"].map({"BTC": 0, "ETH": 1})
    price_df["cmc_is_mineable"] = price_df["base_currency"].map({"BTC": 1, "ETH": 0})
    price_df["cmc_has_defi_tag"] = 0
    price_df["cmc_has_ai_tag"] = 0
    price_df["cmc_has_layer1_tag"] = 1
    price_df["cmc_has_gaming_tag"] = 0
    price_df["cmc_has_meme_tag"] = 0

    feature_df = TechnicalFeatureEngineer().build(price_df)
    last_row = feature_df.iloc[-1]

    assert "breakout_up_20" in feature_df.columns
    assert "range_position_20" in feature_df.columns
    assert "cmc_market_cap_log" in feature_df.columns
    assert last_row["cmc_market_cap_log"] > 0
    assert pd.notna(last_row["breakout_up_20"])
    assert pd.notna(last_row["relative_strength_1"])


def test_coinmarketcap_context_enricher_can_refresh_and_merge_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CMC enricher should build a cached context snapshot and merge it back."""

    price_df = pd.DataFrame(
        [
            {"product_id": "BTC-USD", "base_currency": "BTC", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            {"product_id": "ETH-USD", "base_currency": "ETH", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        ]
    )

    monkeypatch.setenv("COINMARKETCAP_API_KEY", "test-key")
    enricher = CoinMarketCapContextEnricher(
        context_path=tmp_path / "coinMarketCapContext.csv",
        api_base_url="https://pro-api.coinmarketcap.com",
        api_key_env_var="COINMARKETCAP_API_KEY",
        should_refresh_context=True,
        log_progress=False,
    )

    def fake_request_json(endpoint_path: str, query_params: dict) -> dict:
        if endpoint_path == enricher.map_endpoint:
            return {
                "data": [
                    {"id": 1, "name": "Bitcoin", "symbol": "BTC", "slug": "bitcoin", "rank": 1},
                    {"id": 1027, "name": "Ethereum", "symbol": "ETH", "slug": "ethereum", "rank": 2},
                ]
            }

        if endpoint_path == enricher.info_endpoint:
            return {
                "data": {
                    "1": {"category": "coin", "tags": ["mineable", "layer-1"], "platform": None},
                    "1027": {"category": "smart-contracts", "tags": ["defi", "layer-1"], "platform": {"name": "Ethereum"}},
                }
            }

        if endpoint_path == enricher.latest_quotes_endpoint:
            return {
                "data": {
                    "1": {
                        "num_market_pairs": 12000,
                        "circulating_supply": 19_800_000,
                        "max_supply": 21_000_000,
                        "quote": {"USD": {"market_cap": 2_000_000_000_000, "volume_24h": 50_000_000_000, "percent_change_1h": 0.5, "percent_change_24h": 2.0, "percent_change_7d": 4.0, "percent_change_30d": 6.0}},
                    },
                    "1027": {
                        "num_market_pairs": 8000,
                        "circulating_supply": 120_000_000,
                        "max_supply": 0,
                        "quote": {"USD": {"market_cap": 400_000_000_000, "volume_24h": 30_000_000_000, "percent_change_1h": 0.3, "percent_change_24h": 3.0, "percent_change_7d": 7.0, "percent_change_30d": 12.0}},
                    },
                }
            }

        raise AssertionError(f"Unexpected endpoint: {endpoint_path}")

    monkeypatch.setattr(enricher, "_request_json", fake_request_json)

    context_df = enricher.refresh_context(price_df)
    enriched_df = enricher.enrich(price_df)

    assert len(context_df) == 2
    assert int(context_df["cmc_context_available"].sum()) == 2
    assert "cmc_market_cap" in enriched_df.columns
    assert float(enriched_df.loc[enriched_df["product_id"] == "BTC-USD", "cmc_market_cap"].iloc[0]) > 0
    assert int(enriched_df.loc[enriched_df["product_id"] == "ETH-USD", "cmc_has_defi_tag"].iloc[0]) == 1


def test_coinmarketcap_context_refresh_merges_existing_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refreshing one batch of CMC context should not erase earlier cached coins."""

    context_path = tmp_path / "coinMarketCapContext.csv"
    pd.DataFrame(
        [
            {
                "product_id": "BTC-USD",
                "base_currency": "BTC",
                "cmc_context_available": 1,
                "cmc_market_cap": 2_000_000_000_000,
            }
        ]
    ).to_csv(context_path, index=False)

    price_df = pd.DataFrame(
        [
            {"product_id": "ETH-USD", "base_currency": "ETH", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        ]
    )

    monkeypatch.setenv("COINMARKETCAP_API_KEY", "test-key")
    enricher = CoinMarketCapContextEnricher(
        context_path=context_path,
        api_base_url="https://pro-api.coinmarketcap.com",
        api_key_env_var="COINMARKETCAP_API_KEY",
        should_refresh_context=True,
        log_progress=False,
    )

    def fake_request_json(endpoint_path: str, query_params: dict) -> dict:
        if endpoint_path == enricher.map_endpoint:
            return {"data": [{"id": 1027, "name": "Ethereum", "symbol": "ETH", "slug": "ethereum", "rank": 2}]}
        if endpoint_path == enricher.info_endpoint:
            return {"data": {"1027": {"category": "smart-contracts", "tags": ["defi"], "platform": {"name": "Ethereum"}}}}
        if endpoint_path == enricher.latest_quotes_endpoint:
            return {
                "data": {
                    "1027": {
                        "num_market_pairs": 8000,
                        "circulating_supply": 120_000_000,
                        "max_supply": 0,
                        "quote": {"USD": {"market_cap": 400_000_000_000, "volume_24h": 30_000_000_000, "percent_change_1h": 0.3, "percent_change_24h": 3.0, "percent_change_7d": 7.0, "percent_change_30d": 12.0}},
                    }
                }
            }
        raise AssertionError(f"Unexpected endpoint: {endpoint_path}")

    monkeypatch.setattr(enricher, "_request_json", fake_request_json)

    final_context_df = enricher.refresh_context(price_df)

    assert len(final_context_df) == 2
    assert set(final_context_df["product_id"]) == {"BTC-USD", "ETH-USD"}


def test_coinmarketcap_context_enricher_falls_back_to_single_symbol_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad bulk symbol-map request should fall back to single-symbol retries."""

    monkeypatch.setenv("COINMARKETCAP_API_KEY", "test-key")
    enricher = CoinMarketCapContextEnricher(
        context_path=tmp_path / "coinMarketCapContext.csv",
        api_base_url="https://pro-api.coinmarketcap.com",
        api_key_env_var="COINMARKETCAP_API_KEY",
        should_refresh_context=True,
        log_progress=False,
    )
    enricher.map_request_symbol_batch_size = 2

    symbol_ids = {"BTC": 1, "ETH": 1027, "SOL": 5426}
    request_symbols = []

    def fake_request_json(endpoint_path: str, query_params: dict) -> dict:
        assert endpoint_path == enricher.map_endpoint

        requested_symbols = str(query_params["symbol"])
        request_symbols.append(requested_symbols)

        if "," in requested_symbols:
            raise ValueError("bad request")

        return {
            "data": [
                {
                    "id": symbol_ids[requested_symbols],
                    "name": requested_symbols,
                    "symbol": requested_symbols,
                    "slug": requested_symbols.lower(),
                    "rank": 1,
                }
            ]
        }

    monkeypatch.setattr(enricher, "_request_json", fake_request_json)

    map_lookup = enricher._fetch_symbol_map_lookup(["BTC", "ETH", "SOL"])

    assert set(map_lookup) == {"BTC", "ETH", "SOL"}
    assert any("," in request_value for request_value in request_symbols)
    assert "BTC" in request_symbols
    assert "ETH" in request_symbols
    assert "SOL" in request_symbols


def test_coinmarketcap_context_enricher_chunks_keyed_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keyed CMC requests should be chunked so large id sets stay request-safe."""

    monkeypatch.setenv("COINMARKETCAP_API_KEY", "test-key")
    enricher = CoinMarketCapContextEnricher(
        context_path=tmp_path / "coinMarketCapContext.csv",
        api_base_url="https://pro-api.coinmarketcap.com",
        api_key_env_var="COINMARKETCAP_API_KEY",
        should_refresh_context=True,
        log_progress=False,
    )
    enricher.keyed_lookup_batch_size = 2

    requested_id_groups = []

    def fake_request_json(endpoint_path: str, query_params: dict) -> dict:
        assert endpoint_path == enricher.info_endpoint

        requested_ids = str(query_params["id"])
        requested_id_groups.append(requested_ids)

        return {
            "data": {
                requested_id: {"category": "coin", "tags": []}
                for requested_id in requested_ids.split(",")
            }
        }

    monkeypatch.setattr(enricher, "_request_json", fake_request_json)

    keyed_lookup = enricher._fetch_keyed_lookup(
        endpoint_path=enricher.info_endpoint,
        ids=[1, 2, 3],
    )

    assert set(keyed_lookup) == {"1", "2", "3"}
    assert requested_id_groups == ["1,2", "3"]


def test_labeler_shifts_future_returns_within_each_coin() -> None:
    """Future labels should not jump from one coin to another."""

    feature_df = pd.DataFrame(
        [
            {"time_step": 1, "timestamp": "2026-01-01T00:00:00Z", "product_id": "BTC-USD", "close": 100.0},
            {"time_step": 2, "timestamp": "2026-01-01T00:00:00Z", "product_id": "ETH-USD", "close": 200.0},
            {"time_step": 3, "timestamp": "2026-01-01T01:00:00Z", "product_id": "BTC-USD", "close": 110.0},
            {"time_step": 4, "timestamp": "2026-01-01T01:00:00Z", "product_id": "ETH-USD", "close": 220.0},
        ]
    )

    labeled_df = FutureReturnSignalLabeler(
        prediction_horizon=1,
        buy_threshold=0.05,
        sell_threshold=-0.05,
    ).add_labels(feature_df)

    assert labeled_df.iloc[0]["future_return"] == pytest.approx(0.10)
    assert labeled_df.iloc[1]["future_return"] == pytest.approx(0.10)
    assert labeled_df.iloc[0]["target_name"] == "BUY"
    assert labeled_df.iloc[1]["target_name"] == "BUY"
    assert pd.isna(labeled_df.iloc[2]["future_return"])
    assert pd.isna(labeled_df.iloc[3]["future_return"])


def test_market_data_refresh_app_reports_context_failure_without_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CMC refresh problem should not erase a successful market-data refresh."""

    config = TrainingConfig(
        data_file=tmp_path / "marketPrices.csv",
        coinmarketcap_context_file=tmp_path / "coinMarketCapContext.csv",
    )
    refresh_app = MarketDataRefreshApp(config=config)

    class FakeLoader:
        last_refresh_summary = {"rowsDownloaded": 2}

        def refresh_data(self) -> pd.DataFrame:
            return _build_sample_market_frame(total_hours=2)

    class FailingEnricher:
        def refresh_context(self, price_df: pd.DataFrame) -> pd.DataFrame:
            raise RuntimeError("simulated CoinMarketCap failure")

    monkeypatch.setattr(refresh_app, "build_market_data_loader", lambda: FakeLoader())
    monkeypatch.setattr(
        refresh_app,
        "build_coinmarketcap_context_enricher",
        lambda should_refresh_context=False: FailingEnricher(),
    )

    result = refresh_app.run()

    assert result["rowsDownloaded"] > 0
    assert result["coinMarketCapContextStatus"] == "refresh_failed"
    assert "simulated CoinMarketCap failure" in result["coinMarketCapContextError"]


def test_market_universe_refresh_app_continues_after_batch_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed batch should be reported while later batches can still continue."""

    data_path = tmp_path / "marketPrices.csv"
    context_path = tmp_path / "coinMarketCapContext.csv"
    _build_sample_market_frame().to_csv(data_path, index=False)
    pd.DataFrame([{"product_id": "BTC-USD", "base_currency": "BTC"}]).to_csv(context_path, index=False)

    refresh_app = MarketUniverseRefreshApp(
        config=TrainingConfig(
            data_file=data_path,
            coinmarketcap_context_file=context_path,
        )
    )

    class FakeLoader:
        def get_available_products(self) -> list[dict[str, str]]:
            return [
                {"product_id": f"COIN{index}-USD", "base_currency": f"COIN{index}", "quote_currency": "USD"}
                for index in range(75)
            ]

        def get_total_batches(self) -> int:
            return 3

    def fake_batch_run(self: MarketDataRefreshApp) -> dict:
        batch_number = self.config.coinbase_product_batch_number
        if batch_number == 2:
            raise RuntimeError("simulated batch failure")

        return {
            "marketDataSource": "coinbaseExchange",
            "coinMarketCapContextStatus": "refreshed",
            "rowsDownloaded": 10,
            "uniqueProducts": 5,
        }

    monkeypatch.setattr(refresh_app, "build_market_data_loader", lambda: FakeLoader())
    monkeypatch.setattr(MarketDataRefreshApp, "run", fake_batch_run)

    result = refresh_app.run()

    assert result["failedBatches"] == [2]
    assert result["successfulBatches"] == [1, 3]
    assert len(result["batchResults"]) == 3


def test_walk_forward_validation_app_builds_out_of_sample_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The walk-forward app should export fold results and an out-of-sample summary."""

    raw_data_path = tmp_path / "marketPrices.csv"
    _build_mixed_market_frame().to_csv(raw_data_path, index=False)

    config = TrainingConfig(
        data_file=raw_data_path,
        coinmarketcap_use_context=False,
        model_type="logisticRegressionSignalModel",
        walkforward_min_train_size=0.50,
        walkforward_test_size=0.25,
        walkforward_step_size=0.25,
        logistic_max_iter=200,
    )
    validation_app = WalkForwardValidationApp(config=config)

    monkeypatch.setattr(validation_app, "save_dataframe", lambda dataframe, file_path: None)
    monkeypatch.setattr(validation_app, "save_json", lambda payload, file_path: None)

    result = validation_app.run()

    assert result["foldCount"] == 2
    assert result["outOfSampleRows"] > 0
    assert result["walkForwardSummaryPath"].endswith("walkForwardSummary.json")
    assert result["walkForwardBacktestSummaryPath"].endswith("walkForwardBacktestSummary.json")


def test_signal_parameter_tuning_app_returns_best_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tuning app should evaluate candidate settings and return a winner."""

    raw_data_path = tmp_path / "marketPrices.csv"
    _build_mixed_market_frame().to_csv(raw_data_path, index=False)

    config = TrainingConfig(
        data_file=raw_data_path,
        coinmarketcap_use_context=False,
        model_type="logisticRegressionSignalModel",
        walkforward_min_train_size=0.50,
        walkforward_test_size=0.25,
        walkforward_step_size=0.25,
        tuning_prediction_horizon_candidates=(2, 3),
        tuning_buy_threshold_candidates=(0.01, 0.015),
        tuning_backtest_confidence_candidates=(0.0, 0.55),
        logistic_max_iter=200,
    )
    tuning_app = SignalParameterTuningApp(config=config)

    monkeypatch.setattr(tuning_app, "save_dataframe", lambda dataframe, file_path: None)
    monkeypatch.setattr(tuning_app, "save_json", lambda payload, file_path: None)

    result = tuning_app.run()

    assert result["bestPredictionHorizon"] in {2, 3}
    assert result["bestBuyThreshold"] in {0.01, 0.015}
    assert result["bestBacktestMinConfidence"] in {0.0, 0.55}
    assert result["confidenceResultsPath"].endswith("signalConfidenceTuningResults.csv")


def test_signal_summaries_explain_buy_and_take_profit_actions() -> None:
    """Signal summaries should expose spot actions and human-readable reasons."""

    prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "BTC-USD",
                "base_currency": "BTC",
                "quote_currency": "USD",
                "close": 110.0,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.82,
                "prob_take_profit": 0.08,
                "prob_hold": 0.10,
                "prob_buy": 0.82,
                "breakout_up_20": 0.03,
                "close_vs_ema_5": 0.02,
                "relative_strength_1": 0.01,
                "relative_strength_5": 0.03,
                "momentum_10": 0.08,
                "range_position_20": 0.91,
                "rsi_14": 62.0,
                "cmc_context_available": 1,
                "cmc_percent_change_24h": 0.05,
                "cmc_percent_change_7d": 0.12,
                "cmc_percent_change_30d": 0.18,
                "cmc_has_ai_tag": 1,
                "cmc_has_defi_tag": 0,
                "cmc_has_layer1_tag": 0,
                "cmc_has_gaming_tag": 0,
                "cmc_has_meme_tag": 0,
                "cmc_name": "Bitcoin",
                "cmc_category": "coin",
            },
            {
                "time_step": 2,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "ETH-USD",
                "base_currency": "ETH",
                "quote_currency": "USD",
                "close": 200.0,
                "predicted_signal": -1,
                "predicted_name": "TAKE_PROFIT",
                "confidence": 0.77,
                "prob_take_profit": 0.77,
                "prob_hold": 0.15,
                "prob_buy": 0.08,
                "breakout_up_20": -0.01,
                "breakout_down_20": -0.02,
                "close_vs_ema_5": -0.03,
                "relative_strength_1": -0.02,
                "relative_strength_5": -0.04,
                "momentum_10": -0.06,
                "range_position_20": 0.18,
                "rsi_14": 74.0,
                "cmc_context_available": 1,
                "cmc_percent_change_24h": -0.04,
                "cmc_percent_change_7d": 0.06,
                "cmc_percent_change_30d": 0.09,
                "cmc_has_ai_tag": 0,
                "cmc_has_defi_tag": 1,
                "cmc_has_layer1_tag": 0,
                "cmc_has_gaming_tag": 0,
                "cmc_has_meme_tag": 0,
                "cmc_name": "Ethereum",
                "cmc_category": "smart-contracts",
            },
        ]
    )

    latest_signals = build_latest_signal_summaries(prediction_df)
    actionable_signals = build_actionable_signal_summaries(latest_signals)

    assert latest_signals[0]["signal_name"] == "BUY"
    assert latest_signals[0]["symbol"] == "BTC"
    assert latest_signals[0]["coinSymbol"] == "BTC"
    assert latest_signals[0]["pairSymbol"] == "BTC-USD"
    assert latest_signals[0]["spotAction"] == "buy"
    assert "BUY setup" in latest_signals[0]["signalChat"]
    assert actionable_signals[1]["signal_name"] == "TAKE_PROFIT"
    assert actionable_signals[1]["symbol"] == "ETH"
    assert actionable_signals[1]["spotAction"] == "take_profit"
    assert "spot take-profit signal" in actionable_signals[1]["signalChat"]


def test_frontend_snapshot_store_serves_cached_signal_views(tmp_path: Path) -> None:
    """The frontend snapshot store should return overview, filters, and product detail quickly."""

    latest_signals = [
        {
            "productId": "BTC-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.82,
            "setupScore": 4.0,
            "signalChat": "BTC-USD is a BUY setup.",
        },
        {
            "productId": "ETH-USD",
            "signal_name": "TAKE_PROFIT",
            "spotAction": "take_profit",
            "confidence": 0.71,
            "setupScore": 3.0,
            "signalChat": "ETH-USD is a TAKE_PROFIT setup for spot trading.",
        },
        {
            "productId": "SOL-USD",
            "signal_name": "HOLD",
            "spotAction": "wait",
            "confidence": 0.66,
            "setupScore": 0.0,
            "signalChat": "SOL-USD is a HOLD setup.",
        },
    ]
    actionable_signals = latest_signals[:2]
    snapshot_payload = build_frontend_signal_snapshot(
        model_type="randomForestSignalModel",
        primary_signal=latest_signals[0],
        latest_signals=latest_signals,
        actionable_signals=actionable_signals,
    )

    snapshot_path = tmp_path / "frontendSignalSnapshot.json"
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    snapshot_store = SignalSnapshotStore(snapshot_path)
    overview = snapshot_store.get_overview()
    buy_signals = snapshot_store.list_signals(action="buy")
    actionable_list = snapshot_store.list_signals(action="actionable")
    eth_signal = snapshot_store.get_signal_by_product("eth-usd")

    assert overview["marketSummary"]["signalCounts"]["buy"] == 1
    assert overview["marketSummary"]["signalCounts"]["take_profit"] == 1
    assert overview["marketSummary"]["signalCounts"]["wait"] == 1
    assert len(buy_signals) == 1
    assert buy_signals[0]["productId"] == "BTC-USD"
    assert len(actionable_list) == 2
    assert eth_signal is not None
    assert eth_signal["signal_name"] == "TAKE_PROFIT"


def test_backtester_builds_strategy_summary() -> None:
    """The spot backtester should only open trades on BUY signals."""

    prediction_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "BTC-USD",
                "future_return": 0.04,
                "predicted_signal": 1,
                "confidence": 0.80,
            },
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "ETH-USD",
                "future_return": -0.02,
                "predicted_signal": -1,
                "confidence": 0.70,
            },
            {
                "timestamp": "2026-01-01T01:00:00Z",
                "product_id": "SOL-USD",
                "future_return": 0.01,
                "predicted_signal": 0,
                "confidence": 0.55,
            },
        ]
    )

    backtester = EqualWeightSignalBacktester(
        TrainingConfig(
            backtest_initial_capital=10000.0,
            backtest_trading_fee_rate=0.001,
            backtest_slippage_rate=0.0005,
            backtest_min_confidence=0.0,
            backtest_max_positions_per_timestamp=2,
        )
    )
    result = backtester.run(prediction_df)

    assert not result["trade_df"].empty
    assert not result["period_df"].empty
    assert result["summary"]["tradeCount"] == 1
    assert "strategyTotalReturn" in result["summary"]


def test_load_env_file_reads_simple_key_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The local .env loader should populate os.environ for missing keys."""

    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment line\n"
        "COINMARKETCAP_API_KEY=test-value\n"
        "export EXTRA_FLAG=\"enabled\"\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("COINMARKETCAP_API_KEY", raising=False)
    monkeypatch.delenv("EXTRA_FLAG", raising=False)

    loaded_values = load_env_file(env_path)

    assert loaded_values["COINMARKETCAP_API_KEY"] == "test-value"
    assert loaded_values["EXTRA_FLAG"] == "enabled"
