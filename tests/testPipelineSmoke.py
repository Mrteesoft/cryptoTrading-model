"""Very small smoke test for the dataset-building pipeline."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd
import pytest
from pandas import DataFrame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_signal_ml.config import TrainingConfig, apply_runtime_market_data_settings  # noqa: E402
from crypto_signal_ml.application import (  # noqa: E402
    MarketEventsRefreshApp,
    MarketDataRefreshApp,
    MarketUniverseRefreshApp,
    SignalGenerationApp,
    SignalParameterTuningApp,
    TrainingApp,
    WalkForwardValidationApp,
)
from crypto_signal_ml.chat import ConversationSessionStore  # noqa: E402
from crypto_signal_ml.backtesting import EqualWeightSignalBacktester  # noqa: E402
from crypto_signal_ml.environment import load_env_file  # noqa: E402
from crypto_signal_ml.frontend import (  # noqa: E402
    SignalSnapshotStore,
    WatchlistPoolStore,
    build_frontend_signal_snapshot,
    build_watchlist_pool_snapshot,
)
from crypto_signal_ml.live import LiveSignalEngine  # noqa: E402
from crypto_signal_ml.ml import (  # noqa: E402
    BaseSignalModel,
    BinancePublicDataPriceDataLoader,
    CoinbaseExchangePriceDataLoader,
    CoinMarketCalEventEnricher,
    CoinMarketCapRateLimitError,
    CoinMarketCapContextEnricher,
    CoinMarketCapLatestQuotesPriceDataLoader,
    CoinMarketCapMarketIntelligenceEnricher,
    CoinMarketCapOhlcvPriceDataLoader,
    CryptoDatasetBuilder,
    CsvPriceDataLoader,
    FutureReturnSignalLabeler,
    HistGradientBoostingSignalModel,
    KrakenOhlcPriceDataLoader,
    LogisticRegressionSignalModel,
    RandomForestSignalModel,
    TechnicalFeatureEngineer,
    TripleBarrierSignalLabeler,
    create_market_data_loader,
    create_model_from_config,
)
from crypto_signal_ml.portfolio_core import TraderBrain  # noqa: E402
from crypto_signal_ml.retrieval import RagKnowledgeStore  # noqa: E402
from crypto_signal_ml.service import SignalMonitorService  # noqa: E402
from crypto_signal_ml.source_refresh import (  # noqa: E402
    ActiveUniversePlan,
    CoinMarketCapUniverseRefreshService,
    SignalUniverseCoordinator,
)
from crypto_signal_ml.trading import (  # noqa: E402
    TradingPortfolioStore,
    TradingSignalStore,
    WatchlistStateStore,
    apply_signal_trade_context,
    build_actionable_signal_summaries,
    build_latest_signal_summaries,
    filter_published_signal_summaries,
    is_signal_eligible_base_currency,
    is_signal_product_excluded,
    select_primary_signal,
)


def _build_sample_market_frame(total_hours: int = 96) -> DataFrame:
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


def _build_mixed_market_frame(total_hours: int = 120) -> DataFrame:
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


def test_config_can_read_signal_exclusions_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Signal exclusions should be configurable from one comma-separated env var."""

    monkeypatch.setenv("SIGNAL_EXCLUDED_BASE_CURRENCIES", "btc, eth, usdt, usdc, btc")

    config = TrainingConfig()

    assert config.signal_excluded_base_currencies == ("BTC", "ETH", "USDT", "USDC")


def test_default_signal_exclusions_block_btc_eth_and_common_stablecoins() -> None:
    """Default signal settings should keep majors and stablecoins out of surfaced signals."""

    config = TrainingConfig()

    assert is_signal_product_excluded(product_id="BTC-USD", config=config) is True
    assert is_signal_product_excluded(product_id="ETH-USD", config=config) is True
    assert is_signal_product_excluded(product_id="USDT-USD", config=config) is True
    assert is_signal_product_excluded(product_id="DAI-USD", config=config) is True
    assert is_signal_product_excluded(product_id="FDUSD-USD", config=config) is True
    assert is_signal_product_excluded(product_id="SOL-USD", config=config) is False


def test_config_can_create_hist_gradient_boosting_model() -> None:
    """The config model_type should map to the gradient-boosting subclass."""

    config = TrainingConfig(model_type="histGradientBoostingSignalModel")
    model = create_model_from_config(config=config, feature_columns=["return_1"])

    assert isinstance(model, HistGradientBoostingSignalModel)


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

    assert any(isinstance(model, HistGradientBoostingSignalModel) for model in created_models)
    assert any(isinstance(model, RandomForestSignalModel) for model in created_models)
    assert any(isinstance(model, LogisticRegressionSignalModel) for model in created_models)


def test_recency_weighting_prioritizes_newer_training_rows() -> None:
    """Recency weighting should make newer candles count more than older ones."""

    model = HistGradientBoostingSignalModel(
        config=TrainingConfig(
            recency_weighting_enabled=True,
            recency_weighting_halflife_hours=24.0,
        ),
        feature_columns=["return_1"],
    )
    train_df = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "return_1": 0.01, "target_signal": 0},
            {"timestamp": "2026-01-03T00:00:00Z", "return_1": 0.02, "target_signal": 1},
        ]
    )

    sample_weight = model._build_sample_weight(train_df)

    assert sample_weight is not None
    assert sample_weight[1] > sample_weight[0]


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


def test_walk_forward_splits_can_purge_recent_training_timestamps() -> None:
    """The purge gap should remove the timestamps nearest the test block."""

    dataset = _build_sample_market_frame(total_hours=24)
    walk_forward_splits = BaseSignalModel.split_walk_forward_by_time(
        dataset=dataset,
        min_train_size=0.50,
        test_size=0.25,
        step_size=0.25,
        purge_gap_timestamps=2,
    )

    first_split = walk_forward_splits[0]

    assert first_split["purgeGapTimestamps"] == 2
    assert first_split["purgedTimestampCount"] == 2
    assert first_split["train_df"]["timestamp"].nunique() == 10
    assert first_split["train_df"]["timestamp"].max() < first_split["test_df"]["timestamp"].min()


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


def test_coinmarketcap_loader_can_normalize_historical_quote_rows() -> None:
    """The CoinMarketCap loader should normalize nested historical OHLCV rows."""

    loader = CoinMarketCapOhlcvPriceDataLoader(
        data_path=Path("marketPrices.csv"),
        api_base_url="https://pro-api.coinmarketcap.com",
        api_key_env_var="COINMARKETCAP_API_KEY",
        quote_currency="USD",
        granularity_seconds=3600,
        total_candles=2,
        should_save_downloaded_data=False,
    )

    raw_rows = loader._extract_historical_quote_rows(
        response_payload={
            "data": {
                "btc": {
                    "symbol": "BTC",
                    "quotes": [
                        {
                            "time_close": "2026-01-01T00:00:00Z",
                            "quote": {
                                "USD": {
                                    "open": 100.0,
                                    "high": 105.0,
                                    "low": 99.0,
                                    "close": 104.0,
                                    "volume": 1250.0,
                                }
                            },
                        },
                        {
                            "time_close": "2026-01-01T01:00:00Z",
                            "quote": {
                                "USD": {
                                    "open": 104.0,
                                    "high": 106.0,
                                    "low": 103.0,
                                    "close": 105.0,
                                    "volume": 1180.0,
                                }
                            },
                        },
                    ],
                }
            }
        },
        base_currency="BTC",
    )
    normalized_rows = loader._normalize_candle_rows(
        product_id="BTC-USD",
        base_currency="BTC",
        quote_currency="USD",
        candle_rows=raw_rows,
    )
    price_df = loader._build_price_frame(normalized_rows)

    assert len(raw_rows) == 2
    assert list(price_df.columns[:6]) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(price_df) == 2
    assert price_df.iloc[0]["timestamp"] < price_df.iloc[1]["timestamp"]


def test_coinmarketcap_latest_quotes_loader_can_merge_snapshot_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Supported latest-quote snapshots should append cleanly to the local market history."""

    data_path = tmp_path / "marketPrices.csv"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "product_id": "BTC-USD",
                "base_currency": "BTC",
                "quote_currency": "USD",
                "granularity_seconds": 3600,
                "source": "coinbaseExchange",
            }
        ]
    ).to_csv(data_path, index=False)

    monkeypatch.setenv("COINMARKETCAP_API_KEY", "test-key")
    loader = CoinMarketCapLatestQuotesPriceDataLoader(
        data_path=data_path,
        api_base_url="https://pro-api.coinmarketcap.com",
        api_key_env_var="COINMARKETCAP_API_KEY",
        quote_currency="USD",
        granularity_seconds=3600,
        total_candles=120,
        should_save_downloaded_data=False,
        product_ids=("SOL-USD",),
        fetch_all_quote_products=False,
        log_progress=False,
    )

    def fake_request_json(endpoint_path: str, query_params: dict) -> dict:
        assert endpoint_path == loader.quotes_latest_endpoint
        assert query_params["symbol"] == "SOL"
        return {
            "status": {
                "timestamp": "2026-01-01T01:00:00Z",
                "error_code": 0,
            },
            "data": {
                "SOL": [
                    {
                        "id": 5426,
                        "name": "Solana",
                        "symbol": "SOL",
                        "circulating_supply": 480_000_000,
                        "total_supply": 600_000_000,
                        "max_supply": None,
                        "quote": {
                            "USD": {
                                "price": 110.5,
                                "volume_24h": 3_400_000_000,
                                "volume_change_24h": 5.2,
                                "market_cap": 53_000_000_000,
                                "fully_diluted_market_cap": 66_000_000_000,
                                "percent_change_1h": 0.4,
                                "percent_change_24h": 3.1,
                                "percent_change_7d": 8.4,
                                "percent_change_30d": 18.2,
                                "last_updated": "2026-01-01T01:00:00Z",
                            }
                        },
                    }
                ]
            },
        }

    monkeypatch.setattr(loader, "_request_json", fake_request_json)

    price_df = loader.load()

    assert set(price_df["product_id"]) == {"BTC-USD", "SOL-USD"}
    sol_row = price_df.loc[price_df["product_id"] == "SOL-USD"].iloc[0]
    assert float(sol_row["open"]) == pytest.approx(110.5)
    assert float(sol_row["high"]) == pytest.approx(110.5)
    assert float(sol_row["low"]) == pytest.approx(110.5)
    assert float(sol_row["close"]) == pytest.approx(110.5)
    assert float(sol_row["cmc_market_cap"]) == pytest.approx(53_000_000_000)
    assert float(sol_row["cmc_percent_change_7d"]) == pytest.approx(8.4)
    assert str(sol_row["source"]) == "coinmarketcapLatestQuotes"


def test_market_data_loader_factory_can_create_coinmarketcap_loader() -> None:
    """The source factory should build the CoinMarketCap loader when configured."""

    loader = create_market_data_loader(
        TrainingConfig(
            market_data_source="coinmarketcap",
            coinmarketcap_use_context=False,
        ),
        should_save_downloaded_data=False,
    )

    assert isinstance(loader, CoinMarketCapOhlcvPriceDataLoader)


def test_market_data_loader_factory_can_create_coinmarketcap_latest_quotes_loader() -> None:
    """The source factory should build the supported CMC latest-quotes loader when configured."""

    loader = create_market_data_loader(
        TrainingConfig(
            market_data_source="coinmarketcapLatestQuotes",
            coinmarketcap_use_context=False,
        ),
        should_save_downloaded_data=False,
    )

    assert isinstance(loader, CoinMarketCapLatestQuotesPriceDataLoader)


def test_signal_symbol_policy_rejects_numeric_only_symbols() -> None:
    """Numeric-only tickers should not be exposed as public trading symbols."""

    assert is_signal_eligible_base_currency("SOL") is True
    assert is_signal_eligible_base_currency("1INCH") is True
    assert is_signal_eligible_base_currency("00") is False
    assert is_signal_eligible_base_currency("1234") is False


def test_coinmarketcap_latest_quotes_loader_filters_numeric_only_symbols() -> None:
    """The CMC auto-universe should skip numeric-only map entries like `00`."""

    loader = CoinMarketCapLatestQuotesPriceDataLoader(
        data_path=Path("unused.csv"),
        api_base_url="https://pro-api.coinmarketcap.com",
        api_key_env_var="COINMARKETCAP_API_KEY",
        quote_currency="USD",
        granularity_seconds=3600,
        total_candles=120,
        should_save_downloaded_data=False,
        fetch_all_quote_products=True,
        request_pause_seconds=0.0,
        product_batch_size=None,
        log_progress=False,
    )

    def fake_request_json(endpoint_path: str, query_params: dict[str, object]) -> dict[str, object]:
        assert endpoint_path == "/v1/cryptocurrency/map"
        return {
            "data": [
                {"id": 1, "name": "00 Token", "symbol": "00", "rank": 1000},
                {"id": 2, "name": "1inch", "symbol": "1INCH", "rank": 200},
                {"id": 3, "name": "Solana", "symbol": "SOL", "rank": 5},
            ]
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(loader, "_request_json", fake_request_json)
    try:
        filtered_products = loader._fetch_filtered_products()
    finally:
        monkeypatch.undo()

    assert [product["product_id"] for product in filtered_products] == ["SOL-USD", "1INCH-USD"]


def test_runtime_market_data_settings_can_override_saved_model_source() -> None:
    """Runtime settings should be able to switch a saved model off unsupported CMC OHLCV mode."""

    saved_config = TrainingConfig(
        market_data_source="coinmarketcap",
        coinmarketcap_use_context=False,
        prediction_horizon=3,
    )
    runtime_config = TrainingConfig(
        market_data_source="coinmarketcapLatestQuotes",
        coinmarketcap_use_context=True,
        coinmarketcap_fetch_all_quote_products=False,
        coinmarketcap_product_ids=("SOL-USD",),
    )

    merged_config = apply_runtime_market_data_settings(
        base_config=saved_config,
        runtime_config=runtime_config,
    )

    assert merged_config.market_data_source == "coinmarketcapLatestQuotes"
    assert merged_config.coinmarketcap_use_context is True
    assert merged_config.coinmarketcap_product_ids == ("SOL-USD",)
    assert merged_config.backtest_min_confidence == runtime_config.backtest_min_confidence
    assert merged_config.prediction_horizon == 3


def test_coinmarketcal_event_enricher_adds_upcoming_event_context(tmp_path: Path) -> None:
    """The CoinMarketCal enricher should attach simple upcoming-event features from cache."""

    events_path = tmp_path / "coinMarketCalEvents.csv"
    pd.DataFrame(
        [
            {
                "event_id": "1",
                "event_title": "Mainnet upgrade",
                "event_category": "protocol",
                "event_start": "2026-01-05T00:00:00Z",
                "base_currency": "BTC",
            },
            {
                "event_id": "2",
                "event_title": "Conference",
                "event_category": "community",
                "event_start": "2026-01-20T00:00:00Z",
                "base_currency": "BTC",
            },
        ]
    ).to_csv(events_path, index=False)

    price_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "BTC-USD",
                "base_currency": "BTC",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
            },
            {
                "timestamp": "2026-01-10T00:00:00Z",
                "product_id": "BTC-USD",
                "base_currency": "BTC",
                "open": 110.0,
                "high": 111.0,
                "low": 109.0,
                "close": 110.0,
                "volume": 12.0,
            },
        ]
    )

    enricher = CoinMarketCalEventEnricher(
        events_path=events_path,
        api_base_url="https://developers.coinmarketcal.com/v1",
        api_key_env_var="COINMARKETCAL_API_KEY",
        should_refresh_events=False,
        log_progress=False,
    )
    enriched_df = enricher.enrich(price_df)

    assert enriched_df.loc[0, "cmcal_event_count_next_7d"] == 1
    assert enriched_df.loc[0, "cmcal_has_event_next_7d"] == 1
    assert enriched_df.loc[0, "cmcal_event_count_next_30d"] == 2
    assert enriched_df.loc[1, "cmcal_event_count_next_7d"] == 0
    assert enriched_df.loc[1, "cmcal_event_count_next_30d"] == 1


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
    price_df["cmc_market_intelligence_available"] = 1
    price_df["cmc_market_total_market_cap"] = 2_300_000_000_000
    price_df["cmc_market_total_volume_24h"] = 75_000_000_000
    price_df["cmc_market_total_market_cap_change_24h"] = 1.8
    price_df["cmc_market_total_volume_change_24h"] = -5.2
    price_df["cmc_market_altcoin_share"] = 0.42
    price_df["cmc_market_btc_dominance"] = 58.0
    price_df["cmc_market_btc_dominance_change_24h"] = -0.8
    price_df["cmc_market_eth_dominance"] = 11.0
    price_df["cmc_market_stablecoin_share"] = 0.12
    price_df["cmc_market_defi_market_cap"] = 60_000_000_000
    price_df["cmc_market_defi_volume_24h"] = 8_000_000_000
    price_df["cmc_market_derivatives_volume_24h"] = 580_000_000_000
    price_df["cmc_market_fear_greed_value"] = 68.0
    price_df["cmc_market_fear_greed_classification"] = "Greed"

    feature_df = TechnicalFeatureEngineer().build(price_df)
    last_row = feature_df.iloc[-1]

    assert "breakout_up_20" in feature_df.columns
    assert "range_position_20" in feature_df.columns
    assert "cmc_market_cap_log" in feature_df.columns
    assert "cmc_market_total_market_cap_log" in feature_df.columns
    assert "cmc_market_fear_greed_score" in feature_df.columns
    assert last_row["cmc_market_cap_log"] > 0
    assert last_row["cmc_market_total_market_cap_log"] > 0
    assert last_row["cmc_market_fear_greed_score"] == pytest.approx(0.68)
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


def test_coinmarketcap_market_intelligence_enricher_can_refresh_and_merge_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CMC market-intelligence enricher should cache and broadcast global signals."""

    price_df = pd.DataFrame(
        [
            {"product_id": "BTC-USD", "base_currency": "BTC", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            {"product_id": "ETH-USD", "base_currency": "ETH", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        ]
    )

    monkeypatch.setenv("COINMARKETCAP_API_KEY", "test-key")
    enricher = CoinMarketCapMarketIntelligenceEnricher(
        intelligence_path=tmp_path / "coinMarketCapMarketIntelligence.csv",
        api_base_url="https://pro-api.coinmarketcap.com",
        api_key_env_var="COINMARKETCAP_API_KEY",
        should_refresh_market_intelligence=True,
        log_progress=False,
    )

    def fake_request_json(endpoint_path: str, query_params: dict) -> dict:
        if endpoint_path == enricher.global_metrics_endpoint:
            return {
                "data": {
                    "btc_dominance": 58.0,
                    "btc_dominance_24h_percentage_change": -0.8,
                    "eth_dominance": 11.0,
                    "last_updated": "2026-01-01T00:00:00Z",
                    "quote": {
                        "USD": {
                            "total_market_cap": 2_300_000_000_000,
                            "total_volume_24h": 75_000_000_000,
                            "altcoin_market_cap": 966_000_000_000,
                            "stablecoin_market_cap": 280_000_000_000,
                            "defi_market_cap": 60_000_000_000,
                            "defi_volume_24h": 8_000_000_000,
                            "derivatives_volume_24h": 580_000_000_000,
                            "total_market_cap_yesterday_percentage_change": 1.8,
                            "total_volume_24h_yesterday_percentage_change": -5.2,
                            "last_updated": "2026-01-01T00:00:00Z",
                        }
                    },
                }
            }
        if endpoint_path == enricher.fear_greed_latest_endpoint:
            return {
                "data": {
                    "value": 68,
                    "value_classification": "Greed",
                    "update_time": "2026-01-01T00:00:00Z",
                }
            }

        raise AssertionError(f"Unexpected endpoint: {endpoint_path}")

    monkeypatch.setattr(enricher, "_request_json", fake_request_json)

    intelligence_df = enricher.refresh_market_intelligence()
    enriched_df = enricher.enrich(price_df)

    assert len(intelligence_df) == 1
    assert float(intelligence_df.iloc[0]["cmc_market_fear_greed_value"]) == 68.0
    assert "cmc_market_btc_dominance" in enriched_df.columns
    assert float(enriched_df.loc[0, "cmc_market_btc_dominance"]) == 58.0
    assert float(enriched_df.loc[1, "cmc_market_altcoin_share"]) == pytest.approx(966_000_000_000 / 2_300_000_000_000)


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


def test_coinmarketcap_context_enricher_does_not_retry_single_ids_after_rate_limit(
    tmp_path: Path,
) -> None:
    """A CMC 429 should not fan out into one request per id."""

    enricher = CoinMarketCapContextEnricher(
        context_path=tmp_path / "coinMarketCapContext.csv",
        api_base_url="https://pro-api.coinmarketcap.com",
        api_key_env_var="COINMARKETCAP_API_KEY",
        should_refresh_context=True,
        log_progress=False,
    )
    enricher.keyed_lookup_batch_size = 5

    requested_id_groups: list[list[int]] = []

    def fake_request_keyed_lookup_chunk(
        endpoint_path: str,
        ids: Sequence[int],
        extra_query_params: dict | None = None,
    ) -> dict[str, dict[str, object]]:
        requested_id_groups.append([int(cmc_id) for cmc_id in ids])
        raise CoinMarketCapRateLimitError(
            "CoinMarketCap request failed with status 429 for /v2/cryptocurrency/quotes/latest."
        )

    enricher._request_keyed_lookup_chunk = fake_request_keyed_lookup_chunk  # type: ignore[method-assign]

    with pytest.raises(CoinMarketCapRateLimitError):
        enricher._fetch_keyed_lookup(
            endpoint_path=enricher.latest_quotes_endpoint,
            ids=[1, 2, 3],
            extra_query_params={"convert": "USD"},
        )

    assert requested_id_groups == [[1, 2, 3]]


def test_coinmarketcap_context_refresh_reuses_cache_after_rate_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CMC context refreshes should reuse the cached snapshot after a 429."""

    context_path = tmp_path / "coinMarketCapContext.csv"
    cached_context_df = pd.DataFrame(
        [
            {
                "product_id": "BTC-USD",
                "base_currency": "BTC",
                "cmc_context_available": 1,
                "cmc_market_cap": 2_000_000_000_000,
            }
        ]
    )
    cached_context_df.to_csv(context_path, index=False)

    price_df = pd.DataFrame(
        [
            {"product_id": "BTC-USD", "base_currency": "BTC", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
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

    def fake_fetch_symbol_map_lookup(symbols: list[str]) -> dict[str, dict[str, object]]:
        raise CoinMarketCapRateLimitError(
            "CoinMarketCap request failed with status 429 for /v1/cryptocurrency/map."
        )

    monkeypatch.setattr(enricher, "_fetch_symbol_map_lookup", fake_fetch_symbol_map_lookup)

    refreshed_df = enricher.refresh_context(price_df)

    assert refreshed_df.to_dict("records") == cached_context_df.to_dict("records")
    assert enricher.last_context_summary["usedCachedSnapshot"] is True
    assert "status 429" in str(enricher.last_context_summary["warning"])


def test_coinmarketcap_market_intelligence_reuses_cache_after_rate_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CMC market intelligence should reuse the cached snapshot after a 429."""

    intelligence_path = tmp_path / "coinMarketCapMarketIntelligence.csv"
    cached_intelligence_df = pd.DataFrame(
        [
            {
                "cmc_market_intelligence_available": 1,
                "cmc_market_quote_currency": "USD",
                "cmc_market_last_updated": "2026-01-01T00:00:00Z",
                "cmc_market_total_market_cap": 2_300_000_000_000,
                "cmc_market_total_volume_24h": 75_000_000_000,
                "cmc_market_total_market_cap_change_24h": 1.8,
                "cmc_market_total_volume_change_24h": -5.2,
                "cmc_market_altcoin_market_cap": 966_000_000_000,
                "cmc_market_altcoin_share": 0.42,
                "cmc_market_btc_dominance": 58.0,
                "cmc_market_btc_dominance_change_24h": -0.8,
                "cmc_market_eth_dominance": 11.0,
                "cmc_market_stablecoin_market_cap": 280_000_000_000,
                "cmc_market_stablecoin_share": 0.12,
                "cmc_market_defi_market_cap": 60_000_000_000,
                "cmc_market_defi_volume_24h": 8_000_000_000,
                "cmc_market_derivatives_volume_24h": 580_000_000_000,
                "cmc_market_fear_greed_value": 68.0,
                "cmc_market_fear_greed_classification": "Greed",
            }
        ]
    )
    cached_intelligence_df.to_csv(intelligence_path, index=False)

    monkeypatch.setenv("COINMARKETCAP_API_KEY", "test-key")
    enricher = CoinMarketCapMarketIntelligenceEnricher(
        intelligence_path=intelligence_path,
        api_base_url="https://pro-api.coinmarketcap.com",
        api_key_env_var="COINMARKETCAP_API_KEY",
        should_refresh_market_intelligence=True,
        log_progress=False,
    )

    def fake_request_json(endpoint_path: str, query_params: dict) -> dict:
        raise CoinMarketCapRateLimitError(
            f"CoinMarketCap request failed with status 429 for {endpoint_path}."
        )

    monkeypatch.setattr(enricher, "_request_json", fake_request_json)

    refreshed_df = enricher.refresh_market_intelligence()

    assert refreshed_df.to_dict("records") == cached_intelligence_df.to_dict("records")
    assert enricher.last_market_intelligence_summary["usedCachedSnapshot"] is True
    assert "status 429" in str(enricher.last_market_intelligence_summary["warning"])


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


def test_triple_barrier_labeler_uses_first_barrier_hit_per_coin() -> None:
    """Triple-barrier labels should react to the first path event for each asset."""

    feature_df = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "product_id": "BTC-USD", "close": 100.0, "high": 101.0, "low": 99.0},
            {"timestamp": "2026-01-01T01:00:00Z", "product_id": "BTC-USD", "close": 101.0, "high": 106.0, "low": 100.0},
            {"timestamp": "2026-01-01T02:00:00Z", "product_id": "BTC-USD", "close": 103.0, "high": 104.0, "low": 102.0},
            {"timestamp": "2026-01-01T03:00:00Z", "product_id": "BTC-USD", "close": 104.0, "high": 105.0, "low": 103.0},
            {"timestamp": "2026-01-01T00:00:00Z", "product_id": "ETH-USD", "close": 200.0, "high": 201.0, "low": 199.0},
            {"timestamp": "2026-01-01T01:00:00Z", "product_id": "ETH-USD", "close": 198.0, "high": 199.0, "low": 188.0},
            {"timestamp": "2026-01-01T02:00:00Z", "product_id": "ETH-USD", "close": 197.0, "high": 198.0, "low": 196.0},
            {"timestamp": "2026-01-01T03:00:00Z", "product_id": "ETH-USD", "close": 201.0, "high": 202.0, "low": 200.0},
        ]
    )

    labeled_df = TripleBarrierSignalLabeler(
        prediction_horizon=2,
        buy_threshold=0.05,
        sell_threshold=-0.05,
    ).add_labels(feature_df)

    assert labeled_df.iloc[0]["target_name"] == "BUY"
    assert labeled_df.iloc[0]["future_return"] == pytest.approx(0.05)
    assert labeled_df.iloc[0]["label_barrier"] == "upper"
    assert labeled_df.iloc[4]["target_name"] == "TAKE_PROFIT"
    assert labeled_df.iloc[4]["future_return"] == pytest.approx(-0.05)
    assert labeled_df.iloc[4]["label_barrier"] == "lower"


def test_market_data_refresh_app_reports_context_failure_without_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CMC refresh problem should not erase a successful market-data refresh."""

    config = TrainingConfig(
        data_file=tmp_path / "marketPrices.csv",
        coinmarketcap_context_file=tmp_path / "coinMarketCapContext.csv",
        coinmarketcal_use_events=False,
        coinmarketcap_refresh_context_after_market_refresh=True,
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


def test_kraken_loader_normalizes_public_ohlc_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kraken OHLC rows should land in the shared market-price schema."""

    loader = KrakenOhlcPriceDataLoader(
        data_path=tmp_path / "marketPrices.csv",
        product_ids=("BTC-USD",),
        product_id="",
        granularity_seconds=3600,
        total_candles=2,
        log_progress=False,
    )

    def fake_request_ohlc_rows(exchange_pair: str) -> list[list[object]]:
        assert exchange_pair == "XBTUSD"
        return [
            [1767225600, "100.0", "110.0", "90.0", "105.0", "101.0", "12.5", 20],
            [1767229200, "105.0", "115.0", "95.0", "112.0", "108.0", "8.0", 18],
        ]

    monkeypatch.setattr(loader, "_request_ohlc_rows", fake_request_ohlc_rows)

    price_df = loader.refresh_data()

    assert list(price_df["product_id"].unique()) == ["BTC-USD"]
    assert price_df.iloc[0]["open"] == pytest.approx(100.0)
    assert price_df.iloc[1]["close"] == pytest.approx(112.0)
    assert price_df.iloc[0]["source"] == "kraken"


def test_binance_public_data_loader_normalizes_archive_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binance archive rows should be normalized as exchange OHLCV data."""

    loader = BinancePublicDataPriceDataLoader(
        data_path=tmp_path / "marketPrices.csv",
        product_ids=("BTC-USD",),
        product_id="",
        quote_currency="USDT",
        interval="1h",
        granularity_seconds=3600,
        total_candles=2,
        log_progress=False,
    )

    monkeypatch.setattr(loader, "_resolve_months_to_download", lambda: ["2026-01"])

    def fake_request_month_rows(
        product_details: dict[str, str],
        year_month: str,
    ) -> list[dict[str, object]]:
        assert product_details["exchange_symbol"] == "BTCUSDT"
        assert year_month == "2026-01"
        return [
            loader._normalize_kline_row(
                product_details,
                ["1767225600000", "100.0", "110.0", "90.0", "105.0", "12.5"],
            ),
            loader._normalize_kline_row(
                product_details,
                ["1767229200000", "105.0", "115.0", "95.0", "112.0", "8.0"],
            ),
        ]

    monkeypatch.setattr(loader, "_request_month_rows", fake_request_month_rows)

    price_df = loader.refresh_data()

    assert list(price_df["product_id"].unique()) == ["BTC-USDT"]
    assert price_df.iloc[0]["quote_currency"] == "USDT"
    assert price_df.iloc[0]["high"] == pytest.approx(110.0)
    assert price_df.iloc[1]["source"] == "binancePublicData"


def test_binance_public_data_loader_discovers_quote_universe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binance discovery should keep trading quote pairs and exclude stable bases."""

    loader = BinancePublicDataPriceDataLoader(
        data_path=tmp_path / "marketPrices.csv",
        fetch_all_quote_products=True,
        quote_currency="USDT",
        excluded_base_currencies=("USDT", "USDC"),
        max_products=2,
        log_progress=False,
    )

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "symbols": [
                        {
                            "symbol": "ADAUSDT",
                            "status": "TRADING",
                            "baseAsset": "ADA",
                            "quoteAsset": "USDT",
                            "isSpotTradingAllowed": True,
                        },
                        {
                            "symbol": "USDCUSDT",
                            "status": "TRADING",
                            "baseAsset": "USDC",
                            "quoteAsset": "USDT",
                            "isSpotTradingAllowed": True,
                        },
                        {
                            "symbol": "ETHBTC",
                            "status": "TRADING",
                            "baseAsset": "ETH",
                            "quoteAsset": "BTC",
                            "isSpotTradingAllowed": True,
                        },
                        {
                            "symbol": "SOLUSDT",
                            "status": "TRADING",
                            "baseAsset": "SOL",
                            "quoteAsset": "USDT",
                            "isSpotTradingAllowed": True,
                        },
                    ]
                }
            ).encode("utf-8")

    monkeypatch.setattr("crypto_signal_ml.data.urlopen", lambda request, timeout=30: FakeResponse())

    products = loader.get_available_products()

    assert [product["product_id"] for product in products] == ["ADA-USDT", "SOL-USDT"]
    assert [product["exchange_symbol"] for product in products] == ["ADAUSDT", "SOLUSDT"]


def test_market_data_refresh_app_reports_market_intelligence_failure_without_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CMC market-intelligence failure should not erase a successful market-data refresh."""

    config = TrainingConfig(
        data_file=tmp_path / "marketPrices.csv",
        coinmarketcap_use_context=False,
        coinmarketcap_market_intelligence_file=tmp_path / "coinMarketCapMarketIntelligence.csv",
        coinmarketcal_use_events=False,
        coinmarketcap_refresh_market_intelligence_after_market_refresh=True,
    )
    refresh_app = MarketDataRefreshApp(config=config)

    class FakeLoader:
        last_refresh_summary = {"rowsDownloaded": 2}

        def refresh_data(self) -> pd.DataFrame:
            return _build_sample_market_frame(total_hours=2)

    class FailingEnricher:
        def refresh_market_intelligence(self) -> pd.DataFrame:
            raise RuntimeError("simulated CoinMarketCap market intelligence failure")

    monkeypatch.setattr(refresh_app, "build_market_data_loader", lambda: FakeLoader())
    monkeypatch.setattr(
        refresh_app,
        "build_coinmarketcap_market_intelligence_enricher",
        lambda should_refresh_market_intelligence=False: FailingEnricher(),
    )

    result = refresh_app.run()

    assert result["rowsDownloaded"] > 0
    assert result["coinMarketCapMarketIntelligenceStatus"] == "refresh_failed"
    assert "market intelligence failure" in result["coinMarketCapMarketIntelligenceError"]


def test_market_data_refresh_app_rotates_market_batches_between_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-batch refreshes should advance through ranked market batches automatically."""

    import crypto_signal_ml.app as app_module  # noqa: WPS433

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_module, "OUTPUTS_DIR", outputs_dir)

    (outputs_dir / "signalMarketDataRefresh.json").write_text(
        json.dumps(
            {
                "refresh": {
                    "downloadSummary": {
                        "batchNumber": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    state_path = tmp_path / "marketProductBatchState.json"
    refresh_app = MarketDataRefreshApp(
        config=TrainingConfig(
            data_file=tmp_path / "marketPrices.csv",
            coinmarketcap_use_context=False,
            coinmarketcap_use_market_intelligence=False,
            coinmarketcal_use_events=False,
            market_product_batch_state_file=state_path,
            market_product_batch_rotation_enabled=True,
        )
    )

    refreshed_batches: list[int] = []

    class FakeLoader:
        def __init__(self, batch_number: int) -> None:
            self.batch_number = batch_number
            self.last_refresh_summary = {
                "batchNumber": batch_number,
                "rowsDownloaded": 4,
            }

        def get_total_batches(self) -> int:
            return 4

        def refresh_data(self) -> pd.DataFrame:
            refreshed_batches.append(self.batch_number)
            return _build_sample_market_frame(total_hours=2)

    monkeypatch.setattr(refresh_app, "build_market_data_loader", lambda: FakeLoader(1))
    monkeypatch.setattr(
        refresh_app,
        "_build_market_data_loader_for_config",
        lambda config: FakeLoader(config.coinmarketcap_product_batch_number),
    )

    first_result = refresh_app.run()
    second_result = refresh_app.run()

    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    source_states = state_payload["sources"]
    rotation_state = next(iter(source_states.values()))

    assert refreshed_batches == [2, 3]
    assert first_result["batchNumber"] == 2
    assert first_result["nextBatchNumber"] == 3
    assert second_result["batchNumber"] == 3
    assert second_result["nextBatchNumber"] == 4
    assert rotation_state["lastCompletedBatch"] == 3
    assert rotation_state["nextBatch"] == 4
    assert rotation_state["totalBatches"] == 4


def test_training_app_writes_model_metadata_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary training should publish a JSON sidecar for deployment metadata."""

    raw_data_path = tmp_path / "marketPrices.csv"
    _build_sample_market_frame(total_hours=12).to_csv(raw_data_path, index=False)

    import crypto_signal_ml.app as app_module  # noqa: WPS433

    monkeypatch.setattr(app_module, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(app_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(app_module, "OUTPUTS_DIR", tmp_path / "outputs")

    training_df = pd.DataFrame(
        [
            {"return_1": -0.04, "momentum_3": -0.09, "target_signal": -1},
            {"return_1": -0.01, "momentum_3": -0.02, "target_signal": 0},
            {"return_1": 0.05, "momentum_3": 0.08, "target_signal": 1},
            {"return_1": -0.03, "momentum_3": -0.04, "target_signal": -1},
            {"return_1": 0.00, "momentum_3": 0.01, "target_signal": 0},
            {"return_1": 0.04, "momentum_3": 0.07, "target_signal": 1},
        ]
    )

    class StubDatasetBuilder:
        def build_labeled_dataset(self) -> tuple[pd.DataFrame, list[str]]:
            return training_df, ["return_1", "momentum_3"]

    training_app = TrainingApp(
        config=TrainingConfig(
            data_file=raw_data_path,
            coinmarketcap_use_context=False,
            n_estimators=12,
            max_depth=3,
            random_state=7,
        ),
        dataset_builder=StubDatasetBuilder(),
    )

    result = training_app.run()
    metadata_path = Path(result["metadataPath"])
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata_path.exists()
    assert metadata_payload["modelType"] == result["modelType"]
    assert metadata_payload["metrics"]["balancedAccuracy"] == pytest.approx(result["balancedAccuracy"])
    assert metadata_payload["trainingDataPath"] == str(raw_data_path)
    assert metadata_payload["artifacts"]["metricsPath"] == result["metricsPath"]


def test_training_app_bootstraps_missing_market_data_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training should auto-refresh raw market data when the CSV is missing."""

    raw_data_path = tmp_path / "marketPrices.csv"

    import crypto_signal_ml.app as app_module  # noqa: WPS433

    monkeypatch.setattr(app_module, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(app_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(app_module, "OUTPUTS_DIR", tmp_path / "outputs")

    training_df = pd.DataFrame(
        [
            {"return_1": -0.04, "momentum_3": -0.09, "target_signal": -1},
            {"return_1": -0.01, "momentum_3": -0.02, "target_signal": 0},
            {"return_1": 0.05, "momentum_3": 0.08, "target_signal": 1},
            {"return_1": -0.03, "momentum_3": -0.04, "target_signal": -1},
            {"return_1": 0.00, "momentum_3": 0.01, "target_signal": 0},
            {"return_1": 0.04, "momentum_3": 0.07, "target_signal": 1},
        ]
    )

    class StubDatasetBuilder:
        def build_labeled_dataset(self) -> tuple[pd.DataFrame, list[str]]:
            assert raw_data_path.exists()
            return training_df, ["return_1", "momentum_3"]

    def fake_refresh_run(self: MarketDataRefreshApp) -> dict:
        _build_sample_market_frame(total_hours=12).to_csv(raw_data_path, index=False)
        return {
            "savedPath": str(raw_data_path),
            "rowsDownloaded": 24,
            "marketDataSource": "coinbaseExchange",
        }

    monkeypatch.setattr(app_module.MarketDataRefreshApp, "run", fake_refresh_run)

    training_app = TrainingApp(
        config=TrainingConfig(
            data_file=raw_data_path,
            coinmarketcap_use_context=False,
            n_estimators=12,
            max_depth=3,
            random_state=7,
        ),
        dataset_builder=StubDatasetBuilder(),
    )

    training_app.run()

    refresh_summary_path = tmp_path / "outputs" / "marketDataRefreshOnDemand.json"
    refresh_summary = json.loads(refresh_summary_path.read_text(encoding="utf-8"))

    assert raw_data_path.exists()
    assert refresh_summary["trigger"] == "automatic_training_bootstrap"
    assert refresh_summary["refresh"]["savedPath"] == str(raw_data_path)


def test_rag_knowledge_store_can_ingest_and_search_text_sources(tmp_path: Path) -> None:
    """The lightweight knowledge store should return relevant external chunks."""

    store = RagKnowledgeStore(
        db_path=tmp_path / "assistantKnowledge.sqlite3",
        chunk_size_chars=180,
        chunk_overlap_chars=40,
    )

    source = store.ingest_text(
        title="Bitcoin ETF research note",
        content=(
            "Spot bitcoin ETF inflows can tighten liquid supply and improve market structure over time. "
            "When inflows accelerate, short-term momentum and sentiment can change quickly."
        ),
        source_uri="https://example.com/bitcoin-etf-note",
    )
    search_results = store.search("bitcoin ETF inflows", limit=3)

    assert source["title"] == "Bitcoin ETF research note"
    assert store.get_status()["sourceCount"] == 1
    assert store.get_status()["storageBackend"] == "sqlite"
    assert search_results
    assert search_results[0]["title"] == "Bitcoin ETF research note"
    assert "inflows" in search_results[0]["content"].lower()


def test_conversation_session_store_can_round_trip_messages(tmp_path: Path) -> None:
    """Assistant session memory should persist and retrieve messages in order."""

    store = ConversationSessionStore(
        db_path=tmp_path / "assistantSessions.sqlite3",
    )

    session = store.create_session(title="Research desk")
    user_message = store.add_message(
        session_id=session["sessionId"],
        role="user",
        content="What does the model think about BTC-USD?",
        metadata={"productId": "BTC-USD"},
    )
    assistant_message = store.add_message(
        session_id=session["sessionId"],
        role="assistant",
        content="BTC-USD is currently a BUY setup.",
        metadata={"liveSource": "cached"},
    )
    messages = store.list_messages(session["sessionId"], limit=10)

    assert session["title"] == "Research desk"
    assert user_message["role"] == "user"
    assert assistant_message["role"] == "assistant"
    assert len(messages) == 2
    assert messages[0]["content"] == "What does the model think about BTC-USD?"
    assert messages[1]["content"] == "BTC-USD is currently a BUY setup."


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
        batch_number = self.config.coinmarketcap_product_batch_number
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


def test_market_events_refresh_app_refreshes_cached_coinmarketcal_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standalone event-refresh app should update the cached CoinMarketCal snapshot."""

    data_path = tmp_path / "marketPrices.csv"
    events_path = tmp_path / "coinMarketCalEvents.csv"
    sample_df = _build_sample_market_frame()
    sample_df["base_currency"] = sample_df["product_id"].str.split("-").str[0]
    sample_df["quote_currency"] = "USD"
    sample_df.to_csv(data_path, index=False)

    refresh_app = MarketEventsRefreshApp(
        config=TrainingConfig(
            data_file=data_path,
            coinmarketcap_use_context=False,
            coinmarketcal_events_file=events_path,
        )
    )

    class FakeEventEnricher:
        last_events_summary = {
            "eventsPath": str(events_path),
            "coinsRequested": 2,
            "rowsSaved": 1,
            "finalRowsSaved": 1,
        }

        def refresh_events(self, price_df: pd.DataFrame) -> pd.DataFrame:
            assert len(price_df) > 0
            return pd.DataFrame(
                [
                    {
                        "event_id": "evt-1",
                        "event_title": "Protocol upgrade",
                        "event_category": "protocol",
                        "event_start": "2026-01-03T00:00:00Z",
                        "base_currency": "BTC",
                    }
                ]
            )

    monkeypatch.setattr(
        refresh_app,
        "build_coinmarketcal_event_enricher",
        lambda should_refresh_events: FakeEventEnricher(),
    )

    result = refresh_app.run()

    assert result["status"] == "refreshed"
    assert result["trackedProducts"] == 2
    assert result["eventsRows"] == 1
    assert result["refreshSummary"]["coinsRequested"] == 2


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
        model_type="randomForestSignalModel",
        walkforward_min_train_size=0.50,
        walkforward_test_size=0.25,
        walkforward_step_size=0.25,
        n_estimators=25,
    )
    validation_app = WalkForwardValidationApp(config=config)

    monkeypatch.setattr(validation_app, "save_dataframe", lambda dataframe, file_path: None)
    monkeypatch.setattr(validation_app, "save_json", lambda payload, file_path: None)

    result = validation_app.run()

    assert result["foldCount"] == 2
    assert result["outOfSampleRows"] > 0
    assert result["walkForwardSummaryPath"].endswith("walkForwardSummary.json")
    assert result["walkForwardBacktestSummaryPath"].endswith("walkForwardBacktestSummary.json")


def test_walk_forward_validation_app_scopes_outputs_to_feature_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Walk-forward ablations should be saved under a run directory keyed by feature-family selection."""

    raw_data_path = tmp_path / "marketPrices.csv"
    _build_mixed_market_frame().to_csv(raw_data_path, index=False)

    config = TrainingConfig(
        data_file=raw_data_path,
        coinmarketcap_use_context=False,
        model_type="randomForestSignalModel",
        walkforward_min_train_size=0.50,
        walkforward_test_size=0.25,
        walkforward_step_size=0.25,
        n_estimators=25,
        feature_pack="core",
        feature_include_groups=("market_context",),
        feature_exclude_groups=("regime",),
    )
    validation_app = WalkForwardValidationApp(config=config)

    monkeypatch.setattr(validation_app, "save_dataframe", lambda dataframe, file_path: None)
    monkeypatch.setattr(validation_app, "save_json", lambda payload, file_path: None)

    result = validation_app.run()
    run_directory = Path(result["runDirectory"])

    assert result["runLabel"] == "pack-core__inc-market-context__exc-regime"
    assert run_directory.parent.name == "walkForwardRuns"
    assert run_directory.name.endswith("__pack-core__inc-market-context__exc-regime")
    assert Path(result["walkForwardSummaryPath"]).parent == run_directory
    assert Path(result["walkForwardBacktestSummaryPath"]).parent == run_directory


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
        walkforward_purge_gap_timestamps=2,
        tuning_prediction_horizon_candidates=(2, 3),
        tuning_buy_threshold_candidates=(0.01, 0.015),
        tuning_sell_threshold_candidates=(-0.01, -0.015),
        tuning_backtest_confidence_candidates=(0.0, 0.55),
        logistic_max_iter=200,
    )
    tuning_app = SignalParameterTuningApp(config=config)

    monkeypatch.setattr(tuning_app, "save_dataframe", lambda dataframe, file_path: None)
    monkeypatch.setattr(tuning_app, "save_json", lambda payload, file_path: None)

    result = tuning_app.run()

    assert result["bestPredictionHorizon"] in {2, 3}
    assert result["bestBuyThreshold"] in {0.01, 0.015}
    assert result["bestSellThreshold"] in {-0.01, -0.015}
    assert result["bestBacktestMinConfidence"] in {0.0, 0.55}
    assert result["confidenceResultsPath"].endswith("signalConfidenceTuningResults.csv")


def test_signal_generation_app_refreshes_market_data_before_publishing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saved signal files should be published from a fresh market refresh, not stale cache."""

    prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-04-03T16:00:00Z",
                "product_id": "SOL-USD",
                "base_currency": "SOL",
                "quote_currency": "USD",
                "close": 80.5,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.81,
                "prob_take_profit": 0.05,
                "prob_hold": 0.14,
                "prob_buy": 0.81,
            }
        ]
    )

    class FakeDatasetBuilder:
        def build_feature_table(self) -> pd.DataFrame:
            return pd.DataFrame([{"feature_1": 1.0}])

    class FakeModel:
        model_type = "histGradientBoostingSignalModel"

        def __init__(self, config: TrainingConfig) -> None:
            self.config = config
            self.feature_columns = ["feature_1"]

        def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
            assert list(feature_df.columns) == ["feature_1"]
            return prediction_df.copy()

    refresh_calls: list[bool] = []
    saved_json_payloads: dict[str, dict] = {}

    def fake_refresh_run(self: MarketDataRefreshApp) -> dict:
        refresh_calls.append(True)
        return {
            "marketDataSource": "coinbase",
            "savedPath": "marketPrices.csv",
            "rowsDownloaded": 1,
            "uniqueProducts": 1,
            "firstTimestamp": "2026-04-03T16:00:00Z",
            "lastTimestamp": "2026-04-03T16:00:00Z",
        }

    config = TrainingConfig(
        coinmarketcap_use_context=False,
        market_data_source="coinbase",
        signal_excluded_base_currencies=(),
        signal_refresh_market_data_before_generation=True,
    )
    app = SignalGenerationApp(
        config=config,
        dataset_builder=FakeDatasetBuilder(),
        model=FakeModel(config=config),
    )

    monkeypatch.setattr(MarketDataRefreshApp, "run", fake_refresh_run)
    monkeypatch.setattr(app, "save_dataframe", lambda dataframe, file_path: None)
    monkeypatch.setattr(
        app,
        "save_json",
        lambda payload, file_path: saved_json_payloads.__setitem__(str(file_path.name), dict(payload)),
    )

    result = app.run()

    assert refresh_calls == [True]
    assert result["signalSource"] == "live-market-refresh"
    assert result["marketDataRefresh"]["lastTimestamp"] == "2026-04-03T16:00:00Z"
    assert saved_json_payloads["latestSignal.json"]["signalSource"] == "live-market-refresh"
    assert saved_json_payloads["latestSignal.json"]["marketDataLastTimestamp"] == "2026-04-03T16:00:00Z"


def test_signal_generation_app_can_publish_from_prioritized_active_universe_without_broad_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CMC publication should publish from the active universe without a separate broad refresh."""

    historical_prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-04-03T16:00:00Z",
                "product_id": "BTC-USD",
                "base_currency": "BTC",
                "quote_currency": "USD",
                "close": 82000.0,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.83,
                "prob_take_profit": 0.05,
                "prob_hold": 0.12,
                "prob_buy": 0.83,
            }
        ]
    )
    fresh_prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-04-05T12:00:00Z",
                "product_id": "SOL-USD",
                "base_currency": "SOL",
                "quote_currency": "USD",
                "close": 80.5,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.81,
                "prob_take_profit": 0.05,
                "prob_hold": 0.14,
                "prob_buy": 0.81,
            }
        ]
    )

    class FakeDatasetBuilder:
        def build_feature_table(self) -> pd.DataFrame:
            return pd.DataFrame([{"feature_1": 1.0}])

    class FakeModel:
        model_type = "histGradientBoostingSignalModel"

        def __init__(self, config: TrainingConfig) -> None:
            self.config = config
            self.feature_columns = ["feature_1"]

        def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
            assert list(feature_df.columns) == ["feature_1"]
            return historical_prediction_df.copy()

    refresh_calls: list[bool] = []
    saved_json_payloads: dict[str, dict] = {}

    def fake_refresh_run(self: MarketDataRefreshApp) -> dict:
        refresh_calls.append(True)
        return {
            "marketDataSource": "coinmarketcapLatestQuotes",
            "savedPath": "marketPrices.csv",
            "rowsDownloaded": 25,
            "uniqueProducts": 25,
            "firstTimestamp": "2026-04-05T12:00:00Z",
            "lastTimestamp": "2026-04-05T12:00:00Z",
        }

    config = TrainingConfig(
        coinmarketcap_use_context=False,
        coinmarketcal_use_events=False,
        coinmarketcap_use_market_intelligence=False,
        market_data_source="coinmarketcapLatestQuotes",
        coinmarketcap_fetch_all_quote_products=True,
        live_fetch_all_quote_products=True,
        live_max_products=25,
        signal_excluded_base_currencies=("BTC",),
        signal_refresh_market_data_before_generation=True,
        portfolio_store_path=tmp_path / "traderPortfolio.sqlite3",
    )
    app = SignalGenerationApp(
        config=config,
        dataset_builder=FakeDatasetBuilder(),
        model=FakeModel(config=config),
    )

    monkeypatch.setattr(MarketDataRefreshApp, "run", fake_refresh_run)
    monkeypatch.setattr(
        app,
        "_build_fresh_signal_prediction_frame",
        lambda: (
            fresh_prediction_df.copy(),
            {
                "mode": "prioritized-active-universe",
                "warning": "",
                "maxProducts": 25,
                "productsRequested": 3,
                "totalAvailableProducts": 25,
                "rowsScored": 1,
                "productsScored": 1,
                "protectedProductIds": ["SOL-USD"],
                "activeUniverse": {
                    "mode": "prioritized-active-universe",
                    "selectedProductIds": ["SOL-USD"],
                },
                "sourceRefresh": {
                    "sourceStatus": "cache_valid",
                    "productCount": 25,
                },
            },
        ),
    )
    monkeypatch.setattr(app, "save_dataframe", lambda dataframe, file_path: None)
    monkeypatch.setattr(
        app,
        "save_json",
        lambda payload, file_path: saved_json_payloads.__setitem__(str(file_path.name), dict(payload)),
    )

    result = app.run()

    assert refresh_calls == []
    assert result["signalName"] == "BUY"
    assert result["signalSource"] == "live-active-universe-refresh"
    assert result["signalInference"]["mode"] == "prioritized-active-universe"
    assert result["signalInference"]["productsScored"] == 1
    assert saved_json_payloads["latestSignal.json"]["productId"] == "SOL-USD"
    assert saved_json_payloads["frontendSignalSnapshot.json"]["signalInference"]["mode"] == "prioritized-active-universe"


def test_signal_generation_app_persists_current_signals_to_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Publishing signals should persist the current set in the live-signal database."""

    prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-04-05T12:00:00Z",
                "product_id": "ALGO-USD",
                "base_currency": "ALGO",
                "quote_currency": "USD",
                "close": 0.25,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.86,
                "prob_take_profit": 0.04,
                "prob_hold": 0.10,
                "prob_buy": 0.86,
                "breakout_up_20": 0.04,
                "breakout_down_20": 0.01,
                "close_vs_ema_5": 0.03,
                "relative_strength_1": 0.02,
                "relative_strength_5": 0.05,
                "momentum_10": 0.07,
                "range_position_20": 0.93,
                "rsi_14": 63.0,
                "trend_score": 0.03,
                "volatility_ratio": 0.95,
                "regime_label": "trend_up",
                "regime_code": 2,
                "regime_is_trending": 1,
                "regime_is_high_volatility": 0,
                "cmc_context_available": 0,
            }
        ]
    )

    class FakeDatasetBuilder:
        def build_feature_table(self) -> pd.DataFrame:
            return pd.DataFrame([{"feature_1": 1.0}])

    class FakeModel:
        model_type = "histGradientBoostingSignalModel"

        def __init__(self, config: TrainingConfig) -> None:
            self.config = config
            self.feature_columns = ["feature_1"]

        def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
            assert list(feature_df.columns) == ["feature_1"]
            return prediction_df.copy()

    data_path = tmp_path / "marketPrices.csv"
    data_path.write_text("timestamp,product_id,close\n", encoding="utf-8")
    config = TrainingConfig(
        data_file=data_path,
        coinmarketcap_use_context=False,
        market_data_source="coinmarketcapLatestQuotes",
        signal_excluded_base_currencies=(),
        signal_refresh_market_data_before_generation=False,
        portfolio_store_path=tmp_path / "traderPortfolio.sqlite3",
        signal_store_path=tmp_path / "liveSignals.sqlite3",
    )
    app = SignalGenerationApp(
        config=config,
        dataset_builder=FakeDatasetBuilder(),
        model=FakeModel(config=config),
    )

    monkeypatch.setattr(app, "save_dataframe", lambda dataframe, file_path: None)
    monkeypatch.setattr(app, "save_json", lambda payload, file_path: None)

    with caplog.at_level("INFO"):
        result = app.run()

    signal_store = TradingSignalStore(db_path=config.signal_store_path)
    current_signal = signal_store.get_current_signal()
    current_signals = signal_store.list_current_signals(limit=10)
    signal_history = signal_store.list_signal_history(limit=10)

    assert result["signalStore"]["signalCount"] == 1
    assert result["signalStore"]["storageBackend"] == "sqlite"
    assert current_signal is not None
    assert current_signal["productId"] == "ALGO-USD"
    assert current_signal["signal_name"] == "BUY"
    assert current_signal["isPrimary"] is True
    assert len(current_signals) == 1
    assert len(signal_history) == 1
    assert signal_history[0]["productId"] == "ALGO-USD"
    assert "ALGO-USD | live=BUY | action=buy" in caplog.text


def test_signal_generation_app_can_emit_watchlist_fallback_after_quiet_period(
    tmp_path: Path,
) -> None:
    """After a long quiet stretch, the feed should surface the strongest watchlist candidate."""

    app = SignalGenerationApp(
        config=TrainingConfig(
            signal_watchlist_fallback_enabled=True,
            signal_watchlist_fallback_hours=24.0,
            signal_watchlist_min_decision_score=0.30,
            signal_watchlist_min_confidence=0.55,
        ),
    )
    app.primary_signal_history_path = tmp_path / "primarySignalHistory.json"
    app._should_publish_watchlist_fallback = lambda: True  # type: ignore[method-assign]

    selected_signal = app._select_watchlist_fallback_signal(
        [
            {
                "productId": "ABT-USD",
                "signal_name": "HOLD",
                "modelSignalName": "HOLD",
                "tradeReadiness": "standby",
                "confidence": 0.91,
                "policyScore": 1.76,
                "reasonItems": ["ABT-USD stays on the watchlist."],
                "signalChat": "ABT-USD stays on the watchlist.",
                "brain": {
                    "decision": "watchlist",
                    "decisionScore": 0.62,
                },
                "tradeContext": {
                    "hasActiveTrade": False,
                },
            },
            {
                "productId": "SOL-USD",
                "signal_name": "HOLD",
                "modelSignalName": "BUY",
                "tradeReadiness": "medium",
                "confidence": 0.67,
                "policyScore": 0.98,
                "reasonItems": ["SOL-USD is close to a breakout entry."],
                "signalChat": "SOL-USD is close to a breakout entry.",
                "brain": {
                    "decision": "watchlist",
                    "decisionScore": 0.41,
                },
                "tradeContext": {
                    "hasActiveTrade": False,
                },
            },
        ]
    )

    assert selected_signal is not None
    assert selected_signal["productId"] == "SOL-USD"
    assert selected_signal["watchlistFallback"] is True
    assert selected_signal["publicSignalType"] == "watchlist"
    assert selected_signal["spotAction"] == "wait"
    assert selected_signal["reasonSummary"].startswith("No actionable trade cleared the live gate recently")


def test_signal_generation_app_can_supplement_one_public_signal_with_watchlist_candidate() -> None:
    """A thin public feed should be padded with the strongest watchlist idea."""

    app = SignalGenerationApp(
        config=TrainingConfig(
            signal_watchlist_min_decision_score=0.30,
            signal_watchlist_min_confidence=0.55,
            signal_watchlist_min_published_signals=2,
        ),
    )

    supplemented_signals = app._supplement_published_signals_with_watchlist_candidates(
        published_signals=[
            {
                "productId": "ALGO-USD",
                "signal_name": "LOSS",
                "spotAction": "cut_loss",
                "actionable": True,
            }
        ],
        signal_summaries=[
            {
                "productId": "ALGO-USD",
                "signal_name": "LOSS",
                "modelSignalName": "HOLD",
                "tradeReadiness": "high",
                "confidence": 0.71,
                "policyScore": 1.25,
                "reasonItems": ["ALGO-USD is protecting capital."],
                "signalChat": "ALGO-USD is protecting capital.",
                "brain": {
                    "decision": "avoid_long",
                    "decisionScore": 0.36,
                },
                "tradeContext": {
                    "hasActiveTrade": True,
                },
            },
            {
                "productId": "ABT-USD",
                "signal_name": "HOLD",
                "modelSignalName": "HOLD",
                "tradeReadiness": "standby",
                "confidence": 0.91,
                "policyScore": 1.76,
                "reasonItems": ["ABT-USD stays on the watchlist."],
                "signalChat": "ABT-USD stays on the watchlist.",
                "brain": {
                    "decision": "watchlist",
                    "decisionScore": 0.62,
                },
                "tradeContext": {
                    "hasActiveTrade": False,
                },
            },
        ],
    )

    assert [signal["productId"] for signal in supplemented_signals] == ["ALGO-USD", "ABT-USD"]
    assert supplemented_signals[1]["watchlistFallback"] is True
    assert supplemented_signals[1]["publicSignalType"] == "watchlist"
    assert supplemented_signals[1]["spotAction"] == "wait"


def test_signal_generation_app_persists_watchlist_pool_snapshot(tmp_path: Path) -> None:
    """Signal generation should persist the strongest watchlist names for live monitoring."""

    watchlist_pool_path = tmp_path / "watchlistPool.json"
    app = SignalGenerationApp(
        config=TrainingConfig(
            signal_watchlist_pool_enabled=True,
            signal_watchlist_pool_max_products=2,
            signal_watchlist_pool_path=watchlist_pool_path,
        )
    )

    app._save_watchlist_pool_snapshot(
        [
            {
                "productId": "ABT-USD",
                "signal_name": "HOLD",
                "modelSignalName": "HOLD",
                "tradeReadiness": "standby",
                "confidence": 0.91,
                "policyScore": 1.76,
                "setupScore": 1.0,
                "brain": {
                    "decision": "watchlist",
                    "decisionScore": 0.62,
                    "summaryLine": "ABT-USD stays on the watchlist.",
                },
                "reasonSummary": "ABT-USD stays on the watchlist.",
            },
            {
                "productId": "AGLD-USD",
                "signal_name": "HOLD",
                "modelSignalName": "BUY",
                "tradeReadiness": "blocked",
                "confidence": 0.60,
                "policyScore": 0.52,
                "setupScore": 0.25,
                "brain": {
                    "decision": "watchlist",
                    "decisionScore": 0.41,
                    "summaryLine": "AGLD-USD is close to a breakout entry.",
                },
                "reasonSummary": "AGLD-USD is close to a breakout entry.",
            },
            {
                "productId": "ADA-USD",
                "signal_name": "HOLD",
                "modelSignalName": "HOLD",
                "tradeReadiness": "standby",
                "confidence": 0.69,
                "policyScore": 1.24,
                "setupScore": 1.5,
                "brain": {
                    "decision": "watchlist",
                    "decisionScore": 0.37,
                    "summaryLine": "ADA-USD stays in observation mode.",
                },
                "reasonSummary": "ADA-USD stays in observation mode.",
            },
        ]
    )

    pool_store = WatchlistPoolStore(watchlist_pool_path)
    snapshot = pool_store.get_snapshot()

    assert snapshot["count"] == 2
    assert snapshot["productIds"] == ["AGLD-USD", "ABT-USD"]
    assert pool_store.get_monitored_product_ids() == ["AGLD-USD", "ABT-USD"]


def test_coinmarketcap_universe_refresh_service_reuses_cached_universe_after_rate_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale tracked-universe cache should be reused when CoinMarketCap returns a rate-limit error."""

    cache_path = tmp_path / "coinmarketcapUniverse.json"
    stale_fetched_at = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    stale_expires_at = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
    cache_path.write_text(
        json.dumps(
            {
                "source": "coinmarketcapUniverse",
                "marketDataSource": "coinmarketcapLatestQuotes",
                "quoteCurrency": "USD",
                "lastFetchedAt": stale_fetched_at,
                "expiresAt": stale_expires_at,
                "ttlSeconds": 21600,
                "sourceStatus": "refreshed",
                "lastError": "",
                "rateLimitedUntil": None,
                "productCount": 1,
                "productIds": ["ADA-USD"],
                "products": [
                    {
                        "productId": "ADA-USD",
                        "baseCurrency": "ADA",
                        "quoteCurrency": "USD",
                        "marketCapRank": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = TrainingConfig(
        market_data_source="coinmarketcapLatestQuotes",
        coinmarketcap_fetch_all_quote_products=True,
        coinmarketcap_universe_cache_file=cache_path,
    )
    service = CoinMarketCapUniverseRefreshService(config)
    monkeypatch.setenv(config.coinmarketcap_api_key_env_var, "test-key")

    def fake_fetch_ranked_products(self) -> list[dict[str, object]]:
        del self
        raise CoinMarketCapRateLimitError("CoinMarketCap request failed with status 429.")

    monkeypatch.setattr(CoinMarketCapUniverseRefreshService, "_fetch_ranked_products", fake_fetch_ranked_products)

    payload = service.resolve()
    persisted_payload = json.loads(cache_path.read_text(encoding="utf-8"))

    assert payload["sourceStatus"] == "cached_rate_limited"
    assert payload["usedCachedSnapshot"] is True
    assert payload["refreshAttempted"] is True
    assert payload["productIds"] == ["ADA-USD"]
    assert payload["rateLimitedUntil"] is not None
    assert persisted_payload["productIds"] == ["ADA-USD"]
    assert persisted_payload["sourceStatus"] == "cached_rate_limited"


def test_market_refresh_uses_cached_coinmarketcap_universe_after_rate_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Market refresh should reuse the CMC universe cache instead of crashing on map 429."""

    cache_path = tmp_path / "coinmarketcapUniverse.json"
    stale_fetched_at = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    stale_expires_at = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
    cache_path.write_text(
        json.dumps(
            {
                "source": "coinmarketcapUniverse",
                "marketDataSource": "coinmarketcapLatestQuotes",
                "quoteCurrency": "USD",
                "lastFetchedAt": stale_fetched_at,
                "expiresAt": stale_expires_at,
                "ttlSeconds": 21600,
                "sourceStatus": "refreshed",
                "lastError": "",
                "rateLimitedUntil": None,
                "productCount": 2,
                "productIds": ["ADA-USD", "BTC-USD"],
                "products": [
                    {
                        "productId": "ADA-USD",
                        "baseCurrency": "ADA",
                        "quoteCurrency": "USD",
                        "marketCapRank": 1,
                    },
                    {
                        "productId": "BTC-USD",
                        "baseCurrency": "BTC",
                        "quoteCurrency": "USD",
                        "marketCapRank": 2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = TrainingConfig(
        data_file=tmp_path / "marketPrices.csv",
        market_data_source="coinmarketcapLatestQuotes",
        coinmarketcap_fetch_all_quote_products=True,
        coinmarketcap_product_batch_size=1,
        coinmarketcap_product_batch_number=1,
        coinmarketcap_universe_cache_file=cache_path,
        coinmarketcap_use_context=False,
        coinmarketcap_use_market_intelligence=False,
        coinmarketcal_use_events=False,
        market_product_batch_rotation_enabled=True,
        market_product_batch_state_file=tmp_path / "marketProductBatchState.json",
    )
    refresh_app = MarketDataRefreshApp(config=config)
    monkeypatch.setenv(config.coinmarketcap_api_key_env_var, "test-key")

    def fake_fetch_ranked_products(self) -> list[dict[str, object]]:
        del self
        raise CoinMarketCapRateLimitError("CoinMarketCap request failed with status 429.")

    def fail_fetch_filtered_products(self) -> list[dict[str, str]]:
        del self
        raise AssertionError("market refresh should use cached explicit CMC products")

    def fake_request_latest_quote_rows(
        self: CoinMarketCapLatestQuotesPriceDataLoader,
        selected_products: Sequence[dict[str, str]],
    ) -> list[dict[str, object]]:
        product = selected_products[0]
        return [
            {
                "timestamp": pd.Timestamp("2026-05-03T00:00:00Z"),
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 100.0,
                "product_id": product["product_id"],
                "base_currency": product["base_currency"],
                "quote_currency": product["quote_currency"],
                "granularity_seconds": self.granularity_seconds,
                "source": self.api_name,
            }
        ]

    monkeypatch.setattr(CoinMarketCapUniverseRefreshService, "_fetch_ranked_products", fake_fetch_ranked_products)
    monkeypatch.setattr(CoinMarketCapLatestQuotesPriceDataLoader, "_fetch_filtered_products", fail_fetch_filtered_products)
    monkeypatch.setattr(
        CoinMarketCapLatestQuotesPriceDataLoader,
        "_request_latest_quote_rows",
        fake_request_latest_quote_rows,
    )

    result = refresh_app.run()

    assert result["rowsDownloaded"] == 1
    assert result["totalBatches"] == 2
    assert result["universeRefresh"]["sourceStatus"] == "cached_rate_limited"
    assert result["universeRefresh"]["usedCachedSnapshot"] is True
    assert result["downloadSummary"]["totalAvailableProducts"] == 2


def test_signal_universe_coordinator_prioritizes_follow_up_products_before_cached_discovery(
    tmp_path: Path,
) -> None:
    """Open positions, watchlist names, and published signals should be selected before cached discovery names."""

    watchlist_pool_path = tmp_path / "watchlistPool.json"
    watchlist_pool_path.write_text(
        json.dumps(
            {
                "generatedAt": "2026-04-08T10:00:00+00:00",
                "count": 1,
                "productIds": ["SOL-USD"],
                "products": [
                    {
                        "productId": "SOL-USD",
                        "signalName": "HOLD",
                        "modelSignalName": "BUY",
                        "brainDecision": "watchlist",
                        "decisionScore": 0.52,
                        "confidence": 0.71,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    universe_cache_path = tmp_path / "coinmarketcapUniverse.json"
    fetched_at = datetime.now(timezone.utc).isoformat()
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
    universe_cache_path.write_text(
        json.dumps(
            {
                "source": "coinmarketcapUniverse",
                "marketDataSource": "coinmarketcapLatestQuotes",
                "quoteCurrency": "USD",
                "lastFetchedAt": fetched_at,
                "expiresAt": expires_at,
                "ttlSeconds": 21600,
                "sourceStatus": "refreshed",
                "lastError": "",
                "rateLimitedUntil": None,
                "productCount": 3,
                "productIds": ["ADA-USD", "XRP-USD", "DOGE-USD"],
                "products": [
                    {
                        "productId": "ADA-USD",
                        "baseCurrency": "ADA",
                        "quoteCurrency": "USD",
                        "marketCapRank": 1,
                    },
                    {
                        "productId": "XRP-USD",
                        "baseCurrency": "XRP",
                        "quoteCurrency": "USD",
                        "marketCapRank": 2,
                    },
                    {
                        "productId": "DOGE-USD",
                        "baseCurrency": "DOGE",
                        "quoteCurrency": "USD",
                        "marketCapRank": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    portfolio_store = TradingPortfolioStore(db_path=tmp_path / "portfolio.sqlite3")
    portfolio_store.upsert_position(
        product_id="BTC-USD",
        quantity=1.0,
        entry_price=65000.0,
        current_price=65200.0,
    )

    signal_store = TradingSignalStore(db_path=tmp_path / "signals.sqlite3")
    signal_store.replace_current_signals(
        signal_summaries=[
            {
                "productId": "ETH-USD",
                "signal_name": "HOLD",
                "spotAction": "wait",
                "actionable": False,
                "confidence": 0.62,
                "close": 3300.0,
                "timestamp": "2026-04-08T10:00:00+00:00",
            }
        ],
        primary_signal={
            "productId": "ETH-USD",
            "signal_name": "HOLD",
            "spotAction": "wait",
            "actionable": False,
            "confidence": 0.62,
            "close": 3300.0,
            "timestamp": "2026-04-08T10:00:00+00:00",
        },
        generated_at="2026-04-08T10:00:00+00:00",
    )

    config = TrainingConfig(
        data_file=tmp_path / "marketPrices.csv",
        market_data_source="coinmarketcapLatestQuotes",
        coinmarketcap_fetch_all_quote_products=True,
        live_fetch_all_quote_products=True,
        live_max_products=4,
        signal_watchlist_pool_enabled=True,
        signal_watchlist_pool_max_products=3,
        signal_watchlist_pool_path=watchlist_pool_path,
        signal_watchlist_state_path=tmp_path / "watchlistState.json",
        coinmarketcap_universe_cache_file=universe_cache_path,
        portfolio_store_path=tmp_path / "portfolio.sqlite3",
        signal_store_path=tmp_path / "signals.sqlite3",
        signal_excluded_base_currencies=("BTC", "ETH", "USDT", "USDC"),
    )

    plan = SignalUniverseCoordinator(config).resolve_active_universe(max_products=4)

    assert plan.product_ids == ["BTC-USD", "SOL-USD", "ETH-USD", "ADA-USD"]
    assert plan.protected_product_ids == ["BTC-USD", "SOL-USD", "ETH-USD"]
    assert plan.source_refresh["sourceStatus"] == "cache_valid"
    assert plan.summary["selectedCounts"]["openPositions"] == 1
    assert plan.summary["selectedCounts"]["watchlist"] == 1
    assert plan.summary["selectedCounts"]["publishedSignals"] == 1
    assert plan.summary["selectedCounts"]["trackedUniverse"] == 1


def test_signal_generation_app_uses_prioritized_active_universe_for_cmc_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CMC fresh scoring should request one explicit prioritized universe instead of a broad quote refresh."""

    import crypto_signal_ml.app as app_module

    watchlist_pool_path = tmp_path / "watchlistPool.json"
    watchlist_pool_path.write_text(
        json.dumps(
            {
                "generatedAt": "2026-04-07T14:45:00+00:00",
                "count": 1,
                "productIds": ["AGLD-USD"],
                "products": [
                    {
                        "productId": "AGLD-USD",
                        "signalName": "HOLD",
                        "modelSignalName": "BUY",
                        "brainDecision": "watchlist",
                        "decisionScore": 0.41,
                        "confidence": 0.60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = TrainingConfig(
        market_data_source="coinmarketcapLatestQuotes",
        signal_watchlist_pool_enabled=True,
        signal_watchlist_pool_max_products=3,
        signal_watchlist_pool_path=watchlist_pool_path,
        live_max_products=100,
    )

    class FakeModel:
        feature_columns = ("close",)
        model_type = "fakeModel"

        def __init__(self, config: TrainingConfig) -> None:
            self.config = config

        def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
            return feature_df.copy()

    class FakeLoader:
        def __init__(self, frame: pd.DataFrame, summary: dict[str, int]) -> None:
            self._frame = frame
            self.last_refresh_summary = summary

        def refresh_data(self) -> pd.DataFrame:
            return self._frame.copy()

    class FakeDatasetBuilder:
        def __init__(
            self,
            config: TrainingConfig,
            feature_columns: Sequence[str] | None = None,
            data_loader: FakeLoader | None = None,
        ) -> None:
            del config
            del feature_columns
            self.data_loader = data_loader

        def build_feature_table(self) -> pd.DataFrame:
            if self.data_loader is None:
                raise AssertionError("This fake dataset builder should only be used with an explicit loader.")
            return self.data_loader.refresh_data()

    prioritized_prediction_df = pd.DataFrame(
        [
            {
                "time_step": 10,
                "timestamp": "2026-04-07T14:39:00Z",
                "product_id": "AGLD-USD",
                "base_currency": "AGLD",
                "quote_currency": "USD",
                "close": 0.31,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.82,
                "prob_take_profit": 0.08,
                "prob_hold": 0.10,
                "prob_buy": 0.82,
            },
            {
                "time_step": 10,
                "timestamp": "2026-04-07T14:39:00Z",
                "product_id": "ABT-USD",
                "base_currency": "ABT",
                "quote_currency": "USD",
                "close": 0.38,
                "predicted_signal": 0,
                "predicted_name": "HOLD",
                "confidence": 0.91,
                "prob_take_profit": 0.04,
                "prob_hold": 0.91,
                "prob_buy": 0.05,
            },
        ]
    )

    def fake_create_market_data_loader(
        config: TrainingConfig,
        data_path: Path,
        should_save_downloaded_data: bool,
        product_ids: Sequence[str] | None = None,
        fetch_all_quote_products: bool | None = None,
        max_products: int | None = None,
        granularity_seconds: int | None = None,
        total_candles: int | None = None,
        request_pause_seconds: float | None = None,
        save_progress_every_products: int | None = None,
        log_progress: bool | None = None,
    ) -> FakeLoader:
        del config
        del data_path
        del should_save_downloaded_data
        del max_products
        del granularity_seconds
        del total_candles
        del request_pause_seconds
        del save_progress_every_products
        del log_progress
        assert fetch_all_quote_products is False
        assert tuple(product_ids or ()) == ("AGLD-USD", "ABT-USD")
        return FakeLoader(
            prioritized_prediction_df,
            {"productsDownloaded": 2, "totalAvailableProducts": 100},
        )

    def fake_resolve_active_universe(self, *, max_products: int | None = None) -> ActiveUniversePlan:
        del self
        del max_products
        return ActiveUniversePlan(
            product_ids=["AGLD-USD", "ABT-USD"],
            protected_product_ids=["AGLD-USD"],
            summary={
                "mode": "prioritized-active-universe",
                "effectiveLimit": 2,
                "selectedProductIds": ["AGLD-USD", "ABT-USD"],
                "watchlist": {
                    "count": 1,
                    "productIds": ["AGLD-USD"],
                },
                "trackedUniverse": {
                    "count": 100,
                },
            },
            source_refresh={
                "sourceStatus": "cache_valid",
                "productCount": 100,
                "usedCachedSnapshot": True,
            },
        )

    monkeypatch.setattr(app_module, "create_market_data_loader", fake_create_market_data_loader)
    monkeypatch.setattr(app_module, "CryptoDatasetBuilder", FakeDatasetBuilder)
    monkeypatch.setattr(app_module.SignalUniverseCoordinator, "resolve_active_universe", fake_resolve_active_universe)

    app = SignalGenerationApp(
        config=config,
        model=FakeModel(config=config),
    )

    prediction_df, summary = app._build_fresh_signal_prediction_frame()

    assert set(prediction_df["product_id"]) == {"ABT-USD", "AGLD-USD"}
    assert summary["mode"] == "prioritized-active-universe"
    assert summary["productsRequested"] == 2
    assert summary["sourceRefresh"]["sourceStatus"] == "cache_valid"
    assert summary["activeUniverse"]["watchlist"]["productIds"] == ["AGLD-USD"]


def test_signal_generation_app_auto_tracks_new_entry_signals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generating a fresh BUY signal should automatically create a tracked trade record."""

    prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-04-05T12:00:00Z",
                "product_id": "ALGO-USD",
                "base_currency": "ALGO",
                "quote_currency": "USD",
                "close": 0.25,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.86,
                "prob_take_profit": 0.04,
                "prob_hold": 0.10,
                "prob_buy": 0.86,
                "breakout_up_20": 0.04,
                "breakout_down_20": 0.01,
                "close_vs_ema_5": 0.03,
                "relative_strength_1": 0.02,
                "relative_strength_5": 0.05,
                "momentum_10": 0.07,
                "range_position_20": 0.93,
                "rsi_14": 63.0,
                "trend_score": 0.03,
                "volatility_ratio": 0.95,
                "regime_label": "trend_up",
                "regime_code": 2,
                "regime_is_trending": 1,
                "regime_is_high_volatility": 0,
                "cmc_context_available": 0,
            }
        ]
    )

    class FakeDatasetBuilder:
        def build_feature_table(self) -> pd.DataFrame:
            return pd.DataFrame([{"feature_1": 1.0}])

    class FakeModel:
        model_type = "histGradientBoostingSignalModel"

        def __init__(self, config: TrainingConfig) -> None:
            self.config = config
            self.feature_columns = ["feature_1"]

        def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
            assert list(feature_df.columns) == ["feature_1"]
            return prediction_df.copy()

    config = TrainingConfig(
        coinmarketcap_use_context=False,
        market_data_source="coinmarketcapLatestQuotes",
        signal_excluded_base_currencies=(),
        signal_refresh_market_data_before_generation=False,
        signal_track_generated_trades=True,
        signal_generated_trade_status="planned",
        portfolio_store_path=tmp_path / "traderPortfolio.sqlite3",
    )
    app = SignalGenerationApp(
        config=config,
        dataset_builder=FakeDatasetBuilder(),
        model=FakeModel(config=config),
    )

    monkeypatch.setattr(app, "save_dataframe", lambda dataframe, file_path: None)
    monkeypatch.setattr(app, "save_json", lambda payload, file_path: None)

    result = app.run()

    portfolio_store = TradingPortfolioStore(
        db_path=config.portfolio_store_path,
        default_capital=config.portfolio_default_capital,
    )
    tracked_trade = portfolio_store.get_active_trade_for_product("ALGO-USD")

    assert result["trackedTradeSync"]["enabled"] is True
    assert result["trackedTradeSync"]["createdCount"] == 1
    assert result["trackedTradeSync"]["refreshedCount"] == 0
    assert tracked_trade is not None
    assert tracked_trade["entryPrice"] == 0.25
    assert tracked_trade["signalName"] == "BUY"
    assert tracked_trade["status"] == "planned"
    assert tracked_trade["metadata"]["autoTrackedFromSignalGeneration"] is True


def test_signal_generation_app_refreshes_existing_tracked_trade_instead_of_duplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeating the same BUY setup should refresh the tracked trade, not create duplicates."""

    prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-04-05T13:00:00Z",
                "product_id": "ALGO-USD",
                "base_currency": "ALGO",
                "quote_currency": "USD",
                "close": 0.30,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.88,
                "prob_take_profit": 0.03,
                "prob_hold": 0.09,
                "prob_buy": 0.88,
                "breakout_up_20": 0.05,
                "breakout_down_20": 0.01,
                "close_vs_ema_5": 0.03,
                "relative_strength_1": 0.02,
                "relative_strength_5": 0.05,
                "momentum_10": 0.07,
                "range_position_20": 0.94,
                "rsi_14": 64.0,
                "trend_score": 0.03,
                "volatility_ratio": 0.90,
                "regime_label": "trend_up",
                "regime_code": 2,
                "regime_is_trending": 1,
                "regime_is_high_volatility": 0,
                "cmc_context_available": 0,
            }
        ]
    )

    class FakeDatasetBuilder:
        def build_feature_table(self) -> pd.DataFrame:
            return pd.DataFrame([{"feature_1": 1.0}])

    class FakeModel:
        model_type = "histGradientBoostingSignalModel"

        def __init__(self, config: TrainingConfig) -> None:
            self.config = config
            self.feature_columns = ["feature_1"]

        def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
            return prediction_df.copy()

    config = TrainingConfig(
        coinmarketcap_use_context=False,
        market_data_source="coinmarketcapLatestQuotes",
        signal_excluded_base_currencies=(),
        signal_refresh_market_data_before_generation=False,
        signal_track_generated_trades=True,
        signal_generated_trade_status="planned",
        portfolio_store_path=tmp_path / "traderPortfolio.sqlite3",
    )
    portfolio_store = TradingPortfolioStore(
        db_path=config.portfolio_store_path,
        default_capital=config.portfolio_default_capital,
    )
    existing_trade = portfolio_store.create_trade(
        product_id="ALGO-USD",
        entry_price=0.25,
        take_profit_price=0.28,
        stop_loss_price=0.23,
        current_price=0.25,
        signal_name="BUY",
        status="planned",
        metadata={"autoTrackedFromSignalGeneration": True},
    )

    app = SignalGenerationApp(
        config=config,
        dataset_builder=FakeDatasetBuilder(),
        model=FakeModel(config=config),
    )
    monkeypatch.setattr(app, "save_dataframe", lambda dataframe, file_path: None)
    monkeypatch.setattr(app, "save_json", lambda payload, file_path: None)

    result = app.run()

    refreshed_trade = portfolio_store.get_active_trade_for_product("ALGO-USD")
    all_trades = portfolio_store.list_trades(limit=10)

    assert result["trackedTradeSync"]["createdCount"] == 0
    assert result["trackedTradeSync"]["refreshedCount"] == 1
    assert refreshed_trade is not None
    assert refreshed_trade["tradeId"] == existing_trade["tradeId"]
    assert refreshed_trade["currentPrice"] == 0.30
    assert len(all_trades) == 1


def test_signal_generation_app_tracks_published_buy_even_when_brain_stays_watchlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Published BUY calls should still start lifecycle tracking when the brain is cautious."""

    prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-04-05T13:00:00Z",
                "product_id": "ALGO-USD",
                "base_currency": "ALGO",
                "quote_currency": "USD",
                "close": 0.25,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.52,
                "prob_take_profit": 0.21,
                "prob_hold": 0.27,
                "prob_buy": 0.52,
                "breakout_up_20": 0.04,
                "breakout_down_20": 0.01,
                "close_vs_ema_5": 0.01,
                "relative_strength_1": 0.01,
                "relative_strength_5": 0.03,
                "momentum_10": 0.04,
                "range_position_20": 0.83,
                "rsi_14": 58.0,
                "trend_score": 0.02,
                "volatility_ratio": 1.0,
                "regime_label": "trend_up",
                "regime_code": 2,
                "regime_is_trending": 1,
                "regime_is_high_volatility": 0,
                "cmc_context_available": 1,
                "cmc_market_intelligence_available": 1,
                "cmc_market_fear_greed_value": 38.0,
                "cmc_market_fear_greed_classification": "Fear",
                "cmc_market_btc_dominance": 0.58,
                "cmc_market_btc_dominance_change_24h": 0.002,
                "cmc_market_altcoin_share": 0.42,
                "cmc_market_stablecoin_share": 0.12,
                "cmc_market_total_market_cap_change_24h": 0.03,
                "cmc_market_total_volume_change_24h": 0.8,
            }
        ]
    )

    class FakeDatasetBuilder:
        def build_feature_table(self) -> pd.DataFrame:
            return pd.DataFrame([{"feature_1": 1.0}])

    class FakeModel:
        model_type = "histGradientBoostingSignalModel"

        def __init__(self, config: TrainingConfig) -> None:
            self.config = config
            self.feature_columns = ["feature_1"]

        def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
            return prediction_df.copy()

    config = TrainingConfig(
        coinmarketcap_use_context=False,
        market_data_source="coinmarketcapLatestQuotes",
        signal_excluded_base_currencies=(),
        signal_refresh_market_data_before_generation=False,
        signal_track_generated_trades=True,
        signal_generated_trade_status="planned",
        backtest_min_confidence=0.50,
        portfolio_store_path=tmp_path / "traderPortfolio.sqlite3",
    )
    app = SignalGenerationApp(
        config=config,
        dataset_builder=FakeDatasetBuilder(),
        model=FakeModel(config=config),
    )

    monkeypatch.setattr(app, "save_dataframe", lambda dataframe, file_path: None)
    monkeypatch.setattr(app, "save_json", lambda payload, file_path: None)

    result = app.run()

    portfolio_store = TradingPortfolioStore(
        db_path=config.portfolio_store_path,
        default_capital=config.portfolio_default_capital,
    )
    tracked_trade = portfolio_store.get_active_trade_for_product("ALGO-USD")

    assert result["trackedTradeSync"]["enabled"] is True
    assert result["trackedTradeSync"]["createdCount"] == 1
    assert tracked_trade is not None
    assert tracked_trade["signalName"] == "BUY"
    assert tracked_trade["status"] == "planned"


def test_portfolio_store_builds_trade_learning_snapshot(tmp_path: Path) -> None:
    """Closed tracked trades should be summarized into reusable decision memory."""

    store = TradingPortfolioStore(db_path=tmp_path / "tradeLearning.sqlite3")
    first_trade = store.create_trade(
        product_id="ALGO-USD",
        entry_price=1.00,
        take_profit_price=1.12,
        stop_loss_price=0.94,
        quantity=10.0,
        signal_name="BUY",
        status="open",
        opened_at="2026-01-01T00:00:00Z",
    )
    second_trade = store.create_trade(
        product_id="ALGO-USD",
        entry_price=1.00,
        take_profit_price=1.14,
        stop_loss_price=0.95,
        quantity=10.0,
        signal_name="BUY",
        status="open",
        opened_at="2026-01-03T00:00:00Z",
    )
    third_trade = store.create_trade(
        product_id="ALGO-USD",
        entry_price=1.00,
        take_profit_price=1.16,
        stop_loss_price=0.95,
        quantity=10.0,
        signal_name="BUY",
        status="open",
        opened_at="2026-01-05T00:00:00Z",
    )

    store.close_trade(
        trade_id=first_trade["tradeId"],
        exit_price=1.08,
        closed_at="2026-01-02T00:00:00Z",
        close_reason="take_profit",
    )
    store.close_trade(
        trade_id=second_trade["tradeId"],
        exit_price=0.96,
        closed_at="2026-01-04T00:00:00Z",
        close_reason="stop_loss",
    )
    store.close_trade(
        trade_id=third_trade["tradeId"],
        exit_price=1.10,
        closed_at="2026-01-06T00:00:00Z",
        close_reason="take_profit",
    )

    snapshot = store.get_trade_learning_snapshot("ALGO-USD", signal_name="BUY")
    learning_map = store.build_trade_learning_map(
        [{"productId": "ALGO-USD", "signal_name": "BUY"}]
    )

    assert snapshot["available"] is True
    assert snapshot["closedTradeCount"] == 3
    assert snapshot["winCount"] == 2
    assert snapshot["lossCount"] == 1
    assert snapshot["scope"] == "product+signal"
    assert snapshot["sampleAdequate"] is True
    assert snapshot["lastOutcome"] == "win"
    assert snapshot["recentOutcomes"][:3] == ["win", "loss", "win"]
    assert learning_map["ALGO-USD"]["closedTradeCount"] == 3


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

    latest_signals = build_latest_signal_summaries(
        prediction_df,
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            signal_excluded_base_currencies=(),
        ),
    )
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


def test_signal_policy_downgrades_downtrend_buy_to_hold() -> None:
    """The trading policy should block fresh BUY signals during a downtrend regime."""

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
                "confidence": 0.84,
                "prob_take_profit": 0.06,
                "prob_hold": 0.10,
                "prob_buy": 0.84,
                "breakout_up_20": 0.03,
                "close_vs_ema_5": 0.02,
                "relative_strength_1": 0.01,
                "relative_strength_5": 0.03,
                "momentum_10": 0.08,
                "range_position_20": 0.91,
                "rsi_14": 62.0,
                "market_regime_label": "trend_down",
                "market_regime_code": 3.0,
                "regime_trend_score": -0.03,
                "regime_volatility_ratio": 1.0,
                "regime_is_trending": 1.0,
                "regime_is_high_volatility": 0.0,
                "cmcal_has_event_next_7d": 0,
                "cmcal_event_count_next_7d": 0,
                "cmcal_event_count_next_30d": 0,
            }
        ]
    )

    latest_signal = build_latest_signal_summaries(
        prediction_df,
        minimum_action_confidence=0.65,
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            signal_excluded_base_currencies=(),
        ),
    )[0]

    assert latest_signal["modelSignalName"] == "BUY"
    assert latest_signal["signal_name"] == "HOLD"
    assert latest_signal["confidenceGateApplied"] is False
    assert latest_signal["riskGateApplied"] is True
    assert latest_signal["tradeReadiness"] == "blocked"
    assert "downtrend" in latest_signal["gateReasons"][0].lower()


def test_signal_summaries_exclude_configured_base_currencies() -> None:
    """Configured majors and stablecoins should not surface in the signal output."""

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
                "prob_take_profit": 0.06,
                "prob_hold": 0.12,
                "prob_buy": 0.82,
            },
            {
                "time_step": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "ETH-USD",
                "base_currency": "ETH",
                "quote_currency": "USD",
                "close": 210.0,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.79,
                "prob_take_profit": 0.08,
                "prob_hold": 0.13,
                "prob_buy": 0.79,
            },
            {
                "time_step": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "USDT-USD",
                "base_currency": "USDT",
                "quote_currency": "USD",
                "close": 1.0,
                "predicted_signal": 0,
                "predicted_name": "HOLD",
                "confidence": 0.90,
                "prob_take_profit": 0.02,
                "prob_hold": 0.90,
                "prob_buy": 0.08,
            },
            {
                "time_step": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "SOL-USD",
                "base_currency": "SOL",
                "quote_currency": "USD",
                "close": 150.0,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.76,
                "prob_take_profit": 0.09,
                "prob_hold": 0.15,
                "prob_buy": 0.76,
            },
        ]
    )

    latest_signals = build_latest_signal_summaries(
        prediction_df,
        minimum_action_confidence=0.65,
        config=TrainingConfig(coinmarketcap_use_context=False),
    )

    assert [signal_summary["productId"] for signal_summary in latest_signals] == ["SOL-USD"]


def test_build_latest_signal_summaries_keeps_protected_follow_up_products() -> None:
    """Protected follow-up products should survive the normal excluded-base filter."""

    prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "BTC-USD",
                "base_currency": "BTC",
                "quote_currency": "USD",
                "close": 98000.0,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.79,
                "prob_take_profit": 0.06,
                "prob_hold": 0.15,
                "prob_buy": 0.79,
            }
        ]
    )

    latest_signals = build_latest_signal_summaries(
        prediction_df,
        minimum_action_confidence=0.65,
        config=TrainingConfig(coinmarketcap_use_context=False),
        protected_product_ids=["BTC-USD"],
    )

    assert [signal_summary["productId"] for signal_summary in latest_signals] == ["BTC-USD"]


def test_live_signal_engine_filters_excluded_default_watchlist_products() -> None:
    """Live watchlists should drop excluded bases before a refresh request is built."""

    live_engine = LiveSignalEngine(
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_fetch_all_quote_products=False,
            live_product_ids=("BTC-USD", "ETH-USD", "SOL-USD", "USDC-USD"),
        )
    )

    assert live_engine._resolve_requested_products(product_id=None, product_ids=None) == ["SOL-USD"]


def test_live_signal_engine_prefers_quote_universe_when_enabled() -> None:
    """Live mode should use the broader quote universe by default when enabled."""

    live_engine = LiveSignalEngine(
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            live_fetch_all_quote_products=True,
            live_product_ids=("BTC-USD", "ETH-USD", "SOL-USD"),
        )
    )

    assert live_engine._should_use_quote_universe(product_id=None, product_ids=None) is True
    assert live_engine._resolve_requested_products(product_id=None, product_ids=None) == []


def test_live_signal_engine_uses_prioritized_active_universe_for_cmc_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default CMC live requests should score the prioritized active universe instead of the broad quote set."""

    import crypto_signal_ml.live as live_module

    watchlist_pool_path = tmp_path / "watchlistPool.json"
    watchlist_pool_path.write_text(
        json.dumps(
            build_watchlist_pool_snapshot(
                [
                    {
                        "productId": "AGLD-USD",
                        "signal_name": "HOLD",
                        "modelSignalName": "BUY",
                        "tradeReadiness": "blocked",
                        "confidence": 0.60,
                        "policyScore": 0.52,
                        "setupScore": 0.25,
                        "brain": {
                            "decision": "watchlist",
                            "decisionScore": 0.41,
                            "summaryLine": "AGLD-USD is close to a breakout entry.",
                        },
                        "reasonSummary": "AGLD-USD is close to a breakout entry.",
                    }
                ],
                max_products=3,
            )
        ),
        encoding="utf-8",
    )

    config = TrainingConfig(
        market_data_source="coinmarketcapLatestQuotes",
        signal_watchlist_pool_enabled=True,
        signal_watchlist_pool_max_products=3,
        signal_watchlist_pool_path=watchlist_pool_path,
        live_fetch_all_quote_products=True,
        live_max_products=100,
        live_signal_cache_seconds=60,
        live_watchlist_pool_cache_seconds=15,
        portfolio_store_path=tmp_path / "portfolio.sqlite3",
    )

    class FakeModel:
        feature_columns = ("close",)
        model_type = "fakeModel"

        def __init__(self, config: TrainingConfig) -> None:
            self.config = config

        def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
            return feature_df.copy()

    class FakeLoader:
        def __init__(self, frame: pd.DataFrame) -> None:
            self._frame = frame

        def refresh_data(self) -> pd.DataFrame:
            return self._frame.copy()

    class FakeDatasetBuilder:
        def __init__(
            self,
            config: TrainingConfig,
            feature_columns: Sequence[str] | None = None,
            data_loader: FakeLoader | None = None,
        ) -> None:
            del config
            del feature_columns
            self.data_loader = data_loader

        def build_feature_table(self) -> pd.DataFrame:
            if self.data_loader is None:
                raise AssertionError("This fake dataset builder should only be used with an explicit loader.")
            return self.data_loader.refresh_data()

    prioritized_prediction_df = pd.DataFrame(
        [
            {
                "time_step": 10,
                "timestamp": "2026-04-07T14:39:00Z",
                "product_id": "AGLD-USD",
                "base_currency": "AGLD",
                "quote_currency": "USD",
                "close": 0.31,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.82,
                "prob_take_profit": 0.08,
                "prob_hold": 0.10,
                "prob_buy": 0.82,
                "market_regime_label": "trend_up",
                "market_regime_code": 2,
                "regime_is_trending": 1,
                "regime_is_high_volatility": 0,
                "cmcal_has_event_next_7d": 0,
            },
            {
                "time_step": 10,
                "timestamp": "2026-04-07T14:39:00Z",
                "product_id": "ABT-USD",
                "base_currency": "ABT",
                "quote_currency": "USD",
                "close": 0.38,
                "predicted_signal": 0,
                "predicted_name": "HOLD",
                "confidence": 0.91,
                "prob_take_profit": 0.04,
                "prob_hold": 0.91,
                "prob_buy": 0.05,
                "market_regime_label": "range",
                "market_regime_code": 0,
                "regime_is_high_volatility": 0,
                "cmcal_has_event_next_7d": 0,
            },
        ]
    )

    def fake_create_market_data_loader(
        config: TrainingConfig,
        data_path: Path,
        should_save_downloaded_data: bool,
        product_ids: Sequence[str] | None = None,
        fetch_all_quote_products: bool | None = None,
        max_products: int | None = None,
        granularity_seconds: int | None = None,
        total_candles: int | None = None,
        request_pause_seconds: float | None = None,
        save_progress_every_products: int | None = None,
        log_progress: bool | None = None,
    ) -> FakeLoader:
        del config
        del data_path
        del should_save_downloaded_data
        del max_products
        del granularity_seconds
        del total_candles
        del request_pause_seconds
        del save_progress_every_products
        del log_progress
        assert fetch_all_quote_products is False
        assert tuple(product_ids or ()) == ("AGLD-USD", "ABT-USD")
        return FakeLoader(prioritized_prediction_df)

    def fake_resolve_active_universe(self, *, max_products: int | None = None) -> ActiveUniversePlan:
        del self
        del max_products
        return ActiveUniversePlan(
            product_ids=["AGLD-USD", "ABT-USD"],
            protected_product_ids=["AGLD-USD"],
            summary={
                "mode": "prioritized-active-universe",
                "effectiveLimit": 2,
                "selectedProductIds": ["AGLD-USD", "ABT-USD"],
                "watchlist": {
                    "count": 1,
                    "productIds": ["AGLD-USD"],
                },
            },
            source_refresh={
                "sourceStatus": "cache_valid",
                "productCount": 100,
                "usedCachedSnapshot": True,
            },
        )

    live_engine = LiveSignalEngine(config=config)

    monkeypatch.setattr(live_module, "create_market_data_loader", fake_create_market_data_loader)
    monkeypatch.setattr(live_module, "CryptoDatasetBuilder", FakeDatasetBuilder)
    monkeypatch.setattr(live_module.SignalUniverseCoordinator, "resolve_active_universe", fake_resolve_active_universe)
    monkeypatch.setattr(live_engine, "_load_model", lambda: FakeModel(config=config))

    snapshot = live_engine.get_live_snapshot(force_refresh=True)

    assert snapshot["requestMode"] == "prioritized-active-universe"
    assert snapshot["liveSignalCacheSeconds"] == 15
    assert snapshot["watchlistPool"]["active"] is True
    assert snapshot["watchlistPool"]["productIds"] == ["AGLD-USD"]
    assert snapshot["marketSummary"]["actionableSignals"] == 1
    assert snapshot["primarySignal"]["productId"] == "AGLD-USD"
    assert snapshot["primarySignal"]["signal_name"] == "BUY"
    assert snapshot["signals"][0]["productId"] == "AGLD-USD"
    assert snapshot["signalInference"]["mode"] == "prioritized-active-universe"
    assert snapshot["signalInference"]["sourceRefresh"]["sourceStatus"] == "cache_valid"


def test_signal_monitor_service_runs_initial_generation_before_serving() -> None:
    """The single-command monitor should seed signals before starting the live API."""

    refresh_calls: list[str] = []
    create_app_calls: list[dict[str, object]] = []
    uvicorn_calls: list[dict[str, object]] = []

    class FakeSignalGenerationApp:
        def __init__(self, config: TrainingConfig) -> None:
            self.config = config

        def run(self) -> dict[str, object]:
            refresh_calls.append("run")
            return {
                "signalsGenerated": 1,
                "actionableSignalsGenerated": 0,
                "signalName": "BUY",
            }

    fake_app = object()

    def fake_create_app(**kwargs):
        create_app_calls.append(kwargs)
        return fake_app

    import crypto_signal_ml.monitor as monitor_module

    original_uvicorn_run = monitor_module.uvicorn.run

    def fake_uvicorn_run(app, host: str, port: int) -> None:
        uvicorn_calls.append(
            {
                "app": app,
                "host": host,
                "port": port,
            }
        )

    monitor_module.uvicorn.run = fake_uvicorn_run
    try:
        service = SignalMonitorService(
            config=TrainingConfig(
                signal_monitor_run_initial_generation=True,
                signal_monitor_refresh_interval_seconds=0,
            ),
            signal_generation_app_factory=FakeSignalGenerationApp,
            app_factory=fake_create_app,
        )

        service.serve()
    finally:
        monitor_module.uvicorn.run = original_uvicorn_run

    assert refresh_calls == ["run"]
    assert len(create_app_calls) == 1
    assert create_app_calls[0]["config"].signal_monitor_run_initial_generation is True
    assert len(uvicorn_calls) == 1
    assert uvicorn_calls[0] == {
        "app": fake_app,
        "host": "127.0.0.1",
        "port": 8100,
    }


def test_signal_summaries_exclude_numeric_only_base_currencies() -> None:
    """Numeric-only tickers should be filtered even when stale rows exist locally."""

    prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "00-USD",
                "base_currency": "00",
                "quote_currency": "USD",
                "close": 0.0045,
                "predicted_signal": -1,
                "predicted_name": "TAKE_PROFIT",
                "confidence": 0.84,
                "prob_take_profit": 0.84,
                "prob_hold": 0.08,
                "prob_buy": 0.08,
            },
            {
                "time_step": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "SOL-USD",
                "base_currency": "SOL",
                "quote_currency": "USD",
                "close": 150.0,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.76,
                "prob_take_profit": 0.09,
                "prob_hold": 0.15,
                "prob_buy": 0.76,
            },
        ]
    )

    latest_signals = build_latest_signal_summaries(
        prediction_df,
        minimum_action_confidence=0.65,
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            signal_excluded_base_currencies=(),
        ),
    )

    assert [signal_summary["productId"] for signal_summary in latest_signals] == ["SOL-USD"]


def test_signal_summaries_exclude_short_digit_tickers_from_public_output() -> None:
    """Short digit-mixed tickers like 2Z should not surface in public signals."""

    prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "2Z-USD",
                "base_currency": "2Z",
                "quote_currency": "USD",
                "close": 0.08,
                "predicted_signal": -1,
                "predicted_name": "TAKE_PROFIT",
                "confidence": 0.91,
                "prob_take_profit": 0.91,
                "prob_hold": 0.04,
                "prob_buy": 0.05,
                "cmc_name": "DoubleZero",
                "cmc_category": "token",
            },
            {
                "time_step": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "SOL-USD",
                "base_currency": "SOL",
                "quote_currency": "USD",
                "close": 150.0,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.76,
                "prob_take_profit": 0.09,
                "prob_hold": 0.15,
                "prob_buy": 0.76,
                "cmc_name": "Solana",
                "cmc_category": "coin",
            },
        ]
    )

    latest_signals = build_latest_signal_summaries(
        prediction_df,
        minimum_action_confidence=0.65,
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            signal_excluded_base_currencies=(),
        ),
    )

    assert [signal_summary["productId"] for signal_summary in latest_signals] == ["SOL-USD"]


def test_signal_summaries_exclude_usd_pegged_assets_even_when_not_in_base_exclusions() -> None:
    """USD-pegged assets should not surface as public opportunities."""

    prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "USDS-USD",
                "base_currency": "USDS",
                "quote_currency": "USD",
                "close": 1.001,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.81,
                "prob_take_profit": 0.08,
                "prob_hold": 0.11,
                "prob_buy": 0.81,
                "cmc_name": "USD Stable",
                "cmc_category": "stablecoin",
                "cmc_percent_change_24h": 0.001,
                "cmc_percent_change_7d": 0.003,
            },
            {
                "time_step": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "ADA-USD",
                "base_currency": "ADA",
                "quote_currency": "USD",
                "close": 1.12,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.77,
                "prob_take_profit": 0.09,
                "prob_hold": 0.14,
                "prob_buy": 0.77,
                "cmc_name": "Cardano",
                "cmc_category": "coin",
                "cmc_percent_change_24h": 0.05,
                "cmc_percent_change_7d": 0.14,
            },
        ]
    )

    latest_signals = build_latest_signal_summaries(
        prediction_df,
        minimum_action_confidence=0.65,
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            signal_excluded_base_currencies=(),
        ),
    )

    assert [signal_summary["productId"] for signal_summary in latest_signals] == ["ADA-USD"]


def test_take_profit_is_suppressed_without_prior_trade_context() -> None:
    """Exit-only calls without a tracked trade should stay internal and unpublishable."""

    contextualized_signals = apply_signal_trade_context(
        [
            {
                "productId": "ALGO-USD",
                "signal_name": "TAKE_PROFIT",
                "spotAction": "take_profit",
                "actionable": True,
                "tradeReadiness": "medium",
                "reasonItems": ["The raw model wants to take profit."],
                "reasonSummary": "The raw model wants to take profit.",
                "signalChat": "ALGO-USD is a TAKE_PROFIT setup for spot trading.",
                "close": 0.25,
                "quoteCurrency": "USD",
                "baseCurrency": "ALGO",
                "coinName": "Algorand",
                "coinCategory": "coin",
                "marketContext": {
                    "cmcPercentChange24h": 0.03,
                    "cmcPercentChange7d": 0.08,
                },
            }
        ],
        active_trade_product_ids=[],
    )

    assert contextualized_signals[0]["signal_name"] == "HOLD"
    assert contextualized_signals[0]["spotAction"] == "wait"
    assert contextualized_signals[0]["actionable"] is False
    assert contextualized_signals[0]["takeProfitSuppressed"] is True
    assert contextualized_signals[0]["takeProfitEligible"] is False
    assert filter_published_signal_summaries(contextualized_signals) == []


def test_take_profit_survives_when_prior_trade_context_exists() -> None:
    """Tracked positions should keep TAKE_PROFIT public and actionable."""

    contextualized_signals = apply_signal_trade_context(
        [
            {
                "productId": "ALGO-USD",
                "signal_name": "TAKE_PROFIT",
                "spotAction": "take_profit",
                "actionable": True,
                "tradeReadiness": "medium",
                "reasonItems": ["The raw model wants to take profit."],
                "reasonSummary": "The raw model wants to take profit.",
                "signalChat": "ALGO-USD is a TAKE_PROFIT setup for spot trading.",
                "close": 0.25,
                "quoteCurrency": "USD",
                "baseCurrency": "ALGO",
                "coinName": "Algorand",
                "coinCategory": "coin",
                "marketContext": {
                    "cmcPercentChange24h": 0.03,
                    "cmcPercentChange7d": 0.08,
                },
            }
        ],
        active_trade_product_ids=["ALGO-USD"],
    )

    assert contextualized_signals[0]["signal_name"] == "TAKE_PROFIT"
    assert contextualized_signals[0]["spotAction"] == "take_profit"
    assert contextualized_signals[0]["actionable"] is True
    assert contextualized_signals[0]["takeProfitEligible"] is True


def test_buy_signal_becomes_hold_when_trade_is_already_active() -> None:
    """Fresh BUY calls should become HOLD once the system is already tracking the trade."""

    contextualized_signals = apply_signal_trade_context(
        [
            {
                "productId": "ALGO-USD",
                "signal_name": "BUY",
                "spotAction": "buy",
                "actionable": True,
                "tradeReadiness": "medium",
                "reasonItems": ["The raw model wants to buy."],
                "reasonSummary": "The raw model wants to buy.",
                "signalChat": "ALGO-USD is a BUY setup.",
                "close": 0.25,
                "quoteCurrency": "USD",
                "baseCurrency": "ALGO",
                "coinName": "Algorand",
                "coinCategory": "coin",
            }
        ],
        active_trade_product_ids=["ALGO-USD"],
    )

    assert contextualized_signals[0]["signal_name"] == "HOLD"
    assert contextualized_signals[0]["spotAction"] == "wait"
    assert contextualized_signals[0]["actionable"] is False
    assert contextualized_signals[0]["tradeLifecycleApplied"] is True
    assert contextualized_signals[0]["tradeLifecycleSignalName"] == "HOLD"
    published_signals = filter_published_signal_summaries(contextualized_signals)
    assert len(published_signals) == 1
    assert published_signals[0]["signal_name"] == "HOLD"


def test_profitable_active_trade_switches_public_signal_to_take_profit() -> None:
    """Tracked trades in profit should publish TAKE_PROFIT instead of a fresh BUY."""

    contextualized_signals = apply_signal_trade_context(
        [
            {
                "productId": "SOL-USD",
                "signal_name": "BUY",
                "spotAction": "buy",
                "actionable": True,
                "tradeReadiness": "medium",
                "reasonItems": ["The raw model still likes the long."],
                "reasonSummary": "The raw model still likes the long.",
                "signalChat": "SOL-USD is a BUY setup.",
                "close": 109.0,
                "quoteCurrency": "USD",
                "baseCurrency": "SOL",
                "coinName": "Solana",
                "coinCategory": "coin",
            }
        ],
        active_trade_product_ids=["SOL-USD"],
        active_signal_context_by_product={
            "SOL-USD": {
                "entryPrice": 100.0,
                "takeProfitPrice": 108.0,
            }
        },
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            signal_excluded_base_currencies=(),
            brain_profit_lock_threshold=0.08,
        ),
    )

    assert contextualized_signals[0]["signal_name"] == "TAKE_PROFIT"
    assert contextualized_signals[0]["spotAction"] == "take_profit"
    assert contextualized_signals[0]["actionable"] is True
    assert contextualized_signals[0]["tradeContext"]["profitLockTriggered"] is True


def test_losing_active_trade_switches_public_signal_to_loss() -> None:
    """Tracked trades below the loss budget should publish LOSS for risk control."""

    contextualized_signals = apply_signal_trade_context(
        [
            {
                "productId": "ADA-USD",
                "signal_name": "HOLD",
                "spotAction": "wait",
                "actionable": False,
                "tradeReadiness": "standby",
                "reasonItems": ["The raw model is indecisive."],
                "reasonSummary": "The raw model is indecisive.",
                "signalChat": "ADA-USD is a HOLD setup.",
                "close": 94.0,
                "quoteCurrency": "USD",
                "baseCurrency": "ADA",
                "coinName": "Cardano",
                "coinCategory": "coin",
            }
        ],
        active_trade_product_ids=["ADA-USD"],
        active_signal_context_by_product={
            "ADA-USD": {
                "entryPrice": 100.0,
                "stopLossPrice": 95.0,
            }
        },
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            signal_excluded_base_currencies=(),
            brain_loss_cut_threshold=-0.05,
        ),
    )

    assert contextualized_signals[0]["signal_name"] == "LOSS"
    assert contextualized_signals[0]["spotAction"] == "cut_loss"
    assert contextualized_signals[0]["actionable"] is True
    assert contextualized_signals[0]["tradeReadiness"] == "high"


def test_watchlist_hold_without_trade_context_stays_internal_only() -> None:
    """Neutral watchlist rows should not appear in the published public feed."""

    contextualized_signals = apply_signal_trade_context(
        [
            {
                "productId": "LINK-USD",
                "signal_name": "HOLD",
                "spotAction": "wait",
                "actionable": False,
                "tradeReadiness": "standby",
                "reasonItems": ["The setup is still forming."],
                "reasonSummary": "The setup is still forming.",
                "signalChat": "LINK-USD is a HOLD setup.",
                "close": 18.5,
                "quoteCurrency": "USD",
                "baseCurrency": "LINK",
                "coinName": "Chainlink",
                "coinCategory": "coin",
            }
        ],
        active_trade_product_ids=[],
    )

    assert len(contextualized_signals) == 1
    assert filter_published_signal_summaries(contextualized_signals) == []


def test_signal_summaries_drop_stale_products_after_batch_rotation() -> None:
    """Older products should not surface as live signals once a fresher batch has been refreshed."""

    prediction_df = pd.DataFrame(
        [
            {
                "time_step": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "ABT-USD",
                "base_currency": "ABT",
                "quote_currency": "USD",
                "close": 1.2,
                "predicted_signal": -1,
                "predicted_name": "TAKE_PROFIT",
                "confidence": 0.91,
                "prob_take_profit": 0.91,
                "prob_hold": 0.05,
                "prob_buy": 0.04,
            },
            {
                "time_step": 2,
                "timestamp": "2026-01-02T12:00:00Z",
                "product_id": "ALGO-USD",
                "base_currency": "ALGO",
                "quote_currency": "USD",
                "close": 0.12,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.72,
                "prob_take_profit": 0.11,
                "prob_hold": 0.17,
                "prob_buy": 0.72,
            },
        ]
    )

    latest_signals = build_latest_signal_summaries(
        prediction_df,
        minimum_action_confidence=0.65,
        config=TrainingConfig(
            coinmarketcap_use_context=False,
            signal_excluded_base_currencies=(),
            signal_max_staleness_hours=3.0,
        ),
    )

    assert [signal_summary["productId"] for signal_summary in latest_signals] == ["ALGO-USD"]


def test_backtester_skips_buys_blocked_by_policy() -> None:
    """The backtester should not open positions that the trading policy downgraded to HOLD."""

    prediction_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "BTC-USD",
                "future_return": -0.04,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.82,
                "prob_take_profit": 0.07,
                "prob_hold": 0.11,
                "prob_buy": 0.82,
                "market_regime_label": "trend_down",
                "market_regime_code": 3.0,
                "regime_is_trending": 1.0,
                "regime_is_high_volatility": 0.0,
                "cmcal_has_event_next_7d": 0,
            },
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "product_id": "ETH-USD",
                "future_return": 0.05,
                "predicted_signal": 1,
                "predicted_name": "BUY",
                "confidence": 0.83,
                "prob_take_profit": 0.05,
                "prob_hold": 0.12,
                "prob_buy": 0.83,
                "market_regime_label": "trend_up",
                "market_regime_code": 2.0,
                "regime_is_trending": 1.0,
                "regime_is_high_volatility": 0.0,
                "cmcal_has_event_next_7d": 0,
            },
        ]
    )

    backtester = EqualWeightSignalBacktester(
        TrainingConfig(
            coinmarketcap_use_context=False,
            backtest_min_confidence=0.65,
            backtest_max_positions_per_timestamp=2,
        )
    )
    result = backtester.run(prediction_df)

    assert result["summary"]["tradeCount"] == 1
    assert result["trade_df"].iloc[0]["product_id"] == "ETH-USD"


def test_trader_brain_builds_offensive_entry_plan() -> None:
    """The trader brain should rank and size fresh long candidates."""

    signal_summaries = [
        {
            "productId": "BTC-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.84,
            "probabilityMargin": 0.22,
            "setupScore": 4.6,
            "policyScore": 1.02,
            "tradeReadiness": "high",
            "marketState": {
                "label": "trend_up",
                "isTrending": True,
                "isHighVolatility": False,
                "volatilityRatio": 1.02,
            },
            "eventContext": {"hasEventNext7d": False},
            "close": 64000.0,
        },
        {
            "productId": "ETH-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.78,
            "probabilityMargin": 0.15,
            "setupScore": 3.8,
            "policyScore": 0.90,
            "tradeReadiness": "medium",
            "marketState": {
                "label": "trend_up",
                "isTrending": True,
                "isHighVolatility": False,
                "volatilityRatio": 1.05,
            },
            "eventContext": {"hasEventNext7d": False},
            "close": 3200.0,
        },
        {
            "productId": "SOL-USD",
            "signal_name": "HOLD",
            "spotAction": "wait",
            "confidence": 0.66,
            "probabilityMargin": 0.05,
            "setupScore": 0.0,
            "policyScore": 0.45,
            "tradeReadiness": "standby",
            "marketState": {
                "label": "trend_up",
                "isTrending": True,
                "isHighVolatility": False,
                "volatilityRatio": 0.98,
            },
            "eventContext": {"hasEventNext7d": False},
            "close": 140.0,
        },
    ]

    plan = TraderBrain(
        TrainingConfig(
            coinmarketcap_use_context=False,
            brain_max_entry_positions=2,
            brain_max_portfolio_risk_fraction=0.25,
        )
    ).build_plan(signal_summaries=signal_summaries, capital=10000.0)

    assert plan["marketStance"] == "offensive"
    assert plan["plan"]["newEntryCount"] == 2
    assert plan["plan"]["entries"][0]["productId"] == "BTC-USD"
    assert plan["plan"]["entries"][0]["allocationFraction"] > 0
    assert plan["signals"][0]["brain"]["decision"] == "enter_long"


def test_primary_signal_rotation_can_feature_a_different_strong_coin() -> None:
    """The featured signal should rotate away from the same recent coin when alternatives are close."""

    signal_summaries = [
        {
            "productId": "ALGO-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.76,
            "setupScore": 4.4,
            "policyScore": 1.04,
            "tradeReadiness": "high",
        },
        {
            "productId": "APE-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.73,
            "setupScore": 4.1,
            "policyScore": 0.97,
            "tradeReadiness": "high",
        },
        {
            "productId": "ABT-USD",
            "signal_name": "TAKE_PROFIT",
            "spotAction": "take_profit",
            "confidence": 0.71,
            "setupScore": 3.7,
            "policyScore": 0.86,
            "tradeReadiness": "medium",
        },
    ]

    primary_signal = select_primary_signal(
        signal_summaries,
        config=TrainingConfig(
            signal_primary_rotation_enabled=True,
            signal_primary_rotation_lookback=2,
            signal_primary_rotation_candidate_window=3,
            signal_primary_rotation_min_score_ratio=0.88,
        ),
        recent_primary_product_ids=["ALGO-USD"],
    )

    assert primary_signal["productId"] == "APE-USD"


def test_primary_signal_rotation_keeps_the_top_coin_when_others_are_much_weaker() -> None:
    """Rotation should not demote the headline signal when the gap is too large."""

    signal_summaries = [
        {
            "productId": "ALGO-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.82,
            "setupScore": 5.0,
            "policyScore": 1.18,
            "tradeReadiness": "high",
        },
        {
            "productId": "APE-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.66,
            "setupScore": 2.5,
            "policyScore": 0.55,
            "tradeReadiness": "medium",
        },
    ]

    primary_signal = select_primary_signal(
        signal_summaries,
        config=TrainingConfig(
            signal_primary_rotation_enabled=True,
            signal_primary_rotation_lookback=2,
            signal_primary_rotation_candidate_window=3,
            signal_primary_rotation_min_score_ratio=0.92,
        ),
        recent_primary_product_ids=["ALGO-USD"],
    )

    assert primary_signal["productId"] == "ALGO-USD"


def _build_watchlist_test_config(tmp_path: Path, **overrides: object) -> TrainingConfig:
    """Build one small config tuned for watchlist lifecycle tests."""

    config_kwargs: dict[str, object] = {
        "coinmarketcap_use_context": False,
        "coinmarketcap_use_market_intelligence": True,
        "signal_excluded_base_currencies": (),
        "signal_watchlist_state_path": tmp_path / "watchlistState.json",
    }
    config_kwargs.update(overrides)
    return TrainingConfig(**config_kwargs)


def _build_watchlist_follow_up_signal(
    *,
    product_id: str = "ALCX-USD",
    signal_name: str = "HOLD",
    confidence: float = 0.74,
    probability_margin: float = 0.18,
    setup_score: float = 4.8,
    policy_score: float = 1.10,
    trade_readiness: str = "high",
    regime_label: str = "range",
    is_high_volatility: bool = False,
    trend_score: float = 0.02,
    topic_trend_score: float = 0.40,
    news_sentiment: float = 0.0,
    news_relevance: float = 0.0,
    breakout_confirmed: bool = False,
    retest_hold_confirmed: bool = False,
    near_resistance: bool = False,
    resistance_distance_pct: float = 0.03,
    event_window_active: bool = False,
    post_event_cooldown_active: bool = False,
    macro_event_risk: bool = False,
    has_event_next_7d: bool = False,
    macro_risk_mode: str = "neutral",
    close: float = 5.10,
) -> dict[str, object]:
    """Create one reusable signal summary for watchlist lifecycle tests."""

    return {
        "productId": product_id,
        "signal_name": signal_name,
        "spotAction": "buy" if signal_name == "BUY" else "wait",
        "confidence": confidence,
        "probabilityMargin": probability_margin,
        "setupScore": setup_score,
        "policyScore": policy_score,
        "tradeReadiness": trade_readiness,
        "marketState": {
            "label": regime_label,
            "isTrending": regime_label.startswith("trend_up"),
            "isHighVolatility": is_high_volatility,
            "volatilityRatio": 1.30 if is_high_volatility else 1.02,
            "trendScore": trend_score,
        },
        "marketContext": {
            "marketIntelligence": {
                "available": True,
                "fearGreedValue": 34 if macro_risk_mode == "risk_off" else 61,
                "fearGreedClassification": "Fear" if macro_risk_mode == "risk_off" else "Greed",
                "btcDominance": 0.58 if macro_risk_mode == "risk_off" else 0.52,
                "btcDominanceChange24h": 0.01 if macro_risk_mode == "risk_off" else -0.01,
                "riskMode": macro_risk_mode,
            }
        },
        "eventContext": {
            "hasEventNext7d": has_event_next_7d,
            "eventWindowActive": event_window_active,
            "postEventCooldownActive": post_event_cooldown_active,
            "macroEventRiskFlag": macro_event_risk,
        },
        "newsContext": {
            "newsSentiment1h": news_sentiment,
            "newsRelevanceScore": news_relevance,
        },
        "trendContext": {
            "topicTrendScore": topic_trend_score,
            "trendPersistenceScore": 0.45 if topic_trend_score > 0 else 0.20,
        },
        "chartContext": {
            "breakoutConfirmed": breakout_confirmed,
            "retestHoldConfirmed": retest_hold_confirmed,
            "nearResistance": near_resistance,
            "resistanceDistancePct": resistance_distance_pct,
            "structureLabel": "range",
        },
        "close": close,
    }


def test_watchlist_first_check_stays_in_watchlist_for_alcx_calibration_case(tmp_path: Path) -> None:
    """The latest ALCX-like output should stay on the watchlist on its first review."""

    config = _build_watchlist_test_config(tmp_path)
    store = WatchlistStateStore(config)

    state, promotion = store.update_from_signal(
        signal_summary=_build_watchlist_follow_up_signal(
            confidence=0.8413942553341991,
            probability_margin=0.7433841429810573,
            setup_score=0.0,
            policy_score=1.5847783983152564,
            trade_readiness="standby",
            regime_label="range_high_volatility",
            is_high_volatility=True,
            trend_score=0.0006934317752682162,
            topic_trend_score=-0.6,
            breakout_confirmed=False,
            retest_hold_confirmed=False,
            resistance_distance_pct=0.025988787692436892,
            close=5.113558864614706,
            macro_risk_mode="risk_off",
        ),
        decision_score=0.5373042434966011,
        market_context={"marketStance": "defensive", "macroRiskMode": "risk_off"},
    )

    assert state["stage"] == "watchlist"
    assert state["checks"] == 1
    assert state["positiveChecks"] == 0
    assert promotion.promotion_ready is False
    assert promotion.hard_blocks == ()
    assert promotion.soft_penalties == ("defensive_market", "macro_risk_off", "high_volatility")
    assert promotion.hold_reason == "wait_for_setup_building"


def test_watchlist_soft_reviews_confirmed_buy_slightly_below_confidence_cutoff(tmp_path: Path) -> None:
    """Confirmed BUY setups slightly below the confidence cutoff should stay in review, not invalidate."""

    config = _build_watchlist_test_config(
        tmp_path,
        signal_watchlist_invalidation_confidence=0.25,
        signal_watchlist_soft_review_confidence_buffer=0.05,
        signal_watchlist_soft_review_min_raw_confidence=0.50,
        signal_watchlist_soft_review_min_probability_margin=0.12,
    )
    store = WatchlistStateStore(config)

    weak_state, weak_promotion = store.update_from_signal(
        signal_summary={
            **_build_watchlist_follow_up_signal(
                product_id="AERO-USD",
                signal_name="HOLD",
                confidence=0.18,
                probability_margin=0.04,
                trade_readiness="blocked",
                breakout_confirmed=False,
                near_resistance=True,
            ),
            "modelSignalName": "BUY",
            "chartDecision": "blocked",
            "chartPatternLabel": "near_resistance",
            "confidenceCalibration": {
                "calibratedConfidence": 0.18,
            },
        },
        decision_score=0.12,
        market_context={"marketStance": "balanced", "macroRiskMode": "neutral"},
    )

    assert weak_state["stage"] == "invalidated"
    assert weak_promotion.blocked_reason == "signal_quality_veto"

    state, promotion = store.update_from_signal(
        signal_summary={
            **_build_watchlist_follow_up_signal(
                product_id="AERO-USD",
                signal_name="HOLD",
                confidence=0.52,
                probability_margin=0.15,
                trade_readiness="standby",
                breakout_confirmed=True,
                resistance_distance_pct=0.04,
                near_resistance=False,
            ),
            "modelSignalName": "BUY",
            "chartDecision": "confirmed",
            "chartPatternLabel": "breakout_confirmed",
            "confidenceCalibration": {
                "calibratedConfidence": 0.23,
                "reliabilityScore": 0.54,
                "riskPenaltyScore": 0.18,
                "executionPenaltyScore": 0.04,
                "chartAlignmentScore": 0.32,
                "contextAlignmentScore": 0.18,
                "confidenceQuality": "fragile",
            },
            "executionContext": {
                "executionQualityScore": 0.46,
                "decisionPenalty": 0.04,
                "isThinLiquidity": False,
                "hasElevatedCost": False,
                "isExecutionBlocked": False,
            },
        },
        decision_score=0.36,
        market_context={"marketStance": "balanced", "macroRiskMode": "neutral"},
    )

    assert state["stage"] == "watchlist"
    assert state["recoveredFromInvalidation"] is True
    assert state["softLowConfidenceReviewApplied"] is True
    assert state["signalQualitySoftReasons"] == ["soft_low_calibrated_confidence"]
    assert "low_calibrated_confidence" not in state["signalQualityReasons"]
    assert promotion.blocked_reason is None
    assert promotion.hold_reason == "needs_more_signal_confirmation"
    assert promotion.review_reason == "needs_more_signal_confirmation"


def test_watchlist_follow_up_promotes_on_second_and_third_checks(tmp_path: Path) -> None:
    """Repeated positive follow-up checks should move from watchlist into explicit setup stages."""

    config = _build_watchlist_test_config(tmp_path)
    store = WatchlistStateStore(config)
    market_context = {"marketStance": "balanced", "macroRiskMode": "neutral"}

    store.update_from_signal(
        signal_summary=_build_watchlist_follow_up_signal(
            confidence=0.66,
            trade_readiness="medium",
            topic_trend_score=0.20,
        ),
        decision_score=0.56,
        market_context=market_context,
    )
    second_state, second_promotion = store.update_from_signal(
        signal_summary=_build_watchlist_follow_up_signal(
            confidence=0.70,
            trade_readiness="medium",
            topic_trend_score=0.30,
        ),
        decision_score=0.62,
        market_context=market_context,
    )
    third_state, third_promotion = store.update_from_signal(
        signal_summary=_build_watchlist_follow_up_signal(
            confidence=0.74,
            trade_readiness="medium",
            topic_trend_score=0.46,
        ),
        decision_score=0.66,
        market_context=market_context,
    )

    assert second_state["stage"] == "setup_building"
    assert second_promotion.hold_reason == "wait_for_setup_confirmation"
    assert second_promotion.promotion_ready is False
    assert second_state["reviewOutcome"] == "advanced"
    assert second_state["reviewReason"] == "wait_for_setup_confirmation"
    assert third_state["stage"] == "setup_confirmed"
    assert third_promotion.hold_reason == "wait_for_breakout_confirmation"
    assert third_promotion.promotion_ready is False
    assert third_state["previousStage"] == "setup_building"
    assert third_state["lastTransition"]["toStage"] == "setup_confirmed"

    store.save()
    saved_payload = json.loads((tmp_path / "watchlistState.json").read_text(encoding="utf-8"))
    assert saved_payload["lastCycleSummary"]["reviewedCount"] == 3
    assert saved_payload["lastCycleSummary"]["advancedThisCycle"] == 2
    assert saved_payload["lastCycleSummary"]["stageCounts"]["setup_confirmed"] == 1
    assert saved_payload["lastCycleSummary"]["transitions"][0]["productId"] == "ALCX-USD"


def test_watchlist_entry_ready_separates_soft_risk_from_hard_blocks(tmp_path: Path) -> None:
    """Soft risk-off pressure should differ from hard event-style blocks at entry-ready stage."""

    soft_store = WatchlistStateStore(
        _build_watchlist_test_config(
            tmp_path,
            signal_watchlist_state_path=tmp_path / "softWatchlistState.json",
            signal_watchlist_soft_risk_override_min_confirmation=1.01,
        )
    )
    soft_market_context = {"marketStance": "defensive", "macroRiskMode": "risk_off"}
    for confidence, decision_score, breakout_confirmed in (
        (0.74, 0.58, False),
        (0.78, 0.63, False),
        (0.82, 0.71, True),
    ):
        soft_state, soft_promotion = soft_store.update_from_signal(
            signal_summary=_build_watchlist_follow_up_signal(
                confidence=confidence,
                trade_readiness="high",
                topic_trend_score=0.46,
                breakout_confirmed=breakout_confirmed,
                macro_risk_mode="risk_off",
            ),
            decision_score=decision_score,
            market_context=soft_market_context,
        )

    assert soft_state["stage"] == "entry_ready"
    assert soft_promotion.promotion_ready is False
    assert soft_promotion.hard_blocks == ()
    assert "macro_risk_off" in soft_promotion.soft_penalties
    assert soft_promotion.blocked_reason == "market_regime_veto"
    assert soft_promotion.exceptional_override_applied is False

    hard_store = WatchlistStateStore(
        _build_watchlist_test_config(
            tmp_path,
            signal_watchlist_state_path=tmp_path / "hardWatchlistState.json",
        )
    )
    hard_market_context = {"marketStance": "balanced", "macroRiskMode": "neutral"}
    for confidence, decision_score, breakout_confirmed, event_window_active in (
        (0.74, 0.58, False, False),
        (0.78, 0.63, False, False),
        (0.82, 0.71, True, True),
    ):
        hard_state, hard_promotion = hard_store.update_from_signal(
            signal_summary=_build_watchlist_follow_up_signal(
                confidence=confidence,
                trade_readiness="high",
                topic_trend_score=0.46,
                breakout_confirmed=breakout_confirmed,
                event_window_active=event_window_active,
            ),
            decision_score=decision_score,
            market_context=hard_market_context,
        )

    assert hard_state["stage"] == "entry_ready"
    assert hard_promotion.promotion_ready is False
    assert "blocked_by_event_risk" in hard_promotion.hard_blocks
    assert hard_promotion.blocked_reason == "market_regime_veto"
    assert hard_promotion.blocked_reason_detail == "blocked_by_event_risk"


def test_watchlist_preserves_strong_blocked_buy_and_recovers_from_invalidation(tmp_path: Path) -> None:
    """Strong blocked BUY candidates should recover into watchlist review instead of staying invalidated."""

    config = _build_watchlist_test_config(
        tmp_path,
        signal_watchlist_state_path=tmp_path / "aeroWatchlistState.json",
        signal_watchlist_preserve_strong_blocked_buys=True,
        signal_watchlist_strong_buy_min_raw_confidence=0.72,
        signal_watchlist_strong_buy_min_probability_margin=0.18,
        signal_watchlist_invalidation_confidence=0.25,
        signal_watchlist_invalidation_min_probability_margin=0.08,
    )
    store = WatchlistStateStore(config)
    market_context = {"marketStance": "defensive", "macroRiskMode": "risk_off"}

    weak_state, weak_promotion = store.update_from_signal(
        signal_summary={
            **_build_watchlist_follow_up_signal(
                product_id="AERO-USD",
                signal_name="HOLD",
                confidence=0.18,
                probability_margin=0.03,
                trade_readiness="blocked",
                regime_label="trend_down_high_volatility",
                is_high_volatility=True,
                macro_risk_mode="risk_off",
            ),
            "modelSignalName": "BUY",
            "confidenceCalibration": {
                "calibratedConfidence": 0.18,
            },
            "executionContext": {
                "executionQualityScore": 0.0,
                "decisionPenalty": 0.08,
                "isThinLiquidity": True,
                "hasElevatedCost": True,
                "isExecutionBlocked": True,
            },
        },
        decision_score=0.12,
        market_context=market_context,
    )

    assert weak_state["stage"] == "invalidated"
    assert weak_promotion.blocked_reason == "signal_quality_veto"

    strong_state, strong_promotion = store.update_from_signal(
        signal_summary={
            **_build_watchlist_follow_up_signal(
                product_id="AERO-USD",
                signal_name="HOLD",
                confidence=0.7865,
                probability_margin=0.6455,
                trade_readiness="blocked",
                regime_label="trend_down_high_volatility",
                is_high_volatility=True,
                macro_risk_mode="risk_off",
            ),
            "modelSignalName": "BUY",
            "confidenceCalibration": {
                "calibratedConfidence": 0.4782,
                "chartAlignmentScore": 0.12,
                "newsAlignmentScore": 0.0,
                "trendAlignmentScore": 0.04,
                "contextAlignmentScore": 0.10,
            },
            "executionContext": {
                "executionQualityScore": 0.0,
                "decisionPenalty": 0.08,
                "isThinLiquidity": True,
                "hasElevatedCost": True,
                "isExecutionBlocked": True,
            },
        },
        decision_score=0.84,
        market_context=market_context,
    )

    assert strong_state["stage"] == "setup_building"
    assert strong_state["lastModelSignalName"] == "BUY"
    assert strong_state["recoveredFromInvalidation"] is True
    assert strong_state["strongBlockedBuyPreserved"] is True
    assert strong_state["primaryVetoBucket"] == "market_regime_veto"
    assert set(strong_state["vetoBuckets"]) == {"market_regime_veto", "execution_veto"}
    assert "blocked_by_risk" not in strong_state["hardBlocks"]
    assert strong_promotion.blocked_reason == "market_regime_veto"
    assert strong_promotion.blocked_reason_detail == "severe_downtrend_regime"
    assert strong_promotion.hold_reason == "blocked_high_risk"
    assert strong_promotion.strong_blocked_buy_preserved is True


def test_trader_brain_promotes_breakout_confirmed_watchlist_on_follow_up_cycle(tmp_path: Path) -> None:
    """Strong breakout confirmation should promote a matured watchlist despite mild risk-off conditions."""

    config = _build_watchlist_test_config(tmp_path)
    brain = TraderBrain(config)

    first_plan = brain.build_plan(
        signal_summaries=[
            _build_watchlist_follow_up_signal(
                confidence=0.76,
                probability_margin=0.18,
                setup_score=5.0,
                policy_score=1.12,
                trade_readiness="high",
                topic_trend_score=0.40,
                macro_risk_mode="risk_off",
            )
        ],
        capital=10000.0,
    )
    second_plan = brain.build_plan(
        signal_summaries=[
            _build_watchlist_follow_up_signal(
                confidence=0.79,
                probability_margin=0.18,
                setup_score=5.0,
                policy_score=1.15,
                trade_readiness="high",
                topic_trend_score=0.43,
                macro_risk_mode="risk_off",
            )
        ],
        capital=10000.0,
    )
    third_plan = brain.build_plan(
        signal_summaries=[
            _build_watchlist_follow_up_signal(
                confidence=0.83,
                probability_margin=0.19,
                setup_score=5.0,
                policy_score=1.18,
                trade_readiness="high",
                topic_trend_score=0.46,
                breakout_confirmed=True,
                macro_risk_mode="risk_off",
            )
        ],
        capital=10000.0,
    )

    assert first_plan["signals"][0]["brain"]["decision"] == "watchlist"
    assert second_plan["signals"][0]["brain"]["decision"] == "watchlist"
    assert third_plan["signals"][0]["watchlistState"]["stage"] == "entry_ready"
    assert third_plan["signals"][0]["watchlistPromotion"]["exceptionalOverrideApplied"] is True
    assert third_plan["signals"][0]["brain"]["decision"] == "enter_long"
    assert third_plan["signals"][0]["brain"]["macroRiskMode"] == "risk_off"
    assert third_plan["signals"][0]["brain"]["evidence"]["watchlistSoftRiskOverride"] is True


def test_trader_brain_respects_risk_off_market_intelligence() -> None:
    """Risk-off CMC intelligence should downgrade fresh entries into watchlist mode."""

    signal_summaries = [
        {
            "productId": "SOL-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.79,
            "probabilityMargin": 0.16,
            "setupScore": 4.2,
            "policyScore": 0.96,
            "tradeReadiness": "medium",
            "marketState": {
                "label": "trend_up",
                "isTrending": True,
                "isHighVolatility": False,
                "volatilityRatio": 1.01,
            },
            "marketContext": {
                "marketIntelligence": {
                    "available": True,
                    "fearGreedValue": 22,
                    "fearGreedClassification": "Extreme Fear",
                    "btcDominance": 0.58,
                    "btcDominanceChange24h": 0.01,
                    "riskMode": "risk_off",
                }
            },
            "eventContext": {"hasEventNext7d": False},
            "close": 145.0,
        }
    ]

    plan = TraderBrain(
        TrainingConfig(
            coinmarketcap_use_context=False,
            coinmarketcap_use_market_intelligence=True,
        )
    ).build_plan(signal_summaries=signal_summaries, capital=10000.0)

    assert plan["marketStance"] == "defensive"
    assert plan["signals"][0]["brain"]["decision"] == "watchlist"
    assert plan["signals"][0]["brain"]["macroRiskMode"] == "risk_off"


def test_trader_brain_surfaces_structured_decision_evidence() -> None:
    """Each signal should now carry a structured evidence, memo, and critic review block."""

    signal_summaries = [
        {
            "productId": "AAVE-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.83,
            "probabilityMargin": 0.17,
            "setupScore": 4.6,
            "policyScore": 1.02,
            "tradeReadiness": "high",
            "marketState": {
                "label": "trend_up",
                "isTrending": True,
                "isHighVolatility": False,
                "volatilityRatio": 1.02,
            },
            "marketContext": {
                "marketIntelligence": {
                    "available": True,
                    "fearGreedValue": 67,
                    "fearGreedClassification": "Greed",
                    "btcDominance": 0.52,
                    "btcDominanceChange24h": -0.01,
                    "riskMode": "risk_on",
                }
            },
            "eventContext": {"hasEventNext7d": False},
            "close": 108.0,
        }
    ]

    plan = TraderBrain(
        TrainingConfig(
            coinmarketcap_use_context=False,
            coinmarketcap_use_market_intelligence=True,
        )
    ).build_plan(signal_summaries=signal_summaries, capital=10000.0)

    brain = plan["signals"][0]["brain"]

    assert brain["evidence"]["edgeScore"] > 0.70
    assert brain["decisionMemo"]["thesis"]
    assert brain["criticReview"]["verdict"] in {"approve", "caution"}
    assert brain["decision"] in {"enter_long", "watchlist"}


def test_trader_brain_critic_blocks_entry_when_trade_memory_and_risk_conflict() -> None:
    """Weak tracked-trade history should help block fragile fresh entries."""

    signal_summaries = [
        {
            "productId": "SOL-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.84,
            "probabilityMargin": 0.18,
            "setupScore": 4.5,
            "policyScore": 1.00,
            "tradeReadiness": "high",
            "marketState": {
                "label": "trend_up_high_volatility",
                "isTrending": True,
                "isHighVolatility": True,
                "volatilityRatio": 1.28,
            },
            "marketContext": {
                "marketIntelligence": {
                    "available": True,
                    "fearGreedValue": 25,
                    "fearGreedClassification": "Fear",
                    "btcDominance": 0.58,
                    "btcDominanceChange24h": 0.01,
                    "riskMode": "risk_off",
                }
            },
            "eventContext": {"hasEventNext7d": False},
            "close": 145.0,
        }
    ]
    trade_memory_by_product = {
        "SOL-USD": {
            "available": True,
            "scope": "product+signal",
            "closedTradeCount": 4,
            "winRate": 0.25,
            "averageRealizedReturn": -0.05,
            "recentLossStreak": 1,
            "lastOutcome": "loss",
            "sampleAdequate": True,
        }
    }

    plan = TraderBrain(
        TrainingConfig(
            coinmarketcap_use_context=False,
            coinmarketcap_use_market_intelligence=True,
        )
    ).build_plan(
        signal_summaries=signal_summaries,
        capital=10000.0,
        trade_memory_by_product=trade_memory_by_product,
    )

    brain = plan["signals"][0]["brain"]

    assert brain["decision"] == "watchlist"
    assert brain["criticReview"]["verdict"] == "block"
    assert brain["criticReview"]["approvedDecision"] == "watchlist"
    assert brain["evidence"]["tradeMemory"]["bias"] == "cautious"


def test_trader_brain_can_exit_existing_position_on_take_profit_signal() -> None:
    """The trader brain should escalate to an exit when risk rises against an open position."""

    signal_summaries = [
        {
            "productId": "BTC-USD",
            "signal_name": "TAKE_PROFIT",
            "spotAction": "take_profit",
            "confidence": 0.81,
            "probabilityMargin": 0.19,
            "setupScore": 4.0,
            "policyScore": 0.95,
            "tradeReadiness": "medium",
            "marketState": {
                "label": "trend_down_high_volatility",
                "isTrending": True,
                "isHighVolatility": True,
                "volatilityRatio": 1.35,
            },
            "eventContext": {"hasEventNext7d": False},
            "close": 61000.0,
        }
    ]

    plan = TraderBrain(TrainingConfig(coinmarketcap_use_context=False)).build_plan(
        signal_summaries=signal_summaries,
        positions=[
            {
                "productId": "BTC-USD",
                "quantity": 0.2,
                "entryPrice": 59000.0,
                "currentPrice": 61000.0,
                "positionFraction": 0.12,
            }
        ],
        capital=10000.0,
    )

    assert plan["plan"]["exitCount"] == 1
    assert plan["signals"][0]["brain"]["decision"] == "exit_position"
    assert plan["signals"][0]["brain"]["suggestedReduceFraction"] == 1.0


def test_trader_brain_cuts_loser_when_drawdown_breaks_loss_limit() -> None:
    """A losing position should be exited once the configured loss budget is breached."""

    signal_summaries = [
        {
            "productId": "ETH-USD",
            "signal_name": "BUY",
            "spotAction": "buy",
            "confidence": 0.74,
            "probabilityMargin": 0.10,
            "setupScore": 2.4,
            "policyScore": 0.58,
            "tradeReadiness": "medium",
            "marketState": {
                "label": "trend_down_high_volatility",
                "isTrending": True,
                "isHighVolatility": True,
                "volatilityRatio": 1.42,
            },
            "eventContext": {"hasEventNext7d": False},
            "close": 92.0,
        }
    ]

    plan = TraderBrain(
        TrainingConfig(
            coinmarketcap_use_context=False,
            brain_loss_cut_threshold=-0.05,
        )
    ).build_plan(
        signal_summaries=signal_summaries,
        positions=[
            {
                "productId": "ETH-USD",
                "quantity": 1.0,
                "entryPrice": 100.0,
                "currentPrice": 92.0,
                "positionFraction": 0.10,
                "ageHours": 18.0,
            }
        ],
        capital=10000.0,
    )

    assert plan["plan"]["exitCount"] == 1
    assert plan["signals"][0]["brain"]["decision"] == "exit_position"
    assert plan["signals"][0]["brain"]["lossCutTriggered"] is True
    assert "loss" in plan["signals"][0]["brain"]["reasonSummary"].lower()


def test_trader_brain_reduces_stale_hold_position_without_follow_through() -> None:
    """A stale position without a fresh BUY should be trimmed instead of passively held."""

    signal_summaries = [
        {
            "productId": "SOL-USD",
            "signal_name": "HOLD",
            "spotAction": "wait",
            "confidence": 0.61,
            "probabilityMargin": 0.04,
            "setupScore": 0.0,
            "policyScore": 0.41,
            "tradeReadiness": "standby",
            "marketState": {
                "label": "range",
                "isTrending": False,
                "isHighVolatility": False,
                "volatilityRatio": 0.97,
            },
            "eventContext": {"hasEventNext7d": False},
            "close": 145.0,
        }
    ]

    plan = TraderBrain(
        TrainingConfig(
            coinmarketcap_use_context=False,
            brain_stale_position_age_hours=72.0,
            brain_reduce_fraction=0.40,
        )
    ).build_plan(
        signal_summaries=signal_summaries,
        positions=[
            {
                "productId": "SOL-USD",
                "quantity": 10.0,
                "entryPrice": 143.0,
                "currentPrice": 145.0,
                "positionFraction": 0.12,
                "ageHours": 96.0,
            }
        ],
        capital=10000.0,
    )

    assert plan["plan"]["reduceCount"] == 1
    assert plan["signals"][0]["brain"]["decision"] == "reduce_position"
    assert plan["signals"][0]["brain"]["thesisAgeIsStale"] is True
    assert plan["signals"][0]["brain"]["suggestedReduceFraction"] == 0.40


def test_trader_brain_locks_in_profit_even_without_high_volatility() -> None:
    """Strong unrealized gains should still exit on a take-profit signal even in calmer tape."""

    signal_summaries = [
        {
            "productId": "BTC-USD",
            "signal_name": "TAKE_PROFIT",
            "spotAction": "take_profit",
            "confidence": 0.70,
            "probabilityMargin": 0.12,
            "setupScore": 2.8,
            "policyScore": 0.70,
            "tradeReadiness": "medium",
            "marketState": {
                "label": "range",
                "isTrending": False,
                "isHighVolatility": False,
                "volatilityRatio": 1.01,
            },
            "eventContext": {"hasEventNext7d": False},
            "close": 111.0,
        }
    ]

    plan = TraderBrain(
        TrainingConfig(
            coinmarketcap_use_context=False,
            brain_profit_lock_threshold=0.08,
        )
    ).build_plan(
        signal_summaries=signal_summaries,
        positions=[
            {
                "productId": "BTC-USD",
                "quantity": 0.1,
                "entryPrice": 100.0,
                "currentPrice": 111.0,
                "positionFraction": 0.11,
                "ageHours": 30.0,
            }
        ],
        capital=10000.0,
    )

    assert plan["plan"]["exitCount"] == 1
    assert plan["signals"][0]["brain"]["decision"] == "exit_position"
    assert plan["signals"][0]["brain"]["profitLockTriggered"] is True


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
            "marketContext": {
                "marketIntelligence": {
                    "available": True,
                    "fearGreedValue": 68,
                    "fearGreedClassification": "Greed",
                    "btcDominance": 0.58,
                    "btcDominanceChange24h": -0.008,
                    "altcoinShare": 0.42,
                    "stablecoinShare": 0.12,
                    "riskMode": "risk_on",
                }
            },
            "marketState": {
                "label": "trend_up",
                "code": 2,
                "trendScore": 0.03,
                "volatilityRatio": 1.10,
                "isTrending": True,
                "isHighVolatility": False,
            },
        },
        {
            "productId": "ETH-USD",
            "signal_name": "TAKE_PROFIT",
            "spotAction": "take_profit",
            "confidence": 0.71,
            "setupScore": 3.0,
            "signalChat": "ETH-USD is a TAKE_PROFIT setup for spot trading.",
            "marketState": {
                "label": "range_high_volatility",
                "code": 4,
                "trendScore": 0.00,
                "volatilityRatio": 1.35,
                "isTrending": False,
                "isHighVolatility": True,
            },
        },
        {
            "productId": "SOL-USD",
            "signal_name": "HOLD",
            "spotAction": "wait",
            "confidence": 0.66,
            "setupScore": 0.0,
            "signalChat": "SOL-USD is a HOLD setup.",
            "marketState": {
                "label": "trend_up",
                "code": 2,
                "trendScore": 0.02,
                "volatilityRatio": 0.95,
                "isTrending": True,
                "isHighVolatility": False,
            },
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
    market_state = snapshot_store.get_market_state()
    buy_signals = snapshot_store.list_signals(action="buy")
    actionable_list = snapshot_store.list_signals(action="actionable")
    eth_signal = snapshot_store.get_signal_by_product("eth-usd")

    assert overview["marketState"]["dominant"]["label"] == "trend_up"
    assert overview["marketSummary"]["signalCounts"]["buy"] == 1
    assert overview["marketSummary"]["signalCounts"]["take_profit"] == 1
    assert overview["marketSummary"]["signalCounts"]["wait"] == 1
    assert overview["marketIntelligence"]["available"] is True
    assert overview["marketIntelligence"]["riskMode"] == "risk_on"
    assert market_state["highVolatilitySignals"] == 1
    assert market_state["trendingSignals"] == 2
    assert len(buy_signals) == 1
    assert buy_signals[0]["productId"] == "BTC-USD"
    assert len(actionable_list) == 2
    assert eth_signal is not None
    assert eth_signal["signal_name"] == "TAKE_PROFIT"
    assert overview["watchlistPromotion"] == {}
    assert overview["traderBrain"] == {}


def test_frontend_snapshot_can_represent_an_empty_public_feed() -> None:
    """The cached snapshot should stay valid even when nothing is ready to publish."""

    snapshot_payload = build_frontend_signal_snapshot(
        model_type="histGradientBoostingSignalModel",
        primary_signal=None,
        latest_signals=[],
        actionable_signals=[],
    )

    assert snapshot_payload["primarySignal"] is None
    assert snapshot_payload["marketSummary"]["totalSignals"] == 0
    assert snapshot_payload["marketSummary"]["actionableSignals"] == 0
    assert snapshot_payload["marketSummary"]["signalCounts"]["wait"] == 0


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
