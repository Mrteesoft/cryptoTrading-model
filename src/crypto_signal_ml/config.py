"""Project-wide configuration values and common paths."""

from dataclasses import asdict, dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .environment import load_env_file


# We calculate the project root from this file so the code works
# even when you run scripts from different folders.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_env_file(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def _env_csv_tuple(env_var_name: str, default_values: Tuple[str, ...]) -> Tuple[str, ...]:
    """Read one comma-separated environment variable into an uppercase tuple."""

    raw_value = str(os.getenv(env_var_name, "")).strip()
    if not raw_value:
        return tuple(default_values)

    normalized_values = []
    seen_values = set()
    for value in raw_value.split(","):
        normalized_value = str(value).strip().upper()
        if not normalized_value or normalized_value in seen_values:
            continue
        seen_values.add(normalized_value)
        normalized_values.append(normalized_value)

    return tuple(normalized_values) if normalized_values else tuple(default_values)


def _env_str(env_var_name: str, default_value: str) -> str:
    """Read one trimmed environment string, falling back when unset."""

    raw_value = str(os.getenv(env_var_name, "")).strip()
    return raw_value or default_value


def _env_bool(env_var_name: str, default_value: bool) -> bool:
    """Read one boolean environment variable with a conservative fallback."""

    raw_value = str(os.getenv(env_var_name, "")).strip().lower()
    if not raw_value:
        return bool(default_value)

    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False

    return bool(default_value)


def _env_optional_int(env_var_name: str, default_value: Optional[int]) -> Optional[int]:
    """Read one optional integer environment variable."""

    raw_value = str(os.getenv(env_var_name, "")).strip()
    if not raw_value:
        return default_value

    try:
        return int(raw_value)
    except ValueError:
        return default_value


def _env_float(env_var_name: str, default_value: float) -> float:
    """Read one float environment variable with a conservative fallback."""

    raw_value = str(os.getenv(env_var_name, "")).strip()
    if not raw_value:
        return float(default_value)

    try:
        return float(raw_value)
    except ValueError:
        return float(default_value)


COINMARKETCAP_MARKET_DATA_SOURCES = ("coinmarketcap", "coinmarketcapLatestQuotes")
COINBASE_MARKET_DATA_SOURCES = ("coinbaseExchange",)


def is_coinmarketcap_market_data_source(market_data_source: str) -> bool:
    """Return whether the configured source belongs to the CoinMarketCap family."""

    return str(market_data_source).strip() in COINMARKETCAP_MARKET_DATA_SOURCES


def is_coinbase_market_data_source(market_data_source: str) -> bool:
    """Return whether the configured source belongs to the Coinbase family."""

    return str(market_data_source).strip() in COINBASE_MARKET_DATA_SOURCES


def apply_runtime_market_data_settings(
    base_config: "TrainingConfig",
    runtime_config: "TrainingConfig",
) -> "TrainingConfig":
    """
    Keep the model's feature/training settings while honoring current runtime market settings.

    Saved model artifacts should continue to own feature-engineering behavior,
    but current deployment settings should be allowed to switch market-data
    providers, refresh behavior, and local integration paths.
    """

    override_field_names = {
        "market_data_source",
        "signal_refresh_market_data_before_generation",
        "signal_excluded_base_currencies",
        "signal_track_generated_trades",
        "signal_generated_trade_status",
        "backtest_min_confidence",
    }
    override_field_names.update(
        field_name
        for field_name in TrainingConfig.__dataclass_fields__
        if field_name.startswith(
            ("coinmarketcap_", "coinmarketcal_", "coinbase_", "live_", "signal_watchlist_")
        )
    )

    override_kwargs = {
        field_name: getattr(runtime_config, field_name)
        for field_name in override_field_names
    }
    return replace(base_config, **override_kwargs)


@dataclass(frozen=True)
class TrainingConfig:
    """
    Store the main training settings in one place.

    Keeping these values together makes the project easier to change later.
    For example, if you want to predict 5 candles ahead instead of 3,
    you only need to change one field here.
    """

    data_file: Path = RAW_DATA_DIR / "marketPrices.csv"
    market_data_source: str = _env_str("MARKET_DATA_SOURCE", "coinmarketcapLatestQuotes")
    coinmarketcap_use_context: bool = _env_bool("COINMARKETCAP_USE_CONTEXT", True)
    coinmarketcap_context_file: Path = RAW_DATA_DIR / "coinMarketCapContext.csv"
    coinmarketcap_use_market_intelligence: bool = _env_bool("COINMARKETCAP_USE_MARKET_INTELLIGENCE", True)
    coinmarketcap_market_intelligence_file: Path = RAW_DATA_DIR / "coinMarketCapMarketIntelligence.csv"
    coinmarketcap_api_base_url: str = "https://pro-api.coinmarketcap.com"
    coinmarketcap_api_key_env_var: str = "COINMARKETCAP_API_KEY"
    coinmarketcap_quote_currency: str = "USD"
    coinmarketcap_refresh_context_on_load: bool = _env_bool("COINMARKETCAP_REFRESH_CONTEXT_ON_LOAD", False)
    coinmarketcap_refresh_context_after_market_refresh: bool = _env_bool(
        "COINMARKETCAP_REFRESH_CONTEXT_AFTER_MARKET_REFRESH",
        True,
    )
    coinmarketcap_refresh_market_intelligence_on_load: bool = _env_bool(
        "COINMARKETCAP_REFRESH_MARKET_INTELLIGENCE_ON_LOAD",
        False,
    )
    coinmarketcap_refresh_market_intelligence_after_market_refresh: bool = _env_bool(
        "COINMARKETCAP_REFRESH_MARKET_INTELLIGENCE_AFTER_MARKET_REFRESH",
        True,
    )
    coinmarketcap_request_pause_seconds: float = _env_float("COINMARKETCAP_REQUEST_PAUSE_SECONDS", 0.2)
    coinmarketcap_log_progress: bool = _env_bool("COINMARKETCAP_LOG_PROGRESS", True)
    coinmarketcap_fetch_all_quote_products: bool = _env_bool("COINMARKETCAP_FETCH_ALL_QUOTE_PRODUCTS", True)
    coinmarketcap_product_ids: Tuple[str, ...] = ("BTC-USD",)
    coinmarketcap_excluded_base_currencies: Tuple[str, ...] = ("USDT", "USDC")
    coinmarketcap_max_products: Optional[int] = None
    coinmarketcap_product_id: str = "BTC-USD"
    coinmarketcap_product_batch_size: Optional[int] = 25
    coinmarketcap_product_batch_number: int = 1
    coinmarketcap_granularity_seconds: int = 3600
    coinmarketcap_total_candles: int = 1800
    coinmarketcap_save_progress_every_products: int = 5
    coinmarketcap_ohlcv_historical_endpoint: str = "/v2/cryptocurrency/ohlcv/historical"
    coinmarketcap_quotes_latest_endpoint: str = "/v2/cryptocurrency/quotes/latest"
    coinmarketcap_map_endpoint: str = "/v1/cryptocurrency/map"
    coinmarketcap_global_metrics_endpoint: str = "/v1/global-metrics/quotes/latest"
    coinmarketcap_fear_greed_latest_endpoint: str = "/v3/fear-and-greed/latest"
    coinmarketcap_market_fear_threshold: float = 30.0
    coinmarketcap_market_greed_threshold: float = 65.0
    coinmarketcap_market_btc_dominance_risk_off_threshold: float = 55.0
    coinmarketcal_use_events: bool = True
    coinmarketcal_events_file: Path = RAW_DATA_DIR / "coinMarketCalEvents.csv"
    coinmarketcal_api_base_url: str = "https://developers.coinmarketcal.com/v1"
    coinmarketcal_api_key_env_var: str = "COINMARKETCAL_API_KEY"
    coinmarketcal_refresh_events_on_load: bool = False
    coinmarketcal_refresh_events_after_market_refresh: bool = True
    coinmarketcal_request_pause_seconds: float = 0.2
    coinmarketcal_log_progress: bool = True
    coinmarketcal_lookahead_days: int = 30
    coinbase_fetch_all_quote_products: bool = True
    coinbase_quote_currency: str = "USD"
    coinbase_product_ids: Tuple[str, ...] = ("BTC-USD",)
    coinbase_excluded_base_currencies: Tuple[str, ...] = ("USDT", "USDC")
    coinbase_max_products: Optional[int] = None
    coinbase_product_id: str = "BTC-USD"
    coinbase_product_batch_size: Optional[int] = 25
    coinbase_product_batch_number: int = 1
    coinbase_granularity_seconds: int = 3600
    coinbase_total_candles: int = 1800
    coinbase_request_pause_seconds: float = 0.2
    coinbase_save_progress_every_products: int = 5
    coinbase_log_progress: bool = True
    live_product_ids: Tuple[str, ...] = field(
        default_factory=lambda: _env_csv_tuple(
            "LIVE_PRODUCT_IDS",
            ("BTC-USD", "ETH-USD", "SOL-USD"),
        )
    )
    live_fetch_all_quote_products: bool = _env_bool("LIVE_FETCH_ALL_QUOTE_PRODUCTS", True)
    live_max_products: Optional[int] = _env_optional_int("LIVE_MAX_PRODUCTS", 25)
    live_granularity_seconds: int = 3600
    live_total_candles: int = 120
    live_request_pause_seconds: float = 0.05
    live_signal_cache_seconds: int = 60
    signal_monitor_run_initial_generation: bool = _env_bool("SIGNAL_MONITOR_RUN_INITIAL_GENERATION", True)
    signal_monitor_refresh_interval_seconds: int = (
        _env_optional_int("SIGNAL_MONITOR_REFRESH_INTERVAL_SECONDS", 900) or 900
    )
    market_product_batch_rotation_enabled: bool = _env_bool("MARKET_PRODUCT_BATCH_ROTATION_ENABLED", True)
    market_product_batch_state_file: Path = OUTPUTS_DIR / "marketProductBatchState.json"
    signal_refresh_market_data_before_generation: bool = True
    signal_max_staleness_hours: float = 3.0
    signal_primary_rotation_enabled: bool = True
    signal_primary_rotation_lookback: int = 3
    signal_primary_rotation_candidate_window: int = 4
    signal_primary_rotation_min_score_ratio: float = 0.88
    signal_watchlist_fallback_enabled: bool = _env_bool("SIGNAL_WATCHLIST_FALLBACK_ENABLED", True)
    signal_watchlist_fallback_hours: float = _env_float("SIGNAL_WATCHLIST_FALLBACK_HOURS", 12.0)
    signal_watchlist_min_decision_score: float = _env_float("SIGNAL_WATCHLIST_MIN_DECISION_SCORE", 0.30)
    signal_watchlist_min_confidence: float = _env_float("SIGNAL_WATCHLIST_MIN_CONFIDENCE", 0.55)
    signal_watchlist_min_published_signals: int = (
        _env_optional_int("SIGNAL_WATCHLIST_MIN_PUBLISHED_SIGNALS", 2) or 2
    )
    signal_watchlist_pool_enabled: bool = _env_bool("SIGNAL_WATCHLIST_POOL_ENABLED", True)
    signal_watchlist_pool_max_products: int = _env_optional_int("SIGNAL_WATCHLIST_POOL_MAX_PRODUCTS", 12) or 12
    signal_watchlist_pool_path: Path = OUTPUTS_DIR / "watchlistPool.json"
    signal_track_generated_trades: bool = _env_bool("SIGNAL_TRACK_GENERATED_TRADES", False)
    signal_generated_trade_status: str = _env_str("SIGNAL_GENERATED_TRADE_STATUS", "planned")
    live_watchlist_pool_cache_seconds: int = _env_optional_int("LIVE_WATCHLIST_POOL_CACHE_SECONDS", 15) or 15
    assistant_system_name: str = "Crypto Signal Copilot"
    assistant_enable_retrieval: bool = True
    assistant_memory_message_limit: int = 12
    assistant_retrieval_item_limit: int = 4
    assistant_use_llm: bool = _env_bool("ASSISTANT_USE_LLM", False)
    assistant_store_path: Path = OUTPUTS_DIR / "assistantSessions.sqlite3"
    assistant_store_url: Optional[str] = os.getenv("ASSISTANT_DATABASE_URL") or os.getenv("DATABASE_URL")
    llm_provider: str = _env_str("LLM_PROVIDER", "deterministic")
    openai_model: str = _env_str("OPENAI_MODEL", "")
    openai_api_key_env_var: str = _env_str("OPENAI_API_KEY_ENV_VAR", "OPENAI_API_KEY")
    portfolio_store_path: Path = OUTPUTS_DIR / "traderPortfolio.sqlite3"
    portfolio_store_url: Optional[str] = os.getenv("PORTFOLIO_DATABASE_URL") or os.getenv("DATABASE_URL")
    signal_store_path: Path = OUTPUTS_DIR / "liveSignals.sqlite3"
    signal_store_url: Optional[str] = os.getenv("SIGNAL_DATABASE_URL") or os.getenv("DATABASE_URL")
    portfolio_default_capital: float = 10000.0
    rag_enabled: bool = True
    rag_store_path: Path = OUTPUTS_DIR / "assistantKnowledge.sqlite3"
    rag_store_url: Optional[str] = os.getenv("RAG_DATABASE_URL") or os.getenv("DATABASE_URL")
    rag_chunk_size_chars: int = 900
    rag_chunk_overlap_chars: int = 120
    rag_fetch_timeout_seconds: float = 15.0
    rag_fetch_max_chars: int = 50000
    rag_search_limit: int = 6
    signal_excluded_base_currencies: Tuple[str, ...] = field(
        default_factory=lambda: _env_csv_tuple(
            "SIGNAL_EXCLUDED_BASE_CURRENCIES",
            ("BTC", "ETH", "USDT", "USDC"),
        )
    )
    feature_context_timeframes: Tuple[str, ...] = ("4h", "1d")
    regime_features_enabled: bool = True
    regime_trend_strength_threshold: float = 0.0125
    regime_high_volatility_ratio_threshold: float = 1.20
    regime_low_volatility_ratio_threshold: float = 0.85
    model_type: str = "histGradientBoostingSignalModel"
    comparison_model_types: Tuple[str, ...] = (
        "histGradientBoostingSignalModel",
        "randomForestSignalModel",
        "logisticRegressionSignalModel",
    )
    train_size: float = 0.80
    walkforward_min_train_size: float = 0.50
    walkforward_test_size: float = 0.10
    walkforward_step_size: float = 0.10
    tuning_prediction_horizon_candidates: Tuple[int, ...] = (2, 3)
    tuning_buy_threshold_candidates: Tuple[float, ...] = (0.01, 0.0125, 0.015)
    tuning_sell_threshold_candidates: Tuple[float, ...] = (-0.01, -0.0125, -0.015)
    tuning_backtest_confidence_candidates: Tuple[float, ...] = (0.0, 0.55, 0.60, 0.65)
    labeling_strategy: str = "triple_barrier"
    prediction_horizon: int = 2
    buy_threshold: float = 0.01
    sell_threshold: float = -0.015
    triple_barrier_use_high_low: bool = True
    triple_barrier_tie_break: str = "stop_loss"
    recency_weighting_enabled: bool = True
    recency_weighting_halflife_hours: float = 336.0
    n_estimators: int = 300
    max_depth: int = 6
    min_samples_leaf: int = 3
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    hist_gradient_learning_rate: float = 0.05
    hist_gradient_max_iter: int = 300
    hist_gradient_max_depth: int = 6
    hist_gradient_min_samples_leaf: int = 20
    hist_gradient_l2_regularization: float = 0.0
    random_state: int = 42
    backtest_initial_capital: float = 10000.0
    backtest_trading_fee_rate: float = 0.001
    backtest_slippage_rate: float = 0.0005
    backtest_min_confidence: float = _env_float("BACKTEST_MIN_CONFIDENCE", 0.65)
    backtest_max_positions_per_timestamp: int = 3
    decision_policy_enabled: bool = True
    decision_block_downtrend_buys: bool = True
    decision_min_probability_margin: float = 0.08
    decision_high_volatility_confidence_buffer: float = 0.07
    decision_event_risk_confidence_buffer: float = 0.05
    brain_enabled: bool = True
    brain_max_entry_positions: int = 3
    brain_base_position_fraction: float = 0.12
    brain_min_position_fraction: float = 0.04
    brain_max_position_fraction: float = 0.18
    brain_scale_in_fraction: float = 0.05
    brain_reduce_fraction: float = 0.50
    brain_max_portfolio_risk_fraction: float = 0.35
    brain_stale_position_age_hours: float = 72.0
    brain_loss_cut_threshold: float = -0.05
    brain_profit_lock_threshold: float = 0.08
    walkforward_purge_gap_timestamps: Optional[int] = None
    production_model_max_age_hours: float = 24.0
    production_snapshot_max_age_hours: float = 6.0


def ensure_project_directories() -> None:
    """
    Create folders that the pipeline writes to.

    We call this before saving files so the project works on a fresh clone.
    """

    for folder in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def config_to_dict(config: TrainingConfig) -> Dict[str, object]:
    """
    Convert the dataclass into a JSON-friendly dictionary.

    `Path` objects are convenient inside Python, but JSON files do not know
    how to save them directly, so we convert them to strings.
    """

    config_dict = asdict(config)
    config_dict["data_file"] = str(config.data_file)
    config_dict["coinmarketcap_context_file"] = str(config.coinmarketcap_context_file)
    config_dict["coinmarketcap_market_intelligence_file"] = str(config.coinmarketcap_market_intelligence_file)
    config_dict["coinmarketcal_events_file"] = str(config.coinmarketcal_events_file)
    config_dict["market_product_batch_state_file"] = str(config.market_product_batch_state_file)
    config_dict["assistant_store_path"] = str(config.assistant_store_path)
    config_dict["portfolio_store_path"] = str(config.portfolio_store_path)
    config_dict["signal_store_path"] = str(config.signal_store_path)
    config_dict["rag_store_path"] = str(config.rag_store_path)
    return config_dict


def dict_to_config(config_dict: Dict[str, Any]) -> TrainingConfig:
    """
    Convert a saved dictionary back into a TrainingConfig object.

    We use this when loading a model from disk because saved model files
    store plain JSON-friendly values instead of Python Path objects.
    """

    restored_config = dict(config_dict)
    restored_config["data_file"] = Path(str(restored_config["data_file"]))
    if "coinmarketcap_context_file" in restored_config:
        restored_config["coinmarketcap_context_file"] = Path(str(restored_config["coinmarketcap_context_file"]))
    if "coinmarketcap_market_intelligence_file" in restored_config:
        restored_config["coinmarketcap_market_intelligence_file"] = Path(
            str(restored_config["coinmarketcap_market_intelligence_file"])
        )
    if "coinmarketcal_events_file" in restored_config:
        restored_config["coinmarketcal_events_file"] = Path(str(restored_config["coinmarketcal_events_file"]))
    if "market_product_batch_state_file" in restored_config:
        restored_config["market_product_batch_state_file"] = Path(
            str(restored_config["market_product_batch_state_file"])
        )
    if "assistant_store_path" in restored_config:
        restored_config["assistant_store_path"] = Path(str(restored_config["assistant_store_path"]))
    if "portfolio_store_path" in restored_config:
        restored_config["portfolio_store_path"] = Path(str(restored_config["portfolio_store_path"]))
    if "signal_store_path" in restored_config:
        restored_config["signal_store_path"] = Path(str(restored_config["signal_store_path"]))
    if "rag_store_path" in restored_config:
        restored_config["rag_store_path"] = Path(str(restored_config["rag_store_path"]))
    if "comparison_model_types" in restored_config:
        restored_config["comparison_model_types"] = tuple(restored_config["comparison_model_types"])
    if "feature_context_timeframes" in restored_config:
        restored_config["feature_context_timeframes"] = tuple(restored_config["feature_context_timeframes"])
    if "signal_excluded_base_currencies" in restored_config:
        restored_config["signal_excluded_base_currencies"] = tuple(
            restored_config["signal_excluded_base_currencies"]
        )
    if "coinmarketcap_product_ids" in restored_config:
        restored_config["coinmarketcap_product_ids"] = tuple(restored_config["coinmarketcap_product_ids"])
    if "coinmarketcap_excluded_base_currencies" in restored_config:
        restored_config["coinmarketcap_excluded_base_currencies"] = tuple(
            restored_config["coinmarketcap_excluded_base_currencies"]
        )
    if "coinbase_product_ids" in restored_config:
        restored_config["coinbase_product_ids"] = tuple(restored_config["coinbase_product_ids"])
    if "live_product_ids" in restored_config:
        restored_config["live_product_ids"] = tuple(restored_config["live_product_ids"])
    if "coinbase_excluded_base_currencies" in restored_config:
        restored_config["coinbase_excluded_base_currencies"] = tuple(
            restored_config["coinbase_excluded_base_currencies"]
        )
    if "tuning_prediction_horizon_candidates" in restored_config:
        restored_config["tuning_prediction_horizon_candidates"] = tuple(
            restored_config["tuning_prediction_horizon_candidates"]
        )
    if "tuning_buy_threshold_candidates" in restored_config:
        restored_config["tuning_buy_threshold_candidates"] = tuple(
            restored_config["tuning_buy_threshold_candidates"]
        )
    if "tuning_sell_threshold_candidates" in restored_config:
        restored_config["tuning_sell_threshold_candidates"] = tuple(
            restored_config["tuning_sell_threshold_candidates"]
        )
    if "tuning_backtest_confidence_candidates" in restored_config:
        restored_config["tuning_backtest_confidence_candidates"] = tuple(
            restored_config["tuning_backtest_confidence_candidates"]
        )
    return TrainingConfig(**restored_config)
