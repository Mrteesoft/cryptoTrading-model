"""Project-wide configuration values and common paths."""

from dataclasses import asdict, dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .asset_policy import STABLECOIN_BASE_CURRENCIES
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


def _env_csv_str_tuple(env_var_name: str, default_values: Tuple[str, ...]) -> Tuple[str, ...]:
    """Read one comma-separated environment variable into a trimmed tuple."""

    raw_value = str(os.getenv(env_var_name, "")).strip()
    if not raw_value:
        return tuple(default_values)

    normalized_values = []
    seen_values = set()
    for value in raw_value.split(","):
        normalized_value = str(value).strip()
        if not normalized_value or normalized_value in seen_values:
            continue
        seen_values.add(normalized_value)
        normalized_values.append(normalized_value)

    return tuple(normalized_values) if normalized_values else tuple(default_values)


def _env_str(env_var_name: str, default_value: str) -> str:
    """Read one trimmed environment string, falling back when unset."""

    raw_value = str(os.getenv(env_var_name, "")).strip()
    return raw_value or default_value


def _env_optional_str(env_var_name: str, default_value: Optional[str] = None) -> Optional[str]:
    """Read one optional trimmed environment string."""

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
KRAKEN_MARKET_DATA_SOURCES = ("kraken",)
BINANCE_MARKET_DATA_SOURCES = ("binancePublicData",)


def is_coinmarketcap_market_data_source(market_data_source: str) -> bool:
    """Return whether the configured source belongs to the CoinMarketCap family."""

    return str(market_data_source).strip() in COINMARKETCAP_MARKET_DATA_SOURCES


def is_coinbase_market_data_source(market_data_source: str) -> bool:
    """Return whether the configured source belongs to the Coinbase family."""

    return str(market_data_source).strip() in COINBASE_MARKET_DATA_SOURCES


def is_kraken_market_data_source(market_data_source: str) -> bool:
    """Return whether the configured source belongs to the Kraken family."""

    return str(market_data_source).strip() in KRAKEN_MARKET_DATA_SOURCES


def is_binance_market_data_source(market_data_source: str) -> bool:
    """Return whether the configured source belongs to the Binance public-data family."""

    return str(market_data_source).strip() in BINANCE_MARKET_DATA_SOURCES


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
            (
                "coinmarketcap_",
                "coinmarketcal_",
                "coinbase_",
                "kraken_",
                "binance_",
                "live_",
                "signal_watchlist_",
            )
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
    coinmarketcap_universe_cache_file: Path = OUTPUTS_DIR / "coinmarketcapUniverse.json"
    coinmarketcap_universe_ttl_seconds: int = _env_optional_int("COINMARKETCAP_UNIVERSE_TTL_SECONDS", 21600) or 21600
    coinmarketcap_universe_rate_limit_cooldown_seconds: int = (
        _env_optional_int("COINMARKETCAP_UNIVERSE_RATE_LIMIT_COOLDOWN_SECONDS", 1800) or 1800
    )
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
    coinmarketcap_excluded_base_currencies: Tuple[str, ...] = STABLECOIN_BASE_CURRENCIES
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
    coinmarketcal_use_events: bool = _env_bool("COINMARKETCAL_USE_EVENTS", False)
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
    coinbase_excluded_base_currencies: Tuple[str, ...] = STABLECOIN_BASE_CURRENCIES
    coinbase_max_products: Optional[int] = None
    coinbase_product_id: str = "BTC-USD"
    coinbase_product_batch_size: Optional[int] = 25
    coinbase_product_batch_number: int = 1
    coinbase_granularity_seconds: int = 3600
    coinbase_total_candles: int = 1800
    coinbase_request_pause_seconds: float = 0.2
    coinbase_save_progress_every_products: int = 5
    coinbase_log_progress: bool = True
    kraken_product_ids: Tuple[str, ...] = field(
        default_factory=lambda: _env_csv_str_tuple(
            "KRAKEN_PRODUCT_IDS",
            ("BTC-USD", "ETH-USD", "SOL-USD"),
        )
    )
    kraken_product_id: str = _env_str("KRAKEN_PRODUCT_ID", "BTC-USD")
    kraken_quote_currency: str = _env_str("KRAKEN_QUOTE_CURRENCY", "USD")
    kraken_granularity_seconds: int = _env_optional_int("KRAKEN_GRANULARITY_SECONDS", 3600) or 3600
    kraken_total_candles: int = _env_optional_int("KRAKEN_TOTAL_CANDLES", 720) or 720
    kraken_request_pause_seconds: float = _env_float("KRAKEN_REQUEST_PAUSE_SECONDS", 0.2)
    kraken_product_batch_size: Optional[int] = _env_optional_int("KRAKEN_PRODUCT_BATCH_SIZE", 25)
    kraken_product_batch_number: int = _env_optional_int("KRAKEN_PRODUCT_BATCH_NUMBER", 1) or 1
    kraken_save_progress_every_products: int = _env_optional_int("KRAKEN_SAVE_PROGRESS_EVERY_PRODUCTS", 5) or 5
    kraken_log_progress: bool = _env_bool("KRAKEN_LOG_PROGRESS", True)
    binance_product_ids: Tuple[str, ...] = field(
        default_factory=lambda: _env_csv_str_tuple(
            "BINANCE_PRODUCT_IDS",
            ("BTC-USD", "ETH-USD", "SOL-USD"),
        )
    )
    binance_product_id: str = _env_str("BINANCE_PRODUCT_ID", "BTC-USD")
    binance_fetch_all_quote_products: bool = _env_bool("BINANCE_FETCH_ALL_QUOTE_PRODUCTS", True)
    binance_quote_currency: str = _env_str("BINANCE_QUOTE_CURRENCY", "USDT")
    binance_excluded_base_currencies: Tuple[str, ...] = STABLECOIN_BASE_CURRENCIES
    binance_max_products: Optional[int] = _env_optional_int("BINANCE_MAX_PRODUCTS", 150)
    binance_interval: str = _env_str("BINANCE_INTERVAL", "1h")
    binance_granularity_seconds: int = _env_optional_int("BINANCE_GRANULARITY_SECONDS", 3600) or 3600
    binance_total_candles: int = _env_optional_int("BINANCE_TOTAL_CANDLES", 4320) or 4320
    binance_archive_lookback_months: int = _env_optional_int("BINANCE_ARCHIVE_LOOKBACK_MONTHS", 36) or 36
    binance_request_pause_seconds: float = _env_float("BINANCE_REQUEST_PAUSE_SECONDS", 0.1)
    binance_product_batch_size: Optional[int] = _env_optional_int("BINANCE_PRODUCT_BATCH_SIZE", 25)
    binance_product_batch_number: int = _env_optional_int("BINANCE_PRODUCT_BATCH_NUMBER", 1) or 1
    binance_save_progress_every_products: int = _env_optional_int("BINANCE_SAVE_PROGRESS_EVERY_PRODUCTS", 5) or 5
    binance_log_progress: bool = _env_bool("BINANCE_LOG_PROGRESS", True)
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
    live_buy_signals_only: bool = _env_bool("LIVE_BUY_SIGNALS_ONLY", True)
    live_auto_clear_loss_signals: bool = _env_bool("LIVE_AUTO_CLEAR_LOSS_SIGNALS", True)
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
    signal_log_symbol_details: bool = _env_bool("SIGNAL_LOG_SYMBOL_DETAILS", True)
    signal_log_symbol_limit: int = _env_optional_int("SIGNAL_LOG_SYMBOL_LIMIT", 25) or 25
    signal_watchlist_state_path: Path = OUTPUTS_DIR / "watchlistState.json"
    signal_watchlist_diagnostics_enabled: bool = _env_bool("SIGNAL_WATCHLIST_DIAGNOSTICS_ENABLED", True)
    signal_watchlist_diagnostics_max_transitions: int = (
        _env_optional_int("SIGNAL_WATCHLIST_DIAGNOSTICS_MAX_TRANSITIONS", 6) or 6
    )
    signal_watchlist_diagnostics_top_reasons: int = (
        _env_optional_int("SIGNAL_WATCHLIST_DIAGNOSTICS_TOP_REASONS", 3) or 3
    )
    signal_watchlist_history_max: int = _env_optional_int("SIGNAL_WATCHLIST_HISTORY_MAX", 6) or 6
    signal_watchlist_breakout_pct: float = _env_float("SIGNAL_WATCHLIST_BREAKOUT_PCT", 0.01)
    signal_watchlist_invalidation_pct: float = _env_float("SIGNAL_WATCHLIST_INVALIDATION_PCT", 0.02)
    signal_watchlist_invalidation_confidence: float = _env_float("SIGNAL_WATCHLIST_INVALIDATION_CONFIDENCE", 0.25)
    signal_watchlist_invalidation_min_probability_margin: float = _env_float(
        "SIGNAL_WATCHLIST_INVALIDATION_MIN_PROBABILITY_MARGIN",
        0.08,
    )
    signal_watchlist_soft_review_confidence_buffer: float = _env_float(
        "SIGNAL_WATCHLIST_SOFT_REVIEW_CONFIDENCE_BUFFER",
        0.08,
    )
    signal_watchlist_soft_review_min_raw_confidence: float = _env_float(
        "SIGNAL_WATCHLIST_SOFT_REVIEW_MIN_RAW_CONFIDENCE",
        0.50,
    )
    signal_watchlist_soft_review_min_probability_margin: float = _env_float(
        "SIGNAL_WATCHLIST_SOFT_REVIEW_MIN_PROBABILITY_MARGIN",
        0.12,
    )
    signal_watchlist_preserve_strong_blocked_buys: bool = _env_bool(
        "SIGNAL_WATCHLIST_PRESERVE_STRONG_BLOCKED_BUYS",
        True,
    )
    signal_watchlist_strong_buy_min_raw_confidence: float = _env_float(
        "SIGNAL_WATCHLIST_STRONG_BUY_MIN_RAW_CONFIDENCE",
        0.72,
    )
    signal_watchlist_strong_buy_min_probability_margin: float = _env_float(
        "SIGNAL_WATCHLIST_STRONG_BUY_MIN_PROBABILITY_MARGIN",
        0.18,
    )
    signal_watchlist_promotion_min_confidence: float = _env_float(
        "SIGNAL_WATCHLIST_PROMOTION_MIN_CONFIDENCE",
        0.58,
    )
    signal_watchlist_promotion_min_decision_score: float = _env_float(
        "SIGNAL_WATCHLIST_PROMOTION_MIN_DECISION_SCORE",
        0.55,
    )
    signal_watchlist_promotion_min_confidence_gain: float = _env_float(
        "SIGNAL_WATCHLIST_PROMOTION_MIN_CONFIDENCE_GAIN",
        0.05,
    )
    signal_watchlist_promotion_min_decision_score_gain: float = _env_float(
        "SIGNAL_WATCHLIST_PROMOTION_MIN_DECISION_SCORE_GAIN",
        0.08,
    )
    signal_watchlist_promotion_min_positive_checks: int = _env_optional_int(
        "SIGNAL_WATCHLIST_PROMOTION_MIN_POSITIVE_CHECKS",
        2,
    ) or 2
    signal_watchlist_setup_building_min_checks: int = _env_optional_int(
        "SIGNAL_WATCHLIST_SETUP_BUILDING_MIN_CHECKS",
        2,
    ) or 2
    signal_watchlist_entry_ready_min_positive_checks: int = _env_optional_int(
        "SIGNAL_WATCHLIST_ENTRY_READY_MIN_POSITIVE_CHECKS",
        2,
    ) or 2
    signal_watchlist_entry_ready_min_confidence: float = _env_float(
        "SIGNAL_WATCHLIST_ENTRY_READY_MIN_CONFIDENCE",
        0.62,
    )
    signal_watchlist_entry_ready_min_decision_score: float = _env_float(
        "SIGNAL_WATCHLIST_ENTRY_READY_MIN_DECISION_SCORE",
        0.62,
    )
    signal_watchlist_entry_ready_min_resistance_distance_pct: float = _env_float(
        "SIGNAL_WATCHLIST_ENTRY_READY_MIN_RESISTANCE_DISTANCE_PCT",
        0.015,
    )
    signal_watchlist_soft_risk_override_min_confirmation: float = _env_float(
        "SIGNAL_WATCHLIST_SOFT_RISK_OVERRIDE_MIN_CONFIRMATION",
        0.72,
    )
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
    broker_redis_url: Optional[str] = _env_optional_str("REDIS_URL") or _env_optional_str("BROKER_REDIS_URL")
    job_state_ttl_seconds: int = _env_optional_int("JOB_STATE_TTL_SECONDS", 86400) or 86400
    rabbitmq_url: Optional[str] = _env_optional_str("RABBITMQ_URL") or _env_optional_str("BROKER_RABBITMQ_URL")
    rabbitmq_command_exchange: str = _env_str("RABBITMQ_COMMAND_EXCHANGE", "commands.topic")
    kafka_brokers: Tuple[str, ...] = field(
        default_factory=lambda: tuple(
            value.strip()
            for value in str(os.getenv("KAFKA_BROKERS", "")).split(",")
            if value.strip()
        )
    )
    kafka_client_id: str = _env_str("KAFKA_CLIENT_ID", "model-service-worker")
    kafka_topic_prefix: str = _env_str("KAFKA_TOPIC_PREFIX", "")
    worker_prefetch_count: int = _env_optional_int("WORKER_PREFETCH_COUNT", 2) or 2
    news_store_path: Path = OUTPUTS_DIR / "newsFeed.json"
    event_window_minutes: float = _env_float("EVENT_WINDOW_MINUTES", 360.0)
    event_post_window_minutes: float = _env_float("EVENT_POST_WINDOW_MINUTES", 240.0)
    event_high_impact_threshold: float = _env_float("EVENT_HIGH_IMPACT_THRESHOLD", 0.65)
    event_risk_decision_penalty: float = _env_float("EVENT_RISK_DECISION_PENALTY", 0.08)
    news_positive_sentiment_threshold: float = _env_float("NEWS_POSITIVE_SENTIMENT_THRESHOLD", 0.20)
    news_negative_sentiment_threshold: float = _env_float("NEWS_NEGATIVE_SENTIMENT_THRESHOLD", -0.20)
    news_positive_decision_boost: float = _env_float("NEWS_POSITIVE_DECISION_BOOST", 0.04)
    news_negative_decision_penalty: float = _env_float("NEWS_NEGATIVE_DECISION_PENALTY", 0.08)
    trend_support_threshold: float = _env_float("TREND_SUPPORT_THRESHOLD", 0.35)
    chart_feature_window: int = _env_optional_int("CHART_FEATURE_WINDOW", 60) or 60
    chart_breakout_buffer_pct: float = _env_float("CHART_BREAKOUT_BUFFER_PCT", 0.005)
    chart_retest_tolerance_pct: float = _env_float("CHART_RETEST_TOLERANCE_PCT", 0.003)
    chart_near_resistance_pct: float = _env_float("CHART_NEAR_RESISTANCE_PCT", 0.01)
    chart_confirmation_confirmed_min_score: float = _env_float(
        "CHART_CONFIRMATION_CONFIRMED_MIN_SCORE",
        0.30,
    )
    chart_confirmation_early_min_score: float = _env_float(
        "CHART_CONFIRMATION_EARLY_MIN_SCORE",
        0.12,
    )
    chart_confirmation_invalid_max_score: float = _env_float(
        "CHART_CONFIRMATION_INVALID_MAX_SCORE",
        -0.10,
    )
    chart_positive_decision_boost: float = _env_float("CHART_POSITIVE_DECISION_BOOST", 0.05)
    chart_negative_decision_penalty: float = _env_float("CHART_NEGATIVE_DECISION_PENALTY", 0.08)
    chart_snapshot_enabled: bool = _env_bool("CHART_SNAPSHOT_ENABLED", False)
    chart_snapshot_max_signals: int = _env_optional_int("CHART_SNAPSHOT_MAX_SIGNALS", 6) or 6
    chart_snapshot_dir: Path = OUTPUTS_DIR / "chartSnapshots"
    chart_eval_max_rows_total: int = _env_optional_int("CHART_EVAL_MAX_ROWS_TOTAL", 50000) or 50000
    signal_excluded_base_currencies: Tuple[str, ...] = field(
        default_factory=lambda: _env_csv_tuple(
            "SIGNAL_EXCLUDED_BASE_CURRENCIES",
            ("BTC", "ETH", *STABLECOIN_BASE_CURRENCIES),
        )
    )
    feature_context_timeframes: Tuple[str, ...] = ("4h", "1d")
    feature_pack: str = _env_str("FEATURE_PACK", "all")
    feature_include_groups: Tuple[str, ...] = field(
        default_factory=lambda: _env_csv_str_tuple("FEATURE_INCLUDE_GROUPS", ())
    )
    feature_exclude_groups: Tuple[str, ...] = field(
        default_factory=lambda: _env_csv_str_tuple("FEATURE_EXCLUDE_GROUPS", ())
    )
    feature_pack_candidates: Tuple[str, ...] = field(
        default_factory=lambda: _env_csv_str_tuple(
            "FEATURE_PACK_CANDIDATES",
            ("core", "core_plus_market", "core_plus_context", "core_plus_fundamentals", "all"),
        )
    )
    feature_audit_correlation_threshold: float = _env_float("FEATURE_AUDIT_CORRELATION_THRESHOLD", 0.95)
    regime_features_enabled: bool = True
    regime_trend_strength_threshold: float = 0.0125
    regime_high_volatility_ratio_threshold: float = 1.20
    regime_low_volatility_ratio_threshold: float = 0.85
    signal_model_family: str = _env_str("SIGNAL_MODEL_FAMILY", "baseline_current")
    signal_model_variant: str = _env_str("SIGNAL_MODEL_VARIANT", "default")
    enable_online_drift_detection: bool = _env_bool("ENABLE_ONLINE_DRIFT_DETECTION", False)
    enable_watchlist_progression_model: bool = _env_bool("ENABLE_WATCHLIST_PROGRESSION_MODEL", False)
    enable_tft_experiments: bool = _env_bool("ENABLE_TFT_EXPERIMENTS", False)
    comparison_run_walk_forward: bool = _env_bool("COMPARISON_RUN_WALK_FORWARD", False)
    comparison_min_trade_count: int = _env_optional_int("COMPARISON_MIN_TRADE_COUNT", 5) or 5
    model_type: str = "histGradientBoostingSignalModel"
    comparison_model_types: Tuple[str, ...] = (
        "histGradientBoostingSignalModel",
        "randomForestSignalModel",
        "logisticRegressionSignalModel",
        "lightgbmClassifierSignalModel",
        "xgboostClassifierSignalModel",
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
    triple_barrier_use_atr: bool = _env_bool("TRIPLE_BARRIER_USE_ATR", True)
    triple_barrier_atr_period: int = _env_optional_int("TRIPLE_BARRIER_ATR_PERIOD", 14) or 14
    triple_barrier_buy_atr_multiplier: float = _env_float("TRIPLE_BARRIER_BUY_ATR_MULTIPLIER", 1.25)
    triple_barrier_sell_atr_multiplier: float = _env_float("TRIPLE_BARRIER_SELL_ATR_MULTIPLIER", 1.00)
    calibration_enabled: bool = _env_bool("CALIBRATION_ENABLED", True)
    calibration_holdout_fraction: float = _env_float("CALIBRATION_HOLDOUT_FRACTION", 0.15)
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
    lightgbm_n_estimators: int = 400
    lightgbm_learning_rate: float = 0.05
    lightgbm_num_leaves: int = 31
    lightgbm_max_depth: int = -1
    lightgbm_min_child_samples: int = 20
    lightgbm_subsample: float = 0.90
    lightgbm_colsample_bytree: float = 0.90
    xgboost_n_estimators: int = 400
    xgboost_learning_rate: float = 0.05
    xgboost_max_depth: int = 6
    xgboost_min_child_weight: float = 1.0
    xgboost_subsample: float = 0.90
    xgboost_colsample_bytree: float = 0.90
    xgboost_gamma: float = 0.0
    tft_max_epochs: int = 25
    tft_hidden_size: int = 16
    tft_attention_head_size: int = 4
    tft_dropout: float = 0.10
    tft_hidden_continuous_size: int = 8
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
    for field_name in TrainingConfig.__dataclass_fields__:
        field_value = getattr(config, field_name)
        if isinstance(field_value, Path):
            config_dict[field_name] = str(field_value)
    return config_dict


def dict_to_config(config_dict: Dict[str, Any]) -> TrainingConfig:
    """
    Convert a saved dictionary back into a TrainingConfig object.

    We use this when loading a model from disk because saved model files
    store plain JSON-friendly values instead of Python Path objects.
    """

    restored_config = dict(config_dict)
    default_config = TrainingConfig()
    for field_name in TrainingConfig.__dataclass_fields__:
        if field_name not in restored_config:
            continue
        if isinstance(getattr(default_config, field_name), Path):
            restored_config[field_name] = Path(str(restored_config[field_name]))
    if "comparison_model_types" in restored_config:
        restored_config["comparison_model_types"] = tuple(restored_config["comparison_model_types"])
    if "feature_context_timeframes" in restored_config:
        restored_config["feature_context_timeframes"] = tuple(restored_config["feature_context_timeframes"])
    if "feature_include_groups" in restored_config:
        restored_config["feature_include_groups"] = tuple(restored_config["feature_include_groups"])
    if "feature_exclude_groups" in restored_config:
        restored_config["feature_exclude_groups"] = tuple(restored_config["feature_exclude_groups"])
    if "feature_pack_candidates" in restored_config:
        restored_config["feature_pack_candidates"] = tuple(restored_config["feature_pack_candidates"])
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
