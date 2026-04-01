"""Project-wide configuration values and common paths."""

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class TrainingConfig:
    """
    Store the main training settings in one place.

    Keeping these values together makes the project easier to change later.
    For example, if you want to predict 5 candles ahead instead of 3,
    you only need to change one field here.
    """

    data_file: Path = RAW_DATA_DIR / "marketPrices.csv"
    market_data_source: str = "coinbaseExchange"
    coinmarketcap_use_context: bool = True
    coinmarketcap_context_file: Path = RAW_DATA_DIR / "coinMarketCapContext.csv"
    coinmarketcap_api_base_url: str = "https://pro-api.coinmarketcap.com"
    coinmarketcap_api_key_env_var: str = "COINMARKETCAP_API_KEY"
    coinmarketcap_quote_currency: str = "USD"
    coinmarketcap_refresh_context_on_load: bool = False
    coinmarketcap_refresh_context_after_market_refresh: bool = True
    coinmarketcap_request_pause_seconds: float = 0.2
    coinmarketcap_log_progress: bool = True
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
    live_product_ids: Tuple[str, ...] = ("BTC-USD", "ETH-USD", "SOL-USD")
    live_fetch_all_quote_products: bool = False
    live_max_products: Optional[int] = 12
    live_granularity_seconds: int = 3600
    live_total_candles: int = 120
    live_request_pause_seconds: float = 0.05
    live_signal_cache_seconds: int = 60
    assistant_system_name: str = "Crypto Signal Copilot"
    assistant_enable_retrieval: bool = True
    assistant_memory_message_limit: int = 12
    assistant_retrieval_item_limit: int = 4
    rag_enabled: bool = True
    rag_store_path: Path = OUTPUTS_DIR / "assistantKnowledge.sqlite3"
    rag_chunk_size_chars: int = 900
    rag_chunk_overlap_chars: int = 120
    rag_fetch_timeout_seconds: float = 15.0
    rag_fetch_max_chars: int = 50000
    rag_search_limit: int = 6
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
    prediction_horizon: int = 3
    buy_threshold: float = 0.015
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
    backtest_min_confidence: float = 0.0
    backtest_max_positions_per_timestamp: int = 3
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
    if "rag_store_path" in restored_config:
        restored_config["rag_store_path"] = Path(str(restored_config["rag_store_path"]))
    if "comparison_model_types" in restored_config:
        restored_config["comparison_model_types"] = tuple(restored_config["comparison_model_types"])
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
