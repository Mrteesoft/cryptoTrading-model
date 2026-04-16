"""Class-based dataset-building pipeline for training and prediction."""

from abc import ABC
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import TrainingConfig, is_coinmarketcap_market_data_source
from .data import (
    BasePriceDataEnricher,
    BasePriceDataLoader,
    CoinMarketCalEventEnricher,
    CoinMarketCapContextEnricher,
    CoinMarketCapMarketIntelligenceEnricher,
    CsvPriceDataLoader,
    EnrichedPriceDataLoader,
    align_multi_timeframe_context,
)
from .features import (
    BaseFeatureEngineer,
    FEATURE_COLUMNS,
    MULTI_TIMEFRAME_FEATURE_COLUMNS,
    TechnicalFeatureEngineer,
    get_feature_pack_columns,
)
from .labels import BaseSignalLabeler, MarketRegimeLabeler, create_labeler_from_config, create_regime_labeler_from_config
from .regimes import MarketRegimeDetector


class BaseDatasetBuilder(ABC):
    """
    Base class for building ML-ready datasets.

    The base class owns the shared pipeline steps:
    - load raw data
    - build features
    - add labels
    - clean unusable rows

    Concrete dataset builders only need to provide the right components.
    """

    def __init__(
        self,
        config: TrainingConfig,
        data_loader: BasePriceDataLoader,
        feature_engineer: BaseFeatureEngineer,
        labeler: BaseSignalLabeler,
        regime_detector: MarketRegimeDetector | None = None,
        regime_labeler: MarketRegimeLabeler | None = None,
    ) -> None:
        self.config = config
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.labeler = labeler
        self.regime_detector = regime_detector
        self.regime_labeler = regime_labeler
        self.feature_columns = list(self.feature_engineer.feature_columns)

    def build_feature_table(self) -> pd.DataFrame:
        """
        Build the feature table without labels.

        This is useful when generating the newest live-like signal,
        because the newest candle does not have a known future label yet.
        """

        price_df = self.data_loader.load()
        price_df = align_multi_timeframe_context(
            price_df=price_df,
            timeframes=self.config.feature_context_timeframes,
            base_granularity_seconds=self._resolve_base_granularity_seconds(price_df),
        )
        feature_df = self.feature_engineer.build(price_df)
        if self.regime_detector is not None:
            feature_df = self.regime_detector.enrich_feature_table(feature_df)
        feature_df = self._ensure_feature_columns_exist(feature_df)
        return feature_df

    def build_labeled_dataset(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Build the full supervised-learning dataset.

        Steps:
        1. load raw data
        2. create features
        3. create labels using the labeler strategy
        4. drop rows where features or future labels are missing
        """

        dataset_bundle = self.build_labeled_dataset_bundle()
        return dataset_bundle["dataset"], dataset_bundle["feature_columns"]

    def build_labeled_dataset_bundle(self) -> Dict[str, Any]:
        """Build the labeled dataset plus pre-clean frames for audit workflows."""

        feature_df = self.build_feature_table()
        if self.regime_labeler is not None:
            feature_df = self.regime_labeler.add_labels(feature_df)
        labeled_df = self.labeler.add_labels(feature_df)
        cleaned_df = self._clean_labeled_dataset(labeled_df)

        return {
            "feature_df": feature_df,
            "labeled_df": labeled_df,
            "dataset": cleaned_df,
            "feature_columns": list(self.feature_columns),
        }

    def _clean_labeled_dataset(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that are unusable for supervised learning.

        Rolling indicators create NaN values at the beginning of the dataset.
        Labelers still create NaN values at the end of the dataset when there
        are not enough future candles to score a row.
        We drop both sections here once, in one shared place.
        """

        required_columns = list(self.feature_columns) + ["future_return"]
        if self.regime_labeler is not None:
            required_columns.append("target_market_regime_code")

        return labeled_df.dropna(
            subset=required_columns
        ).reset_index(drop=True)

    def _ensure_feature_columns_exist(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure every configured feature column exists before labels or inference use it."""

        completed_df = feature_df.copy()
        optional_context_columns = set(MULTI_TIMEFRAME_FEATURE_COLUMNS)
        optional_context_columns.update(self._regime_feature_columns())

        for feature_column in self.feature_columns:
            if feature_column not in completed_df.columns:
                completed_df[feature_column] = 0.0
            elif feature_column in optional_context_columns:
                completed_df[feature_column] = pd.to_numeric(
                    completed_df[feature_column],
                    errors="coerce",
                ).fillna(0.0)

        return completed_df

    def _resolve_base_granularity_seconds(self, price_df: pd.DataFrame) -> int:
        """Resolve the candle size from the loaded data when possible."""

        if "granularity_seconds" in price_df.columns:
            granularity_series = pd.to_numeric(price_df["granularity_seconds"], errors="coerce").dropna()
            if not granularity_series.empty:
                return int(granularity_series.mode().iloc[0])

        if is_coinmarketcap_market_data_source(self.config.market_data_source):
            return int(self.config.coinmarketcap_granularity_seconds)

        return int(self.config.coinbase_granularity_seconds)

    def _regime_feature_columns(self) -> list[str]:
        """Return the numeric regime feature columns when the detector is enabled."""

        if self.regime_detector is None:
            return []

        return [
            "regime_trend_score",
            "regime_volatility_ratio",
            "regime_is_trending",
            "regime_is_high_volatility",
            "market_regime_code",
        ]


class CryptoDatasetBuilder(BaseDatasetBuilder):
    """
    Concrete dataset builder for the current crypto signal project.

    The subclass wires together:
    - CSV loading
    - technical indicator feature engineering
    - configurable labeling, defaulting to triple barrier
    """

    def __init__(
        self,
        config: TrainingConfig,
        feature_columns: List[str] = None,
        data_loader: BasePriceDataLoader = None,
        feature_engineer: BaseFeatureEngineer = None,
        labeler: BaseSignalLabeler = None,
    ) -> None:
        if feature_columns is not None:
            selected_feature_columns = list(feature_columns)
        else:
            selected_feature_columns = get_feature_pack_columns(config.feature_pack)
            if not selected_feature_columns:
                selected_feature_columns = list(FEATURE_COLUMNS)

        if data_loader is None:
            data_loader = CsvPriceDataLoader(config.data_file)

        enrichers: List[BasePriceDataEnricher] = []
        if config.coinmarketcap_use_context:
            enrichers.append(
                CoinMarketCapContextEnricher(
                    context_path=config.coinmarketcap_context_file,
                    api_base_url=config.coinmarketcap_api_base_url,
                    api_key_env_var=config.coinmarketcap_api_key_env_var,
                    quote_currency=config.coinmarketcap_quote_currency,
                    request_pause_seconds=config.coinmarketcap_request_pause_seconds,
                    should_refresh_context=config.coinmarketcap_refresh_context_on_load,
                    log_progress=config.coinmarketcap_log_progress,
                )
            )
        if config.coinmarketcap_use_market_intelligence:
            enrichers.append(
                CoinMarketCapMarketIntelligenceEnricher(
                    intelligence_path=config.coinmarketcap_market_intelligence_file,
                    api_base_url=config.coinmarketcap_api_base_url,
                    api_key_env_var=config.coinmarketcap_api_key_env_var,
                    quote_currency=config.coinmarketcap_quote_currency,
                    request_pause_seconds=config.coinmarketcap_request_pause_seconds,
                    should_refresh_market_intelligence=config.coinmarketcap_refresh_market_intelligence_on_load,
                    log_progress=config.coinmarketcap_log_progress,
                    global_metrics_endpoint=config.coinmarketcap_global_metrics_endpoint,
                    fear_greed_latest_endpoint=config.coinmarketcap_fear_greed_latest_endpoint,
                )
            )
        if config.coinmarketcal_use_events:
            enrichers.append(
                CoinMarketCalEventEnricher(
                    events_path=config.coinmarketcal_events_file,
                    api_base_url=config.coinmarketcal_api_base_url,
                    api_key_env_var=config.coinmarketcal_api_key_env_var,
                    lookahead_days=config.coinmarketcal_lookahead_days,
                    request_pause_seconds=config.coinmarketcal_request_pause_seconds,
                    should_refresh_events=config.coinmarketcal_refresh_events_on_load,
                    log_progress=config.coinmarketcal_log_progress,
                )
            )

        if enrichers:
            data_loader = EnrichedPriceDataLoader(
                base_loader=data_loader,
                enrichers=enrichers,
            )

        feature_engineer = feature_engineer or TechnicalFeatureEngineer(
            feature_columns=selected_feature_columns,
        )
        labeler = labeler or create_labeler_from_config(config)
        regime_detector = MarketRegimeDetector(config) if config.regime_features_enabled else None
        regime_labeler = create_regime_labeler_from_config(config) if config.regime_features_enabled else None

        super().__init__(
            config=config,
            data_loader=data_loader,
            feature_engineer=feature_engineer,
            labeler=labeler,
            regime_detector=regime_detector,
            regime_labeler=regime_labeler,
        )


def build_feature_table(config: TrainingConfig) -> pd.DataFrame:
    """
    Backward-compatible helper that delegates to the concrete builder class.
    """

    return CryptoDatasetBuilder(config).build_feature_table()


def build_labeled_dataset(config: TrainingConfig) -> Tuple[pd.DataFrame, List[str]]:
    """
    Backward-compatible helper that delegates to the concrete builder class.
    """

    return CryptoDatasetBuilder(config).build_labeled_dataset()
