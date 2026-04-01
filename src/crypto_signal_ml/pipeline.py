"""Class-based dataset-building pipeline for training and prediction."""

from abc import ABC
from typing import List, Tuple

import pandas as pd

from .config import TrainingConfig
from .data import (
    BasePriceDataEnricher,
    BasePriceDataLoader,
    CoinMarketCapContextEnricher,
    CsvPriceDataLoader,
    EnrichedCsvPriceDataLoader,
)
from .features import BaseFeatureEngineer, FEATURE_COLUMNS, TechnicalFeatureEngineer
from .labels import BaseSignalLabeler, create_labeler_from_config


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
    ) -> None:
        self.config = config
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.labeler = labeler
        self.feature_columns = list(self.feature_engineer.feature_columns)

    def build_feature_table(self) -> pd.DataFrame:
        """
        Build the feature table without labels.

        This is useful when generating the newest live-like signal,
        because the newest candle does not have a known future label yet.
        """

        price_df = self.data_loader.load()
        feature_df = self.feature_engineer.build(price_df)
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

        feature_df = self.build_feature_table()
        labeled_df = self.labeler.add_labels(feature_df)
        cleaned_df = self._clean_labeled_dataset(labeled_df)

        return cleaned_df, self.feature_columns

    def _clean_labeled_dataset(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that are unusable for supervised learning.

        Rolling indicators create NaN values at the beginning of the dataset.
        Labelers still create NaN values at the end of the dataset when there
        are not enough future candles to score a row.
        We drop both sections here once, in one shared place.
        """

        return labeled_df.dropna(
            subset=self.feature_columns + ["future_return"]
        ).reset_index(drop=True)


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
        selected_feature_columns = feature_columns or list(FEATURE_COLUMNS)

        if data_loader is None:
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

            if enrichers:
                data_loader = EnrichedCsvPriceDataLoader(
                    data_path=config.data_file,
                    enrichers=enrichers,
                )
            else:
                data_loader = CsvPriceDataLoader(config.data_file)

        feature_engineer = feature_engineer or TechnicalFeatureEngineer(
            feature_columns=selected_feature_columns,
        )
        labeler = labeler or create_labeler_from_config(config)

        super().__init__(
            config=config,
            data_loader=data_loader,
            feature_engineer=feature_engineer,
            labeler=labeler,
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
