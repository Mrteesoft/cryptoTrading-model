"""Application-level classes for training, comparison, and signal generation."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Type

import pandas as pd

from .backtesting import BaseSignalBacktester, EqualWeightSignalBacktester
from .config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    TrainingConfig,
    apply_runtime_market_data_settings,
    config_to_dict,
    ensure_project_directories,
    is_coinmarketcap_market_data_source,
)
from .data import (
    CoinMarketCalEventEnricher,
    CoinMarketCapContextEnricher,
    CoinMarketCapMarketIntelligenceEnricher,
    CsvPriceDataLoader,
    create_market_data_loader,
)
from .frontend import WatchlistPoolStore
from .labels import create_labeler_from_config, create_regime_labeler_from_config
from .modeling import BaseSignalModel, create_model_from_config, get_model_class
from .pipeline import CryptoDatasetBuilder
from .source_refresh import SignalUniverseCoordinator
from .trading.portfolio import TradingPortfolioStore
from .trading.signal_store import TradingSignalStore
from .regime_modeling import MarketRegimeModel
from .application import (
    PrimarySignalHistoryStore,
    SignalContextEnrichmentStage,
    SignalDecisionStage,
    SignalEnrichmentStage,
    SignalGenerationCoordinator,
    SignalInferenceStage,
    SignalPublicationStage,
)


LOGGER = logging.getLogger(__name__)


class BaseSignalApp:
    """
    Shared app-level behavior for command-line workflows.

    Training, comparison, and signal generation all need:
    - project directory setup
    - access to config
    - access to the dataset builder
    - file save helpers

    Putting that here keeps the command scripts very small.
    """

    def __init__(
        self,
        config: TrainingConfig = None,
        dataset_builder: CryptoDatasetBuilder = None,
        model: BaseSignalModel = None,
        model_class: Type[BaseSignalModel] = None,
    ) -> None:
        self.config = config or TrainingConfig()
        if model_class is None:
            from .ml.models.registry import ensure_registry_loaded, resolve_model_type

            ensure_registry_loaded()
            resolved_model_type = resolve_model_type(self.config)
            self.model_class = get_model_class(resolved_model_type)
        else:
            self.model_class = model_class
        self.dataset_builder = dataset_builder or CryptoDatasetBuilder(self.config)
        self.model = model

        ensure_project_directories()

    @property
    def model_path(self) -> Path:
        """Return the path of the model artifact for the chosen model class."""

        active_model_class = self.model_class
        if self.model is not None:
            active_model_class = self.model.__class__

        return MODELS_DIR / active_model_class.default_model_filename

    @property
    def dataset_path(self) -> Path:
        """
        Return the prepared dataset path used by training and comparison.

        The name is intentionally generic because the project now supports
        many coins, not only BTC.
        """

        return PROCESSED_DATA_DIR / "marketFeaturesAndLabels.csv"

    def save_dataframe(self, dataframe: pd.DataFrame, file_path: Path) -> None:
        """Save a DataFrame to CSV."""

        self._write_file_atomically(
            file_path=file_path,
            writer=lambda temp_path: dataframe.to_csv(temp_path, index=False),
        )

    def save_json(self, payload: Dict[str, Any], file_path: Path) -> None:
        """Save a dictionary to JSON with readable indentation."""

        def write_json(temp_path: Path) -> None:
            with temp_path.open("w", encoding="utf-8") as output_file:
                json.dump(payload, output_file, indent=2)

        self._write_file_atomically(file_path=file_path, writer=write_json)

    def _write_file_atomically(
        self,
        file_path: Path,
        writer,
    ) -> None:
        """Write one output file through a temp file before replacing the target."""

        file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file_descriptor, temp_file_name = tempfile.mkstemp(
            dir=file_path.parent,
            prefix=f".{file_path.stem}-",
            suffix=".tmp",
        )
        os.close(temp_file_descriptor)
        temp_path = Path(temp_file_name)

        try:
            writer(temp_path)
            temp_path.replace(file_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def model_metadata_path(model_path: Path) -> Path:
        """Return the JSON sidecar path for a saved model artifact."""

        return model_path.with_suffix(".metadata.json")

    @staticmethod
    def _file_timestamp_to_isoformat(file_path: Path) -> str | None:
        """Return the file's modified timestamp as a UTC ISO string when it exists."""

        if not file_path.exists():
            return None

        return datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()

    @staticmethod
    def _resolve_walkforward_purge_gap(config: TrainingConfig) -> int:
        """Choose the gap that purges train rows nearest the validation window."""

        if config.walkforward_purge_gap_timestamps is not None:
            return int(config.walkforward_purge_gap_timestamps)

        return int(config.prediction_horizon)

    def _resolve_market_quote_currency(self) -> str:
        """Return the active quote currency for the configured market source."""

        if is_coinmarketcap_market_data_source(self.config.market_data_source):
            return self.config.coinmarketcap_quote_currency

        return self.config.coinbase_quote_currency

    def _resolve_market_fetch_all_quote_products(self) -> bool:
        """Return whether the active market source is in quote-universe mode."""

        if is_coinmarketcap_market_data_source(self.config.market_data_source):
            return bool(self.config.coinmarketcap_fetch_all_quote_products)

        return bool(self.config.coinbase_fetch_all_quote_products)

    def _resolve_market_granularity_seconds(self) -> int:
        """Return the active base-candle size for the configured market source."""

        if is_coinmarketcap_market_data_source(self.config.market_data_source):
            return int(self.config.coinmarketcap_granularity_seconds)

        return int(self.config.coinbase_granularity_seconds)

    def _resolve_market_product_batch_number(self) -> int:
        """Return the active product-batch index for the configured market source."""

        if is_coinmarketcap_market_data_source(self.config.market_data_source):
            return int(self.config.coinmarketcap_product_batch_number)

        return int(self.config.coinbase_product_batch_number)

    def _resolve_market_product_batch_size(self) -> int | None:
        """Return the active product-batch size for the configured market source."""

        if is_coinmarketcap_market_data_source(self.config.market_data_source):
            batch_size = self.config.coinmarketcap_product_batch_size
        else:
            batch_size = self.config.coinbase_product_batch_size

        return None if batch_size is None else int(batch_size)

    def _with_market_product_batch_number(
        self,
        batch_number: int,
    ) -> TrainingConfig:
        """Clone the config with the active source's product batch number updated."""

        if is_coinmarketcap_market_data_source(self.config.market_data_source):
            return replace(
                self.config,
                coinmarketcap_product_batch_number=int(batch_number),
            )

        return replace(
            self.config,
            coinbase_product_batch_number=int(batch_number),
        )

    def _market_product_batch_state_key(self) -> str:
        """Build one stable key for batch rotation state across different market sources."""

        batch_size = self._resolve_market_product_batch_size()
        return (
            f"{self.config.market_data_source}:"
            f"{self._resolve_market_quote_currency().upper()}:"
            f"{batch_size if batch_size is not None else 'all'}"
        )

    def _load_market_product_batch_state(self) -> dict[str, Any]:
        """Load the persisted market-batch rotation state when present."""

        state_path = Path(self.config.market_product_batch_state_file)
        if not state_path.exists():
            return {}

        try:
            with state_path.open("r", encoding="utf-8") as state_file:
                payload = json.load(state_file)
        except (OSError, json.JSONDecodeError):
            return {}

        return payload if isinstance(payload, dict) else {}

    def _load_last_market_refresh_batch_number(self) -> int | None:
        """Read the most recent published refresh batch when no explicit rotation state exists yet."""

        candidate_paths = (
            OUTPUTS_DIR / "signalMarketDataRefresh.json",
            OUTPUTS_DIR / "marketDataRefreshOnDemand.json",
        )

        for candidate_path in candidate_paths:
            if not candidate_path.exists():
                continue

            try:
                with candidate_path.open("r", encoding="utf-8") as refresh_file:
                    payload = json.load(refresh_file)
            except (OSError, json.JSONDecodeError):
                continue

            if not isinstance(payload, dict):
                continue

            refresh_payload = payload.get("refresh", payload)
            if not isinstance(refresh_payload, dict):
                continue

            download_summary = refresh_payload.get("downloadSummary", {})
            if not isinstance(download_summary, dict):
                continue

            try:
                batch_number = int(download_summary.get("batchNumber") or 0)
            except (TypeError, ValueError):
                batch_number = 0

            if batch_number > 0:
                return batch_number

        return None

    def _resolve_rotated_market_product_batch_number(
        self,
        total_batches: int,
    ) -> tuple[int, dict[str, Any]]:
        """Resolve the active batch number for one rotating market refresh."""

        configured_batch_number = self._resolve_market_product_batch_number()
        if total_batches <= 0:
            return configured_batch_number, {
                "enabled": False,
                "activeBatchNumber": configured_batch_number,
                "nextBatchNumber": configured_batch_number,
                "totalBatches": 0,
            }

        normalized_configured_batch = ((configured_batch_number - 1) % total_batches) + 1
        state_payload = self._load_market_product_batch_state()
        source_states = state_payload.get("sources", {}) if isinstance(state_payload, dict) else {}
        source_state = (
            source_states.get(self._market_product_batch_state_key(), {})
            if isinstance(source_states, dict)
            else {}
        )

        active_batch_number = 0
        if isinstance(source_state, dict):
            try:
                active_batch_number = int(source_state.get("nextBatch") or 0)
            except (TypeError, ValueError):
                active_batch_number = 0

        if active_batch_number <= 0:
            last_refresh_batch_number = self._load_last_market_refresh_batch_number()
            if last_refresh_batch_number is not None:
                active_batch_number = (int(last_refresh_batch_number) % total_batches) + 1

        if active_batch_number <= 0:
            active_batch_number = normalized_configured_batch

        active_batch_number = ((active_batch_number - 1) % total_batches) + 1
        next_batch_number = (active_batch_number % total_batches) + 1

        return active_batch_number, {
            "enabled": True,
            "stateKey": self._market_product_batch_state_key(),
            "statePath": str(self.config.market_product_batch_state_file),
            "configuredBatchNumber": normalized_configured_batch,
            "activeBatchNumber": active_batch_number,
            "nextBatchNumber": next_batch_number,
            "totalBatches": int(total_batches),
        }

    def _save_market_product_batch_state(
        self,
        active_batch_number: int,
        total_batches: int,
    ) -> dict[str, Any]:
        """Persist the next batch that should be used for the following refresh."""

        state_path = Path(self.config.market_product_batch_state_file)
        state_payload = self._load_market_product_batch_state()
        source_states = state_payload.get("sources", {}) if isinstance(state_payload, dict) else {}
        if not isinstance(source_states, dict):
            source_states = {}

        next_batch_number = (int(active_batch_number) % int(total_batches)) + 1
        source_state = {
            "marketDataSource": str(self.config.market_data_source),
            "quoteCurrency": self._resolve_market_quote_currency().upper(),
            "batchSize": self._resolve_market_product_batch_size(),
            "lastCompletedBatch": int(active_batch_number),
            "nextBatch": int(next_batch_number),
            "totalBatches": int(total_batches),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }

        source_states[self._market_product_batch_state_key()] = source_state
        persisted_payload = {
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "sources": source_states,
        }
        self.save_json(persisted_payload, state_path)
        return source_state

    def _ensure_market_data_available(self) -> None:
        """
        Bootstrap the raw market CSV when a training-style workflow starts empty.

        This lets the CLI recover from a missing `marketPrices.csv` by
        refreshing market data once and recording that bootstrap step.
        """

        if self.config.data_file.exists():
            return

        print(
            f"Raw market data not found at {self.config.data_file}. "
            "Refreshing market data before continuing."
        )
        refresh_result = MarketDataRefreshApp(config=self.config).run()
        self.save_json(
            {
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "trigger": "automatic_training_bootstrap",
                "refresh": refresh_result,
            },
            OUTPUTS_DIR / "marketDataRefreshOnDemand.json",
        )

        if not self.config.data_file.exists():
            raise FileNotFoundError(
                f"Expected market data after refresh, but the file is still missing: {self.config.data_file}"
            )

    def save_model_artifact_outputs(
        self,
        model: BaseSignalModel,
        metrics: Dict[str, Any],
        prediction_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        dataset_path: Path,
        model_path: Path,
        metrics_path: Path,
        predictions_path: Path,
        feature_importance_path: Path,
    ) -> Dict[str, str]:
        """Persist the model bundle plus a JSON sidecar for operations metadata."""

        model.save(model_path)
        self.save_json(metrics, metrics_path)
        self.save_dataframe(prediction_df, predictions_path)
        self.save_dataframe(model.get_feature_importance_frame(), feature_importance_path)

        metadata_path = self.model_metadata_path(model_path)
        metadata_payload = {
            "artifactCreatedAt": datetime.now(timezone.utc).isoformat(),
            "artifactPath": str(model_path),
            "modelType": model.model_type,
            "featureCount": len(model.feature_columns),
            "featurePreview": list(model.feature_columns[:10]),
            "labeling": {
                "strategy": model.config.labeling_strategy,
                "predictionHorizon": int(model.config.prediction_horizon),
                "buyThreshold": float(model.config.buy_threshold),
                "sellThreshold": float(model.config.sell_threshold),
                "walkForwardPurgeGapTimestamps": int(self._resolve_walkforward_purge_gap(model.config)),
            },
            "datasetPath": str(dataset_path),
            "trainingDataPath": str(model.config.data_file),
            "trainingDataLastModified": self._file_timestamp_to_isoformat(Path(model.config.data_file)),
            "trainRows": int(len(train_df)),
            "testRows": int(len(test_df)),
            "metrics": {
                "accuracy": float(metrics["accuracy"]),
                "balancedAccuracy": float(metrics["balanced_accuracy"]),
            },
            "artifacts": {
                "metricsPath": str(metrics_path),
                "predictionsPath": str(predictions_path),
                "featureImportancePath": str(feature_importance_path),
            },
            "config": config_to_dict(model.config),
        }
        self.save_json(metadata_payload, metadata_path)

        return {
            "modelPath": str(model_path),
            "metricsPath": str(metrics_path),
            "predictionsPath": str(predictions_path),
            "featureImportancePath": str(feature_importance_path),
            "metadataPath": str(metadata_path),
        }

    def build_market_data_loader(self):
        """
        Create the configured real-market-data loader.

        Keeping this factory in the base app avoids rebuilding the same
        loader-selection logic in multiple places.
        """

        return create_market_data_loader(
            config=self.config,
            data_path=self.config.data_file,
            should_save_downloaded_data=True,
        )

    def _build_market_data_loader_for_config(
        self,
        config: TrainingConfig,
    ):
        """Create one market loader for an explicit config snapshot."""

        return create_market_data_loader(
            config=config,
            data_path=config.data_file,
            should_save_downloaded_data=True,
        )

    def build_coinmarketcap_context_enricher(
        self,
        should_refresh_context: bool = False,
    ) -> CoinMarketCapContextEnricher:
        """
        Create the configured CoinMarketCap enrichment helper.

        We keep this as a reusable factory so both:
        - market refresh workflows
        - training dataset loaders

        can share the same context settings.
        """

        return CoinMarketCapContextEnricher(
            context_path=self.config.coinmarketcap_context_file,
            api_base_url=self.config.coinmarketcap_api_base_url,
            api_key_env_var=self.config.coinmarketcap_api_key_env_var,
            quote_currency=self.config.coinmarketcap_quote_currency,
            request_pause_seconds=self.config.coinmarketcap_request_pause_seconds,
            should_refresh_context=should_refresh_context,
            log_progress=self.config.coinmarketcap_log_progress,
        )

    def build_coinmarketcap_market_intelligence_enricher(
        self,
        should_refresh_market_intelligence: bool = False,
    ) -> CoinMarketCapMarketIntelligenceEnricher:
        """Create the configured CoinMarketCap market-intelligence helper."""

        return CoinMarketCapMarketIntelligenceEnricher(
            intelligence_path=self.config.coinmarketcap_market_intelligence_file,
            api_base_url=self.config.coinmarketcap_api_base_url,
            api_key_env_var=self.config.coinmarketcap_api_key_env_var,
            quote_currency=self.config.coinmarketcap_quote_currency,
            request_pause_seconds=self.config.coinmarketcap_request_pause_seconds,
            should_refresh_market_intelligence=should_refresh_market_intelligence,
            log_progress=self.config.coinmarketcap_log_progress,
            global_metrics_endpoint=self.config.coinmarketcap_global_metrics_endpoint,
            fear_greed_latest_endpoint=self.config.coinmarketcap_fear_greed_latest_endpoint,
        )

    def build_coinmarketcal_event_enricher(
        self,
        should_refresh_events: bool = False,
    ) -> CoinMarketCalEventEnricher:
        """Create the configured CoinMarketCal event enrichment helper."""

        return CoinMarketCalEventEnricher(
            events_path=self.config.coinmarketcal_events_file,
            api_base_url=self.config.coinmarketcal_api_base_url,
            api_key_env_var=self.config.coinmarketcal_api_key_env_var,
            lookahead_days=self.config.coinmarketcal_lookahead_days,
            request_pause_seconds=self.config.coinmarketcal_request_pause_seconds,
            should_refresh_events=should_refresh_events,
            log_progress=self.config.coinmarketcal_log_progress,
        )

    @staticmethod
    def _model_name_suffix(model_type: str) -> str:
        """
        Convert a camelCase model type into a suffix for filenames.

        Example:
        `randomForestSignalModel` becomes `RandomForestSignalModel`
        """

        return model_type[:1].upper() + model_type[1:]

    def _comparison_output_paths(self, model_type: str) -> Dict[str, Path]:
        """
        Build model-specific output paths for the comparison workflow.
        """

        suffix = self._model_name_suffix(model_type)
        model_class = get_model_class(model_type)

        return {
            "modelPath": MODELS_DIR / model_class.default_model_filename,
            "metricsPath": OUTPUTS_DIR / f"trainingMetrics{suffix}.json",
            "predictionsPath": OUTPUTS_DIR / f"testPredictions{suffix}.csv",
            "featureImportancePath": OUTPUTS_DIR / f"featureImportance{suffix}.csv",
        }


class TrainingApp(BaseSignalApp):
    """Train a model, evaluate it, and save the generated artifacts."""

    def _build_dataset(self) -> tuple[pd.DataFrame, List[str]]:
        """
        Build and save the labeled dataset used for training.
        """

        self._ensure_market_data_available()
        dataset, feature_columns = self.dataset_builder.build_labeled_dataset()
        self.save_dataframe(dataset, self.dataset_path)
        return dataset, feature_columns

    def _train_model_for_type(
        self,
        dataset: pd.DataFrame,
        feature_columns: List[str],
        model_type: str = None,
    ) -> Dict[str, Any]:
        """
        Train and evaluate one model type on the provided dataset.

        This helper is shared by:
        - the normal single-model training flow
        - the multi-model comparison flow

        Keeping this in one place avoids repeating the fit/evaluate logic.
        """

        train_df, test_df = BaseSignalModel.split_train_test_by_time(dataset, self.config.train_size)
        return self._train_model_on_split(
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            model_type=model_type,
        )

    def _train_model_on_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: List[str],
        training_config: TrainingConfig = None,
        model_type: str = None,
    ) -> Dict[str, Any]:
        """
        Train and evaluate one model on an explicit train/test split.

        This is the core reusable training unit for:
        - one normal time split
        - multi-model comparison
        - walk-forward validation folds
        """

        training_config = training_config or self.config
        if model_type is not None:
            training_config = replace(training_config, model_type=model_type)
            training_config = replace(
                training_config,
                signal_model_family="baseline_current",
                signal_model_variant="default",
            )

        model = create_model_from_config(
            config=training_config,
            feature_columns=feature_columns,
        )

        model.fit(train_df)
        prediction_df, metrics = model.evaluate(train_df, test_df)
        metrics["config"] = config_to_dict(training_config)

        return {
            "config": training_config,
            "model": model,
            "train_df": train_df,
            "test_df": test_df,
            "prediction_df": prediction_df,
            "metrics": metrics,
        }

    def _save_primary_training_outputs(
        self,
        training_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Save the normal single-model training artifacts.

        This helper keeps training and backtesting DRY because both workflows
        need the same model, metrics, predictions, and feature-importance files.
        """

        self.model = training_result["model"]
        prediction_df = training_result["prediction_df"]
        metrics = training_result["metrics"]
        train_df = training_result["train_df"]
        test_df = training_result["test_df"]
        output_paths = self.save_model_artifact_outputs(
            model=self.model,
            metrics=metrics,
            prediction_df=prediction_df,
            train_df=train_df,
            test_df=test_df,
            dataset_path=self.dataset_path,
            model_path=self.model_path,
            metrics_path=OUTPUTS_DIR / "trainingMetrics.json",
            predictions_path=OUTPUTS_DIR / "testPredictions.csv",
            feature_importance_path=OUTPUTS_DIR / "featureImportance.csv",
        )

        return {
            "modelType": self.model.model_type,
            "datasetPath": str(self.dataset_path),
            **output_paths,
            "trainRows": len(train_df),
            "testRows": len(test_df),
            "accuracy": metrics["accuracy"],
            "balancedAccuracy": metrics["balanced_accuracy"],
        }

    def run(self) -> Dict[str, Any]:
        """
        Execute the full training workflow and return a compact summary.
        """

        dataset, feature_columns = self._build_dataset()
        training_result = self._train_model_for_type(
            dataset=dataset,
            feature_columns=feature_columns,
            model_type=self.config.model_type,
        )
        return self._save_primary_training_outputs(training_result)


class WalkForwardValidationApp(TrainingApp):
    """Validate one configured model across multiple expanding time folds."""

    def __init__(
        self,
        config: TrainingConfig = None,
        dataset_builder: CryptoDatasetBuilder = None,
        model: BaseSignalModel = None,
        model_class: Type[BaseSignalModel] = None,
        backtester: BaseSignalBacktester = None,
    ) -> None:
        super().__init__(
            config=config,
            dataset_builder=dataset_builder,
            model=model,
            model_class=model_class,
        )
        self.backtester = backtester or EqualWeightSignalBacktester(self.config)

    def run(self) -> Dict[str, Any]:
        """
        Evaluate the configured model over multiple out-of-sample folds.

        This gives a stricter view of model quality because every prediction in
        the exported walk-forward files comes from a model that only saw older
        data during training.
        """

        dataset, feature_columns = self._build_dataset()
        walk_forward_result = self._run_walk_forward_validation(
            dataset=dataset,
            feature_columns=feature_columns,
            validation_config=self.config,
            backtester=self.backtester,
        )
        return self._save_walk_forward_outputs(walk_forward_result)

    def _run_walk_forward_validation(
        self,
        dataset: pd.DataFrame,
        feature_columns: List[str],
        validation_config: TrainingConfig,
        backtester: BaseSignalBacktester = None,
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation on a provided dataset and config.

        This is the reusable engine for both:
        - the normal walk-forward script
        - parameter tuning over multiple label settings
        """

        fold_splits = BaseSignalModel.split_walk_forward_by_time(
            dataset=dataset,
            min_train_size=validation_config.walkforward_min_train_size,
            test_size=validation_config.walkforward_test_size,
            step_size=validation_config.walkforward_step_size,
            purge_gap_timestamps=self._resolve_walkforward_purge_gap(validation_config),
        )

        fold_metric_rows = []
        prediction_frames = []
        feature_importance_frames = []

        for fold_split in fold_splits:
            training_result = self._train_model_on_split(
                train_df=fold_split["train_df"],
                test_df=fold_split["test_df"],
                feature_columns=feature_columns,
                training_config=validation_config,
                model_type=validation_config.model_type,
            )
            model = training_result["model"]
            metrics = training_result["metrics"]
            prediction_df = training_result["prediction_df"].copy()
            prediction_df["fold_number"] = fold_split["fold_number"]
            prediction_frames.append(prediction_df)

            feature_importance_df = model.get_feature_importance_frame()
            feature_importance_df["fold_number"] = fold_split["fold_number"]
            feature_importance_frames.append(feature_importance_df)

            fold_metric_rows.append(
                {
                    "foldNumber": fold_split["fold_number"],
                    "trainRows": int(len(fold_split["train_df"])),
                    "testRows": int(len(fold_split["test_df"])),
                    "trainTimestampCount": fold_split["train_timestamp_count"],
                    "testTimestampCount": fold_split["test_timestamp_count"],
                    "trainStartTimestamp": str(fold_split["train_start_timestamp"]),
                    "trainEndTimestamp": str(fold_split["train_end_timestamp"]),
                    "testStartTimestamp": str(fold_split["test_start_timestamp"]),
                    "testEndTimestamp": str(fold_split["test_end_timestamp"]),
                    "purgeGapTimestamps": int(fold_split["purgeGapTimestamps"]),
                    "purgedTimestampCount": int(fold_split["purgedTimestampCount"]),
                    "accuracy": metrics["accuracy"],
                    "balancedAccuracy": metrics["balanced_accuracy"],
                }
            )
            self._save_walk_forward_progress(
                fold_metric_rows=fold_metric_rows,
                prediction_frames=prediction_frames,
                feature_importance_frames=feature_importance_frames,
            )

        walk_forward_predictions_df = pd.concat(prediction_frames, ignore_index=True)
        walk_forward_predictions_df = walk_forward_predictions_df.sort_values(
            by=["timestamp", "product_id"] if "product_id" in walk_forward_predictions_df.columns else ["timestamp"]
        ).reset_index(drop=True)
        fold_metrics_df = pd.DataFrame(fold_metric_rows)
        average_feature_importance_df = self._average_feature_importance(feature_importance_frames)

        aggregate_metrics = BaseSignalModel.build_classification_metrics(
            actual_signals=walk_forward_predictions_df["target_signal"],
            predicted_signals=walk_forward_predictions_df["predicted_signal"],
        )
        aggregate_metrics.update(
            {
                "model_type": validation_config.model_type,
                "fold_count": int(len(fold_metrics_df)),
                "out_of_sample_rows": int(len(walk_forward_predictions_df)),
                "average_fold_accuracy": float(fold_metrics_df["accuracy"].mean()),
                "average_fold_balanced_accuracy": float(fold_metrics_df["balancedAccuracy"].mean()),
                "best_fold_balanced_accuracy": float(fold_metrics_df["balancedAccuracy"].max()),
                "worst_fold_balanced_accuracy": float(fold_metrics_df["balancedAccuracy"].min()),
                "test_class_distribution": BaseSignalModel.build_signal_distribution(
                    walk_forward_predictions_df["target_signal"]
                ),
                "walkforward_settings": {
                    "minTrainSize": float(validation_config.walkforward_min_train_size),
                    "testSize": float(validation_config.walkforward_test_size),
                    "stepSize": float(validation_config.walkforward_step_size),
                    "purgeGapTimestamps": int(self._resolve_walkforward_purge_gap(validation_config)),
                },
                "config": config_to_dict(validation_config),
            }
        )

        local_backtester = backtester or EqualWeightSignalBacktester(validation_config)
        backtest_result = local_backtester.run(walk_forward_predictions_df)
        walk_forward_summary = {
            **aggregate_metrics,
            "backtest_summary": backtest_result["summary"],
        }

        return {
            "config": validation_config,
            "fold_metrics_df": fold_metrics_df,
            "walk_forward_predictions_df": walk_forward_predictions_df,
            "average_feature_importance_df": average_feature_importance_df,
            "summary": walk_forward_summary,
            "backtest_result": backtest_result,
        }

    def _save_walk_forward_outputs(
        self,
        walk_forward_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save the standard walk-forward outputs to the outputs folder."""

        fold_metrics_df = walk_forward_result["fold_metrics_df"]
        walk_forward_predictions_df = walk_forward_result["walk_forward_predictions_df"]
        average_feature_importance_df = walk_forward_result["average_feature_importance_df"]
        walk_forward_summary = walk_forward_result["summary"]
        backtest_result = walk_forward_result["backtest_result"]

        fold_metrics_path = OUTPUTS_DIR / "walkForwardFoldMetrics.csv"
        predictions_path = OUTPUTS_DIR / "walkForwardPredictions.csv"
        feature_importance_path = OUTPUTS_DIR / "walkForwardFeatureImportance.csv"
        summary_path = OUTPUTS_DIR / "walkForwardSummary.json"
        backtest_trades_path = OUTPUTS_DIR / "walkForwardBacktestTrades.csv"
        backtest_periods_path = OUTPUTS_DIR / "walkForwardBacktestPeriods.csv"
        backtest_summary_path = OUTPUTS_DIR / "walkForwardBacktestSummary.json"

        self.save_dataframe(fold_metrics_df, fold_metrics_path)
        self.save_dataframe(walk_forward_predictions_df, predictions_path)
        self.save_dataframe(average_feature_importance_df, feature_importance_path)
        self.save_json(walk_forward_summary, summary_path)
        self.save_dataframe(backtest_result["trade_df"], backtest_trades_path)
        self.save_dataframe(backtest_result["period_df"], backtest_periods_path)
        self.save_json(backtest_result["summary"], backtest_summary_path)

        return {
            "modelType": self.config.model_type,
            "datasetPath": str(self.dataset_path),
            "walkForwardFoldMetricsPath": str(fold_metrics_path),
            "walkForwardPredictionsPath": str(predictions_path),
            "walkForwardFeatureImportancePath": str(feature_importance_path),
            "walkForwardSummaryPath": str(summary_path),
            "walkForwardBacktestTradesPath": str(backtest_trades_path),
            "walkForwardBacktestPeriodsPath": str(backtest_periods_path),
            "walkForwardBacktestSummaryPath": str(backtest_summary_path),
            "foldCount": int(len(fold_metrics_df)),
            "outOfSampleRows": int(len(walk_forward_predictions_df)),
            "accuracy": walk_forward_summary["accuracy"],
            "balancedAccuracy": walk_forward_summary["balanced_accuracy"],
            "averageFoldBalancedAccuracy": walk_forward_summary["average_fold_balanced_accuracy"],
            "tradeCount": backtest_result["summary"]["tradeCount"],
            "strategyTotalReturn": backtest_result["summary"]["strategyTotalReturn"],
            "benchmarkTotalReturn": backtest_result["summary"]["benchmarkTotalReturn"],
            "maxDrawdown": backtest_result["summary"]["maxDrawdown"],
        }

    def _average_feature_importance(
        self,
        feature_importance_frames: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """Average feature importance across all walk-forward folds."""

        valid_feature_importance_frames = [
            feature_importance_df
            for feature_importance_df in feature_importance_frames
            if not feature_importance_df.empty
            and {"feature", "importance"}.issubset(feature_importance_df.columns)
        ]

        if not valid_feature_importance_frames:
            return pd.DataFrame(columns=["feature", "importance"])

        combined_feature_importance_df = pd.concat(valid_feature_importance_frames, ignore_index=True)
        return (
            combined_feature_importance_df.groupby("feature", as_index=False)
            .agg(importance=("importance", "mean"))
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _save_walk_forward_progress(
        self,
        fold_metric_rows: List[Dict[str, Any]],
        prediction_frames: List[pd.DataFrame],
        feature_importance_frames: List[pd.DataFrame],
    ) -> None:
        """Persist partial walk-forward results while the folds are still running."""

        if not fold_metric_rows:
            return

        self.save_dataframe(
            pd.DataFrame(fold_metric_rows),
            OUTPUTS_DIR / "walkForwardFoldMetrics.partial.csv",
        )

        if prediction_frames:
            partial_predictions_df = pd.concat(prediction_frames, ignore_index=True)
            partial_predictions_df = partial_predictions_df.sort_values(
                by=["timestamp", "product_id"] if "product_id" in partial_predictions_df.columns else ["timestamp"]
            ).reset_index(drop=True)
            self.save_dataframe(
                partial_predictions_df,
                OUTPUTS_DIR / "walkForwardPredictions.partial.csv",
            )

        self.save_dataframe(
            self._average_feature_importance(feature_importance_frames),
            OUTPUTS_DIR / "walkForwardFeatureImportance.partial.csv",
        )
        self.save_json(
            {
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "completedFolds": int(len(fold_metric_rows)),
            },
            OUTPUTS_DIR / "walkForwardProgress.json",
        )


class RegimeTrainingApp(BaseSignalApp):
    """Train a dedicated model that forecasts the next market-regime label."""

    def __init__(
        self,
        config: TrainingConfig = None,
        dataset_builder: CryptoDatasetBuilder = None,
        model: MarketRegimeModel = None,
    ) -> None:
        super().__init__(
            config=config,
            dataset_builder=dataset_builder,
            model=model,
        )
        self.regime_labeler = create_regime_labeler_from_config(self.config)

    @property
    def model_path(self) -> Path:
        """Return the saved artifact path for the standalone regime model."""

        return MODELS_DIR / MarketRegimeModel.default_model_filename

    @property
    def dataset_path(self) -> Path:
        """Return the prepared regime dataset path."""

        return PROCESSED_DATA_DIR / "marketFeaturesAndRegimes.csv"

    def _build_dataset(self) -> tuple[pd.DataFrame, List[str]]:
        """Build and save the explicit regime-label dataset."""

        self._ensure_market_data_available()
        feature_df = self.dataset_builder.build_feature_table()
        feature_columns = list(self.dataset_builder.feature_columns)
        labeled_df = self.regime_labeler.add_labels(feature_df)
        cleaned_df = labeled_df.dropna(
            subset=feature_columns + [MarketRegimeModel.target_column]
        ).reset_index(drop=True)
        self.save_dataframe(cleaned_df, self.dataset_path)
        return cleaned_df, feature_columns

    def _train_model_on_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: List[str],
        training_config: TrainingConfig = None,
    ) -> Dict[str, Any]:
        """Train and evaluate the dedicated regime model on one time split."""

        training_config = training_config or self.config
        model = MarketRegimeModel(
            config=training_config,
            feature_columns=feature_columns,
        )
        model.fit(train_df)
        prediction_df, metrics = model.evaluate(train_df, test_df)
        metrics["config"] = config_to_dict(training_config)

        return {
            "config": training_config,
            "model": model,
            "train_df": train_df,
            "test_df": test_df,
            "prediction_df": prediction_df,
            "metrics": metrics,
        }

    def _save_training_outputs(
        self,
        training_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save the standalone regime-model artifacts and enrich the metadata sidecar."""

        self.model = training_result["model"]
        prediction_df = training_result["prediction_df"]
        metrics = training_result["metrics"]
        train_df = training_result["train_df"]
        test_df = training_result["test_df"]
        output_paths = self.save_model_artifact_outputs(
            model=self.model,
            metrics=metrics,
            prediction_df=prediction_df,
            train_df=train_df,
            test_df=test_df,
            dataset_path=self.dataset_path,
            model_path=self.model_path,
            metrics_path=OUTPUTS_DIR / "regimeTrainingMetrics.json",
            predictions_path=OUTPUTS_DIR / "regimeTestPredictions.csv",
            feature_importance_path=OUTPUTS_DIR / "regimeFeatureImportance.csv",
        )

        metadata_path = Path(output_paths["metadataPath"])
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_payload.update(
            {
                "artifactType": "market_regime_model",
                "estimatorType": self.model.estimator_type,
                "target": {
                    "targetColumn": MarketRegimeModel.target_column,
                    "targetLabelColumn": MarketRegimeModel.target_label_column,
                    "predictionHorizon": int(self.regime_labeler.prediction_horizon),
                },
                "regimeClassDistribution": {
                    "train": metrics.get("train_class_distribution", {}),
                    "test": metrics.get("test_class_distribution", {}),
                },
            }
        )
        self.save_json(metadata_payload, metadata_path)

        return {
            "modelType": self.model.model_type,
            "estimatorType": self.model.estimator_type,
            "datasetPath": str(self.dataset_path),
            **output_paths,
            "trainRows": int(len(train_df)),
            "testRows": int(len(test_df)),
            "accuracy": float(metrics["accuracy"]),
            "balancedAccuracy": float(metrics["balanced_accuracy"]),
        }

    def run(self) -> Dict[str, Any]:
        """Train the regime model on the configured time split and save outputs."""

        dataset, feature_columns = self._build_dataset()
        train_df, test_df = MarketRegimeModel.split_train_test_by_time(
            dataset=dataset,
            train_size=self.config.train_size,
        )
        training_result = self._train_model_on_split(
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            training_config=self.config,
        )
        return self._save_training_outputs(training_result)


class RegimeWalkForwardValidationApp(RegimeTrainingApp):
    """Validate the standalone regime model across expanding time folds."""

    def run(self) -> Dict[str, Any]:
        """Execute walk-forward validation for the dedicated regime model."""

        dataset, feature_columns = self._build_dataset()
        walk_forward_result = self._run_walk_forward_validation(
            dataset=dataset,
            feature_columns=feature_columns,
            validation_config=self.config,
        )
        return self._save_walk_forward_outputs(walk_forward_result)

    def _run_walk_forward_validation(
        self,
        dataset: pd.DataFrame,
        feature_columns: List[str],
        validation_config: TrainingConfig,
    ) -> Dict[str, Any]:
        """Run walk-forward validation for the standalone regime classifier."""

        fold_splits = MarketRegimeModel.split_walk_forward_by_time(
            dataset=dataset,
            min_train_size=validation_config.walkforward_min_train_size,
            test_size=validation_config.walkforward_test_size,
            step_size=validation_config.walkforward_step_size,
            purge_gap_timestamps=self._resolve_walkforward_purge_gap(validation_config),
        )

        fold_metric_rows = []
        prediction_frames = []
        feature_importance_frames = []

        for fold_split in fold_splits:
            training_result = self._train_model_on_split(
                train_df=fold_split["train_df"],
                test_df=fold_split["test_df"],
                feature_columns=feature_columns,
                training_config=validation_config,
            )
            model = training_result["model"]
            metrics = training_result["metrics"]
            prediction_df = training_result["prediction_df"].copy()
            prediction_df["fold_number"] = fold_split["fold_number"]
            prediction_frames.append(prediction_df)

            feature_importance_df = model.get_feature_importance_frame()
            feature_importance_df["fold_number"] = fold_split["fold_number"]
            feature_importance_frames.append(feature_importance_df)

            fold_metric_rows.append(
                {
                    "foldNumber": fold_split["fold_number"],
                    "trainRows": int(len(fold_split["train_df"])),
                    "testRows": int(len(fold_split["test_df"])),
                    "trainTimestampCount": fold_split["train_timestamp_count"],
                    "testTimestampCount": fold_split["test_timestamp_count"],
                    "trainStartTimestamp": str(fold_split["train_start_timestamp"]),
                    "trainEndTimestamp": str(fold_split["train_end_timestamp"]),
                    "testStartTimestamp": str(fold_split["test_start_timestamp"]),
                    "testEndTimestamp": str(fold_split["test_end_timestamp"]),
                    "purgeGapTimestamps": int(fold_split["purgeGapTimestamps"]),
                    "purgedTimestampCount": int(fold_split["purgedTimestampCount"]),
                    "accuracy": float(metrics["accuracy"]),
                    "balancedAccuracy": float(metrics["balanced_accuracy"]),
                }
            )
            self._save_walk_forward_progress(
                fold_metric_rows=fold_metric_rows,
                prediction_frames=prediction_frames,
                feature_importance_frames=feature_importance_frames,
            )

        walk_forward_predictions_df = pd.concat(prediction_frames, ignore_index=True)
        walk_forward_predictions_df = walk_forward_predictions_df.sort_values(
            by=["timestamp", "product_id"] if "product_id" in walk_forward_predictions_df.columns else ["timestamp"]
        ).reset_index(drop=True)
        fold_metrics_df = pd.DataFrame(fold_metric_rows)
        average_feature_importance_df = self._average_feature_importance(feature_importance_frames)

        aggregate_metrics = MarketRegimeModel.build_classification_metrics(
            actual_labels=walk_forward_predictions_df[MarketRegimeModel.target_column],
            predicted_labels=walk_forward_predictions_df["predicted_market_regime_code"],
        )
        aggregate_metrics.update(
            {
                "model_type": MarketRegimeModel.model_type,
                "estimator_type": validation_config.model_type,
                "fold_count": int(len(fold_metrics_df)),
                "out_of_sample_rows": int(len(walk_forward_predictions_df)),
                "average_fold_accuracy": float(fold_metrics_df["accuracy"].mean()),
                "average_fold_balanced_accuracy": float(fold_metrics_df["balancedAccuracy"].mean()),
                "best_fold_balanced_accuracy": float(fold_metrics_df["balancedAccuracy"].max()),
                "worst_fold_balanced_accuracy": float(fold_metrics_df["balancedAccuracy"].min()),
                "test_class_distribution": MarketRegimeModel.build_class_distribution(
                    walk_forward_predictions_df[MarketRegimeModel.target_column]
                ),
                "walkforward_settings": {
                    "minTrainSize": float(validation_config.walkforward_min_train_size),
                    "testSize": float(validation_config.walkforward_test_size),
                    "stepSize": float(validation_config.walkforward_step_size),
                    "purgeGapTimestamps": int(self._resolve_walkforward_purge_gap(validation_config)),
                },
                "config": config_to_dict(validation_config),
            }
        )

        return {
            "config": validation_config,
            "fold_metrics_df": fold_metrics_df,
            "walk_forward_predictions_df": walk_forward_predictions_df,
            "average_feature_importance_df": average_feature_importance_df,
            "summary": aggregate_metrics,
        }

    def _save_walk_forward_outputs(
        self,
        walk_forward_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save the standalone regime walk-forward outputs."""

        fold_metrics_df = walk_forward_result["fold_metrics_df"]
        walk_forward_predictions_df = walk_forward_result["walk_forward_predictions_df"]
        average_feature_importance_df = walk_forward_result["average_feature_importance_df"]
        walk_forward_summary = walk_forward_result["summary"]

        fold_metrics_path = OUTPUTS_DIR / "regimeWalkForwardFoldMetrics.csv"
        predictions_path = OUTPUTS_DIR / "regimeWalkForwardPredictions.csv"
        feature_importance_path = OUTPUTS_DIR / "regimeWalkForwardFeatureImportance.csv"
        summary_path = OUTPUTS_DIR / "regimeWalkForwardSummary.json"

        self.save_dataframe(fold_metrics_df, fold_metrics_path)
        self.save_dataframe(walk_forward_predictions_df, predictions_path)
        self.save_dataframe(average_feature_importance_df, feature_importance_path)
        self.save_json(walk_forward_summary, summary_path)

        return {
            "modelType": MarketRegimeModel.model_type,
            "estimatorType": self.config.model_type,
            "datasetPath": str(self.dataset_path),
            "walkForwardFoldMetricsPath": str(fold_metrics_path),
            "walkForwardPredictionsPath": str(predictions_path),
            "walkForwardFeatureImportancePath": str(feature_importance_path),
            "walkForwardSummaryPath": str(summary_path),
            "foldCount": int(len(fold_metrics_df)),
            "outOfSampleRows": int(len(walk_forward_predictions_df)),
            "accuracy": float(walk_forward_summary["accuracy"]),
            "balancedAccuracy": float(walk_forward_summary["balanced_accuracy"]),
            "averageFoldBalancedAccuracy": float(walk_forward_summary["average_fold_balanced_accuracy"]),
        }

    def _average_feature_importance(
        self,
        feature_importance_frames: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """Average feature importance across regime-validation folds."""

        valid_feature_importance_frames = [
            feature_importance_df
            for feature_importance_df in feature_importance_frames
            if not feature_importance_df.empty
            and {"feature", "importance"}.issubset(feature_importance_df.columns)
        ]

        if not valid_feature_importance_frames:
            return pd.DataFrame(columns=["feature", "importance"])

        combined_feature_importance_df = pd.concat(valid_feature_importance_frames, ignore_index=True)
        return (
            combined_feature_importance_df.groupby("feature", as_index=False)
            .agg(importance=("importance", "mean"))
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _save_walk_forward_progress(
        self,
        fold_metric_rows: List[Dict[str, Any]],
        prediction_frames: List[pd.DataFrame],
        feature_importance_frames: List[pd.DataFrame],
    ) -> None:
        """Persist partial regime walk-forward outputs while folds are still running."""

        if not fold_metric_rows:
            return

        self.save_dataframe(
            pd.DataFrame(fold_metric_rows),
            OUTPUTS_DIR / "regimeWalkForwardFoldMetrics.partial.csv",
        )

        if prediction_frames:
            partial_predictions_df = pd.concat(prediction_frames, ignore_index=True)
            partial_predictions_df = partial_predictions_df.sort_values(
                by=["timestamp", "product_id"] if "product_id" in partial_predictions_df.columns else ["timestamp"]
            ).reset_index(drop=True)
            self.save_dataframe(
                partial_predictions_df,
                OUTPUTS_DIR / "regimeWalkForwardPredictions.partial.csv",
            )

        self.save_dataframe(
            self._average_feature_importance(feature_importance_frames),
            OUTPUTS_DIR / "regimeWalkForwardFeatureImportance.partial.csv",
        )
        self.save_json(
            {
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "completedFolds": int(len(fold_metric_rows)),
            },
            OUTPUTS_DIR / "regimeWalkForwardProgress.json",
        )


class SignalParameterTuningApp(WalkForwardValidationApp):
    """Tune label thresholds and backtest confidence with walk-forward scoring."""

    def run(self) -> Dict[str, Any]:
        """
        Search a small parameter grid and recommend the strongest config.

        The tuning flow is intentionally split into two stages:
        1. walk-forward search over label settings that require retraining
        2. confidence sweep on the winning walk-forward predictions

        That keeps the expensive retraining work limited to the label settings,
        because backtest confidence can be tuned cheaply from the already
        generated out-of-sample predictions.
        """

        self._ensure_market_data_available()
        feature_df = self.dataset_builder.build_feature_table()
        feature_columns = list(self.dataset_builder.feature_columns)

        label_result_rows = []
        best_label_result: Dict[str, Any] | None = None

        for prediction_horizon in self.config.tuning_prediction_horizon_candidates:
            for buy_threshold in self.config.tuning_buy_threshold_candidates:
                for sell_threshold in self.config.tuning_sell_threshold_candidates:
                    candidate_config = replace(
                        self.config,
                        prediction_horizon=prediction_horizon,
                        buy_threshold=buy_threshold,
                        sell_threshold=sell_threshold,
                    )
                    tuned_dataset = self._build_tuned_dataset(
                        feature_df=feature_df,
                        feature_columns=feature_columns,
                        tuning_config=candidate_config,
                    )
                    walk_forward_result = self._run_walk_forward_validation(
                        dataset=tuned_dataset,
                        feature_columns=feature_columns,
                        validation_config=candidate_config,
                    )

                    summary = walk_forward_result["summary"]
                    backtest_summary = walk_forward_result["backtest_result"]["summary"]
                    label_result_row = {
                        "labelingStrategy": candidate_config.labeling_strategy,
                        "predictionHorizon": prediction_horizon,
                        "buyThreshold": buy_threshold,
                        "sellThreshold": sell_threshold,
                        "purgeGapTimestamps": int(self._resolve_walkforward_purge_gap(candidate_config)),
                        "datasetRows": int(len(tuned_dataset)),
                        "takeProfitRows": int(summary["test_class_distribution"]["TAKE_PROFIT"]),
                        "holdRows": int(summary["test_class_distribution"]["HOLD"]),
                        "buyRows": int(summary["test_class_distribution"]["BUY"]),
                        "foldCount": int(summary["fold_count"]),
                        "accuracy": float(summary["accuracy"]),
                        "balancedAccuracy": float(summary["balanced_accuracy"]),
                        "averageFoldBalancedAccuracy": float(summary["average_fold_balanced_accuracy"]),
                        "strategyTotalReturn": float(backtest_summary["strategyTotalReturn"]),
                        "benchmarkTotalReturn": float(backtest_summary["benchmarkTotalReturn"]),
                        "maxDrawdown": float(backtest_summary["maxDrawdown"]),
                        "tradeCount": int(backtest_summary["tradeCount"]),
                    }
                    label_result_rows.append(label_result_row)

                    if self._is_better_label_candidate(label_result_row, best_label_result):
                        best_label_result = {
                            "config": candidate_config,
                            "row": label_result_row,
                            "walk_forward_result": walk_forward_result,
                        }
                    self._save_tuning_progress(
                        label_result_rows=label_result_rows,
                        confidence_result_rows=[],
                        best_label_result=best_label_result,
                        best_confidence_result=None,
                    )

        if best_label_result is None:
            raise ValueError("Signal parameter tuning could not evaluate any label candidates.")

        label_results_df = pd.DataFrame(label_result_rows).sort_values(
            by=["averageFoldBalancedAccuracy", "balancedAccuracy", "accuracy"],
            ascending=False,
        ).reset_index(drop=True)

        confidence_result_rows = []
        best_confidence_result: Dict[str, Any] | None = None
        winning_predictions_df = best_label_result["walk_forward_result"]["walk_forward_predictions_df"]

        for min_confidence in self.config.tuning_backtest_confidence_candidates:
            confidence_config = replace(
                best_label_result["config"],
                backtest_min_confidence=min_confidence,
            )
            backtest_result = EqualWeightSignalBacktester(confidence_config).run(winning_predictions_df)
            backtest_summary = backtest_result["summary"]
            confidence_result_row = {
                "backtestMinConfidence": float(min_confidence),
                "tradeCount": int(backtest_summary["tradeCount"]),
                "activePeriods": int(backtest_summary["activePeriods"]),
                "winRate": float(backtest_summary["winRate"]),
                "averageTradeReturn": float(backtest_summary["averageTradeReturn"]),
                "strategyTotalReturn": float(backtest_summary["strategyTotalReturn"]),
                "benchmarkTotalReturn": float(backtest_summary["benchmarkTotalReturn"]),
                "maxDrawdown": float(backtest_summary["maxDrawdown"]),
            }
            confidence_result_rows.append(confidence_result_row)

            if self._is_better_confidence_candidate(confidence_result_row, best_confidence_result):
                best_confidence_result = {
                    "config": confidence_config,
                    "row": confidence_result_row,
                    "backtest_result": backtest_result,
                }
            self._save_tuning_progress(
                label_result_rows=label_result_rows,
                confidence_result_rows=confidence_result_rows,
                best_label_result=best_label_result,
                best_confidence_result=best_confidence_result,
            )

        if best_confidence_result is None:
            raise ValueError("Signal parameter tuning could not evaluate any confidence candidates.")

        confidence_results_df = pd.DataFrame(confidence_result_rows).sort_values(
            by=["strategyTotalReturn", "winRate", "averageTradeReturn"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        tuning_summary = {
            "modelType": self.config.model_type,
            "bestLabelConfig": {
                "labelingStrategy": best_label_result["config"].labeling_strategy,
                "predictionHorizon": int(best_label_result["config"].prediction_horizon),
                "buyThreshold": float(best_label_result["config"].buy_threshold),
                "sellThreshold": float(best_label_result["config"].sell_threshold),
                "purgeGapTimestamps": int(self._resolve_walkforward_purge_gap(best_label_result["config"])),
            },
            "bestConfidenceConfig": {
                "backtestMinConfidence": float(best_confidence_result["config"].backtest_min_confidence),
            },
            "recommendedConfig": config_to_dict(best_confidence_result["config"]),
            "bestLabelMetrics": best_label_result["walk_forward_result"]["summary"],
            "bestConfidenceBacktest": best_confidence_result["backtest_result"]["summary"],
        }

        label_results_path = OUTPUTS_DIR / "signalParameterTuningResults.csv"
        confidence_results_path = OUTPUTS_DIR / "signalConfidenceTuningResults.csv"
        summary_path = OUTPUTS_DIR / "signalParameterTuningSummary.json"

        self.save_dataframe(label_results_df, label_results_path)
        self.save_dataframe(confidence_results_df, confidence_results_path)
        self.save_json(tuning_summary, summary_path)

        return {
            "modelType": self.config.model_type,
            "labelResultsPath": str(label_results_path),
            "confidenceResultsPath": str(confidence_results_path),
            "summaryPath": str(summary_path),
            "bestPredictionHorizon": int(best_label_result["config"].prediction_horizon),
            "bestBuyThreshold": float(best_label_result["config"].buy_threshold),
            "bestSellThreshold": float(best_label_result["config"].sell_threshold),
            "bestBacktestMinConfidence": float(best_confidence_result["config"].backtest_min_confidence),
            "balancedAccuracy": float(best_label_result["walk_forward_result"]["summary"]["balanced_accuracy"]),
            "averageFoldBalancedAccuracy": float(
                best_label_result["walk_forward_result"]["summary"]["average_fold_balanced_accuracy"]
            ),
            "tradeCount": int(best_confidence_result["backtest_result"]["summary"]["tradeCount"]),
            "strategyTotalReturn": float(best_confidence_result["backtest_result"]["summary"]["strategyTotalReturn"]),
            "maxDrawdown": float(best_confidence_result["backtest_result"]["summary"]["maxDrawdown"]),
        }

    def _build_tuned_dataset(
        self,
        feature_df: pd.DataFrame,
        feature_columns: List[str],
        tuning_config: TrainingConfig,
    ) -> pd.DataFrame:
        """Apply candidate label settings to one shared feature table."""

        labeler = create_labeler_from_config(tuning_config)
        labeled_df = labeler.add_labels(feature_df.copy())
        return labeled_df.dropna(subset=feature_columns + ["future_return"]).reset_index(drop=True)

    def _save_tuning_progress(
        self,
        label_result_rows: List[Dict[str, Any]],
        confidence_result_rows: List[Dict[str, Any]],
        best_label_result: Dict[str, Any] | None,
        best_confidence_result: Dict[str, Any] | None,
    ) -> None:
        """Persist partial tuning outputs so long searches are resumable to inspect."""

        if label_result_rows:
            self.save_dataframe(
                pd.DataFrame(label_result_rows),
                OUTPUTS_DIR / "signalParameterTuningResults.partial.csv",
            )

        if confidence_result_rows:
            self.save_dataframe(
                pd.DataFrame(confidence_result_rows),
                OUTPUTS_DIR / "signalConfidenceTuningResults.partial.csv",
            )

        self.save_json(
            {
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "labelCandidatesEvaluated": int(len(label_result_rows)),
                "confidenceCandidatesEvaluated": int(len(confidence_result_rows)),
                "bestLabelRow": best_label_result["row"] if best_label_result is not None else None,
                "bestConfidenceRow": best_confidence_result["row"] if best_confidence_result is not None else None,
            },
            OUTPUTS_DIR / "signalParameterTuningProgress.json",
        )

    def _is_better_label_candidate(
        self,
        candidate_row: Dict[str, Any],
        best_result: Dict[str, Any] | None,
    ) -> bool:
        """Compare two label candidates using walk-forward metrics."""

        if best_result is None:
            return True

        best_row = best_result["row"]
        return (
            candidate_row["averageFoldBalancedAccuracy"],
            candidate_row["balancedAccuracy"],
            candidate_row["accuracy"],
        ) > (
            best_row["averageFoldBalancedAccuracy"],
            best_row["balancedAccuracy"],
            best_row["accuracy"],
        )

    def _is_better_confidence_candidate(
        self,
        candidate_row: Dict[str, Any],
        best_result: Dict[str, Any] | None,
    ) -> bool:
        """Compare two confidence candidates using backtest outcomes."""

        if best_result is None:
            return True

        best_row = best_result["row"]
        return (
            candidate_row["strategyTotalReturn"],
            candidate_row["winRate"],
            candidate_row["averageTradeReturn"],
        ) > (
            best_row["strategyTotalReturn"],
            best_row["winRate"],
            best_row["averageTradeReturn"],
        )


class ModelComparisonApp(TrainingApp):
    """Train multiple registered model subclasses on the same dataset split."""

    @staticmethod
    def _find_regime_column(prediction_df: pd.DataFrame) -> str | None:
        for column in (
            "market_regime_label",
            "market_regime_code",
            "trend_regime_label",
            "volatility_regime_label",
        ):
            if column in prediction_df.columns:
                return column
        return None

    def _build_buy_precision_metrics(self, prediction_df: pd.DataFrame) -> Dict[str, Any]:
        if "target_signal" not in prediction_df.columns or "predicted_name" not in prediction_df.columns:
            return {"available": False}

        predicted_buy = prediction_df["predicted_name"] == "BUY"
        actual_buy = prediction_df["target_signal"] == 1
        buy_count = int(predicted_buy.sum())
        true_positive = int((predicted_buy & actual_buy).sum())
        false_positive = int((predicted_buy & ~actual_buy).sum())

        return {
            "available": True,
            "predictedBuyCount": buy_count,
            "buyPrecision": float(true_positive / max(buy_count, 1)),
            "falsePositiveRate": float(false_positive / max(buy_count, 1)),
        }

    def _build_forward_return_metrics(self, prediction_df: pd.DataFrame) -> Dict[str, Any]:
        if "future_return" not in prediction_df.columns or "predicted_name" not in prediction_df.columns:
            return {"available": False}

        predicted_buy = prediction_df["predicted_name"] == "BUY"
        buy_returns = prediction_df.loc[predicted_buy, "future_return"]
        return {
            "available": True,
            "averageForwardReturn": float(buy_returns.mean()) if not buy_returns.empty else 0.0,
            "medianForwardReturn": float(buy_returns.median()) if not buy_returns.empty else 0.0,
        }

    def _build_promotion_quality_metrics(self, prediction_df: pd.DataFrame) -> Dict[str, Any]:
        required_columns = {"confidence", "prob_buy", "target_signal"}
        if not required_columns.issubset(prediction_df.columns):
            return {"available": False}

        promotion_mask = (
            (prediction_df["confidence"] >= float(self.config.signal_watchlist_promotion_min_confidence))
            & (prediction_df["prob_buy"] >= float(self.config.signal_watchlist_promotion_min_decision_score))
        )
        entry_ready_mask = (
            (prediction_df["confidence"] >= float(self.config.signal_watchlist_entry_ready_min_confidence))
            & (prediction_df["prob_buy"] >= float(self.config.signal_watchlist_entry_ready_min_decision_score))
        )
        actual_buy = prediction_df["target_signal"] == 1

        def summarize(mask: pd.Series) -> Dict[str, Any]:
            count = int(mask.sum())
            true_positive = int((mask & actual_buy).sum())
            avg_return = None
            if "future_return" in prediction_df.columns:
                avg_return = float(prediction_df.loc[mask, "future_return"].mean()) if count > 0 else 0.0
            return {
                "count": count,
                "buyPrecision": float(true_positive / max(count, 1)),
                "averageForwardReturn": avg_return,
            }

        return {
            "available": True,
            "promotionCandidates": summarize(promotion_mask),
            "entryReadyCandidates": summarize(entry_ready_mask),
        }

    def _build_confidence_bucket_metrics(self, prediction_df: pd.DataFrame) -> Dict[str, Any]:
        if "confidence" not in prediction_df.columns or "predicted_name" not in prediction_df.columns:
            return {"available": False}

        buckets = [
            (0.0, 0.40),
            (0.40, 0.55),
            (0.55, 0.70),
            (0.70, 0.85),
            (0.85, 1.01),
        ]
        actual_buy = prediction_df["target_signal"] == 1 if "target_signal" in prediction_df.columns else None

        bucket_rows = []
        for lower, upper in buckets:
            bucket_mask = (prediction_df["confidence"] >= lower) & (prediction_df["confidence"] < upper)
            predicted_buy = bucket_mask & (prediction_df["predicted_name"] == "BUY")
            count = int(predicted_buy.sum())
            precision = None
            if actual_buy is not None:
                precision = float((predicted_buy & actual_buy).sum() / max(count, 1))
            avg_return = None
            if "future_return" in prediction_df.columns:
                avg_return = float(prediction_df.loc[predicted_buy, "future_return"].mean()) if count > 0 else 0.0

            bucket_rows.append(
                {
                    "bucket": f"{lower:.2f}-{upper:.2f}",
                    "predictedBuyCount": count,
                    "buyPrecision": precision,
                    "averageForwardReturn": avg_return,
                }
            )

        return {"available": True, "buckets": bucket_rows}

    def _build_regime_metrics(self, prediction_df: pd.DataFrame) -> Dict[str, Any]:
        regime_column = self._find_regime_column(prediction_df)
        if regime_column is None or "predicted_name" not in prediction_df.columns:
            return {"available": False}

        actual_buy = prediction_df["target_signal"] == 1 if "target_signal" in prediction_df.columns else None
        grouped = prediction_df.groupby(regime_column)
        rows = []
        for regime_value, group in grouped:
            predicted_buy = group["predicted_name"] == "BUY"
            count = int(predicted_buy.sum())
            precision = None
            if actual_buy is not None:
                precision = float((predicted_buy & (group["target_signal"] == 1)).sum() / max(count, 1))
            avg_return = None
            if "future_return" in group.columns:
                avg_return = float(group.loc[predicted_buy, "future_return"].mean()) if count > 0 else 0.0
            rows.append(
                {
                    "regime": str(regime_value),
                    "predictedBuyCount": count,
                    "buyPrecision": precision,
                    "averageForwardReturn": avg_return,
                }
            )

        return {"available": True, "regimeColumn": regime_column, "rows": rows[:8]}

    def run(self) -> Dict[str, Any]:
        """
        Compare multiple model types and save side-by-side outputs.

        Every model sees the same dataset and the same time-based split.
        That makes the comparison fairer than training on different slices.
        """

        if not self.config.comparison_model_types:
            raise ValueError("comparison_model_types is empty. Add at least one model type in the config.")

        dataset, feature_columns = self._build_dataset()
        comparison_rows = []
        comparison_details = []

        for model_type in self.config.comparison_model_types:
            training_result = self._train_model_for_type(
                dataset=dataset,
                feature_columns=feature_columns,
                model_type=model_type,
            )

            model = training_result["model"]
            metrics = training_result["metrics"]
            prediction_df = training_result["prediction_df"]
            train_df = training_result["train_df"]
            test_df = training_result["test_df"]
            output_paths = self._comparison_output_paths(model_type)

            saved_artifact_paths = self.save_model_artifact_outputs(
                model=model,
                metrics=metrics,
                prediction_df=prediction_df,
                train_df=train_df,
                test_df=test_df,
                dataset_path=self.dataset_path,
                model_path=output_paths["modelPath"],
                metrics_path=output_paths["metricsPath"],
                predictions_path=output_paths["predictionsPath"],
                feature_importance_path=output_paths["featureImportancePath"],
            )

            backtest_summary = EqualWeightSignalBacktester(self.config).run(prediction_df)["summary"]
            evaluation_metrics = {
                "buyPrecision": self._build_buy_precision_metrics(prediction_df),
                "forwardReturn": self._build_forward_return_metrics(prediction_df),
                "promotionQuality": self._build_promotion_quality_metrics(prediction_df),
                "confidenceBuckets": self._build_confidence_bucket_metrics(prediction_df),
                "regimePerformance": self._build_regime_metrics(prediction_df),
                "backtestSummary": backtest_summary,
            }

            walk_forward_summary = None
            if self.config.comparison_run_walk_forward:
                walk_forward_app = WalkForwardValidationApp(
                    config=training_result["config"],
                    dataset_builder=self.dataset_builder,
                )
                walk_forward_result = walk_forward_app._run_walk_forward_validation(
                    dataset=dataset,
                    feature_columns=feature_columns,
                    validation_config=training_result["config"],
                    backtester=EqualWeightSignalBacktester(training_result["config"]),
                )
                walk_forward_summary = walk_forward_result["summary"]

            comparison_row = {
                "modelType": model.model_type,
                "accuracy": metrics["accuracy"],
                "balancedAccuracy": metrics["balanced_accuracy"],
                "trainRows": len(train_df),
                "testRows": len(test_df),
            }
            comparison_rows.append(comparison_row)

            comparison_details.append(
                {
                    **comparison_row,
                    **saved_artifact_paths,
                    "evaluation": evaluation_metrics,
                    "walkForwardSummary": walk_forward_summary,
                }
            )

        comparison_df = pd.DataFrame(comparison_rows).sort_values(
            by=["balancedAccuracy", "accuracy"],
            ascending=False,
        ).reset_index(drop=True)

        best_model_type = str(comparison_df.iloc[0]["modelType"])
        comparison_json = {
            "datasetPath": str(self.dataset_path),
            "bestModelType": best_model_type,
            "models": comparison_details,
        }

        comparison_csv_path = OUTPUTS_DIR / "modelComparison.csv"
        comparison_json_path = OUTPUTS_DIR / "modelComparison.json"

        self.save_dataframe(comparison_df, comparison_csv_path)
        self.save_json(comparison_json, comparison_json_path)

        return {
            "datasetPath": str(self.dataset_path),
            "comparisonCsvPath": str(comparison_csv_path),
            "comparisonJsonPath": str(comparison_json_path),
            "bestModelType": best_model_type,
            "comparedModels": len(comparison_df),
        }


class SignalGenerationApp(BaseSignalApp):
    """Load a trained model and generate historical plus latest signals."""

    primary_signal_history_path = OUTPUTS_DIR / "primarySignalHistory.json"

    def _build_compat_primary_history_store(self) -> PrimarySignalHistoryStore:
        """Create the history store used by compatibility helper methods."""

        return PrimarySignalHistoryStore(
            config=self.config,
            history_path=self.primary_signal_history_path,
            save_json=self.save_json,
        )

    def _build_compat_decision_stage(self) -> SignalDecisionStage:
        """Build one decision stage for legacy helper wrappers used by tests and callers."""

        decision_stage = SignalDecisionStage(
            self.config,
            primary_history_store=self._build_compat_primary_history_store(),
            allow_watchlist_fallback=True,
            allow_watchlist_supplement=True,
        )
        overridden_should_publish = self.__dict__.get("_should_publish_watchlist_fallback")
        if callable(overridden_should_publish):
            decision_stage._should_publish_watchlist_fallback = overridden_should_publish  # type: ignore[method-assign]
        return decision_stage

    def _select_watchlist_fallback_signal(
        self,
        signal_summaries: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Compatibility wrapper for legacy direct watchlist-fallback selection."""

        return self._build_compat_decision_stage()._select_watchlist_fallback_signal(signal_summaries)

    def _supplement_published_signals_with_watchlist_candidates(
        self,
        *,
        published_signals: list[dict[str, Any]],
        signal_summaries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compatibility wrapper for legacy direct watchlist supplementation."""

        return self._build_compat_decision_stage()._supplement_published_signals_with_watchlist_candidates(
            published_signals=published_signals,
            signal_summaries=signal_summaries,
        )

    def _save_watchlist_pool_snapshot(self, signal_summaries: list[dict[str, Any]]) -> None:
        """Compatibility wrapper for persisting the ranked watchlist pool directly."""

        publication_stage = SignalPublicationStage(
            config=self.config,
            save_json=self.save_json,
            save_dataframe=self.save_dataframe,
            signal_store_factory=self._build_signal_store,
            primary_history_store=self._build_compat_primary_history_store(),
        )
        publication_stage.save_watchlist_pool_snapshot(signal_summaries)

    def _should_score_fresh_signal_universe(self) -> bool:
        """Return whether publication should score a fresh top-market universe."""

        return bool(
            self.config.signal_refresh_market_data_before_generation
            and self._resolve_market_fetch_all_quote_products()
            and self.config.live_fetch_all_quote_products
        )

    def _build_unbatched_market_loader_config(self) -> TrainingConfig:
        """Clone the runtime config with product batching disabled for live scoring."""

        if is_coinmarketcap_market_data_source(self.config.market_data_source):
            return replace(
                self.config,
                coinmarketcap_product_batch_size=None,
                coinmarketcap_product_batch_number=1,
            )

        return replace(
            self.config,
            coinbase_product_batch_size=None,
            coinbase_product_batch_number=1,
        )

    def _should_use_prioritized_active_universe(self) -> bool:
        """Return whether CMC live scoring should use the cached tracked universe plus follow-up priorities."""

        return bool(
            is_coinmarketcap_market_data_source(self.config.market_data_source)
            and self._resolve_market_fetch_all_quote_products()
            and self.config.live_fetch_all_quote_products
        )

    def _load_watchlist_pool_product_ids(self) -> list[str]:
        """Return the persisted watchlist pool product ids that deserve extra monitoring."""

        if not bool(getattr(self.config, "signal_watchlist_pool_enabled", True)):
            return []

        max_products = int(getattr(self.config, "signal_watchlist_pool_max_products", 12) or 12)
        pool_store = WatchlistPoolStore(Path(self.config.signal_watchlist_pool_path))
        return pool_store.get_monitored_product_ids(limit=max_products)

    def _build_explicit_signal_prediction_frame(
        self,
        product_ids: List[str],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Score one explicit set of products for watchlist-pool promotion checks."""

        if self.model is None:
            raise ValueError("A loaded model is required before scoring explicit signal products.")

        normalized_product_ids = [
            str(product_id).strip().upper()
            for product_id in product_ids
            if str(product_id).strip()
        ]
        if not normalized_product_ids:
            raise ValueError("At least one explicit product id is required for signal scoring.")

        loader_config = self._build_unbatched_market_loader_config()
        explicit_loader = create_market_data_loader(
            config=loader_config,
            data_path=loader_config.data_file,
            should_save_downloaded_data=False,
            product_ids=tuple(normalized_product_ids),
            fetch_all_quote_products=False,
            max_products=None,
            granularity_seconds=loader_config.live_granularity_seconds,
            total_candles=loader_config.live_total_candles,
            request_pause_seconds=loader_config.live_request_pause_seconds,
            save_progress_every_products=0,
            log_progress=False,
        )
        explicit_dataset_builder = CryptoDatasetBuilder(
            config=loader_config,
            feature_columns=self.model.feature_columns,
            data_loader=explicit_loader,
        )
        explicit_feature_df = explicit_dataset_builder.build_feature_table()
        explicit_prediction_df = self.model.predict(explicit_feature_df)

        return explicit_prediction_df, {
            "productsRequested": len(normalized_product_ids),
            "rowsScored": int(len(explicit_prediction_df)),
            "productsScored": (
                int(explicit_prediction_df["product_id"].nunique())
                if "product_id" in explicit_prediction_df.columns
                else int(len(explicit_prediction_df))
            ),
        }

    def _build_signal_store(self) -> TradingSignalStore:
        """Create the persistent live-signal store for current and historical rows."""

        return TradingSignalStore(
            db_path=self.config.signal_store_path,
            database_url=self.config.signal_store_url,
        )

    def _build_prioritized_signal_prediction_frame(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Score a reduced active universe selected from cached CMC ranking plus current follow-up state."""

        if self.model is None:
            raise ValueError("A loaded model is required before scoring the active signal universe.")

        active_universe_plan = SignalUniverseCoordinator(self.config).resolve_active_universe(
            max_products=self.config.live_max_products,
        )
        if not active_universe_plan.product_ids:
            raise ValueError(
                "No active analysis products were available after applying watchlist, portfolio, "
                "published-signal, and tracked-universe selection."
            )

        active_prediction_df, explicit_summary = self._build_explicit_signal_prediction_frame(
            active_universe_plan.product_ids,
        )
        signal_inference_summary = {
            "mode": "prioritized-active-universe",
            "warning": "",
            "maxProducts": int(active_universe_plan.summary.get("effectiveLimit", len(active_universe_plan.product_ids))),
            "productsRequested": int(explicit_summary["productsRequested"]),
            "totalAvailableProducts": int(
                active_universe_plan.source_refresh.get(
                    "productCount",
                    active_universe_plan.summary.get("trackedUniverse", {}).get("count", 0),
                )
                or 0
            ),
            "rowsScored": int(explicit_summary["rowsScored"]),
            "productsScored": int(explicit_summary["productsScored"]),
            "protectedProductIds": list(active_universe_plan.protected_product_ids),
            "activeUniverse": dict(active_universe_plan.summary),
            "sourceRefresh": dict(active_universe_plan.source_refresh),
        }
        return active_prediction_df, signal_inference_summary

    def _build_fresh_signal_prediction_frame(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Score a fresh top-market universe for the published signal snapshot."""

        if self.model is None:
            raise ValueError("A loaded model is required before scoring the fresh signal universe.")

        if self._should_use_prioritized_active_universe():
            return self._build_prioritized_signal_prediction_frame()

        loader_config = self._build_unbatched_market_loader_config()
        fresh_market_loader = create_market_data_loader(
            config=loader_config,
            data_path=loader_config.data_file,
            should_save_downloaded_data=False,
            fetch_all_quote_products=True,
            max_products=loader_config.live_max_products,
            granularity_seconds=loader_config.live_granularity_seconds,
            total_candles=loader_config.live_total_candles,
            request_pause_seconds=loader_config.live_request_pause_seconds,
            save_progress_every_products=0,
            log_progress=False,
        )
        fresh_dataset_builder = CryptoDatasetBuilder(
            config=loader_config,
            feature_columns=self.model.feature_columns,
            data_loader=fresh_market_loader,
        )
        fresh_feature_df = fresh_dataset_builder.build_feature_table()
        fresh_prediction_df = self.model.predict(fresh_feature_df)
        loader_summary = dict(getattr(fresh_market_loader, "last_refresh_summary", {}))

        signal_inference_summary = {
            "mode": "fresh-top-market-universe",
            "warning": "",
            "maxProducts": (
                int(loader_config.live_max_products)
                if loader_config.live_max_products is not None
                else None
            ),
            "productsRequested": int(loader_summary.get("productsDownloaded", 0) or 0),
            "totalAvailableProducts": int(loader_summary.get("totalAvailableProducts", 0) or 0),
            "rowsScored": int(len(fresh_prediction_df)),
            "productsScored": (
                int(fresh_prediction_df["product_id"].nunique())
                if "product_id" in fresh_prediction_df.columns
                else int(len(fresh_prediction_df))
            ),
        }
        prediction_frames = [fresh_prediction_df]
        watchlist_pool_product_ids = self._load_watchlist_pool_product_ids()
        if watchlist_pool_product_ids:
            explicit_prediction_df, explicit_summary = self._build_explicit_signal_prediction_frame(
                watchlist_pool_product_ids,
            )
            if not explicit_prediction_df.empty:
                prediction_frames.append(explicit_prediction_df)
                signal_inference_summary["mode"] = "fresh-top-market-plus-watchlist-pool"
                signal_inference_summary["watchlistPoolProductsRequested"] = explicit_summary["productsRequested"]
                signal_inference_summary["watchlistPoolRowsScored"] = explicit_summary["rowsScored"]
                signal_inference_summary["watchlistPoolProductsScored"] = explicit_summary["productsScored"]
                signal_inference_summary["watchlistPoolProductIds"] = watchlist_pool_product_ids

        combined_prediction_df = pd.concat(prediction_frames, ignore_index=True)
        duplicate_subset = [
            column_name
            for column_name in ("product_id", "timestamp", "time_step")
            if column_name in combined_prediction_df.columns
        ]
        if duplicate_subset:
            combined_prediction_df = combined_prediction_df.drop_duplicates(
                subset=duplicate_subset,
                keep="last",
            )

        return combined_prediction_df, signal_inference_summary

    def run(self) -> Dict[str, Any]:
        """
        Execute the signal generation workflow.

        If no model object was passed in, we load it from disk first.
        """

        if self.model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    "No trained model was found. Run `python model-service/scripts/trainModel.py` first."
                )

            self.model = BaseSignalModel.load(self.model_path)
            self.config = apply_runtime_market_data_settings(
                base_config=self.model.config,
                runtime_config=self.config,
            )
            self.model.config = self.config
            self.dataset_builder = CryptoDatasetBuilder(
                config=self.config,
                feature_columns=self.model.feature_columns,
            )

        prefetched_signal_prediction_df: pd.DataFrame | None = None
        prefetched_signal_inference_summary: dict[str, Any] | None = None
        prefetched_signal_inference_warning = ""
        if self._should_score_fresh_signal_universe():
            try:
                (
                    prefetched_signal_prediction_df,
                    prefetched_signal_inference_summary,
                ) = self._build_fresh_signal_prediction_frame()
            except Exception as error:
                prefetched_signal_inference_warning = str(error)

        market_data_refresh = None
        market_data_refreshed_at = None
        should_use_prioritized_active_universe = self._should_use_prioritized_active_universe()
        if self.config.signal_refresh_market_data_before_generation and not should_use_prioritized_active_universe:
            market_data_refreshed_at = datetime.now(timezone.utc).isoformat()
            market_data_refresh = MarketDataRefreshApp(config=self.config).run()
            self.save_json(
                {
                    "generatedAt": market_data_refreshed_at,
                    "mode": "fresh-market-refresh",
                    "refresh": market_data_refresh,
                },
                OUTPUTS_DIR / "signalMarketDataRefresh.json",
            )
        elif not self.config.data_file.exists():
            self._ensure_market_data_available()

        feature_df = self.dataset_builder.build_feature_table()
        prediction_df = self.model.predict(feature_df)
        inference_stage = SignalInferenceStage(self.config)
        allow_empty_historical_inference = (
            prefetched_signal_prediction_df is not None
            and prefetched_signal_inference_summary is not None
        )
        inference_artifacts = inference_stage.build_from_prediction_frame(
            prediction_df,
            mode="historical-market-data",
            raise_on_empty=not allow_empty_historical_inference,
        )
        signal_prediction_df = prediction_df
        used_fresh_signal_prediction = False
        if prefetched_signal_prediction_df is not None and prefetched_signal_inference_summary is not None:
            fresh_inference_artifacts = inference_stage.build_from_prediction_frame(
                prefetched_signal_prediction_df,
                summary=prefetched_signal_inference_summary,
                raise_on_empty=False,
                protected_product_ids=prefetched_signal_inference_summary.get("protectedProductIds"),
            )
            if fresh_inference_artifacts.signal_candidates:
                signal_prediction_df = prefetched_signal_prediction_df
                inference_artifacts = fresh_inference_artifacts
                used_fresh_signal_prediction = True
            else:
                inference_artifacts.summary["mode"] = "historical-market-data-fallback"
                inference_artifacts.summary["warning"] = (
                    "Fresh active-universe signal scoring produced no eligible signal rows, "
                    "so publication fell back to the persisted market dataset."
                )
        elif prefetched_signal_inference_warning:
            inference_artifacts.summary["mode"] = "historical-market-data-fallback"
            inference_artifacts.summary["warning"] = prefetched_signal_inference_warning

        if not inference_artifacts.signal_candidates:
            raise ValueError(
                "No signal summaries remained after applying the configured signal-universe exclusions."
            )

        portfolio_store = TradingPortfolioStore(
            db_path=self.config.portfolio_store_path,
            default_capital=self.config.portfolio_default_capital,
            database_url=self.config.portfolio_store_url,
        )
        primary_history_store = PrimarySignalHistoryStore(
            config=self.config,
            history_path=self.primary_signal_history_path,
            save_json=self.save_json,
        )
        decision_stage = SignalDecisionStage(
            self.config,
            primary_history_store=primary_history_store,
            allow_watchlist_fallback=True,
            allow_watchlist_supplement=True,
        )
        publication_stage = SignalPublicationStage(
            config=self.config,
            save_json=self.save_json,
            save_dataframe=self.save_dataframe,
            signal_store_factory=self._build_signal_store,
            primary_history_store=primary_history_store,
        )
        coordinator = SignalGenerationCoordinator(
            inference_stage=inference_stage,
            context_stage=SignalContextEnrichmentStage(self.config),
            enrichment_stage=SignalEnrichmentStage(self.config),
            decision_stage=decision_stage,
            publication_stage=publication_stage,
        )
        pipeline_artifacts = coordinator.run_pipeline(
            inference_artifacts=inference_artifacts,
            portfolio_store=portfolio_store,
            save_watchlist_pool_snapshot=True,
        )
        latest_signals = pipeline_artifacts.decision.published_signals
        actionable_signals = pipeline_artifacts.decision.actionable_signals
        top_signal = pipeline_artifacts.decision.primary_signal
        prediction_timestamps = signal_prediction_df["timestamp"] if "timestamp" in signal_prediction_df.columns else None
        data_first_timestamp = (
            str(prediction_timestamps.min())
            if prediction_timestamps is not None and len(prediction_timestamps) > 0
            else None
        )
        data_last_timestamp = (
            str(prediction_timestamps.max())
            if prediction_timestamps is not None and len(prediction_timestamps) > 0
            else None
        )
        signal_source = (
            "live-market-refresh"
            if market_data_refresh is not None
            else "live-active-universe-refresh"
            if used_fresh_signal_prediction
            else "cached-market-data"
        )
        signal_metadata = {
            "signalSource": signal_source,
            "marketDataSource": str(self.config.market_data_source),
            "marketDataPath": str(self.config.data_file),
            "marketDataFirstTimestamp": (
                market_data_refresh.get("firstTimestamp")
                if market_data_refresh is not None
                else data_first_timestamp
            ),
            "marketDataLastTimestamp": (
                market_data_refresh.get("lastTimestamp")
                if market_data_refresh is not None
                else data_last_timestamp
            ),
            "marketDataRefreshedAt": market_data_refreshed_at,
        }
        publication_artifacts = coordinator.publish_signal_generation(
            model_type=self.model.model_type,
            historical_prediction_df=prediction_df,
            pipeline_artifacts=pipeline_artifacts,
            signal_source=signal_source,
            signal_metadata=signal_metadata,
            market_data_refresh=market_data_refresh,
            market_data_refreshed_at=market_data_refreshed_at,
            portfolio_store=portfolio_store,
        )

        return {
            "modelType": self.model.model_type,
            "historicalSignalsPath": str(OUTPUTS_DIR / "historicalSignals.csv"),
            "latestSignalPath": str(OUTPUTS_DIR / "latestSignal.json"),
            "latestSignalsPath": str(OUTPUTS_DIR / "latestSignals.json"),
            "actionableSignalsPath": str(OUTPUTS_DIR / "actionableSignals.json"),
            "frontendSignalSnapshotPath": str(OUTPUTS_DIR / "frontendSignalSnapshot.json"),
            "signalsGenerated": len(publication_artifacts.latest_signals),
            "actionableSignalsGenerated": len(publication_artifacts.actionable_signals),
            "signalName": (
                publication_artifacts.primary_signal.get("signal_name")
                if publication_artifacts.primary_signal is not None
                else None
            ),
            "confidence": (
                publication_artifacts.primary_signal.get("confidence")
                if publication_artifacts.primary_signal is not None
                else None
            ),
            "signalChat": (
                publication_artifacts.primary_signal.get("signalChat")
                if publication_artifacts.primary_signal is not None
                else None
            ),
            "signalSource": signal_source,
            "marketDataRefresh": market_data_refresh,
            "marketDataRefreshedAt": market_data_refreshed_at,
            "signalInference": pipeline_artifacts.inference.summary,
            "signalStore": publication_artifacts.signal_store_summary,
            "signalStorePath": str(self.config.signal_store_path),
            "trackedTradeSync": publication_artifacts.tracked_trade_sync,
        }


class ProductionCycleApp(BaseSignalApp):
    """Run the continuous refresh, retrain, and publish cycle in one command."""

    def run(self) -> Dict[str, Any]:
        """Refresh data, train the current model, and regenerate the frontend snapshot."""

        market_refresh_result = MarketDataRefreshApp(config=self.config).run()
        training_result = TrainingApp(config=self.config).run()
        signal_generation_result = SignalGenerationApp(
            config=replace(
                self.config,
                signal_refresh_market_data_before_generation=False,
            )
        ).run()

        return {
            "marketRefresh": market_refresh_result,
            "training": training_result,
            "signalGeneration": signal_generation_result,
            "modelType": training_result["modelType"],
            "modelPath": training_result["modelPath"],
            "metadataPath": training_result["metadataPath"],
            "frontendSignalSnapshotPath": signal_generation_result["frontendSignalSnapshotPath"],
        }


class BacktestApp(TrainingApp):
    """Train one model and evaluate its predictions as trades."""

    def __init__(
        self,
        config: TrainingConfig = None,
        dataset_builder: CryptoDatasetBuilder = None,
        model: BaseSignalModel = None,
        model_class: Type[BaseSignalModel] = None,
        backtester: BaseSignalBacktester = None,
    ) -> None:
        super().__init__(
            config=config,
            dataset_builder=dataset_builder,
            model=model,
            model_class=model_class,
        )
        self.backtester = backtester or EqualWeightSignalBacktester(self.config)

    def run(self) -> Dict[str, Any]:
        """
        Train the configured model and save a simple trading backtest.
        """

        dataset, feature_columns = self._build_dataset()
        training_result = self._train_model_for_type(
            dataset=dataset,
            feature_columns=feature_columns,
            model_type=self.config.model_type,
        )
        training_summary = self._save_primary_training_outputs(training_result)

        backtest_result = self.backtester.run(training_result["prediction_df"])
        trade_df = backtest_result["trade_df"]
        period_df = backtest_result["period_df"]
        summary = backtest_result["summary"]

        self.save_dataframe(trade_df, OUTPUTS_DIR / "backtestTrades.csv")
        self.save_dataframe(period_df, OUTPUTS_DIR / "backtestPeriods.csv")
        self.save_json(summary, OUTPUTS_DIR / "backtestSummary.json")

        return {
            **training_summary,
            "backtestTradesPath": str(OUTPUTS_DIR / "backtestTrades.csv"),
            "backtestPeriodsPath": str(OUTPUTS_DIR / "backtestPeriods.csv"),
            "backtestSummaryPath": str(OUTPUTS_DIR / "backtestSummary.json"),
            "tradeCount": summary["tradeCount"],
            "strategyTotalReturn": summary["strategyTotalReturn"],
            "benchmarkTotalReturn": summary["benchmarkTotalReturn"],
            "maxDrawdown": summary["maxDrawdown"],
        }


class MarketDataRefreshApp(BaseSignalApp):
    """Download fresh real-market data and save it to the raw-data CSV."""

    def run(self) -> Dict[str, Any]:
        """
        Refresh the raw market-data file from the configured live source.
        """

        effective_config = self.config
        batch_rotation_summary = {
            "enabled": False,
            "activeBatchNumber": self._resolve_market_product_batch_number(),
            "nextBatchNumber": self._resolve_market_product_batch_number(),
            "totalBatches": 1,
            "statePath": str(self.config.market_product_batch_state_file),
        }
        preview_loader = self.build_market_data_loader()
        data_loader = preview_loader

        should_rotate_batches = bool(
            self.config.market_product_batch_rotation_enabled
            and self._resolve_market_fetch_all_quote_products()
            and self._resolve_market_product_batch_size() is not None
            and hasattr(preview_loader, "get_total_batches")
        )
        if should_rotate_batches:
            total_batches = int(preview_loader.get_total_batches())
            active_batch_number, batch_rotation_summary = self._resolve_rotated_market_product_batch_number(
                total_batches=total_batches,
            )
            if active_batch_number != self._resolve_market_product_batch_number():
                effective_config = self._with_market_product_batch_number(active_batch_number)
                data_loader = self._build_market_data_loader_for_config(effective_config)

        price_df = data_loader.refresh_data()
        persisted_batch_state: Dict[str, Any] = {}
        if batch_rotation_summary.get("enabled"):
            persisted_batch_state = self._save_market_product_batch_state(
                active_batch_number=int(batch_rotation_summary["activeBatchNumber"]),
                total_batches=int(batch_rotation_summary["totalBatches"]),
            )
        coinmarketcap_context_status = "disabled"
        coinmarketcap_context_rows = 0
        coinmarketcap_context_error = ""
        coinmarketcap_market_intelligence_status = "disabled"
        coinmarketcap_market_intelligence_error = ""
        coinmarketcap_market_intelligence_summary: Dict[str, Any] = {}
        coinmarketcal_events_status = "disabled"
        coinmarketcal_events_rows = 0
        coinmarketcal_events_error = ""

        if self.config.coinmarketcap_use_context:
            context_enricher = self.build_coinmarketcap_context_enricher(
                should_refresh_context=self.config.coinmarketcap_refresh_context_after_market_refresh,
            )

            if self.config.coinmarketcap_refresh_context_after_market_refresh:
                try:
                    context_df = context_enricher.refresh_context(price_df)
                    context_summary = dict(getattr(context_enricher, "last_context_summary", {}))
                    if context_summary.get("usedCachedSnapshot"):
                        coinmarketcap_context_status = "rate_limited_cached_only"
                        coinmarketcap_context_error = str(context_summary.get("warning", ""))
                    else:
                        coinmarketcap_context_status = "refreshed"
                    coinmarketcap_context_rows = len(context_df)
                except Exception as error:
                    error_message = str(error)
                    if "requires an API key" in error_message:
                        coinmarketcap_context_status = "skipped_missing_api_key"
                    else:
                        coinmarketcap_context_status = "refresh_failed"
                        coinmarketcap_context_error = error_message
            else:
                coinmarketcap_context_status = "enabled_cached_only"

        if self.config.coinmarketcap_use_market_intelligence:
            market_intelligence_enricher = self.build_coinmarketcap_market_intelligence_enricher(
                should_refresh_market_intelligence=(
                    self.config.coinmarketcap_refresh_market_intelligence_after_market_refresh
                ),
            )

            if self.config.coinmarketcap_refresh_market_intelligence_after_market_refresh:
                try:
                    market_intelligence_enricher.refresh_market_intelligence()
                    coinmarketcap_market_intelligence_summary = dict(
                        getattr(
                            market_intelligence_enricher,
                            "last_market_intelligence_summary",
                            {},
                        )
                    )
                    if coinmarketcap_market_intelligence_summary.get("usedCachedSnapshot"):
                        coinmarketcap_market_intelligence_status = "rate_limited_cached_only"
                        coinmarketcap_market_intelligence_error = str(
                            coinmarketcap_market_intelligence_summary.get("warning", "")
                        )
                    else:
                        coinmarketcap_market_intelligence_status = "refreshed"
                except Exception as error:
                    error_message = str(error)
                    if "requires an API key" in error_message:
                        coinmarketcap_market_intelligence_status = "skipped_missing_api_key"
                    else:
                        coinmarketcap_market_intelligence_status = "refresh_failed"
                        coinmarketcap_market_intelligence_error = error_message
            else:
                coinmarketcap_market_intelligence_status = "enabled_cached_only"

        if self.config.coinmarketcal_use_events:
            event_enricher = self.build_coinmarketcal_event_enricher(
                should_refresh_events=self.config.coinmarketcal_refresh_events_after_market_refresh,
            )

            if self.config.coinmarketcal_refresh_events_after_market_refresh:
                try:
                    events_df = event_enricher.refresh_events(price_df)
                    coinmarketcal_events_status = "refreshed"
                    coinmarketcal_events_rows = len(events_df)
                except Exception as error:
                    error_message = str(error)
                    if "requires an API key" in error_message:
                        coinmarketcal_events_status = "skipped_missing_api_key"
                    else:
                        coinmarketcal_events_status = "refresh_failed"
                        coinmarketcal_events_error = error_message
            else:
                coinmarketcal_events_status = "enabled_cached_only"

        return {
            "marketDataSource": self.config.market_data_source,
            "productMode": (
                f"all-{self._resolve_market_quote_currency().upper()}-quoted-products"
                if self._resolve_market_fetch_all_quote_products()
                else "explicit-product-list"
            ),
            "batchRotationEnabled": bool(batch_rotation_summary.get("enabled")),
            "batchNumber": int(batch_rotation_summary.get("activeBatchNumber", self._resolve_market_product_batch_number())),
            "nextBatchNumber": int(batch_rotation_summary.get("nextBatchNumber", self._resolve_market_product_batch_number())),
            "totalBatches": int(batch_rotation_summary.get("totalBatches", 1)),
            "batchRotationStatePath": str(batch_rotation_summary.get("statePath", self.config.market_product_batch_state_file)),
            "batchRotationState": persisted_batch_state,
            "granularitySeconds": self._resolve_market_granularity_seconds(),
            "savedPath": str(self.config.data_file),
            "rowsDownloaded": len(price_df),
            "uniqueProducts": int(price_df["product_id"].nunique()) if "product_id" in price_df.columns else 1,
            "firstTimestamp": str(price_df.iloc[0]["timestamp"]),
            "lastTimestamp": str(price_df.iloc[-1]["timestamp"]),
            "downloadSummary": getattr(data_loader, "last_refresh_summary", {}),
            "coinMarketCapContextStatus": coinmarketcap_context_status,
            "coinMarketCapContextRows": coinmarketcap_context_rows,
            "coinMarketCapContextError": coinmarketcap_context_error,
            "coinMarketCapContextPath": str(self.config.coinmarketcap_context_file),
            "coinMarketCapMarketIntelligenceStatus": coinmarketcap_market_intelligence_status,
            "coinMarketCapMarketIntelligenceError": coinmarketcap_market_intelligence_error,
            "coinMarketCapMarketIntelligencePath": str(self.config.coinmarketcap_market_intelligence_file),
            "coinMarketCapMarketIntelligenceSummary": coinmarketcap_market_intelligence_summary,
            "coinMarketCalEventsStatus": coinmarketcal_events_status,
            "coinMarketCalEventsRows": coinmarketcal_events_rows,
            "coinMarketCalEventsError": coinmarketcal_events_error,
            "coinMarketCalEventsPath": str(self.config.coinmarketcal_events_file),
        }


class MarketEventsRefreshApp(BaseSignalApp):
    """Refresh the cached CoinMarketCal event snapshot independently of candle refreshes."""

    def run(self) -> Dict[str, Any]:
        """Load the tracked market universe and refresh its event cache."""

        self._ensure_market_data_available()
        price_df = CsvPriceDataLoader(self.config.data_file).load()
        event_enricher = self.build_coinmarketcal_event_enricher(should_refresh_events=True)
        events_df = event_enricher.refresh_events(price_df)
        refresh_summary = getattr(event_enricher, "last_events_summary", {})

        return {
            "status": "refreshed",
            "marketDataPath": str(self.config.data_file),
            "eventsPath": str(self.config.coinmarketcal_events_file),
            "trackedProducts": (
                int(price_df["product_id"].nunique())
                if "product_id" in price_df.columns
                else 1
            ),
            "trackedBaseCurrencies": (
                int(price_df["base_currency"].astype(str).str.upper().nunique())
                if "base_currency" in price_df.columns
                else (
                    int(price_df["product_id"].astype(str).str.split("-").str[0].nunique())
                    if "product_id" in price_df.columns
                    else 0
                )
            ),
            "eventsRows": int(len(events_df)),
            "refreshSummary": refresh_summary,
        }


class MarketUniverseRefreshApp(BaseSignalApp):
    """Refresh every remaining market batch instead of only one configured batch."""

    def run(self) -> Dict[str, Any]:
        """
        Download the full configured market universe from the current start batch.

        This app is the practical next step once the project grows beyond
        one batch. It keeps the user from manually editing the config and
        rerunning the single-batch script over and over.
        """

        market_loader = self.build_market_data_loader()
        total_products = len(market_loader.get_available_products())
        total_batches = market_loader.get_total_batches()
        start_batch = self._resolve_market_product_batch_number()

        if start_batch > total_batches:
            raise ValueError(
                "The configured market product batch number is beyond the available batch count. "
                f"Start batch: {start_batch}, total batches: {total_batches}."
            )

        batch_results = []
        failed_batches = []
        for batch_number in range(start_batch, total_batches + 1):
            batch_config = replace(
                self._with_market_product_batch_number(batch_number),
                market_product_batch_rotation_enabled=False,
            )
            batch_app = MarketDataRefreshApp(config=batch_config)
            try:
                batch_result = batch_app.run()
                batch_result["batchStatus"] = (
                    "completed_with_context_warning"
                    if batch_result.get("coinMarketCapContextStatus") == "refresh_failed"
                    else "completed"
                )
            except Exception as error:
                batch_result = {
                    "batchNumber": batch_number,
                    "batchStatus": "failed",
                    "error": str(error),
                }
                failed_batches.append(batch_number)

            batch_result["batchNumber"] = batch_number
            batch_results.append(batch_result)

        final_rows_saved = 0
        final_unique_products = 0
        if self.config.data_file.exists():
            final_price_df = pd.read_csv(self.config.data_file)
            final_price_df["timestamp"] = pd.to_datetime(final_price_df["timestamp"], errors="coerce", utc=True)
            final_rows_saved = int(len(final_price_df))
            final_unique_products = (
                int(final_price_df["product_id"].nunique()) if "product_id" in final_price_df.columns else 1
            )

        context_rows_saved = 0
        context_unique_products = 0
        if self.config.coinmarketcap_context_file.exists():
            context_df = pd.read_csv(self.config.coinmarketcap_context_file)
            context_rows_saved = int(len(context_df))
            context_unique_products = (
                int(context_df["product_id"].nunique()) if "product_id" in context_df.columns else 0
            )

        final_summary = {
            "marketDataSource": self.config.market_data_source,
            "startBatch": start_batch,
            "endBatch": total_batches,
            "batchesRun": len(batch_results),
            "totalProductsAvailable": total_products,
            "successfulBatches": [result["batchNumber"] for result in batch_results if result["batchStatus"] != "failed"],
            "failedBatches": failed_batches,
            "finalRowsSaved": final_rows_saved,
            "finalUniqueProducts": final_unique_products,
            "savedPath": str(self.config.data_file),
            "coinMarketCapContextRows": context_rows_saved,
            "coinMarketCapContextUniqueProducts": context_unique_products,
            "coinMarketCapContextPath": str(self.config.coinmarketcap_context_file),
            "batchResults": batch_results,
        }

        return final_summary
