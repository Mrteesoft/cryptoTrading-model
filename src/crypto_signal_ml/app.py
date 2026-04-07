"""Application-level classes for training, comparison, and signal generation."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
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
from .frontend import (
    WatchlistPoolStore,
    build_frontend_signal_snapshot,
    build_watchlist_pool_snapshot,
)
from .labels import create_labeler_from_config, create_regime_labeler_from_config
from .modeling import BaseSignalModel, create_model_from_config, get_model_class
from .pipeline import CryptoDatasetBuilder
from .trading.portfolio import TradingPortfolioStore
from .regime_modeling import MarketRegimeModel
from .trading.signals import (
    apply_signal_trade_context,
    build_actionable_signal_summaries,
    build_latest_signal_summaries,
    filter_published_signal_summaries,
    select_primary_signal,
)
from .trading.trader_brain import TraderBrain


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
        self.model_class = model_class or get_model_class(self.config.model_type)
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

    def _save_watchlist_pool_snapshot(self, signal_summaries: List[Dict[str, Any]]) -> None:
        """Persist the strongest watchlist names for more aggressive live monitoring."""

        if not bool(getattr(self.config, "signal_watchlist_pool_enabled", True)):
            return

        max_products = int(getattr(self.config, "signal_watchlist_pool_max_products", 12) or 12)
        watchlist_pool_snapshot = build_watchlist_pool_snapshot(
            signal_summaries=signal_summaries,
            max_products=max_products,
        )
        self.save_json(
            watchlist_pool_snapshot,
            Path(self.config.signal_watchlist_pool_path),
        )

    def _build_fresh_signal_prediction_frame(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Score a fresh top-market universe for the published signal snapshot."""

        if self.model is None:
            raise ValueError("A loaded model is required before scoring the fresh signal universe.")

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

    def _build_signal_tracking_metadata(
        self,
        signal_summary: Dict[str, Any],
        signal_source: str,
    ) -> dict[str, Any]:
        """Build one compact metadata payload for an auto-tracked signal trade."""

        brain = signal_summary.get("brain", {}) if isinstance(signal_summary.get("brain"), dict) else {}
        return {
            "autoTrackedFromSignalGeneration": True,
            "signalSource": signal_source,
            "signalTimestamp": signal_summary.get("timestamp"),
            "signalName": signal_summary.get("signal_name"),
            "confidence": signal_summary.get("confidence"),
            "productId": signal_summary.get("productId"),
            "marketDataSource": signal_summary.get("marketDataSource"),
            "brainDecision": brain.get("decision"),
            "brainSummary": brain.get("summaryLine"),
        }

    def _sync_generated_signal_trades(
        self,
        latest_signals: list[Dict[str, Any]],
        signal_source: str,
        portfolio_store: TradingPortfolioStore | None = None,
    ) -> dict[str, Any]:
        """Create tracked trade records for fresh BUY signals and refresh active records."""

        if not bool(getattr(self.config, "signal_track_generated_trades", False)):
            return {
                "enabled": False,
                "createdCount": 0,
                "refreshedCount": 0,
                "skippedCount": 0,
                "trackedTradeIds": [],
            }

        portfolio_store = portfolio_store or TradingPortfolioStore(
            db_path=self.config.portfolio_store_path,
            default_capital=self.config.portfolio_default_capital,
            database_url=self.config.portfolio_store_url,
        )
        created_count = 0
        refreshed_count = 0
        skipped_count = 0
        tracked_trade_ids: list[int] = []

        for signal_summary in latest_signals:
            product_id = str(signal_summary.get("productId", "")).strip().upper()
            if not product_id:
                skipped_count += 1
                continue

            signal_name = str(signal_summary.get("signal_name", "")).strip().upper()
            brain = signal_summary.get("brain", {}) if isinstance(signal_summary.get("brain"), dict) else {}
            signal_metadata = self._build_signal_tracking_metadata(signal_summary, signal_source)
            active_trade = portfolio_store.get_active_trade_for_product(product_id)

            if active_trade is not None:
                refreshed_trade = portfolio_store.refresh_trade(
                    trade_id=int(active_trade["tradeId"]),
                    current_price=float(signal_summary.get("close") or 0.0) or None,
                    stop_loss_price=brain.get("stopLossPrice"),
                    take_profit_price=brain.get("takeProfitPrice"),
                    signal_name=signal_name or None,
                    metadata=signal_metadata,
                )
                refreshed_count += 1
                tracked_trade_ids.append(int(refreshed_trade["tradeId"]))
                continue

            if signal_name != "BUY":
                skipped_count += 1
                continue
            if not bool(signal_summary.get("actionable", False)):
                skipped_count += 1
                continue

            entry_price = float(signal_summary.get("close") or 0.0)
            if entry_price <= 0:
                skipped_count += 1
                continue

            tracked_trade = portfolio_store.create_trade(
                product_id=product_id,
                entry_price=entry_price,
                take_profit_price=brain.get("takeProfitPrice"),
                stop_loss_price=brain.get("stopLossPrice"),
                quantity=0.0,
                current_price=entry_price,
                signal_name=signal_name,
                status=str(getattr(self.config, "signal_generated_trade_status", "planned") or "planned"),
                opened_at=str(signal_summary.get("timestamp") or "") or None,
                metadata=signal_metadata,
            )
            created_count += 1
            tracked_trade_ids.append(int(tracked_trade["tradeId"]))

        return {
            "enabled": True,
            "createdCount": created_count,
            "refreshedCount": refreshed_count,
            "skippedCount": skipped_count,
            "trackedTradeIds": tracked_trade_ids,
            "storageBackend": portfolio_store.database.storage_backend,
            "databaseTarget": portfolio_store.database.database_target,
        }

    def _load_recent_primary_product_ids(self) -> list[str]:
        """Load the recently featured primary-signal products from disk when available."""

        if not self.primary_signal_history_path.exists():
            latest_signal_path = OUTPUTS_DIR / "latestSignal.json"
            if not latest_signal_path.exists():
                return []

            try:
                with latest_signal_path.open("r", encoding="utf-8") as latest_signal_file:
                    latest_signal_payload = json.load(latest_signal_file)
            except (OSError, json.JSONDecodeError):
                return []

            latest_product_id = str(latest_signal_payload.get("productId", "")).strip().upper()
            return [latest_product_id] if latest_product_id else []

        try:
            with self.primary_signal_history_path.open("r", encoding="utf-8") as history_file:
                history_payload = json.load(history_file)
        except (OSError, json.JSONDecodeError):
            return []

        history_entries = history_payload.get("entries", []) if isinstance(history_payload, dict) else []
        if not isinstance(history_entries, list):
            return []

        return [
            str(history_entry.get("productId", "")).strip().upper()
            for history_entry in history_entries
            if isinstance(history_entry, dict) and str(history_entry.get("productId", "")).strip()
        ]

    def _save_primary_signal_history(
        self,
        top_signal: Dict[str, Any],
        signal_source: str,
    ) -> None:
        """Persist a small recent history of featured primary signals for rotation."""

        recent_entries: list[dict[str, Any]] = []
        if self.primary_signal_history_path.exists():
            try:
                with self.primary_signal_history_path.open("r", encoding="utf-8") as history_file:
                    history_payload = json.load(history_file)
                previous_entries = history_payload.get("entries", []) if isinstance(history_payload, dict) else []
                if isinstance(previous_entries, list):
                    recent_entries = [
                        entry
                        for entry in previous_entries
                        if isinstance(entry, dict)
                    ]
            except (OSError, json.JSONDecodeError):
                recent_entries = []

        new_entry = {
            "productId": str(top_signal.get("productId", "")),
            "signalName": str(top_signal.get("signal_name", "")),
            "generatedAt": str(top_signal.get("marketDataRefreshedAt") or datetime.now(timezone.utc).isoformat()),
            "signalSource": str(signal_source),
            "confidence": float(top_signal.get("confidence", 0.0) or 0.0),
            "policyScore": float(top_signal.get("policyScore", 0.0) or 0.0),
            "setupScore": float(top_signal.get("setupScore", 0.0) or 0.0),
        }
        recent_entries = [new_entry] + recent_entries

        deduped_entries: list[dict[str, Any]] = []
        seen_products: set[str] = set()
        max_history_entries = max(
            int(getattr(self.config, "signal_primary_rotation_candidate_window", 4)) * 2,
            int(getattr(self.config, "signal_primary_rotation_lookback", 3)),
            6,
        )
        for history_entry in recent_entries:
            product_id = str(history_entry.get("productId", "")).strip().upper()
            if not product_id or product_id in seen_products:
                continue
            seen_products.add(product_id)
            deduped_entries.append(history_entry)
            if len(deduped_entries) >= max_history_entries:
                break

        self.save_json(
            {
                "updatedAt": datetime.now(timezone.utc).isoformat(),
                "entries": deduped_entries,
            },
            self.primary_signal_history_path,
        )

    def _load_last_primary_signal_generated_at(self) -> datetime | None:
        """Return when the most recent non-empty primary signal was published."""

        timestamp_candidates: list[str] = []

        if self.primary_signal_history_path.exists():
            try:
                with self.primary_signal_history_path.open("r", encoding="utf-8") as history_file:
                    history_payload = json.load(history_file)
                history_entries = history_payload.get("entries", []) if isinstance(history_payload, dict) else []
                if isinstance(history_entries, list):
                    timestamp_candidates.extend(
                        str(history_entry.get("generatedAt", "")).strip()
                        for history_entry in history_entries
                        if isinstance(history_entry, dict) and str(history_entry.get("generatedAt", "")).strip()
                    )
            except (OSError, json.JSONDecodeError):
                timestamp_candidates = []

        latest_signal_path = OUTPUTS_DIR / "latestSignal.json"
        if latest_signal_path.exists():
            try:
                with latest_signal_path.open("r", encoding="utf-8") as latest_signal_file:
                    latest_signal_payload = json.load(latest_signal_file)
                if isinstance(latest_signal_payload, dict) and latest_signal_payload:
                    for field_name in ("marketDataRefreshedAt", "generatedAt", "timestamp"):
                        raw_value = str(latest_signal_payload.get(field_name, "")).strip()
                        if raw_value:
                            timestamp_candidates.append(raw_value)
            except (OSError, json.JSONDecodeError):
                pass

        for raw_timestamp in timestamp_candidates:
            parsed_timestamp = pd.to_datetime(raw_timestamp, utc=True, errors="coerce")
            if pd.notna(parsed_timestamp):
                return parsed_timestamp.to_pydatetime()

        return None

    def _should_publish_watchlist_fallback(self) -> bool:
        """Return whether the public feed should emit a watchlist fallback signal."""

        if not bool(getattr(self.config, "signal_watchlist_fallback_enabled", True)):
            return False

        quiet_period_hours = max(
            float(getattr(self.config, "signal_watchlist_fallback_hours", 12.0) or 0.0),
            0.0,
        )
        last_primary_signal_at = self._load_last_primary_signal_generated_at()
        if last_primary_signal_at is None:
            return True
        if quiet_period_hours <= 0:
            return True

        return (datetime.now(timezone.utc) - last_primary_signal_at) >= timedelta(hours=quiet_period_hours)

    def _select_watchlist_fallback_signal(
        self,
        signal_summaries: list[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        """Choose one strong watchlist candidate when the public feed would otherwise be empty."""

        if not self._should_publish_watchlist_fallback():
            return None

        readiness_priority = {
            "high": 0,
            "medium": 1,
            "standby": 2,
            "blocked": 3,
        }
        minimum_decision_score = float(
            getattr(self.config, "signal_watchlist_min_decision_score", 0.30) or 0.30
        )
        minimum_confidence = float(
            getattr(self.config, "signal_watchlist_min_confidence", 0.55) or 0.55
        )
        ranked_candidates: list[tuple[tuple[Any, ...], Dict[str, Any]]] = []

        for signal_summary in signal_summaries:
            brain = signal_summary.get("brain") if isinstance(signal_summary.get("brain"), dict) else {}
            if str(brain.get("decision", "")).strip().lower() != "watchlist":
                continue

            trade_context = (
                signal_summary.get("tradeContext")
                if isinstance(signal_summary.get("tradeContext"), dict)
                else {}
            )
            if bool(trade_context.get("hasActiveTrade", False)):
                continue

            raw_signal_name = str(signal_summary.get("modelSignalName", "")).strip().upper()
            final_signal_name = str(signal_summary.get("signal_name", "")).strip().upper()
            if raw_signal_name == "TAKE_PROFIT" or final_signal_name == "LOSS":
                continue

            confidence = float(signal_summary.get("confidence", 0.0) or 0.0)
            decision_score = float(brain.get("decisionScore", 0.0) or 0.0)
            if confidence < minimum_confidence or decision_score < minimum_decision_score:
                continue

            trade_readiness = str(signal_summary.get("tradeReadiness", "standby")).strip().lower()
            ranked_candidates.append(
                (
                    (
                        0 if raw_signal_name == "BUY" else 1,
                        readiness_priority.get(trade_readiness, 99),
                        -decision_score,
                        -float(signal_summary.get("policyScore", 0.0) or 0.0),
                        -confidence,
                        str(signal_summary.get("productId", "")),
                    ),
                    dict(signal_summary),
                )
            )

        if not ranked_candidates:
            return None

        selected_signal = sorted(ranked_candidates, key=lambda item: item[0])[0][1]
        fallback_note = (
            "No actionable trade cleared the live gate recently, so this strongest watchlist candidate is surfaced instead."
        )
        existing_reason_items = [
            str(reason_item).strip()
            for reason_item in list(selected_signal.get("reasonItems", []))
            if str(reason_item).strip()
        ]
        merged_reason_items = [fallback_note]
        for reason_item in existing_reason_items:
            if reason_item not in merged_reason_items:
                merged_reason_items.append(reason_item)

        fallback_signal = dict(selected_signal)
        fallback_signal["watchlistFallback"] = True
        fallback_signal["publicSignalType"] = "watchlist"
        fallback_signal["actionable"] = False
        fallback_signal["spotAction"] = "wait"
        fallback_signal["reasonItems"] = merged_reason_items[:4]
        fallback_signal["reasonSummary"] = merged_reason_items[0]
        existing_chat = str(fallback_signal.get("signalChat", "")).strip()
        fallback_signal["signalChat"] = (
            f"{fallback_note} {existing_chat}".strip()
            if existing_chat
            else fallback_note
        )

        return fallback_signal

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
        if self.config.signal_refresh_market_data_before_generation:
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
        else:
            self._ensure_market_data_available()

        feature_df = self.dataset_builder.build_feature_table()
        prediction_df = self.model.predict(feature_df)
        latest_signals = build_latest_signal_summaries(
            prediction_df,
            minimum_action_confidence=self.config.backtest_min_confidence,
            config=self.config,
        )
        signal_prediction_df = prediction_df
        signal_inference_summary = {
            "mode": "historical-market-data",
            "warning": "",
            "maxProducts": None,
            "productsRequested": (
                int(prediction_df["product_id"].nunique())
                if "product_id" in prediction_df.columns
                else int(len(prediction_df))
            ),
            "totalAvailableProducts": None,
            "rowsScored": int(len(prediction_df)),
            "productsScored": (
                int(prediction_df["product_id"].nunique())
                if "product_id" in prediction_df.columns
                else int(len(prediction_df))
            ),
        }
        if prefetched_signal_prediction_df is not None and prefetched_signal_inference_summary is not None:
            fresh_latest_signals = build_latest_signal_summaries(
                prefetched_signal_prediction_df,
                minimum_action_confidence=self.config.backtest_min_confidence,
                config=self.config,
            )
            if fresh_latest_signals:
                signal_prediction_df = prefetched_signal_prediction_df
                latest_signals = fresh_latest_signals
                signal_inference_summary = prefetched_signal_inference_summary
            else:
                signal_inference_summary["mode"] = "historical-market-data-fallback"
                signal_inference_summary["warning"] = (
                    "Fresh top-market signal scoring produced no eligible signal rows, "
                    "so publication fell back to the persisted market dataset."
                )
        elif prefetched_signal_inference_warning:
            signal_inference_summary["mode"] = "historical-market-data-fallback"
            signal_inference_summary["warning"] = prefetched_signal_inference_warning
        if not latest_signals:
            raise ValueError(
                "No signal summaries remained after applying the configured signal-universe exclusions."
            )
        portfolio_store = TradingPortfolioStore(
            db_path=self.config.portfolio_store_path,
            default_capital=self.config.portfolio_default_capital,
            database_url=self.config.portfolio_store_url,
        )
        portfolio = portfolio_store.get_portfolio()
        positions_by_product = {
            str(position.get("productId", "")).strip().upper(): position
            for position in list(portfolio.get("positions", []))
            if str(position.get("productId", "")).strip()
        }
        active_signal_context_by_product: dict[str, dict[str, Any]] = {}
        for signal_summary in latest_signals:
            product_id = str(signal_summary.get("productId", "")).strip().upper()
            if not product_id:
                continue
            active_trade = portfolio_store.get_active_trade_for_product(product_id)
            position = positions_by_product.get(product_id)
            if active_trade is None and position is None:
                continue
            active_signal_context_by_product[product_id] = {
                "entryPrice": (
                    position.get("entryPrice")
                    if position is not None and position.get("entryPrice") is not None
                    else (active_trade.get("entryPrice") if active_trade is not None else None)
                ),
                "currentPrice": (
                    position.get("currentPrice")
                    if position is not None and position.get("currentPrice") is not None
                    else (active_trade.get("currentPrice") if active_trade is not None else None)
                ),
                "stopLossPrice": active_trade.get("stopLossPrice") if active_trade is not None else None,
                "takeProfitPrice": active_trade.get("takeProfitPrice") if active_trade is not None else None,
                "positionFraction": position.get("positionFraction") if position is not None else None,
                "quantity": position.get("quantity") if position is not None else None,
                "openedAt": (
                    position.get("openedAt")
                    if position is not None and position.get("openedAt") is not None
                    else (active_trade.get("openedAt") if active_trade is not None else None)
                ),
                "status": active_trade.get("status") if active_trade is not None else None,
            }
        latest_signals = apply_signal_trade_context(
            latest_signals,
            active_trade_product_ids=portfolio_store.get_active_signal_product_ids(),
            active_signal_context_by_product=active_signal_context_by_product,
            config=self.config,
        )
        trader_brain_plan = TraderBrain(config=self.config).build_plan(
            signal_summaries=latest_signals,
            positions=list(portfolio.get("positions", [])),
            capital=float(portfolio["capital"]),
            trade_memory_by_product=portfolio_store.build_trade_learning_map(latest_signals),
        )
        latest_signals = trader_brain_plan["signals"]
        self._save_watchlist_pool_snapshot(latest_signals)
        trader_brain_snapshot = {
            key: value
            for key, value in trader_brain_plan.items()
            if key != "signals"
        }
        latest_signals = filter_published_signal_summaries(latest_signals)
        if not latest_signals:
            watchlist_fallback_signal = self._select_watchlist_fallback_signal(
                trader_brain_plan["signals"],
            )
            if watchlist_fallback_signal is not None:
                latest_signals = [watchlist_fallback_signal]
        actionable_signals = build_actionable_signal_summaries(latest_signals)
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
        signal_source = "live-market-refresh" if market_data_refresh is not None else "cached-market-data"
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
        latest_signals = [
            {
                **signal_summary,
                **signal_metadata,
            }
            for signal_summary in latest_signals
        ]
        actionable_signals = [
            {
                **signal_summary,
                **signal_metadata,
            }
            for signal_summary in actionable_signals
        ]
        recent_primary_product_ids = self._load_recent_primary_product_ids()
        top_signal = None
        if latest_signals:
            top_signal = {
                **select_primary_signal(
                    latest_signals,
                    config=self.config,
                    recent_primary_product_ids=recent_primary_product_ids,
                ),
                **signal_metadata,
            }
        frontend_signal_snapshot = build_frontend_signal_snapshot(
            model_type=self.model.model_type,
            primary_signal=top_signal,
            latest_signals=latest_signals,
            actionable_signals=actionable_signals,
            trader_brain=trader_brain_snapshot,
        )
        frontend_signal_snapshot.update(
            {
                "mode": signal_source,
                "marketDataSource": str(self.config.market_data_source),
                "marketDataPath": str(self.config.data_file),
                "marketDataRefresh": market_data_refresh or {},
                "marketDataRefreshedAt": market_data_refreshed_at,
                "marketDataLastTimestamp": signal_metadata["marketDataLastTimestamp"],
                "marketDataFirstTimestamp": signal_metadata["marketDataFirstTimestamp"],
                "signalInference": signal_inference_summary,
            }
        )

        self.save_dataframe(prediction_df, OUTPUTS_DIR / "historicalSignals.csv")
        self.save_json(top_signal or {}, OUTPUTS_DIR / "latestSignal.json")
        self.save_json({"signals": latest_signals}, OUTPUTS_DIR / "latestSignals.json")
        self.save_json({"signals": actionable_signals}, OUTPUTS_DIR / "actionableSignals.json")
        self.save_json(frontend_signal_snapshot, OUTPUTS_DIR / "frontendSignalSnapshot.json")
        if top_signal is not None:
            self._save_primary_signal_history(top_signal=top_signal, signal_source=signal_source)
        tracked_trade_sync = self._sync_generated_signal_trades(
            latest_signals=latest_signals,
            signal_source=signal_source,
            portfolio_store=portfolio_store,
        )

        return {
            "modelType": self.model.model_type,
            "historicalSignalsPath": str(OUTPUTS_DIR / "historicalSignals.csv"),
            "latestSignalPath": str(OUTPUTS_DIR / "latestSignal.json"),
            "latestSignalsPath": str(OUTPUTS_DIR / "latestSignals.json"),
            "actionableSignalsPath": str(OUTPUTS_DIR / "actionableSignals.json"),
            "frontendSignalSnapshotPath": str(OUTPUTS_DIR / "frontendSignalSnapshot.json"),
            "signalsGenerated": len(latest_signals),
            "actionableSignalsGenerated": len(actionable_signals),
            "signalName": top_signal.get("signal_name") if top_signal is not None else None,
            "confidence": top_signal.get("confidence") if top_signal is not None else None,
            "signalChat": top_signal.get("signalChat") if top_signal is not None else None,
            "signalSource": signal_source,
            "marketDataRefresh": market_data_refresh,
            "marketDataRefreshedAt": market_data_refreshed_at,
            "signalInference": signal_inference_summary,
            "trackedTradeSync": tracked_trade_sync,
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
