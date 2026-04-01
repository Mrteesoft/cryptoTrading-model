"""Application-level classes for training, comparison, and signal generation."""

import json
from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Type

import pandas as pd

from .backtesting import BaseSignalBacktester, EqualWeightSignalBacktester
from .config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, TrainingConfig, config_to_dict, ensure_project_directories
from .data import CoinbaseExchangePriceDataLoader, CoinMarketCapContextEnricher
from .frontend import build_frontend_signal_snapshot
from .labels import create_labeler_from_config
from .modeling import BaseSignalModel, create_model_from_config, get_model_class
from .pipeline import CryptoDatasetBuilder
from .signals import (
    build_actionable_signal_summaries,
    build_latest_signal_summaries,
    select_primary_signal,
)


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

    def build_market_data_loader(self) -> CoinbaseExchangePriceDataLoader:
        """
        Create the configured real-market-data loader.

        For now the real-data path uses Coinbase Exchange public candles.
        Keeping this factory in the base app avoids rebuilding the same
        loader configuration in multiple places.
        """

        if self.config.market_data_source != "coinbaseExchange":
            raise ValueError(
                "Unsupported market_data_source. "
                "Currently supported: coinbaseExchange"
            )

        return CoinbaseExchangePriceDataLoader(
            data_path=self.config.data_file,
            product_id=self.config.coinbase_product_id,
            product_ids=self.config.coinbase_product_ids,
            fetch_all_quote_products=self.config.coinbase_fetch_all_quote_products,
            quote_currency=self.config.coinbase_quote_currency,
            excluded_base_currencies=self.config.coinbase_excluded_base_currencies,
            max_products=self.config.coinbase_max_products,
            product_batch_size=self.config.coinbase_product_batch_size,
            product_batch_number=self.config.coinbase_product_batch_number,
            granularity_seconds=self.config.coinbase_granularity_seconds,
            total_candles=self.config.coinbase_total_candles,
            request_pause_seconds=self.config.coinbase_request_pause_seconds,
            save_progress_every_products=self.config.coinbase_save_progress_every_products,
            log_progress=self.config.coinbase_log_progress,
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

        if not feature_importance_frames:
            return pd.DataFrame(columns=["feature", "importance"])

        combined_feature_importance_df = pd.concat(feature_importance_frames, ignore_index=True)
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

            # The saved model config is the safest source of truth because
            # it reflects the exact settings used during training.
            self.config = self.model.config
            self.dataset_builder = CryptoDatasetBuilder(
                config=self.config,
                feature_columns=self.model.feature_columns,
            )

        self._ensure_market_data_available()
        feature_df = self.dataset_builder.build_feature_table()
        prediction_df = self.model.predict(feature_df)
        latest_signals = build_latest_signal_summaries(prediction_df)
        actionable_signals = build_actionable_signal_summaries(latest_signals)
        top_signal = select_primary_signal(latest_signals)
        frontend_signal_snapshot = build_frontend_signal_snapshot(
            model_type=self.model.model_type,
            primary_signal=top_signal,
            latest_signals=latest_signals,
            actionable_signals=actionable_signals,
        )

        self.save_dataframe(prediction_df, OUTPUTS_DIR / "historicalSignals.csv")
        self.save_json(top_signal, OUTPUTS_DIR / "latestSignal.json")
        self.save_json({"signals": latest_signals}, OUTPUTS_DIR / "latestSignals.json")
        self.save_json({"signals": actionable_signals}, OUTPUTS_DIR / "actionableSignals.json")
        self.save_json(frontend_signal_snapshot, OUTPUTS_DIR / "frontendSignalSnapshot.json")

        return {
            "modelType": self.model.model_type,
            "historicalSignalsPath": str(OUTPUTS_DIR / "historicalSignals.csv"),
            "latestSignalPath": str(OUTPUTS_DIR / "latestSignal.json"),
            "latestSignalsPath": str(OUTPUTS_DIR / "latestSignals.json"),
            "actionableSignalsPath": str(OUTPUTS_DIR / "actionableSignals.json"),
            "frontendSignalSnapshotPath": str(OUTPUTS_DIR / "frontendSignalSnapshot.json"),
            "signalsGenerated": len(latest_signals),
            "actionableSignalsGenerated": len(actionable_signals),
            "signalName": top_signal["signal_name"],
            "confidence": top_signal["confidence"],
            "signalChat": top_signal["signalChat"],
        }


class ProductionCycleApp(BaseSignalApp):
    """Run the continuous refresh, retrain, and publish cycle in one command."""

    def run(self) -> Dict[str, Any]:
        """Refresh data, train the current model, and regenerate the frontend snapshot."""

        market_refresh_result = MarketDataRefreshApp(config=self.config).run()
        training_result = TrainingApp(config=self.config).run()
        signal_generation_result = SignalGenerationApp(config=self.config).run()

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

        data_loader = self.build_market_data_loader()
        price_df = data_loader.refresh_data()
        coinmarketcap_context_status = "disabled"
        coinmarketcap_context_rows = 0
        coinmarketcap_context_error = ""

        if self.config.coinmarketcap_use_context:
            context_enricher = self.build_coinmarketcap_context_enricher(
                should_refresh_context=self.config.coinmarketcap_refresh_context_after_market_refresh,
            )

            if self.config.coinmarketcap_refresh_context_after_market_refresh:
                try:
                    context_df = context_enricher.refresh_context(price_df)
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

        return {
            "marketDataSource": self.config.market_data_source,
            "productMode": (
                f"all-{self.config.coinbase_quote_currency.upper()}-quoted-products"
                if self.config.coinbase_fetch_all_quote_products
                else "explicit-product-list"
            ),
            "granularitySeconds": self.config.coinbase_granularity_seconds,
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
        start_batch = self.config.coinbase_product_batch_number

        if start_batch > total_batches:
            raise ValueError(
                "coinbase_product_batch_number is beyond the available batch count. "
                f"Start batch: {start_batch}, total batches: {total_batches}."
            )

        batch_results = []
        failed_batches = []
        for batch_number in range(start_batch, total_batches + 1):
            batch_config = replace(
                self.config,
                coinbase_product_batch_number=batch_number,
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
