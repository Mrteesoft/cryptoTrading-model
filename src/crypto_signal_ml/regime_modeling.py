"""Dedicated modeling utilities for market-regime classification."""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import TrainingConfig, config_to_dict, dict_to_config
from .modeling import BaseSignalModel
from .regimes import REGIME_LABEL_TO_CODE


REGIME_CODE_TO_LABEL = {
    code: label
    for label, code in REGIME_LABEL_TO_CODE.items()
}


class MarketRegimeModel:
    """Train, evaluate, and persist a dedicated market-regime classifier."""

    model_type = "marketRegimeModel"
    default_model_filename = "marketRegimeModel.pkl"
    target_column = "target_market_regime_code"
    target_label_column = "target_market_regime_label"

    def __init__(self, config: TrainingConfig, feature_columns: List[str] | None = None) -> None:
        self.config = config
        self.feature_columns = feature_columns or []
        self.estimator = None
        self.estimator_type = str(config.model_type)

    @staticmethod
    def split_train_test_by_time(dataset: pd.DataFrame, train_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Delegate the time split logic to the shared signal-model helper."""

        return BaseSignalModel.split_train_test_by_time(dataset=dataset, train_size=train_size)

    @staticmethod
    def split_walk_forward_by_time(
        dataset: pd.DataFrame,
        min_train_size: float,
        test_size: float,
        step_size: float,
        purge_gap_timestamps: int = 0,
    ) -> List[Dict[str, Any]]:
        """Delegate walk-forward split creation to the shared signal-model helper."""

        return BaseSignalModel.split_walk_forward_by_time(
            dataset=dataset,
            min_train_size=min_train_size,
            test_size=test_size,
            step_size=step_size,
            purge_gap_timestamps=purge_gap_timestamps,
        )

    def fit(self, train_df: pd.DataFrame) -> "MarketRegimeModel":
        """Fit the classifier on the configured feature columns and regime target."""

        self._validate_feature_columns()
        self._validate_target_column(train_df)
        self.estimator = self._build_estimator()
        feature_frame = train_df[self.feature_columns]
        target_series = pd.to_numeric(train_df[self.target_column], errors="coerce").astype(int)
        sample_weight = self._build_sample_weight(train_df)
        self._fit_estimator(
            feature_frame=feature_frame,
            target_series=target_series,
            sample_weight=sample_weight,
        )
        return self

    def evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Evaluate the trained regime model on a held-out time split."""

        self._validate_trained_model()
        self._validate_target_column(test_df)

        predictions = self.estimator.predict(test_df[self.feature_columns])
        probabilities = self.estimator.predict_proba(test_df[self.feature_columns])

        prediction_df = test_df.copy()
        prediction_df = self._attach_prediction_columns(
            base_df=prediction_df,
            predictions=predictions,
            probabilities=probabilities,
        )
        prediction_df["prediction_correct"] = (
            prediction_df["predicted_market_regime_code"] == prediction_df[self.target_column].astype(int)
        )

        metrics = {
            "model_type": self.model_type,
            "estimator_type": self.estimator_type,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            **self.build_classification_metrics(
                actual_labels=test_df[self.target_column],
                predicted_labels=predictions,
            ),
            "feature_importance": self.get_feature_importance(),
            "train_class_distribution": self.build_class_distribution(train_df[self.target_column]),
            "test_class_distribution": self.build_class_distribution(test_df[self.target_column]),
        }

        return prediction_df, metrics

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Predict the next regime state for rows with complete feature values."""

        self._validate_trained_model()

        usable_rows = feature_df.dropna(subset=self.feature_columns).copy()
        predictions = self.estimator.predict(usable_rows[self.feature_columns])
        probabilities = self.estimator.predict_proba(usable_rows[self.feature_columns])

        return self._attach_prediction_columns(
            base_df=usable_rows,
            predictions=predictions,
            probabilities=probabilities,
        )

    @staticmethod
    def build_classification_metrics(
        actual_labels: pd.Series,
        predicted_labels: Any,
    ) -> Dict[str, Any]:
        """Build accuracy and per-class metrics for market-regime predictions."""

        actual_series = pd.Series(pd.to_numeric(actual_labels, errors="coerce")).dropna().astype(int)
        predicted_series = pd.Series(pd.to_numeric(predicted_labels, errors="coerce")).dropna().astype(int)
        label_codes = sorted(int(label_code) for label_code in (set(actual_series.unique()) | set(predicted_series.unique())))
        label_names = [REGIME_CODE_TO_LABEL.get(code, f"code_{code}") for code in label_codes]

        return {
            "accuracy": float(accuracy_score(actual_series, predicted_series)),
            "balanced_accuracy": float(balanced_accuracy_score(actual_series, predicted_series)),
            "class_codes": label_codes,
            "class_names": label_names,
            "confusion_matrix": confusion_matrix(
                actual_series,
                predicted_series,
                labels=label_codes,
            ).tolist(),
            "classification_report": classification_report(
                actual_series,
                predicted_series,
                labels=label_codes,
                target_names=label_names,
                zero_division=0,
                output_dict=True,
            ),
        }

    @staticmethod
    def build_class_distribution(label_values: pd.Series) -> Dict[str, int]:
        """Count how often each regime label appears in a target series."""

        label_series = pd.Series(pd.to_numeric(label_values, errors="coerce")).dropna().astype(int)
        class_counter = label_series.value_counts().sort_index()

        return {
            REGIME_CODE_TO_LABEL.get(int(class_code), f"code_{int(class_code)}"): int(count)
            for class_code, count in class_counter.items()
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance when the configured estimator exposes it."""

        self._validate_trained_model()

        if hasattr(self.estimator, "feature_importances_"):
            importance_values = self.estimator.feature_importances_
        elif isinstance(self.estimator, Pipeline) and "classifier" in self.estimator.named_steps:
            classifier = self.estimator.named_steps["classifier"]
            coefficient_matrix = np.abs(classifier.coef_)
            importance_values = coefficient_matrix.mean(axis=0)
        else:
            return {}

        return {
            feature_name: float(importance)
            for feature_name, importance in sorted(
                zip(self.feature_columns, importance_values),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    def get_feature_importance_frame(self) -> pd.DataFrame:
        """Return feature importance as a DataFrame for CSV export."""

        return pd.DataFrame(
            [
                {"feature": feature_name, "importance": importance}
                for feature_name, importance in self.get_feature_importance().items()
            ]
        )

    def save(self, model_path: Path) -> None:
        """Persist the trained regime model and the metadata needed to reload it."""

        self._validate_trained_model()

        model_bundle = {
            "model_type": self.model_type,
            "estimator_type": self.estimator_type,
            "model": self.estimator,
            "feature_columns": self.feature_columns,
            "config": config_to_dict(self.config),
        }

        model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file_descriptor, temp_file_name = tempfile.mkstemp(
            dir=model_path.parent,
            prefix=f".{model_path.stem}-",
            suffix=".tmp",
        )
        os.close(temp_file_descriptor)
        temp_path = Path(temp_file_name)

        try:
            with temp_path.open("wb") as model_file:
                pickle.dump(model_bundle, model_file)
            temp_path.replace(model_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @classmethod
    def load(cls, model_path: Path) -> "MarketRegimeModel":
        """Load a saved regime model artifact."""

        with model_path.open("rb") as model_file:
            model_bundle = pickle.load(model_file)

        config = dict_to_config(model_bundle["config"])
        feature_columns = list(model_bundle["feature_columns"])
        model_instance = cls(config=config, feature_columns=feature_columns)
        model_instance.estimator = model_bundle["model"]
        model_instance.estimator_type = str(model_bundle.get("estimator_type") or config.model_type)
        return model_instance

    def _attach_prediction_columns(
        self,
        base_df: pd.DataFrame,
        predictions: Any,
        probabilities: Any,
    ) -> pd.DataFrame:
        """Attach predicted regime labels and class probabilities to a DataFrame."""

        output_df = base_df.copy()
        predicted_codes = pd.Series(predictions, index=output_df.index).astype(int)
        output_df["predicted_market_regime_code"] = predicted_codes
        output_df["predicted_market_regime_label"] = predicted_codes.map(
            lambda code: REGIME_CODE_TO_LABEL.get(int(code), "unknown")
        )

        probability_columns = []
        for class_label, class_probability in zip(self.estimator.classes_, probabilities.T):
            class_code = int(class_label)
            class_name = REGIME_CODE_TO_LABEL.get(class_code, f"code_{class_code}")
            column_name = f"prob_regime_{class_name}"
            output_df[column_name] = class_probability
            probability_columns.append(column_name)

        output_df["regime_confidence"] = output_df[probability_columns].max(axis=1)
        return output_df

    def _build_estimator(self) -> Any:
        """Create the configured estimator for regime classification."""

        if self.estimator_type == "randomForestSignalModel":
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state,
                class_weight="balanced_subsample",
            )

        if self.estimator_type == "logisticRegressionSignalModel":
            return Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=self.config.logistic_c,
                            max_iter=self.config.logistic_max_iter,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )

        if self.estimator_type != "histGradientBoostingSignalModel":
            raise ValueError(
                "Unsupported estimator type for MarketRegimeModel. "
                "Use histGradientBoostingSignalModel, randomForestSignalModel, or logisticRegressionSignalModel."
            )

        return HistGradientBoostingClassifier(
            learning_rate=self.config.hist_gradient_learning_rate,
            max_iter=self.config.hist_gradient_max_iter,
            max_depth=self.config.hist_gradient_max_depth,
            min_samples_leaf=self.config.hist_gradient_min_samples_leaf,
            l2_regularization=self.config.hist_gradient_l2_regularization,
            random_state=self.config.random_state,
        )

    def _fit_estimator(
        self,
        feature_frame: pd.DataFrame,
        target_series: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit the estimator, routing sample weights when supported."""

        if sample_weight is None:
            self.estimator.fit(feature_frame, target_series)
            return

        if isinstance(self.estimator, Pipeline):
            self.estimator.fit(
                feature_frame,
                target_series,
                classifier__sample_weight=sample_weight,
            )
            return

        try:
            self.estimator.fit(feature_frame, target_series, sample_weight=sample_weight)
        except TypeError:
            self.estimator.fit(feature_frame, target_series)

    def _build_sample_weight(self, train_df: pd.DataFrame) -> np.ndarray | None:
        """Apply the same recency weighting logic used by the main signal model."""

        if not self.config.recency_weighting_enabled:
            return None
        if "timestamp" not in train_df.columns:
            return None

        timestamp_series = pd.to_datetime(train_df["timestamp"], errors="coerce", utc=True)
        if timestamp_series.isna().all():
            return None

        latest_timestamp = timestamp_series.max()
        halflife_hours = max(float(self.config.recency_weighting_halflife_hours), 1.0)
        age_hours = (latest_timestamp - timestamp_series).dt.total_seconds().fillna(0.0) / 3600.0
        recency_weights = np.power(0.5, age_hours / halflife_hours)

        if self.estimator_type == "histGradientBoostingSignalModel" and self.target_column in train_df.columns:
            target_series = pd.Series(pd.to_numeric(train_df[self.target_column], errors="coerce")).dropna().astype(int)
            class_counts = target_series.value_counts()
            class_count_total = max(len(class_counts), 1)
            class_weight = pd.Series(
                pd.to_numeric(train_df[self.target_column], errors="coerce"),
                index=train_df.index,
            ).map(
                lambda class_label: len(target_series) / (class_count_total * class_counts[int(class_label)])
                if pd.notna(class_label) and int(class_label) in class_counts
                else 1.0
            )
            return class_weight.to_numpy(dtype=float) * recency_weights.to_numpy(dtype=float)

        return recency_weights.to_numpy(dtype=float)

    def _validate_feature_columns(self) -> None:
        """Ensure the feature list is present before training or inference."""

        if not self.feature_columns:
            raise ValueError("feature_columns is empty. Pass the feature list before training the regime model.")

    def _validate_target_column(self, dataset: pd.DataFrame) -> None:
        """Ensure the regime target column exists before supervised training."""

        if self.target_column not in dataset.columns:
            raise ValueError(
                "target_market_regime_code is missing from the dataset. "
                "Run the regime labeler before training the regime model."
            )

    def _validate_trained_model(self) -> None:
        """Guard prediction-only operations until a model has been fit or loaded."""

        if self.estimator is None:
            raise ValueError("The regime model has not been trained or loaded yet.")
