"""Model training, evaluation, prediction, and persistence utilities."""

from abc import ABC, abstractmethod
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Dict, List, Tuple, Type

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
from .labels import signal_to_text


CLASS_ORDER = [-1, 0, 1]
CLASS_NAME_ORDER = ["TAKE_PROFIT", "HOLD", "BUY"]
MODEL_REGISTRY = {}


def register_model(model_class: Type["BaseSignalModel"]) -> Type["BaseSignalModel"]:
    """
    Register a model subclass so it can be restored from a saved file.

    We save the model type inside the pickle file.
    On load, this registry tells us which Python subclass to rebuild.
    """

    MODEL_REGISTRY[model_class.model_type] = model_class
    return model_class


def get_model_class(model_type: str) -> Type["BaseSignalModel"]:
    """
    Return the registered subclass for a model type string.

    This keeps model lookup in one place so the app layer does not need
    to know about every individual model subclass.
    """

    model_class = MODEL_REGISTRY.get(model_type)
    if model_class is None:
        available_models = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available model types: {available_models}"
        )

    return model_class


class BaseSignalModel(ABC):
    """
    Shared behavior for every signal model in the project.

    The DRY rule matters here because most ML models need the same steps:
    split, fit, evaluate, predict, save, and load.
    A subclass should only care about the model-specific estimator setup.
    """

    model_type = "baseSignalModel"
    default_model_filename = "signalModel.pkl"

    def __init__(self, config: TrainingConfig, feature_columns: List[str] = None) -> None:
        self.config = config
        self.feature_columns = feature_columns or []
        self.estimator = None

    @staticmethod
    def split_train_test_by_time(
        dataset: pd.DataFrame,
        train_size: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into train and test sections without shuffling.

        Time-series data must keep its order.
        Training on future data and testing on past data would leak information.
        """

        if not 0.50 <= train_size < 1.0:
            raise ValueError("train_size must be between 0.50 and 0.99 for a useful time split.")

        split_index = int(len(dataset) * train_size)
        if split_index <= 0 or split_index >= len(dataset):
            raise ValueError("The time split created an empty train or test set.")

        train_df = dataset.iloc[:split_index].copy()
        test_df = dataset.iloc[split_index:].copy()

        return train_df, test_df

    @staticmethod
    def split_walk_forward_by_time(
        dataset: pd.DataFrame,
        min_train_size: float,
        test_size: float,
        step_size: float,
        purge_gap_timestamps: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Build expanding walk-forward folds using whole timestamps.

        This is stricter than a simple row split because all rows that share
        the same timestamp stay in the same fold. That prevents one coin at a
        timestamp from leaking into train while another coin at that very same
        market moment lands in test.
        """

        if "timestamp" not in dataset.columns:
            raise ValueError("Walk-forward validation requires a timestamp column.")

        for field_name, field_value in {
            "walkforward_min_train_size": min_train_size,
            "walkforward_test_size": test_size,
            "walkforward_step_size": step_size,
        }.items():
            if not 0 < field_value < 1:
                raise ValueError(f"{field_name} must be between 0 and 1.")
        if purge_gap_timestamps < 0:
            raise ValueError("purge_gap_timestamps must be 0 or greater.")

        prepared_df = dataset.copy()
        prepared_df["timestamp"] = pd.to_datetime(prepared_df["timestamp"], errors="coerce", utc=True)
        if prepared_df["timestamp"].isna().any():
            raise ValueError("Walk-forward validation found invalid timestamps in the dataset.")

        sort_columns = ["timestamp"]
        if "product_id" in prepared_df.columns:
            sort_columns.append("product_id")
        prepared_df = prepared_df.sort_values(sort_columns).reset_index(drop=True)

        unique_timestamps = pd.Index(prepared_df["timestamp"].drop_duplicates()).sort_values()
        total_timestamp_count = len(unique_timestamps)
        if total_timestamp_count < 3:
            raise ValueError("Walk-forward validation needs at least 3 timestamps.")

        min_train_timestamp_count = max(1, int(total_timestamp_count * min_train_size))
        test_timestamp_count = max(1, int(total_timestamp_count * test_size))
        step_timestamp_count = max(1, int(total_timestamp_count * step_size))

        if min_train_timestamp_count + test_timestamp_count > total_timestamp_count:
            raise ValueError(
                "Walk-forward settings leave no room for a test fold. "
                "Reduce walkforward_min_train_size or walkforward_test_size."
            )

        walk_forward_splits: List[Dict[str, Any]] = []
        fold_number = 1
        train_end_index = min_train_timestamp_count

        while train_end_index + test_timestamp_count <= total_timestamp_count:
            effective_train_end_index = train_end_index - purge_gap_timestamps
            if effective_train_end_index <= 0:
                raise ValueError(
                    "purge_gap_timestamps is too large for the current walk-forward settings. "
                    "Reduce the purge gap or increase the minimum train size."
                )

            train_end_timestamp = unique_timestamps[effective_train_end_index - 1]
            test_start_timestamp = unique_timestamps[train_end_index]
            test_end_timestamp = unique_timestamps[train_end_index + test_timestamp_count - 1]

            train_df = prepared_df.loc[
                prepared_df["timestamp"] <= train_end_timestamp
            ].copy()
            test_df = prepared_df.loc[
                (prepared_df["timestamp"] >= test_start_timestamp)
                & (prepared_df["timestamp"] <= test_end_timestamp)
            ].copy()

            if train_df.empty or test_df.empty:
                raise ValueError(
                    "Walk-forward validation created an empty train or test fold. "
                    "Adjust the fold sizes."
                )

            walk_forward_splits.append(
                {
                    "fold_number": fold_number,
                    "train_df": train_df,
                    "test_df": test_df,
                    "train_timestamp_count": int(train_df["timestamp"].nunique()),
                    "test_timestamp_count": int(test_df["timestamp"].nunique()),
                    "train_start_timestamp": train_df.iloc[0]["timestamp"],
                    "train_end_timestamp": train_end_timestamp,
                    "test_start_timestamp": test_start_timestamp,
                    "test_end_timestamp": test_end_timestamp,
                    "purgeGapTimestamps": int(purge_gap_timestamps),
                    "purgedTimestampCount": int(purge_gap_timestamps),
                }
            )

            train_end_index += step_timestamp_count
            fold_number += 1

        if not walk_forward_splits:
            raise ValueError("Walk-forward validation could not create any folds.")

        return walk_forward_splits

    @staticmethod
    def build_classification_metrics(
        actual_signals: pd.Series,
        predicted_signals: Any,
    ) -> Dict[str, Any]:
        """
        Build reusable classification metrics from actual and predicted labels.

        This keeps normal train/test evaluation and walk-forward validation
        aligned so both workflows report the same metric definitions.
        """

        return {
            "accuracy": float(accuracy_score(actual_signals, predicted_signals)),
            "balanced_accuracy": float(balanced_accuracy_score(actual_signals, predicted_signals)),
            "class_names": CLASS_NAME_ORDER,
            "confusion_matrix": confusion_matrix(
                actual_signals,
                predicted_signals,
                labels=CLASS_ORDER,
            ).tolist(),
            "classification_report": classification_report(
                actual_signals,
                predicted_signals,
                labels=CLASS_ORDER,
                target_names=CLASS_NAME_ORDER,
                zero_division=0,
                output_dict=True,
            ),
        }

    @staticmethod
    def build_signal_distribution(signal_values: pd.Series) -> Dict[str, int]:
        """Count how many TAKE_PROFIT, HOLD, and BUY labels appear in a series."""

        signal_series = pd.Series(signal_values)

        return {
            class_name: int((signal_series == class_label).sum())
            for class_label, class_name in zip(CLASS_ORDER, CLASS_NAME_ORDER)
        }

    @abstractmethod
    def _build_estimator(self) -> Any:
        """Create the underlying scikit-learn estimator for the subclass."""

    def fit(self, train_df: pd.DataFrame) -> "BaseSignalModel":
        """
        Train the estimator using the configured feature columns.

        The subclass chooses which estimator to build.
        The base class handles the repeated training flow.
        """

        self._validate_feature_columns()
        self.estimator = self._build_estimator()
        feature_frame = train_df[self.feature_columns]
        target_series = train_df["target_signal"]
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
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evaluate the trained model on the test split.

        This method is shared because evaluation should work the same way
        regardless of which concrete model subclass we plug in later.
        """

        self._validate_trained_model()

        predictions = self.estimator.predict(test_df[self.feature_columns])
        probabilities = self.estimator.predict_proba(test_df[self.feature_columns])

        prediction_df = test_df.copy()
        prediction_df = self._attach_prediction_columns(
            base_df=prediction_df,
            predictions=predictions,
            probabilities=probabilities,
        )
        prediction_df["prediction_correct"] = (
            prediction_df["predicted_signal"] == prediction_df["target_signal"]
        )

        metrics = {
            "model_type": self.model_type,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            **self.build_classification_metrics(
                actual_signals=test_df["target_signal"],
                predicted_signals=predictions,
            ),
            "feature_importance": self.get_feature_importance(),
            "train_class_distribution": self.build_signal_distribution(train_df["target_signal"]),
            "test_class_distribution": self.build_signal_distribution(test_df["target_signal"]),
        }

        return prediction_df, metrics

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict signals for rows that have complete feature values.

        Prediction code is shared here so subclasses do not repeat
        the same probability and confidence logic.
        """

        self._validate_trained_model()

        usable_rows = feature_df.dropna(subset=self.feature_columns).copy()
        predictions = self.estimator.predict(usable_rows[self.feature_columns])
        probabilities = self.estimator.predict_proba(usable_rows[self.feature_columns])

        usable_rows = self._attach_prediction_columns(
            base_df=usable_rows,
            predictions=predictions,
            probabilities=probabilities,
        )

        return usable_rows

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance in descending order when the estimator supports it.

        Some models will not have `feature_importances_`.
        In that case we return an empty dictionary instead of crashing.
        """

        self._validate_trained_model()

        if not hasattr(self.estimator, "feature_importances_"):
            return {}

        return {
            feature_name: float(importance)
            for feature_name, importance in sorted(
                zip(self.feature_columns, self.estimator.feature_importances_),
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
        """
        Save the trained model and metadata needed for future inference.

        The bundle stores:
        - which subclass created the model
        - the learned estimator
        - the feature column list
        - the training config
        """

        self._validate_trained_model()

        model_bundle = {
            "model_type": self.model_type,
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
    def load(cls, model_path: Path) -> "BaseSignalModel":
        """
        Load a saved model bundle and restore the correct subclass.
        """

        with model_path.open("rb") as model_file:
            model_bundle = pickle.load(model_file)

        model_type = model_bundle["model_type"]
        model_class = MODEL_REGISTRY.get(model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type found in saved file: {model_type}")

        config = dict_to_config(model_bundle["config"])
        feature_columns = list(model_bundle["feature_columns"])

        model_instance = model_class(config=config, feature_columns=feature_columns)
        model_instance.estimator = model_bundle["model"]

        return model_instance

    def _attach_prediction_columns(
        self,
        base_df: pd.DataFrame,
        predictions: Any,
        probabilities: Any,
    ) -> pd.DataFrame:
        """
        Add predicted class names and probability columns to a DataFrame.

        This keeps evaluation and inference consistent because both rely
        on exactly the same output-shaping logic.
        """

        output_df = base_df.copy()
        output_df["predicted_signal"] = predictions
        output_df["predicted_name"] = output_df["predicted_signal"].map(signal_to_text)

        output_df = self._add_probability_columns(
            base_df=output_df,
            class_labels=list(self.estimator.classes_),
            probability_values=probabilities,
        )
        output_df["confidence"] = output_df[["prob_take_profit", "prob_hold", "prob_buy"]].max(axis=1)

        return output_df

    def _add_probability_columns(
        self,
        base_df: pd.DataFrame,
        class_labels: List[int],
        probability_values: Any,
    ) -> pd.DataFrame:
        """
        Add probability columns in a stable order.

        Some models may not output classes in the order we expect, so we map
        them by class label instead of assuming the first column is TAKE_PROFIT.
        """

        output_df = base_df.copy()
        output_df["prob_take_profit"] = 0.0
        output_df["prob_hold"] = 0.0
        output_df["prob_buy"] = 0.0

        for class_label, class_probability in zip(class_labels, probability_values.T):
            if class_label == -1:
                output_df["prob_take_profit"] = class_probability
            elif class_label == 0:
                output_df["prob_hold"] = class_probability
            elif class_label == 1:
                output_df["prob_buy"] = class_probability

        return output_df

    def _validate_feature_columns(self) -> None:
        """Make sure training knows which columns to read."""

        if not self.feature_columns:
            raise ValueError("feature_columns is empty. Pass the feature list before training the model.")

    def _validate_trained_model(self) -> None:
        """Make sure prediction-only methods are not called too early."""

        if self.estimator is None:
            raise ValueError("The model has not been trained or loaded yet.")

    def _fit_estimator(
        self,
        feature_frame: pd.DataFrame,
        target_series: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit the underlying estimator, passing sample weights when supported."""

        if sample_weight is None:
            self.estimator.fit(feature_frame, target_series)
            return

        try:
            self.estimator.fit(feature_frame, target_series, sample_weight=sample_weight)
        except TypeError:
            self.estimator.fit(feature_frame, target_series)

    def _build_sample_weight(self, train_df: pd.DataFrame) -> np.ndarray | None:
        """
        Weight recent rows more heavily so the model tracks current market regimes.

        Crypto changes fast, so older observations should still contribute but
        gradually matter less than newer candles.
        """

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

        return recency_weights.to_numpy(dtype=float)


@register_model
class RandomForestSignalModel(BaseSignalModel):
    """
    Random forest implementation of the signal model.

    This subclass only handles random-forest-specific creation.
    All generic training, evaluation, saving, and prediction behavior
    stays in BaseSignalModel.
    """

    model_type = "randomForestSignalModel"
    default_model_filename = "randomForestSignalModel.pkl"

    def _build_estimator(self) -> RandomForestClassifier:
        """
        Create the random forest estimator using the project config.

        Why random forest first?
        - it handles non-linear patterns well
        - it works on mixed feature scales without mandatory standardisation
        - it is easier to explain than a deep learning model for a first version
        """

        return RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            class_weight="balanced_subsample",
        )


@register_model
class HistGradientBoostingSignalModel(BaseSignalModel):
    """
    Histogram gradient boosting tuned for tabular market features.

    This is usually a stronger default than plain logistic regression and often
    more adaptive than a basic random forest on structured data.
    """

    model_type = "histGradientBoostingSignalModel"
    default_model_filename = "histGradientBoostingSignalModel.pkl"

    def _build_estimator(self) -> HistGradientBoostingClassifier:
        """Create the boosted-tree estimator from the project config."""

        return HistGradientBoostingClassifier(
            learning_rate=self.config.hist_gradient_learning_rate,
            max_iter=self.config.hist_gradient_max_iter,
            max_depth=self.config.hist_gradient_max_depth,
            min_samples_leaf=self.config.hist_gradient_min_samples_leaf,
            l2_regularization=self.config.hist_gradient_l2_regularization,
            random_state=self.config.random_state,
        )

    def _build_sample_weight(self, train_df: pd.DataFrame) -> np.ndarray | None:
        """
        Combine recency weighting with simple class balancing for boosted trees.

        HistGradientBoostingClassifier does not expose a class_weight argument,
        so sample weights carry both effects.
        """

        recency_weight = super()._build_sample_weight(train_df)
        target_series = pd.Series(train_df["target_signal"])
        class_counts = target_series.value_counts()
        class_count_total = max(len(class_counts), 1)
        class_weight = target_series.map(
            lambda class_label: len(target_series) / (class_count_total * class_counts[class_label])
        ).to_numpy(dtype=float)

        if recency_weight is None:
            return class_weight

        return class_weight * recency_weight


@register_model
class LogisticRegressionSignalModel(BaseSignalModel):
    """
    Logistic regression implementation of the signal model.

    This subclass uses feature scaling internally because logistic regression
    usually behaves better when numeric features are on similar scales.
    """

    model_type = "logisticRegressionSignalModel"
    default_model_filename = "logisticRegressionSignalModel.pkl"

    def _build_estimator(self) -> Pipeline:
        """
        Create a scaled logistic regression pipeline from the config.

        We use a scikit-learn Pipeline so scaling happens as part of the
        model itself. That means the same steps are automatically reused
        during both training and prediction.
        """

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

    def _fit_estimator(
        self,
        feature_frame: pd.DataFrame,
        target_series: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit the pipeline, routing sample weights to the classifier step."""

        if sample_weight is None:
            self.estimator.fit(feature_frame, target_series)
            return

        self.estimator.fit(
            feature_frame,
            target_series,
            classifier__sample_weight=sample_weight,
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Estimate feature importance from logistic regression coefficients.

        Logistic regression does not expose `feature_importances_` like
        tree-based models do, so we derive importance from the absolute
        magnitude of the learned coefficients.
        """

        self._validate_trained_model()

        classifier = self.estimator.named_steps["classifier"]
        coefficient_matrix = np.abs(classifier.coef_)
        average_importance = coefficient_matrix.mean(axis=0)

        return {
            feature_name: float(importance)
            for feature_name, importance in sorted(
                zip(self.feature_columns, average_importance),
                key=lambda item: item[1],
                reverse=True,
            )
        }


def create_model_from_config(
    config: TrainingConfig,
    feature_columns: List[str] = None,
) -> BaseSignalModel:
    """
    Create the correct model subclass from the config.

    The app layer calls this helper instead of hardcoding a specific model,
    which keeps model selection DRY and centralized.
    """

    model_class = get_model_class(config.model_type)
    return model_class(config=config, feature_columns=feature_columns)


def split_train_test_by_time(dataset: pd.DataFrame, train_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible helper that delegates to the base model class.
    """

    return BaseSignalModel.split_train_test_by_time(dataset, train_size)


def split_walk_forward_by_time(
    dataset: pd.DataFrame,
    min_train_size: float,
    test_size: float,
    step_size: float,
    purge_gap_timestamps: int = 0,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible helper that delegates to the base model class.
    """

    return BaseSignalModel.split_walk_forward_by_time(
        dataset=dataset,
        min_train_size=min_train_size,
        test_size=test_size,
        step_size=step_size,
        purge_gap_timestamps=purge_gap_timestamps,
    )
