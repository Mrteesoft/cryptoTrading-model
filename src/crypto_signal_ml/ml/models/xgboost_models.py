"""XGBoost-backed signal model implementations."""

from __future__ import annotations

from typing import Any

from ...modeling import BaseSignalModel, register_model


def _require_xgboost() -> Any:
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for XGBoost models. "
            "Install it with `pip install xgboost`."
        ) from exc
    return xgb


@register_model
class XGBoostClassifierSignalModel(BaseSignalModel):
    """XGBoost classifier for tabular signal prediction."""

    model_type = "xgboostClassifierSignalModel"
    default_model_filename = "xgboostClassifierSignalModel.pkl"

    def _build_estimator(self):
        xgb = _require_xgboost()
        return xgb.XGBClassifier(
            n_estimators=int(self.config.xgboost_n_estimators),
            learning_rate=float(self.config.xgboost_learning_rate),
            max_depth=int(self.config.xgboost_max_depth),
            min_child_weight=float(self.config.xgboost_min_child_weight),
            subsample=float(self.config.xgboost_subsample),
            colsample_bytree=float(self.config.xgboost_colsample_bytree),
            gamma=float(self.config.xgboost_gamma),
            objective="multi:softprob",
            num_class=3,
            random_state=int(self.config.random_state),
            eval_metric="mlogloss",
        )
