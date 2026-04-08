"""Model family registry and implementations for multi-model experimentation."""

from .base import SignalModelContract
from .registry import (
    MODEL_FAMILY_ALIASES,
    MODEL_FAMILY_REGISTRY,
    ensure_registry_loaded,
    resolve_model_type,
)
from .lightgbm_models import LightGBMClassifierSignalModel, LightGBMRankerSignalModel
from .pytorch_forecasting_models import TFTSequenceSignalModel
from .river_models import RiverOnlineSignalModel
from .xgboost_models import XGBoostClassifierSignalModel

__all__ = [
    "MODEL_FAMILY_ALIASES",
    "MODEL_FAMILY_REGISTRY",
    "SignalModelContract",
    "ensure_registry_loaded",
    "resolve_model_type",
    "LightGBMClassifierSignalModel",
    "LightGBMRankerSignalModel",
    "XGBoostClassifierSignalModel",
    "RiverOnlineSignalModel",
    "TFTSequenceSignalModel",
]
