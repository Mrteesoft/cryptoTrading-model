"""Registry helpers for multi-family signal model selection."""

from __future__ import annotations

from typing import Dict


MODEL_FAMILY_REGISTRY: Dict[str, str] = {
    "baseline_current:default": "baseline_current",
    "lightgbm:classifier": "lightgbmClassifierSignalModel",
    "lightgbm:ranker": "lightgbmRankerSignalModel",
    "xgboost:classifier": "xgboostClassifierSignalModel",
    "river:online_scorer": "riverOnlineSignalModel",
    "tft:sequence": "tftSequenceSignalModel",
}

MODEL_FAMILY_ALIASES: Dict[str, str] = {
    "baseline_current": "baseline_current:default",
    "lightgbm_classifier": "lightgbm:classifier",
    "lightgbm_ranker": "lightgbm:ranker",
    "xgboost_classifier": "xgboost:classifier",
    "river_online_scorer": "river:online_scorer",
    "tft_sequence_model": "tft:sequence",
}

_REGISTRY_LOADED = False


def ensure_registry_loaded() -> None:
    """Import model-family modules so they register themselves."""

    global _REGISTRY_LOADED
    if _REGISTRY_LOADED:
        return

    from . import lightgbm_models  # noqa: F401
    from . import xgboost_models  # noqa: F401
    from . import river_models  # noqa: F401
    from . import pytorch_forecasting_models  # noqa: F401

    _REGISTRY_LOADED = True


def resolve_model_type(config) -> str:
    """Resolve the configured model family + variant into a model type string."""

    family = str(getattr(config, "signal_model_family", "") or "baseline_current").strip().lower()
    variant = str(getattr(config, "signal_model_variant", "") or "default").strip().lower()

    if family in MODEL_FAMILY_ALIASES:
        family_key = MODEL_FAMILY_ALIASES[family]
    else:
        family_key = f"{family}:{variant}"

    if family_key == "baseline_current:default":
        return str(getattr(config, "model_type"))

    model_type = MODEL_FAMILY_REGISTRY.get(family_key)
    if model_type is None:
        available = ", ".join(sorted(MODEL_FAMILY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model family selection: {family_key}. "
            f"Available model families: {available}"
        )

    return model_type
