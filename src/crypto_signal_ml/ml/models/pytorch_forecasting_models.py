"""PyTorch Forecasting / TFT experiment scaffolding."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...modeling import BaseSignalModel, register_model


def _require_tft() -> Any:
    try:
        import torch  # noqa: F401
        import pytorch_forecasting  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pytorch-forecasting and torch are required for TFT models. "
            "Install them with `pip install torch pytorch-forecasting`."
        ) from exc

    return True


@register_model
class TFTSequenceSignalModel(BaseSignalModel):
    """Experimental Temporal Fusion Transformer placeholder."""

    model_type = "tftSequenceSignalModel"
    default_model_filename = "tftSequenceSignalModel.pkl"

    def _build_estimator(self):
        _require_tft()
        return None

    def fit(self, train_df: pd.DataFrame) -> "TFTSequenceSignalModel":
        if not bool(getattr(self.config, "enable_tft_experiments", False)):
            raise ValueError("TFT experiments are disabled. Set ENABLE_TFT_EXPERIMENTS=true to proceed.")
        _require_tft()
        raise NotImplementedError(
            "TFT training requires a dedicated sequence dataset. "
            "Use a TFT-specific dataset builder before enabling this model in production."
        )

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("TFT prediction requires a sequence dataset adapter.")

    def predict_proba(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("TFT prediction requires a sequence dataset adapter.")

    def rank(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("TFT ranking requires a sequence dataset adapter.")

    def metadata(self) -> dict[str, Any]:
        return {
            "modelType": self.model_type,
            "featureCount": len(self.feature_columns),
            "featurePreview": list(self.feature_columns[:10]),
            "supportsRank": False,
            "supportsProba": False,
            "experimental": True,
        }
