"""Training-oriented utilities for the staged modeling core."""

from __future__ import annotations

from dataclasses import replace

from ..config import TrainingConfig


def with_model_type(config: TrainingConfig, model_type: str | None) -> TrainingConfig:
    """Clone the active config with one explicit model type when requested."""

    if model_type is None:
        return config

    return replace(
        config,
        model_type=model_type,
        signal_model_family="baseline_current",
        signal_model_variant="default",
    )
