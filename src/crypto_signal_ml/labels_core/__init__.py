"""Labeling contracts, ATR-aware labelers, and diagnostics."""

from .contracts import LabelRecipe
from .diagnostics import build_label_diagnostics
from .triple_barrier import (
    BaseSignalLabeler,
    FutureReturnSignalLabeler,
    MarketRegimeLabeler,
    TripleBarrierSignalLabeler,
    add_regime_labels,
    add_signal_labels,
    create_labeler_from_config,
    create_regime_labeler_from_config,
    signal_to_text,
)

__all__ = [
    "BaseSignalLabeler",
    "FutureReturnSignalLabeler",
    "LabelRecipe",
    "MarketRegimeLabeler",
    "TripleBarrierSignalLabeler",
    "add_regime_labels",
    "add_signal_labels",
    "build_label_diagnostics",
    "create_labeler_from_config",
    "create_regime_labeler_from_config",
    "signal_to_text",
]
