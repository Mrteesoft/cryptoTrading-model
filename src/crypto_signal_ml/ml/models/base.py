"""Common model contract for multi-family signal models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class SignalModelContract(ABC):
    """Minimal contract that all signal model families must follow."""

    @abstractmethod
    def fit(self, train_frame: pd.DataFrame, valid_frame: pd.DataFrame | None = None) -> "SignalModelContract":
        """Train the model on the provided frame."""

    @abstractmethod
    def predict(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Predict signals for the provided feature frame."""

    @abstractmethod
    def predict_proba(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Return probability outputs when supported."""

    @abstractmethod
    def rank(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Return ranking scores when supported."""

    @abstractmethod
    def save(self, path) -> None:
        """Persist the model bundle."""

    @classmethod
    @abstractmethod
    def load(cls, path) -> "SignalModelContract":
        """Restore a previously saved model bundle."""

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return a JSON-friendly metadata summary."""
