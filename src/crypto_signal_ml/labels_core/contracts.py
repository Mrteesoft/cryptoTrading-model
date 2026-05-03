"""Typed label-recipe contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LabelRecipe:
    """Versioned recipe that describes how supervised labels were built."""

    version: str
    strategy: str
    prediction_horizon: int
    buy_threshold: float
    sell_threshold: float
    use_high_low: bool
    tie_break: str
    use_atr_barriers: bool
    atr_period: int
    buy_atr_multiplier: float
    sell_atr_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the recipe into a JSON-friendly dictionary."""

        return asdict(self)
