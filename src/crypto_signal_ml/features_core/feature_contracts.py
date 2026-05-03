"""Typed feature metadata contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one model feature column."""

    name: str
    family: str
    source: str
    required_inputs: tuple[str, ...]
    missing_handling: str
    online_available: bool
    offline_available: bool
    expected_scale: str
    importance_stability: str
    owner: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the spec into a JSON-friendly dictionary."""

        return asdict(self)
