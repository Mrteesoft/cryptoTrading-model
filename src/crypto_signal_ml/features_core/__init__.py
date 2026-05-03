"""Feature-registry and diagnostics layer for the signal platform."""

from .diagnostics import audit_feature_registry_coverage, build_feature_registry_snapshot
from .feature_contracts import FeatureSpec
from .feature_registry import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    FEATURE_GROUP_LOOKUP,
    FEATURE_PACKS,
    FEATURE_REGISTRY,
    MULTI_TIMEFRAME_FEATURE_COLUMNS,
    get_feature_pack_columns,
    get_feature_spec,
    resolve_feature_group,
)

__all__ = [
    "FeatureSpec",
    "FEATURE_COLUMNS",
    "FEATURE_GROUPS",
    "FEATURE_GROUP_LOOKUP",
    "FEATURE_PACKS",
    "FEATURE_REGISTRY",
    "MULTI_TIMEFRAME_FEATURE_COLUMNS",
    "audit_feature_registry_coverage",
    "build_feature_registry_snapshot",
    "get_feature_pack_columns",
    "get_feature_spec",
    "resolve_feature_group",
]
