"""Diagnostics helpers for feature-registry coverage and snapshots."""

from __future__ import annotations

from .feature_registry import FEATURE_COLUMNS, FEATURE_REGISTRY, get_feature_spec


def audit_feature_registry_coverage(feature_columns: list[str] | tuple[str, ...] | None = None) -> dict[str, object]:
    """Return a compact coverage report for the selected feature set."""

    selected_feature_columns = list(feature_columns or FEATURE_COLUMNS)
    missing_features = [
        feature_name
        for feature_name in selected_feature_columns
        if feature_name not in FEATURE_REGISTRY
    ]

    return {
        "selectedFeatureCount": int(len(selected_feature_columns)),
        "registeredFeatureCount": int(len(FEATURE_REGISTRY)),
        "isComplete": not missing_features,
        "missingFeatures": missing_features,
    }


def build_feature_registry_snapshot(feature_columns: list[str] | tuple[str, ...] | None = None) -> dict[str, object]:
    """Build one JSON-friendly registry snapshot for artifact persistence."""

    selected_feature_columns = list(feature_columns or FEATURE_COLUMNS)
    return {
        "version": "feature-registry-v1",
        "featureCount": int(len(selected_feature_columns)),
        "features": [get_feature_spec(feature_name).to_dict() for feature_name in selected_feature_columns],
    }
