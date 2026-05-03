"""Registry-first feature definitions for model training and serving."""

from __future__ import annotations

from collections import OrderedDict

from .feature_contracts import FeatureSpec
from .fundamentals import FUNDAMENTALS_CONTEXT_FEATURES, MARKET_INTELLIGENCE_FEATURES
from .market_structure import MARKET_STRUCTURE_FEATURES
from .news import NEWS_FEATURES
from .regime import REGIME_FEATURES
from .technical import (
    MARKET_CONTEXT_FEATURES,
    MOMENTUM_FEATURES,
    MULTI_TIMEFRAME_FEATURE_COLUMNS,
    RETURNS_FEATURES,
    TIME_CONTEXT_FEATURES,
    TREND_FEATURES,
    VOLUME_FEATURES,
    VOLATILITY_FEATURES,
)


FEATURE_GROUPS: dict[str, list[str]] = OrderedDict(
    {
        "returns": list(RETURNS_FEATURES),
        "momentum": list(MOMENTUM_FEATURES),
        "volume": list(VOLUME_FEATURES),
        "volatility": list(VOLATILITY_FEATURES),
        "trend": list(TREND_FEATURES),
        "market_context": list(MARKET_CONTEXT_FEATURES),
        "market_structure": list(MARKET_STRUCTURE_FEATURES),
        "time_context": list(TIME_CONTEXT_FEATURES),
        "higher_timeframe": list(MULTI_TIMEFRAME_FEATURE_COLUMNS),
        "regime": list(REGIME_FEATURES),
        "fundamentals_context": list(FUNDAMENTALS_CONTEXT_FEATURES),
        "market_intelligence": list(MARKET_INTELLIGENCE_FEATURES),
        "news": list(NEWS_FEATURES),
    }
)

FEATURE_PACKS: dict[str, tuple[str, ...]] = {
    "core": (
        "returns",
        "momentum",
        "volume",
        "volatility",
        "trend",
        "market_structure",
        "regime",
    ),
    "core_plus_market": (
        "returns",
        "momentum",
        "volume",
        "volatility",
        "trend",
        "market_structure",
        "regime",
        "market_context",
    ),
    "core_plus_context": (
        "returns",
        "momentum",
        "volume",
        "volatility",
        "trend",
        "market_structure",
        "regime",
        "market_context",
        "higher_timeframe",
        "time_context",
    ),
    "core_plus_fundamentals": (
        "returns",
        "momentum",
        "volume",
        "volatility",
        "trend",
        "market_structure",
        "regime",
        "market_context",
        "higher_timeframe",
        "time_context",
        "fundamentals_context",
        "market_intelligence",
    ),
    "all": tuple(FEATURE_GROUPS.keys()),
}

FEATURE_COLUMNS = [
    feature_name
    for group_columns in FEATURE_GROUPS.values()
    for feature_name in group_columns
]


def _build_feature_group_lookup() -> dict[str, str]:
    """Build a reverse lookup from feature name to its owning family."""

    feature_group_lookup: dict[str, str] = {}
    for group_name, group_columns in FEATURE_GROUPS.items():
        for feature_name in group_columns:
            if feature_name in feature_group_lookup:
                raise ValueError(
                    f"Feature '{feature_name}' is assigned to multiple groups: "
                    f"{feature_group_lookup[feature_name]} and {group_name}."
                )
            feature_group_lookup[feature_name] = group_name

    missing_features = [feature_name for feature_name in FEATURE_COLUMNS if feature_name not in feature_group_lookup]
    if missing_features:
        raise ValueError(
            "Every feature must belong to exactly one feature group. "
            f"Missing assignments: {missing_features}"
        )

    return feature_group_lookup


FEATURE_GROUP_LOOKUP = _build_feature_group_lookup()


def resolve_feature_group(feature_name: str) -> str:
    """Return the configured feature family for one feature column."""

    return FEATURE_GROUP_LOOKUP.get(str(feature_name), "unknown")


def _infer_feature_source(feature_name: str, group_name: str) -> str:
    """Infer the upstream data source for one feature."""

    if feature_name.startswith("cmc_market_"):
        return "coinmarketcap_market_intelligence"
    if feature_name.startswith("cmc_"):
        return "coinmarketcap_context"
    if feature_name.startswith("cmcal_"):
        return "coinmarketcal"
    if feature_name.startswith("htf_"):
        return "aligned_higher_timeframe"
    if group_name == "regime":
        return "regime_detector"
    return "price_history"


def _infer_required_inputs(feature_name: str) -> tuple[str, ...]:
    """Infer the raw inputs needed to compute one feature."""

    if feature_name.startswith(("cmc_", "cmcal_")):
        return ("context_enrichment",)
    if feature_name.startswith("htf_"):
        return ("timestamp", "close", "volume", "aligned_higher_timeframe")
    if feature_name.startswith(("hour_", "day_")):
        return ("timestamp",)
    if "volume" in feature_name:
        return ("volume", "close")
    if "atr" in feature_name or "range" in feature_name or "drawdown" in feature_name or "rebound" in feature_name:
        return ("open", "high", "low", "close")
    return ("close",)


def _infer_expected_scale(group_name: str) -> str:
    """Infer the expected scale semantics for one feature family."""

    if group_name in {"returns", "momentum", "volatility", "trend", "market_context"}:
        return "continuous_normalized"
    if group_name in {"market_structure", "higher_timeframe", "regime"}:
        return "continuous_mixed"
    if group_name in {"fundamentals_context", "market_intelligence"}:
        return "continuous_contextual"
    if group_name == "time_context":
        return "bounded_periodic"
    return "continuous"


def _build_feature_registry() -> dict[str, FeatureSpec]:
    """Create the canonical feature registry."""

    registry: dict[str, FeatureSpec] = {}
    for feature_name in FEATURE_COLUMNS:
        group_name = resolve_feature_group(feature_name)
        registry[feature_name] = FeatureSpec(
            name=feature_name,
            family=group_name,
            source=_infer_feature_source(feature_name, group_name),
            required_inputs=_infer_required_inputs(feature_name),
            missing_handling="fill_zero_optional_context" if group_name in {"higher_timeframe", "regime"} else "drop_before_model",
            online_available=not feature_name.startswith("news_"),
            offline_available=True,
            expected_scale=_infer_expected_scale(group_name),
            importance_stability="unknown",
            owner="signal-platform",
        )

    return registry


FEATURE_REGISTRY = _build_feature_registry()


def _normalize_group_names(
    group_names: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """Normalize one sequence of feature-family names while preserving order."""

    if not group_names:
        return ()

    normalized_group_names: list[str] = []
    seen_group_names: set[str] = set()
    for group_name in group_names:
        normalized_group_name = str(group_name or "").strip().lower()
        if not normalized_group_name or normalized_group_name in seen_group_names:
            continue
        seen_group_names.add(normalized_group_name)
        normalized_group_names.append(normalized_group_name)
    return tuple(normalized_group_names)


def _validate_group_names(group_names: tuple[str, ...], *, field_name: str) -> None:
    """Ensure every configured family name exists in the registry."""

    unknown_group_names = [
        group_name
        for group_name in group_names
        if group_name not in FEATURE_GROUPS
    ]
    if unknown_group_names:
        available_groups = ", ".join(FEATURE_GROUPS.keys())
        raise ValueError(
            f"Unknown {field_name}: {', '.join(unknown_group_names)}. "
            f"Available feature groups: {available_groups}"
        )


def get_feature_pack_columns(
    feature_pack: str = "all",
    *,
    include_groups: tuple[str, ...] | list[str] | None = None,
    exclude_groups: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """Return the ordered feature list for one pack plus optional family overrides."""

    normalized_pack = str(feature_pack or "all").strip().lower()
    if normalized_pack not in FEATURE_PACKS:
        available_packs = ", ".join(sorted(FEATURE_PACKS.keys()))
        raise ValueError(
            f"Unknown feature pack: {feature_pack}. "
            f"Available feature packs: {available_packs}"
        )

    normalized_include_groups = _normalize_group_names(include_groups)
    normalized_exclude_groups = _normalize_group_names(exclude_groups)
    _validate_group_names(normalized_include_groups, field_name="feature include groups")
    _validate_group_names(normalized_exclude_groups, field_name="feature exclude groups")

    selected_groups = set(FEATURE_PACKS[normalized_pack])
    selected_groups.update(normalized_include_groups)
    selected_groups.difference_update(normalized_exclude_groups)
    return [
        feature_name
        for feature_name in FEATURE_COLUMNS
        if resolve_feature_group(feature_name) in selected_groups
    ]


def get_feature_spec(feature_name: str) -> FeatureSpec:
    """Return metadata for one registered feature."""

    normalized_feature_name = str(feature_name).strip()
    if normalized_feature_name not in FEATURE_REGISTRY:
        raise KeyError(f"Unknown feature: {feature_name}")

    return FEATURE_REGISTRY[normalized_feature_name]
