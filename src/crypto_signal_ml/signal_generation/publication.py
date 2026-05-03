"""Publication helpers extracted from the legacy decision stage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import TrainingConfig
from ..trading.signals import (
    build_actionable_signal_summaries,
    filter_published_signal_summaries,
    select_primary_signal,
)

CHART_CONFIRMATION_PRIORITY = {
    "confirmed": 0,
    "early": 1,
    "blocked": 2,
    "unclear": 3,
    "invalid": 4,
}


@dataclass(frozen=True)
class PublicationSelection:
    """Publication outputs built from finalized signal summaries."""

    signal_summaries: tuple[dict[str, Any], ...]
    published_signals: tuple[dict[str, Any], ...]
    actionable_signals: tuple[dict[str, Any], ...]
    primary_signal: dict[str, Any] | None


def should_publish_watchlist_fallback(
    *,
    config: TrainingConfig,
    last_primary_signal_at: datetime | None,
) -> bool:
    """Return whether the public feed should emit a watchlist fallback row."""

    if not bool(getattr(config, "signal_watchlist_fallback_enabled", True)):
        return False

    quiet_period_hours = max(
        float(getattr(config, "signal_watchlist_fallback_hours", 12.0) or 0.0),
        0.0,
    )
    if last_primary_signal_at is None:
        return True
    if quiet_period_hours <= 0:
        return True

    return (datetime.now(timezone.utc) - last_primary_signal_at) >= timedelta(hours=quiet_period_hours)


def rank_watchlist_fallback_candidates(
    signal_summaries: list[dict[str, Any]],
    *,
    config: TrainingConfig,
    excluded_product_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the strongest watchlist candidates that are safe for public fallback."""

    readiness_priority = {
        "high": 0,
        "medium": 1,
        "standby": 2,
        "blocked": 3,
    }
    minimum_decision_score = float(
        getattr(config, "signal_watchlist_min_decision_score", 0.30) or 0.30
    )
    minimum_confidence = float(
        getattr(config, "signal_watchlist_min_confidence", 0.55) or 0.55
    )
    excluded_product_ids = {
        str(product_id).strip().upper()
        for product_id in list(excluded_product_ids or set())
        if str(product_id).strip()
    }
    ranked_candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    for signal_summary in signal_summaries:
        product_id = str(signal_summary.get("productId", "")).strip().upper()
        if not product_id or product_id in excluded_product_ids:
            continue

        brain = signal_summary.get("brain") if isinstance(signal_summary.get("brain"), dict) else {}
        if str(brain.get("decision", "")).strip().lower() != "watchlist":
            continue

        trade_context = (
            signal_summary.get("tradeContext")
            if isinstance(signal_summary.get("tradeContext"), dict)
            else {}
        )
        if bool(trade_context.get("hasActiveTrade", False)):
            continue

        raw_signal_name = str(signal_summary.get("modelSignalName", "")).strip().upper()
        final_signal_name = str(signal_summary.get("signal_name", "")).strip().upper()
        if raw_signal_name == "TAKE_PROFIT" or final_signal_name == "LOSS":
            continue

        confidence = float(signal_summary.get("confidence", 0.0) or 0.0)
        decision_score = float(brain.get("decisionScore", 0.0) or 0.0)
        if confidence < minimum_confidence or decision_score < minimum_decision_score:
            continue

        trade_readiness = str(signal_summary.get("tradeReadiness", "standby")).strip().lower()
        chart_confirmation_status = str(
            signal_summary.get("chartDecision", signal_summary.get("chartConfirmationStatus", "early")) or "early"
        ).strip().lower()
        if chart_confirmation_status == "blocked":
            continue
        ranked_candidates.append(
            (
                (
                    0 if raw_signal_name == "BUY" else 1,
                    readiness_priority.get(trade_readiness, 99),
                    CHART_CONFIRMATION_PRIORITY.get(chart_confirmation_status, 99),
                    -decision_score,
                    -float(signal_summary.get("policyScore", 0.0) or 0.0),
                    -confidence,
                    product_id,
                ),
                dict(signal_summary),
            )
        )

    return [candidate for _, candidate in sorted(ranked_candidates, key=lambda item: item[0])]


def decorate_watchlist_fallback_signal(selected_signal: dict[str, Any]) -> dict[str, Any]:
    """Label one internal watchlist candidate as a public fallback row."""

    fallback_note = (
        "No actionable trade cleared the live gate recently, so this strongest watchlist candidate is surfaced instead."
    )
    existing_reason_items = [
        str(reason_item).strip()
        for reason_item in list(selected_signal.get("reasonItems", []))
        if str(reason_item).strip()
    ]
    merged_reason_items = [fallback_note]
    for reason_item in existing_reason_items:
        if reason_item not in merged_reason_items:
            merged_reason_items.append(reason_item)

    fallback_signal = dict(selected_signal)
    fallback_signal["watchlistFallback"] = True
    fallback_signal["publicSignalType"] = "watchlist"
    fallback_signal["actionable"] = False
    fallback_signal["spotAction"] = "wait"
    fallback_signal["reasonItems"] = merged_reason_items[:4]
    fallback_signal["reasonSummary"] = merged_reason_items[0]
    existing_chat = str(fallback_signal.get("signalChat", "")).strip()
    fallback_signal["signalChat"] = (
        f"{fallback_note} {existing_chat}".strip()
        if existing_chat
        else fallback_note
    )
    fallback_signal["publicationReason"] = "watchlist_fallback"

    return fallback_signal


def supplement_published_signals_with_watchlist_candidates(
    *,
    published_signals: list[dict[str, Any]],
    signal_summaries: list[dict[str, Any]],
    config: TrainingConfig,
) -> list[dict[str, Any]]:
    """Keep the public feed populated with the strongest watchlist ideas when needed."""

    minimum_published_signals = max(
        int(getattr(config, "signal_watchlist_min_published_signals", 2) or 2),
        1,
    )
    if len(published_signals) >= minimum_published_signals:
        return list(published_signals)
    if not published_signals:
        return []

    existing_product_ids = {
        str(signal_summary.get("productId", "")).strip().upper()
        for signal_summary in published_signals
        if str(signal_summary.get("productId", "")).strip()
    }
    ranked_candidates = rank_watchlist_fallback_candidates(
        signal_summaries,
        config=config,
        excluded_product_ids=existing_product_ids,
    )
    if not ranked_candidates:
        return list(published_signals)

    supplemented_signals = list(published_signals)
    remaining_slots = minimum_published_signals - len(supplemented_signals)
    for candidate in ranked_candidates[:remaining_slots]:
        supplemented_signals.append(decorate_watchlist_fallback_signal(candidate))

    return supplemented_signals


def decide_publication(
    *,
    signal_summaries: list[dict[str, Any]],
    config: TrainingConfig,
    recent_primary_product_ids: list[str] | None = None,
    allow_watchlist_fallback: bool = False,
    allow_watchlist_supplement: bool = False,
    last_primary_signal_at: datetime | None = None,
) -> PublicationSelection:
    """Turn finalized signal summaries into published, actionable, and primary outputs."""

    published_signals = filter_published_signal_summaries(signal_summaries)
    if not published_signals:
        if allow_watchlist_fallback and should_publish_watchlist_fallback(
            config=config,
            last_primary_signal_at=last_primary_signal_at,
        ):
            ranked_candidates = rank_watchlist_fallback_candidates(signal_summaries, config=config)
            if ranked_candidates:
                published_signals = [decorate_watchlist_fallback_signal(ranked_candidates[0])]
    elif allow_watchlist_supplement:
        published_signals = supplement_published_signals_with_watchlist_candidates(
            published_signals=published_signals,
            signal_summaries=signal_summaries,
            config=config,
        )

    actionable_signals = build_actionable_signal_summaries(published_signals)
    primary_signal = None
    if published_signals:
        primary_signal = select_primary_signal(
            published_signals,
            config=config,
            recent_primary_product_ids=recent_primary_product_ids or [],
        )

    return PublicationSelection(
        signal_summaries=tuple(dict(signal_summary) for signal_summary in signal_summaries),
        published_signals=tuple(dict(signal_summary) for signal_summary in published_signals),
        actionable_signals=tuple(dict(signal_summary) for signal_summary in actionable_signals),
        primary_signal=(dict(primary_signal) if primary_signal is not None else None),
    )
