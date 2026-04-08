"""Signal decision stage for publication and primary-signal selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import OUTPUTS_DIR, TrainingConfig
from ..trading.signals import (
    build_actionable_signal_summaries,
    filter_published_signal_summaries,
    select_primary_signal,
)


@dataclass
class SignalDecisionArtifacts:
    """Publication-facing signal decisions derived from enriched signals."""

    signal_summaries: list[dict[str, Any]]
    published_signals: list[dict[str, Any]]
    actionable_signals: list[dict[str, Any]]
    primary_signal: dict[str, Any] | None


class PrimarySignalHistoryStore:
    """Persist and query the small rotating history of featured primary signals."""

    def __init__(
        self,
        *,
        config: TrainingConfig,
        history_path: Path,
        latest_signal_path: Path | None = None,
        save_json: Callable[[dict[str, Any], Path], None],
    ) -> None:
        self.config = config
        self.history_path = Path(history_path)
        self.latest_signal_path = Path(latest_signal_path or (OUTPUTS_DIR / "latestSignal.json"))
        self.save_json = save_json

    def load_recent_product_ids(self) -> list[str]:
        """Load the recently featured primary-signal products from disk when available."""

        if not self.history_path.exists():
            if not self.latest_signal_path.exists():
                return []

            try:
                with self.latest_signal_path.open("r", encoding="utf-8") as latest_signal_file:
                    latest_signal_payload = json.load(latest_signal_file)
            except (OSError, json.JSONDecodeError):
                return []

            latest_product_id = str(latest_signal_payload.get("productId", "")).strip().upper()
            return [latest_product_id] if latest_product_id else []

        try:
            with self.history_path.open("r", encoding="utf-8") as history_file:
                history_payload = json.load(history_file)
        except (OSError, json.JSONDecodeError):
            return []

        history_entries = history_payload.get("entries", []) if isinstance(history_payload, dict) else []
        if not isinstance(history_entries, list):
            return []

        return [
            str(history_entry.get("productId", "")).strip().upper()
            for history_entry in history_entries
            if isinstance(history_entry, dict) and str(history_entry.get("productId", "")).strip()
        ]

    def save_primary_signal(self, primary_signal: dict[str, Any], signal_source: str) -> None:
        """Persist the newest primary signal while keeping a small deduped history."""

        recent_entries: list[dict[str, Any]] = []
        if self.history_path.exists():
            try:
                with self.history_path.open("r", encoding="utf-8") as history_file:
                    history_payload = json.load(history_file)
                previous_entries = history_payload.get("entries", []) if isinstance(history_payload, dict) else []
                if isinstance(previous_entries, list):
                    recent_entries = [
                        entry
                        for entry in previous_entries
                        if isinstance(entry, dict)
                    ]
            except (OSError, json.JSONDecodeError):
                recent_entries = []

        new_entry = {
            "productId": str(primary_signal.get("productId", "")),
            "signalName": str(primary_signal.get("signal_name", "")),
            "generatedAt": str(primary_signal.get("marketDataRefreshedAt") or datetime.now(timezone.utc).isoformat()),
            "signalSource": str(signal_source),
            "confidence": float(primary_signal.get("confidence", 0.0) or 0.0),
            "policyScore": float(primary_signal.get("policyScore", 0.0) or 0.0),
            "setupScore": float(primary_signal.get("setupScore", 0.0) or 0.0),
        }
        recent_entries = [new_entry] + recent_entries

        deduped_entries: list[dict[str, Any]] = []
        seen_products: set[str] = set()
        max_history_entries = max(
            int(getattr(self.config, "signal_primary_rotation_candidate_window", 4)) * 2,
            int(getattr(self.config, "signal_primary_rotation_lookback", 3)),
            6,
        )
        for history_entry in recent_entries:
            product_id = str(history_entry.get("productId", "")).strip().upper()
            if not product_id or product_id in seen_products:
                continue
            seen_products.add(product_id)
            deduped_entries.append(history_entry)
            if len(deduped_entries) >= max_history_entries:
                break

        self.save_json(
            {
                "updatedAt": datetime.now(timezone.utc).isoformat(),
                "entries": deduped_entries,
            },
            self.history_path,
        )

    def load_last_generated_at(self) -> datetime | None:
        """Return when the most recent non-empty primary signal was published."""

        timestamp_candidates: list[str] = []

        if self.history_path.exists():
            try:
                with self.history_path.open("r", encoding="utf-8") as history_file:
                    history_payload = json.load(history_file)
                history_entries = history_payload.get("entries", []) if isinstance(history_payload, dict) else []
                if isinstance(history_entries, list):
                    timestamp_candidates.extend(
                        str(history_entry.get("generatedAt", "")).strip()
                        for history_entry in history_entries
                        if isinstance(history_entry, dict) and str(history_entry.get("generatedAt", "")).strip()
                    )
            except (OSError, json.JSONDecodeError):
                timestamp_candidates = []

        if self.latest_signal_path.exists():
            try:
                with self.latest_signal_path.open("r", encoding="utf-8") as latest_signal_file:
                    latest_signal_payload = json.load(latest_signal_file)
                if isinstance(latest_signal_payload, dict) and latest_signal_payload:
                    for field_name in ("marketDataRefreshedAt", "generatedAt", "timestamp"):
                        raw_value = str(latest_signal_payload.get(field_name, "")).strip()
                        if raw_value:
                            timestamp_candidates.append(raw_value)
            except (OSError, json.JSONDecodeError):
                pass

        for raw_timestamp in timestamp_candidates:
            parsed_timestamp = pd.to_datetime(raw_timestamp, utc=True, errors="coerce")
            if pd.notna(parsed_timestamp):
                return parsed_timestamp.to_pydatetime()

        return None


class SignalDecisionStage:
    """Apply publication and primary-signal decision rules to enriched signals."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        primary_history_store: PrimarySignalHistoryStore | None = None,
        allow_watchlist_fallback: bool = False,
        allow_watchlist_supplement: bool = False,
    ) -> None:
        self.config = config
        self.primary_history_store = primary_history_store
        self.allow_watchlist_fallback = allow_watchlist_fallback
        self.allow_watchlist_supplement = allow_watchlist_supplement

    def decide(self, signal_summaries: list[dict[str, Any]]) -> SignalDecisionArtifacts:
        """Turn enriched signals into published, actionable, and primary outputs."""

        published_signals = filter_published_signal_summaries(signal_summaries)
        if not published_signals:
            if self.allow_watchlist_fallback:
                watchlist_fallback_signal = self._select_watchlist_fallback_signal(signal_summaries)
                if watchlist_fallback_signal is not None:
                    published_signals = [watchlist_fallback_signal]
        elif self.allow_watchlist_supplement:
            published_signals = self._supplement_published_signals_with_watchlist_candidates(
                published_signals=published_signals,
                signal_summaries=signal_summaries,
            )

        actionable_signals = build_actionable_signal_summaries(published_signals)
        recent_primary_product_ids = (
            self.primary_history_store.load_recent_product_ids()
            if self.primary_history_store is not None
            else []
        )
        primary_signal = None
        if published_signals:
            primary_signal = select_primary_signal(
                published_signals,
                config=self.config,
                recent_primary_product_ids=recent_primary_product_ids,
            )

        return SignalDecisionArtifacts(
            signal_summaries=list(signal_summaries),
            published_signals=published_signals,
            actionable_signals=actionable_signals,
            primary_signal=primary_signal,
        )

    def _should_publish_watchlist_fallback(self) -> bool:
        """Return whether the public feed should emit a watchlist fallback signal."""

        if not bool(getattr(self.config, "signal_watchlist_fallback_enabled", True)):
            return False

        quiet_period_hours = max(
            float(getattr(self.config, "signal_watchlist_fallback_hours", 12.0) or 0.0),
            0.0,
        )
        last_primary_signal_at = (
            self.primary_history_store.load_last_generated_at()
            if self.primary_history_store is not None
            else None
        )
        if last_primary_signal_at is None:
            return True
        if quiet_period_hours <= 0:
            return True

        return (datetime.now(timezone.utc) - last_primary_signal_at) >= timedelta(hours=quiet_period_hours)

    def _select_watchlist_fallback_signal(
        self,
        signal_summaries: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Choose one strong watchlist candidate when the public feed would otherwise be empty."""

        if not self._should_publish_watchlist_fallback():
            return None

        ranked_candidates = self._rank_watchlist_fallback_candidates(signal_summaries)
        if not ranked_candidates:
            return None

        return self._decorate_watchlist_fallback_signal(ranked_candidates[0])

    def _rank_watchlist_fallback_candidates(
        self,
        signal_summaries: list[dict[str, Any]],
        excluded_product_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the strongest watchlist candidates that are safe to surface publicly."""

        readiness_priority = {
            "high": 0,
            "medium": 1,
            "standby": 2,
            "blocked": 3,
        }
        minimum_decision_score = float(
            getattr(self.config, "signal_watchlist_min_decision_score", 0.30) or 0.30
        )
        minimum_confidence = float(
            getattr(self.config, "signal_watchlist_min_confidence", 0.55) or 0.55
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
            ranked_candidates.append(
                (
                    (
                        0 if raw_signal_name == "BUY" else 1,
                        readiness_priority.get(trade_readiness, 99),
                        -decision_score,
                        -float(signal_summary.get("policyScore", 0.0) or 0.0),
                        -confidence,
                        product_id,
                    ),
                    dict(signal_summary),
                )
            )

        if not ranked_candidates:
            return []

        return [candidate for _, candidate in sorted(ranked_candidates, key=lambda item: item[0])]

    @staticmethod
    def _decorate_watchlist_fallback_signal(selected_signal: dict[str, Any]) -> dict[str, Any]:
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

        return fallback_signal

    def _supplement_published_signals_with_watchlist_candidates(
        self,
        published_signals: list[dict[str, Any]],
        signal_summaries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep one primary public signal while backfilling a thin feed with watchlist ideas."""

        minimum_published_signals = max(
            int(getattr(self.config, "signal_watchlist_min_published_signals", 2) or 2),
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
        ranked_candidates = self._rank_watchlist_fallback_candidates(
            signal_summaries,
            excluded_product_ids=existing_product_ids,
        )
        if not ranked_candidates:
            return list(published_signals)

        supplemented_signals = list(published_signals)
        remaining_slots = minimum_published_signals - len(supplemented_signals)
        for candidate in ranked_candidates[:remaining_slots]:
            supplemented_signals.append(self._decorate_watchlist_fallback_signal(candidate))

        return supplemented_signals
