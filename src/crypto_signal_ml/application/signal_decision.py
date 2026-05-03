"""Signal decision stage for publication and primary-signal selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import OUTPUTS_DIR, TrainingConfig
from ..signal_generation.publication import (
    decide_publication,
    decorate_watchlist_fallback_signal,
    rank_watchlist_fallback_candidates,
    should_publish_watchlist_fallback,
    supplement_published_signals_with_watchlist_candidates,
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

        publication_selection = decide_publication(
            signal_summaries=signal_summaries,
            config=self.config,
            recent_primary_product_ids=(
                self.primary_history_store.load_recent_product_ids()
                if self.primary_history_store is not None
                else []
            ),
            allow_watchlist_fallback=self.allow_watchlist_fallback,
            allow_watchlist_supplement=self.allow_watchlist_supplement,
            last_primary_signal_at=(
                self.primary_history_store.load_last_generated_at()
                if self.primary_history_store is not None
                else None
            ),
        )

        return SignalDecisionArtifacts(
            signal_summaries=list(publication_selection.signal_summaries),
            published_signals=list(publication_selection.published_signals),
            actionable_signals=list(publication_selection.actionable_signals),
            primary_signal=dict(publication_selection.primary_signal) if publication_selection.primary_signal is not None else None,
        )

    def _should_publish_watchlist_fallback(self) -> bool:
        """Return whether the public feed should emit a watchlist fallback signal."""

        return should_publish_watchlist_fallback(
            config=self.config,
            last_primary_signal_at=(
                self.primary_history_store.load_last_generated_at()
                if self.primary_history_store is not None
                else None
            ),
        )

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

        return rank_watchlist_fallback_candidates(
            signal_summaries,
            config=self.config,
            excluded_product_ids=excluded_product_ids,
        )

    @staticmethod
    def _decorate_watchlist_fallback_signal(selected_signal: dict[str, Any]) -> dict[str, Any]:
        """Label one internal watchlist candidate as a public fallback row."""

        return decorate_watchlist_fallback_signal(selected_signal)

    def _supplement_published_signals_with_watchlist_candidates(
        self,
        published_signals: list[dict[str, Any]],
        signal_summaries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep one primary public signal while backfilling a thin feed with watchlist ideas."""

        return supplement_published_signals_with_watchlist_candidates(
            published_signals=published_signals,
            signal_summaries=signal_summaries,
            config=self.config,
        )
