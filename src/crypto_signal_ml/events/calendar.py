"""Event calendar loader for upcoming event context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
from pathlib import Path
from typing import Iterable, Sequence

from ..config import TrainingConfig
from .scoring import score_event_impact


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class EventEntry:
    """Normalized event row for enrichment features."""

    event_id: str
    title: str
    category: str
    starts_at: datetime
    base_currency: str
    impact_score: float
    macro_event: bool = False


class EventCalendar:
    """Load upcoming events from cached CSV sources."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self._events: list[EventEntry] = []
        self._load()

    def _load(self) -> None:
        events_path = Path(self.config.coinmarketcal_events_file)
        if not events_path.exists():
            return

        events: list[EventEntry] = []
        with events_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                starts_at = _parse_timestamp(row.get("event_start"))
                if starts_at is None:
                    continue
                title = str(row.get("event_title") or "").strip()
                category = str(row.get("event_category") or "").strip().lower()
                base_currency = str(row.get("base_currency") or "").strip().upper()
                if not title and not base_currency:
                    continue

                impact_score = score_event_impact(title=title, category=category)
                macro_event = "macro" in category or "fed" in title.lower()
                events.append(
                    EventEntry(
                        event_id=str(row.get("event_id") or ""),
                        title=title,
                        category=category,
                        starts_at=starts_at,
                        base_currency=base_currency,
                        impact_score=impact_score,
                        macro_event=macro_event,
                    )
                )

        self._events = sorted(events, key=lambda entry: entry.starts_at)

    def iter_events(self) -> Iterable[EventEntry]:
        return list(self._events)

    def next_event_for_base(self, base_currency: str, now: datetime | None = None) -> EventEntry | None:
        normalized_base = str(base_currency or "").strip().upper()
        if not normalized_base:
            return None
        now = now or datetime.now(timezone.utc)
        for entry in self._events:
            if entry.base_currency != normalized_base:
                continue
            if entry.starts_at >= now:
                return entry
        return None

    def last_event_for_base(self, base_currency: str, now: datetime | None = None) -> EventEntry | None:
        normalized_base = str(base_currency or "").strip().upper()
        if not normalized_base:
            return None
        now = now or datetime.now(timezone.utc)
        for entry in reversed(self._events):
            if entry.base_currency != normalized_base:
                continue
            if entry.starts_at <= now:
                return entry
        return None

    def next_macro_event(self, now: datetime | None = None) -> EventEntry | None:
        now = now or datetime.now(timezone.utc)
        for entry in self._events:
            if not entry.macro_event:
                continue
            if entry.starts_at >= now:
                return entry
        return None
