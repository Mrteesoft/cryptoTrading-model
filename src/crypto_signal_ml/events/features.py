"""Event feature builders for signal enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..config import TrainingConfig
from .calendar import EventCalendar, EventEntry


@dataclass(frozen=True)
class EventFeaturePayload:
    """Structured event features attached to a signal."""

    minutes_to_next_high_impact_event: float | None
    next_event_impact_score: float
    coin_specific_event_flag: bool
    macro_event_risk_flag: bool
    event_window_active: bool
    post_event_cooldown_active: bool
    next_event_title: str | None
    next_event_starts_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "minutesToNextHighImpactEvent": self.minutes_to_next_high_impact_event,
            "nextEventImpactScore": float(self.next_event_impact_score),
            "coinSpecificEventFlag": bool(self.coin_specific_event_flag),
            "macroEventRiskFlag": bool(self.macro_event_risk_flag),
            "eventWindowActive": bool(self.event_window_active),
            "postEventCooldownActive": bool(self.post_event_cooldown_active),
            "nextEventTitle": self.next_event_title,
            "nextEventStartsAt": self.next_event_starts_at,
        }


def _minutes_between(start: datetime, end: datetime) -> float:
    return max((end - start).total_seconds() / 60.0, 0.0)


def build_event_features(
    *,
    base_currency: str,
    calendar: EventCalendar,
    config: TrainingConfig,
    now: datetime | None = None,
) -> EventFeaturePayload:
    now = now or datetime.now(timezone.utc)
    event_window_minutes = float(getattr(config, "event_window_minutes", 360.0))
    cooldown_minutes = float(getattr(config, "event_post_window_minutes", 240.0))
    impact_threshold = float(getattr(config, "event_high_impact_threshold", 0.65))

    next_event = calendar.next_event_for_base(base_currency, now=now)
    last_event = calendar.last_event_for_base(base_currency, now=now)
    macro_event = calendar.next_macro_event(now=now)

    minutes_to_next = None
    next_event_impact = 0.0
    next_event_title = None
    next_event_starts_at = None
    event_window_active = False
    post_event_cooldown_active = False
    coin_specific_event_flag = False

    if next_event is not None:
        minutes_to_next = _minutes_between(now, next_event.starts_at)
        next_event_impact = float(next_event.impact_score)
        next_event_title = next_event.title or None
        next_event_starts_at = next_event.starts_at.isoformat()
        coin_specific_event_flag = True
        if minutes_to_next <= event_window_minutes:
            event_window_active = True

    macro_event_risk_flag = False
    if macro_event is not None and macro_event.impact_score >= impact_threshold:
        macro_event_risk_flag = _minutes_between(now, macro_event.starts_at) <= event_window_minutes

    if last_event is not None:
        minutes_since_last = _minutes_between(last_event.starts_at, now)
        if minutes_since_last <= cooldown_minutes:
            post_event_cooldown_active = True

    if next_event_impact < impact_threshold:
        minutes_to_next = None

    return EventFeaturePayload(
        minutes_to_next_high_impact_event=minutes_to_next,
        next_event_impact_score=next_event_impact,
        coin_specific_event_flag=coin_specific_event_flag,
        macro_event_risk_flag=macro_event_risk_flag,
        event_window_active=event_window_active,
        post_event_cooldown_active=post_event_cooldown_active,
        next_event_title=next_event_title,
        next_event_starts_at=next_event_starts_at,
    )
