"""News providers for enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from ..config import OUTPUTS_DIR, TrainingConfig


@dataclass(frozen=True)
class NewsItem:
    """Normalized news item."""

    published_at: datetime
    title: str
    summary: str
    source: str
    url: str
    entities: list[str]


class LocalNewsProvider:
    """Load news items from a local JSON cache if present."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.path = Path(getattr(self.config, "news_store_path", OUTPUTS_DIR / "newsFeed.json"))
        self._items: list[NewsItem] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, list):
            return
        items: list[NewsItem] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            published_at = entry.get("publishedAt") or entry.get("published_at")
            if not published_at:
                continue
            try:
                parsed = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            items.append(
                NewsItem(
                    published_at=parsed.astimezone(timezone.utc),
                    title=str(entry.get("title") or ""),
                    summary=str(entry.get("summary") or entry.get("description") or ""),
                    source=str(entry.get("source") or "local"),
                    url=str(entry.get("url") or ""),
                    entities=[str(entity).upper() for entity in entry.get("entities", []) if str(entity).strip()],
                )
            )
        self._items = sorted(items, key=lambda item: item.published_at, reverse=True)

    def iter_recent(self, since_minutes: float, now: datetime | None = None) -> Iterable[NewsItem]:
        now = now or datetime.now(timezone.utc)
        cutoff = now.timestamp() - (since_minutes * 60.0)
        for item in self._items:
            if item.published_at.timestamp() < cutoff:
                continue
            yield item

    def iter_all(self) -> Iterable[NewsItem]:
        return list(self._items)
