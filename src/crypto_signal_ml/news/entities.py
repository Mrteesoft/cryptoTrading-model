"""Entity extraction helpers for news enrichment."""

from __future__ import annotations

from typing import Iterable


def extract_entities(text: str, known_entities: Iterable[str]) -> list[str]:
    """Return uppercase entity symbols mentioned in the text."""

    normalized_text = str(text or "").upper()
    entities: list[str] = []
    for entity in known_entities:
        candidate = str(entity or "").upper()
        if not candidate:
            continue
        if candidate in normalized_text and candidate not in entities:
            entities.append(candidate)
    return entities
