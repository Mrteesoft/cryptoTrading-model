"""Helpers for validating asset symbols before they reach public signal outputs."""

from __future__ import annotations

import re


_ALPHANUMERIC_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]+$")
_ALPHABETIC_CHARACTER_PATTERN = re.compile(r"[A-Z]")


def normalize_base_currency(base_currency: str | None) -> str:
    """Normalize one base symbol into the uppercase ticker form used in the project."""

    if base_currency is None:
        return ""

    normalized_base_currency = str(base_currency).strip().upper()
    if normalized_base_currency in {"", "NAN", "<NA>", "NONE"}:
        return ""

    return normalized_base_currency


def is_signal_eligible_base_currency(base_currency: str | None) -> bool:
    """Return whether one base symbol is sane enough to expose as a trading asset."""

    normalized_base_currency = normalize_base_currency(base_currency)
    if not normalized_base_currency:
        return False

    if _ALPHANUMERIC_SYMBOL_PATTERN.fullmatch(normalized_base_currency) is None:
        return False

    return _ALPHABETIC_CHARACTER_PATTERN.search(normalized_base_currency) is not None
