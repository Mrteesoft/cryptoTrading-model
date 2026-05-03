"""Helpers for compact, readable console logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


LOGGER_COMPONENT_ALIASES = {
    "crypto_signal_ml.monitor": "monitor",
    "crypto_signal_ml.app": "pipeline",
    "crypto_signal_ml.application.signal_generation": "stages",
    "crypto_signal_ml.application.signal_inference": "inference",
    "crypto_signal_ml.application.signal_publication": "publish",
    "crypto_signal_ml.portfolio_core.trader_brain": "brain",
    "crypto_signal_ml.trading.watchlist_state": "watchlist",
}


def short_logger_name(logger_name: str) -> str:
    """Return one compact display name for a logger."""

    normalized_name = str(logger_name).strip()
    if normalized_name in LOGGER_COMPONENT_ALIASES:
        return LOGGER_COMPONENT_ALIASES[normalized_name]
    if normalized_name.startswith("crypto_signal_ml."):
        normalized_name = normalized_name.removeprefix("crypto_signal_ml.")
    if "." in normalized_name:
        normalized_name = normalized_name.rsplit(".", maxsplit=1)[-1]
    return normalized_name or "app"


def format_path_for_log(path_like: Any) -> str:
    """Render one path relative to the project root when possible."""

    raw_value = str(path_like or "").strip()
    if not raw_value:
        return "-"

    try:
        path = Path(raw_value)
    except (TypeError, ValueError):
        return raw_value

    try:
        if path.exists():
            resolved_path = path.resolve()
        elif path.is_absolute():
            resolved_path = path
        else:
            resolved_path = (PROJECT_ROOT / path).resolve()
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except (OSError, RuntimeError, ValueError):
        try:
            return path.as_posix()
        except (TypeError, ValueError):
            return raw_value


def format_bool_for_log(value: Any) -> str:
    """Render one boolean flag as yes/no for logs."""

    return "yes" if bool(value) else "no"


class CompactConsoleFormatter(logging.Formatter):
    """Render compact one-line console logs for interactive runs."""

    default_msec_format = "%s.%03d"

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%H:%M:%S")
        component = short_logger_name(record.name)
        message = record.getMessage()
        prefix = f"{timestamp} | {component:<9}"
        if record.levelno >= logging.WARNING:
            prefix += f" | {record.levelname:<7}"
        formatted = f"{prefix} | {message}"

        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            formatted += "\n" + self.formatStack(record.stack_info)
        return formatted
