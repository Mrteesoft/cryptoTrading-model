"""Structured model-status tools backed by the deployment artifact summary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable


MODEL_STATUS_TOOL_SCHEMA = {
    "name": "get_model_status",
    "description": "Return the authoritative trained-model status and lifecycle metadata.",
    "inputSchema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

TOOL_SCHEMAS = [MODEL_STATUS_TOOL_SCHEMA]


def _utc_now_iso() -> str:
    """Return one UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


class ModelToolService:
    """Expose a stable model-status tool over one summary provider."""

    tool_schemas = TOOL_SCHEMAS

    def __init__(
        self,
        model_status_provider: Callable[[], dict[str, Any]],
    ) -> None:
        self.model_status_provider = model_status_provider

    def get_model_status(self) -> dict[str, Any]:
        """Return one stable model-status tool payload."""

        try:
            model_summary = self.model_status_provider()
        except Exception as error:
            return {
                "toolName": "get_model_status",
                "requestedAt": _utc_now_iso(),
                "status": "error",
                "error": str(error),
                "model": {},
            }

        return {
            "toolName": "get_model_status",
            "requestedAt": _utc_now_iso(),
            "status": str(model_summary.get("status") or "ok"),
            "error": "",
            "model": model_summary,
        }
