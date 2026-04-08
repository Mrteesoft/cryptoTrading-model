"""Structured signal tools backed by the authoritative live engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from ..application import PublishedSignalViewService
from ..live import LiveSignalEngine


MARKET_OVERVIEW_TOOL_SCHEMA = {
    "name": "get_market_overview",
    "description": "Return the authoritative live or cached market overview from the trading engine.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "force_refresh": {
                "type": "boolean",
                "default": False,
                "description": "When true, bypass the current live-signal cache before reading the overview.",
            },
        },
        "additionalProperties": False,
    },
}

SIGNAL_DETAIL_TOOL_SCHEMA = {
    "name": "get_signal",
    "description": "Return the authoritative live or cached signal for one product id from the trading engine.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "string",
                "description": "Coinbase-style product id such as BTC-USD.",
            },
            "force_refresh": {
                "type": "boolean",
                "default": False,
                "description": "When true, bypass the current live-signal cache before reading the signal.",
            },
        },
        "required": ["product_id"],
        "additionalProperties": False,
    },
}

TOOL_SCHEMAS = [MARKET_OVERVIEW_TOOL_SCHEMA, SIGNAL_DETAIL_TOOL_SCHEMA]


def _utc_now_iso() -> str:
    """Return one UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    """Convert one value into a float when possible."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_snapshot_overview(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Trim a live or cached snapshot into a stable overview payload."""

    if not isinstance(snapshot, dict):
        return {}

    signals = list(snapshot.get("signals", []))
    actionable_signals = list(snapshot.get("actionableSignals", []))
    top_signals = actionable_signals[:3] if actionable_signals else signals[:3]

    return {
        "generatedAt": snapshot.get("generatedAt"),
        "mode": snapshot.get("mode"),
        "marketDataSource": snapshot.get("marketDataSource"),
        "requestMode": snapshot.get("requestMode"),
        "productsCovered": snapshot.get("productsCovered"),
        "granularitySeconds": snapshot.get("granularitySeconds"),
        "primarySignal": snapshot.get("primarySignal"),
        "marketSummary": snapshot.get("marketSummary", {}),
        "marketState": snapshot.get("marketState", {}),
        "traderBrain": snapshot.get("traderBrain", {}),
        "topSignals": top_signals,
    }


class SignalToolService:
    """Expose stable signal and overview tools over the live engine."""

    tool_schemas = TOOL_SCHEMAS

    def __init__(
        self,
        live_signal_engine: LiveSignalEngine | None = None,
        cached_snapshot_provider: Callable[[], dict[str, Any] | None] | None = None,
        published_signal_service: PublishedSignalViewService | None = None,
    ) -> None:
        self.live_signal_engine = live_signal_engine
        self.cached_snapshot_provider = cached_snapshot_provider
        self.published_signal_service = published_signal_service

    def load_market_snapshot(
        self,
        *,
        force_refresh: bool = False,
        product_id: str | None = None,
        product_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Return a live snapshot with a cached fallback and stable status fields."""

        try:
            request_kwargs: dict[str, Any] = {
                "force_refresh": force_refresh,
            }
            if product_id is not None:
                request_kwargs["product_id"] = product_id
            if product_ids is not None:
                request_kwargs["product_ids"] = product_ids

            snapshot = self.live_signal_engine.get_live_snapshot(**request_kwargs)
            return {
                "status": "ok",
                "source": "live",
                "warning": "",
                "error": "",
                "snapshot": snapshot,
            }
        except Exception as error:
            cached_snapshot = self._load_cached_snapshot()
            if cached_snapshot is None:
                return {
                    "status": "error",
                    "source": "unavailable",
                    "warning": "",
                    "error": str(error),
                    "snapshot": None,
                }

            return {
                "status": "ok",
                "source": "cached",
                "warning": str(error),
                "error": "",
                "snapshot": cached_snapshot,
            }

    def get_market_overview(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Return one stable market-overview tool payload."""

        if self.published_signal_service is not None:
            state = self.published_signal_service.load_tool_signal_state(force_refresh=force_refresh)
            if state.get("status") != "ok":
                return {
                    "toolName": "get_market_overview",
                    "requestedAt": _utc_now_iso(),
                    "status": "error",
                    "source": state.get("source"),
                    "forceRefresh": bool(force_refresh),
                    "warning": state.get("warning", ""),
                    "error": state.get("error", "Market overview is unavailable."),
                    "overview": {},
                }
            return {
                "toolName": "get_market_overview",
                "requestedAt": _utc_now_iso(),
                "status": "ok",
                "source": state.get("source"),
                "forceRefresh": bool(force_refresh),
                "warning": state.get("warning", ""),
                "error": "",
                "overview": state.get("overview", {}),
            }

        snapshot_result = self.load_market_snapshot(force_refresh=force_refresh)
        snapshot = snapshot_result.get("snapshot")
        if snapshot is None:
            return {
                "toolName": "get_market_overview",
                "requestedAt": _utc_now_iso(),
                "status": "error",
                "source": snapshot_result.get("source"),
                "forceRefresh": bool(force_refresh),
                "warning": snapshot_result.get("warning", ""),
                "error": snapshot_result.get("error", "Market overview is unavailable."),
                "overview": {},
            }

        return {
            "toolName": "get_market_overview",
            "requestedAt": _utc_now_iso(),
            "status": "ok",
            "source": snapshot_result.get("source"),
            "forceRefresh": bool(force_refresh),
            "warning": snapshot_result.get("warning", ""),
            "error": "",
            "overview": _build_snapshot_overview(snapshot),
        }

    def get_signal(
        self,
        product_id: str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Return one stable single-signal tool payload."""

        normalized_product_id = str(product_id).strip().upper()
        if not normalized_product_id:
            return {
                "toolName": "get_signal",
                "requestedAt": _utc_now_iso(),
                "status": "error",
                "source": "unavailable",
                "productId": None,
                "forceRefresh": bool(force_refresh),
                "warning": "",
                "error": "Product id is empty.",
                "signal": None,
                "overview": {},
            }

        if self.published_signal_service is not None:
            state = self.published_signal_service.load_tool_signal_state(
                force_refresh=force_refresh,
                product_id=normalized_product_id,
            )
            if state.get("status") != "ok":
                return {
                    "toolName": "get_signal",
                    "requestedAt": _utc_now_iso(),
                    "status": "error",
                    "source": state.get("source"),
                    "productId": normalized_product_id,
                    "forceRefresh": bool(force_refresh),
                    "warning": state.get("warning", ""),
                    "error": state.get("error", f"Signal is unavailable for {normalized_product_id}."),
                    "signal": None,
                    "overview": {},
                }

            signal_summary = state.get("signal")
            if not isinstance(signal_summary, dict):
                return {
                    "toolName": "get_signal",
                    "requestedAt": _utc_now_iso(),
                    "status": "not_found",
                    "source": state.get("source"),
                    "productId": normalized_product_id,
                    "forceRefresh": bool(force_refresh),
                    "warning": state.get("warning", ""),
                    "error": "",
                    "signal": None,
                    "overview": state.get("overview", {}),
                }

            return {
                "toolName": "get_signal",
                "requestedAt": _utc_now_iso(),
                "status": "ok",
                "source": state.get("source"),
                "productId": normalized_product_id,
                "forceRefresh": bool(force_refresh),
                "warning": state.get("warning", ""),
                "error": "",
                "signal": {
                    **signal_summary,
                    "confidence": _safe_float(signal_summary.get("confidence")),
                    "close": _safe_float(signal_summary.get("close")),
                },
                "overview": state.get("overview", {}),
            }

        snapshot_result = self.load_market_snapshot(
            force_refresh=force_refresh,
            product_id=normalized_product_id,
        )
        snapshot = snapshot_result.get("snapshot")
        if snapshot is None:
            return {
                "toolName": "get_signal",
                "requestedAt": _utc_now_iso(),
                "status": "error",
                "source": snapshot_result.get("source"),
                "productId": normalized_product_id,
                "forceRefresh": bool(force_refresh),
                "warning": snapshot_result.get("warning", ""),
                "error": snapshot_result.get("error", f"Signal is unavailable for {normalized_product_id}."),
                "signal": None,
                "overview": {},
            }

        signal_summary = self._resolve_signal_from_snapshot(snapshot, normalized_product_id)
        if signal_summary is None:
            return {
                "toolName": "get_signal",
                "requestedAt": _utc_now_iso(),
                "status": "not_found",
                "source": snapshot_result.get("source"),
                "productId": normalized_product_id,
                "forceRefresh": bool(force_refresh),
                "warning": snapshot_result.get("warning", ""),
                "error": "",
                "signal": None,
                "overview": _build_snapshot_overview(snapshot),
            }

        return {
            "toolName": "get_signal",
            "requestedAt": _utc_now_iso(),
            "status": "ok",
            "source": snapshot_result.get("source"),
            "productId": normalized_product_id,
            "forceRefresh": bool(force_refresh),
            "warning": snapshot_result.get("warning", ""),
            "error": "",
            "signal": {
                **signal_summary,
                "confidence": _safe_float(signal_summary.get("confidence")),
                "close": _safe_float(signal_summary.get("close")),
            },
            "overview": _build_snapshot_overview(snapshot),
        }

    def _load_cached_snapshot(self) -> dict[str, Any] | None:
        """Return the cached snapshot when it exists."""

        if self.cached_snapshot_provider is None:
            return None

        try:
            return self.cached_snapshot_provider()
        except Exception:
            return None

    @staticmethod
    def _resolve_signal_from_snapshot(
        snapshot: dict[str, Any],
        normalized_product_id: str,
    ) -> dict[str, Any] | None:
        """Resolve one signal summary from a live or cached snapshot."""

        signal_by_product = snapshot.get("signalsByProduct", {})
        if isinstance(signal_by_product, dict):
            signal_summary = signal_by_product.get(normalized_product_id)
            if isinstance(signal_summary, dict):
                return signal_summary

        for signal_summary in list(snapshot.get("signals", [])):
            if str(signal_summary.get("productId", "")).strip().upper() == normalized_product_id:
                return signal_summary

        return None
