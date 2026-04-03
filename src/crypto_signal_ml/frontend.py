"""Frontend-focused helpers for serving cached signal snapshots quickly."""

from __future__ import annotations

from collections import Counter
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _build_market_state_snapshot(
    primary_signal: Dict[str, Any],
    latest_signals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize the dominant and primary regime state for the cached snapshot."""

    market_state_rows = [
        signal_summary.get("marketState", {})
        for signal_summary in latest_signals
        if isinstance(signal_summary.get("marketState"), dict)
    ]
    regime_counter = Counter(
        str(state_row.get("label", "unknown"))
        for state_row in market_state_rows
    )
    dominant_label = "unknown"
    dominant_count = 0
    if regime_counter:
        dominant_label, dominant_count = sorted(
            regime_counter.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]

    primary_market_state = primary_signal.get("marketState", {}) if isinstance(primary_signal, dict) else {}
    total_signals = max(len(latest_signals), 1)

    return {
        "primary": {
            "label": str(primary_market_state.get("label", "unknown")),
            "code": int(primary_market_state.get("code", 0) or 0),
            "trendScore": float(primary_market_state.get("trendScore", 0.0) or 0.0),
            "volatilityRatio": float(primary_market_state.get("volatilityRatio", 1.0) or 1.0),
            "isTrending": bool(primary_market_state.get("isTrending", False)),
            "isHighVolatility": bool(primary_market_state.get("isHighVolatility", False)),
        },
        "dominant": {
            "label": dominant_label,
            "count": int(dominant_count),
            "share": float(dominant_count / total_signals),
        },
        "regimeCounts": dict(sorted(regime_counter.items())),
        "trendingSignals": int(
            sum(bool(state_row.get("isTrending", False)) for state_row in market_state_rows)
        ),
        "highVolatilitySignals": int(
            sum(bool(state_row.get("isHighVolatility", False)) for state_row in market_state_rows)
        ),
    }


def build_frontend_signal_snapshot(
    model_type: str,
    primary_signal: Dict[str, Any],
    latest_signals: List[Dict[str, Any]],
    actionable_signals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build one frontend-ready JSON payload from the latest signal outputs.

    The key design decision is performance:
    - we do the model work once in the background
    - we publish one cached snapshot file
    - the frontend/API only reads this ready-made payload

    That is much faster and much cheaper than running the ML model for every
    incoming user request.
    """

    buy_signals = [
        signal_summary
        for signal_summary in actionable_signals
        if signal_summary.get("signal_name") == "BUY"
    ]
    take_profit_signals = [
        signal_summary
        for signal_summary in actionable_signals
        if signal_summary.get("signal_name") == "TAKE_PROFIT"
    ]
    hold_signals = [
        signal_summary
        for signal_summary in latest_signals
        if signal_summary.get("signal_name") == "HOLD"
    ]

    signals_by_product = {
        str(signal_summary["productId"]): signal_summary
        for signal_summary in latest_signals
        if "productId" in signal_summary
    }

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "modelType": model_type,
        "primarySignal": primary_signal,
        "marketState": _build_market_state_snapshot(
            primary_signal=primary_signal,
            latest_signals=latest_signals,
        ),
        "marketSummary": {
            "totalSignals": len(latest_signals),
            "actionableSignals": len(actionable_signals),
            "signalCounts": {
                "buy": len(buy_signals),
                "take_profit": len(take_profit_signals),
                "wait": len(hold_signals),
            },
        },
        "topBuys": buy_signals[:25],
        "topTakeProfits": take_profit_signals[:25],
        "signals": latest_signals,
        "actionableSignals": actionable_signals,
        "signalsByProduct": signals_by_product,
    }


class SignalSnapshotStore:
    """
    Small in-memory cache for the frontend signal snapshot file.

    This class is meant for lightweight APIs or server-side frontend routes:
    - keep the parsed JSON in memory
    - only reload it when the file changes
    - answer user-facing reads from the cached snapshot

    That pattern keeps response latency low even with many concurrent users.
    """

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path
        self._cached_snapshot: Optional[Dict[str, Any]] = None
        self._cached_mtime: Optional[float] = None

    def get_snapshot(self) -> Dict[str, Any]:
        """Return the current cached snapshot, reloading only if the file changed."""

        self._reload_if_needed()
        if self._cached_snapshot is None:
            raise FileNotFoundError(
                f"Frontend signal snapshot not found: {self.snapshot_path}. "
                "Run `python model-service/scripts/generateSignals.py` first."
            )

        return self._cached_snapshot

    def get_overview(self) -> Dict[str, Any]:
        """Return the small summary block a frontend dashboard loads first."""

        snapshot = self.get_snapshot()
        return {
            "generatedAt": snapshot["generatedAt"],
            "modelType": snapshot["modelType"],
            "marketState": snapshot.get("marketState", {}),
            "marketSummary": snapshot["marketSummary"],
            "primarySignal": snapshot["primarySignal"],
        }

    def get_market_state(self) -> Dict[str, Any]:
        """Return the cached aggregate market-state block for API consumers."""

        snapshot = self.get_snapshot()
        return dict(snapshot.get("marketState", {}))

    def list_signals(
        self,
        action: str = "all",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return filtered signal lists for the frontend.

        Supported actions:
        - `all`
        - `buy`
        - `take_profit`
        - `wait`
        - `actionable`
        """

        snapshot = self.get_snapshot()
        normalized_action = str(action).strip().lower() or "all"

        if normalized_action == "all":
            signal_rows = list(snapshot["signals"])
        elif normalized_action == "actionable":
            signal_rows = list(snapshot["actionableSignals"])
        else:
            signal_rows = [
                signal_summary
                for signal_summary in snapshot["signals"]
                if str(signal_summary.get("spotAction", "")).lower() == normalized_action
            ]

        if limit is not None:
            signal_rows = signal_rows[: max(0, int(limit))]

        return signal_rows

    def get_signal_by_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Return the newest signal for one requested product id."""

        snapshot = self.get_snapshot()
        normalized_product_id = str(product_id).strip().upper()
        return snapshot["signalsByProduct"].get(normalized_product_id)

    def _reload_if_needed(self) -> None:
        """Reload the JSON snapshot only when its file timestamp changes."""

        if not self.snapshot_path.exists():
            self._cached_snapshot = None
            self._cached_mtime = None
            return

        current_mtime = self.snapshot_path.stat().st_mtime
        if self._cached_snapshot is not None and self._cached_mtime == current_mtime:
            return

        with self.snapshot_path.open("r", encoding="utf-8") as snapshot_file:
            self._cached_snapshot = json.load(snapshot_file)

        self._cached_mtime = current_mtime
