"""Frontend-focused helpers for serving cached signal snapshots quickly."""

from __future__ import annotations

from collections import Counter
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _build_market_state_snapshot(
    primary_signal: Optional[Dict[str, Any]],
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


def _build_market_intelligence_snapshot(
    primary_signal: Optional[Dict[str, Any]],
    latest_signals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the market-wide intelligence block from the newest signal set."""

    candidate_contexts = []
    for signal_summary in [primary_signal, *latest_signals]:
        if not isinstance(signal_summary, dict):
            continue
        market_context = signal_summary.get("marketContext", {})
        if not isinstance(market_context, dict):
            continue
        market_intelligence = market_context.get("marketIntelligence", {})
        if isinstance(market_intelligence, dict):
            candidate_contexts.append(market_intelligence)

    first_available_context = next(
        (
            context
            for context in candidate_contexts
            if bool(context.get("available", False))
        ),
        candidate_contexts[0] if candidate_contexts else {},
    )

    return {
        "available": bool(first_available_context.get("available", False)),
        "lastUpdated": first_available_context.get("lastUpdated"),
        "fearGreedValue": float(first_available_context.get("fearGreedValue", 0.0) or 0.0),
        "fearGreedClassification": str(first_available_context.get("fearGreedClassification", "") or ""),
        "btcDominance": float(first_available_context.get("btcDominance", 0.0) or 0.0),
        "btcDominanceChange24h": float(first_available_context.get("btcDominanceChange24h", 0.0) or 0.0),
        "altcoinShare": float(first_available_context.get("altcoinShare", 0.0) or 0.0),
        "stablecoinShare": float(first_available_context.get("stablecoinShare", 0.0) or 0.0),
        "riskMode": str(first_available_context.get("riskMode", "neutral") or "neutral"),
    }


def build_frontend_signal_snapshot(
    model_type: str,
    primary_signal: Optional[Dict[str, Any]],
    latest_signals: List[Dict[str, Any]],
    actionable_signals: List[Dict[str, Any]],
    trader_brain: Dict[str, Any] | None = None,
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
    loss_signals = [
        signal_summary
        for signal_summary in actionable_signals
        if signal_summary.get("signal_name") == "LOSS"
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
        "marketIntelligence": _build_market_intelligence_snapshot(
            primary_signal=primary_signal,
            latest_signals=latest_signals,
        ),
        "marketSummary": {
            "totalSignals": len(latest_signals),
            "actionableSignals": len(actionable_signals),
            "signalCounts": {
                "buy": len(buy_signals),
                "take_profit": len(take_profit_signals),
                "loss": len(loss_signals),
                "wait": len(hold_signals),
            },
        },
        "topBuys": buy_signals[:25],
        "topTakeProfits": take_profit_signals[:25],
        "topLossCuts": loss_signals[:25],
        "signals": latest_signals,
        "actionableSignals": actionable_signals,
        "signalsByProduct": signals_by_product,
        "traderBrain": trader_brain or {},
    }


def build_watchlist_pool_snapshot(
    signal_summaries: List[Dict[str, Any]],
    max_products: int = 12,
) -> Dict[str, Any]:
    """Build one compact persistent snapshot of the strongest watchlist candidates."""

    normalized_limit = max(int(max_products), 0)
    ranked_candidates = sorted(
        [
            signal_summary
            for signal_summary in signal_summaries
            if str((signal_summary.get("brain") or {}).get("decision", "")).strip().lower()
            in {"watchlist", "avoid_long"}
            and str(signal_summary.get("productId", "")).strip()
        ],
        key=lambda signal_summary: (
            0 if str(signal_summary.get("modelSignalName", "")).strip().upper() == "BUY" else 1,
            0 if str(signal_summary.get("signal_name", "")).strip().upper() == "BUY" else 1,
            -float((signal_summary.get("brain") or {}).get("decisionScore", 0.0) or 0.0),
            -float(signal_summary.get("confidence", 0.0) or 0.0),
            str(signal_summary.get("productId", "")),
        ),
    )

    monitored_products: list[Dict[str, Any]] = []
    seen_product_ids: set[str] = set()
    for signal_summary in ranked_candidates:
        product_id = str(signal_summary.get("productId", "")).strip().upper()
        if not product_id or product_id in seen_product_ids:
            continue

        brain = signal_summary.get("brain") or {}
        market_context = signal_summary.get("marketContext") or {}
        monitored_products.append(
            {
                "productId": product_id,
                "signalName": str(signal_summary.get("signal_name", "") or ""),
                "modelSignalName": str(signal_summary.get("modelSignalName", "") or ""),
                "tradeReadiness": str(signal_summary.get("tradeReadiness", "") or ""),
                "confidence": float(signal_summary.get("confidence", 0.0) or 0.0),
                "policyScore": float(signal_summary.get("policyScore", 0.0) or 0.0),
                "setupScore": float(signal_summary.get("setupScore", 0.0) or 0.0),
                "timestamp": signal_summary.get("timestamp"),
                "brainDecision": str(brain.get("decision", "") or ""),
                "decisionScore": float(brain.get("decisionScore", 0.0) or 0.0),
                "summaryLine": str(brain.get("summaryLine", signal_summary.get("reasonSummary", "")) or ""),
                "reasonSummary": str(signal_summary.get("reasonSummary", "") or ""),
                "watchlistStage": str(brain.get("watchlistStage", "") or ""),
                "marketStance": str(brain.get("marketStance", market_context.get("marketStance", "")) or ""),
                "macroRiskMode": str(
                    brain.get(
                        "macroRiskMode",
                        (market_context.get("marketIntelligence") or {}).get("riskMode", ""),
                    )
                    or ""
                ),
            }
        )
        seen_product_ids.add(product_id)
        if normalized_limit and len(monitored_products) >= normalized_limit:
            break

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "count": len(monitored_products),
        "productIds": [product["productId"] for product in monitored_products],
        "products": monitored_products,
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
            "marketIntelligence": snapshot.get("marketIntelligence", {}),
            "marketSummary": snapshot["marketSummary"],
            "primarySignal": snapshot["primarySignal"],
            "traderBrain": snapshot.get("traderBrain", {}),
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


class WatchlistPoolStore:
    """Read the persisted watchlist-monitoring pool published by signal generation."""

    def __init__(self, pool_path: Path) -> None:
        self.pool_path = Path(pool_path)
        self._cached_snapshot: Optional[Dict[str, Any]] = None
        self._cached_mtime: Optional[float] = None

    def get_snapshot(self) -> Dict[str, Any]:
        """Return the latest watchlist pool snapshot or an empty default payload."""

        self._reload_if_needed()
        return dict(
            self._cached_snapshot
            or {
                "generatedAt": None,
                "count": 0,
                "productIds": [],
                "products": [],
            }
        )

    def get_monitored_products(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the ranked watchlist products currently under aggressive monitoring."""

        snapshot = self.get_snapshot()
        products = list(snapshot.get("products", []))
        if limit is not None:
            products = products[: max(int(limit), 0)]
        return products

    def get_monitored_product_ids(self, limit: Optional[int] = None) -> List[str]:
        """Return the product ids that should be polled more aggressively."""

        products = self.get_monitored_products(limit=limit)
        return [
            str(product.get("productId", "")).strip().upper()
            for product in products
            if str(product.get("productId", "")).strip()
        ]

    def _reload_if_needed(self) -> None:
        """Reload the watchlist pool only when the backing JSON file changes."""

        if not self.pool_path.exists():
            self._cached_snapshot = None
            self._cached_mtime = None
            return

        current_mtime = self.pool_path.stat().st_mtime
        if self._cached_snapshot is not None and self._cached_mtime == current_mtime:
            return

        with self.pool_path.open("r", encoding="utf-8") as pool_file:
            self._cached_snapshot = json.load(pool_file)

        self._cached_mtime = current_mtime
