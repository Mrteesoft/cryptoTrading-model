"""Signal publication and published-view services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import OUTPUTS_DIR, TrainingConfig
from ..frontend import SignalSnapshotStore, build_frontend_signal_snapshot, build_watchlist_pool_snapshot
from ..trading.portfolio import TradingPortfolioStore
from ..trading.signal_store import TradingSignalStore


LOGGER = logging.getLogger(__name__)


def _filter_signal_rows(
    signal_rows: list[dict[str, Any]],
    *,
    action: str,
) -> list[dict[str, Any]]:
    """Apply the public action filter semantics used by signal-serving endpoints."""

    normalized_action = str(action).strip().lower() or "all"
    if normalized_action == "all":
        return list(signal_rows)
    if normalized_action == "actionable":
        return [
            signal_summary
            for signal_summary in signal_rows
            if bool(signal_summary.get("actionable", False))
        ]

    return [
        signal_summary
        for signal_summary in signal_rows
        if str(signal_summary.get("spotAction", "")).strip().lower() == normalized_action
    ]


def _build_snapshot_overview(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Trim a signal snapshot into one stable overview payload."""

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
        "marketIntelligence": snapshot.get("marketIntelligence", {}),
        "traderBrain": snapshot.get("traderBrain", {}),
        "topSignals": top_signals,
        "currentSignalStore": snapshot.get("currentSignalStore", {}),
    }


@dataclass
class SignalPublicationArtifacts:
    """Persisted outputs of one signal-publication cycle."""

    latest_signals: list[dict[str, Any]]
    actionable_signals: list[dict[str, Any]]
    primary_signal: dict[str, Any] | None
    frontend_snapshot: dict[str, Any]
    signal_store_summary: dict[str, Any]
    tracked_trade_sync: dict[str, Any]


class SignalPublicationStage:
    """Persist published signals and secondary artifacts."""

    def __init__(
        self,
        *,
        config: TrainingConfig,
        save_json: Callable[[dict[str, Any], Path], None],
        save_dataframe: Callable[[pd.DataFrame, Path], None],
        signal_store_factory: Callable[[], TradingSignalStore],
        primary_history_store=None,
    ) -> None:
        self.config = config
        self.save_json = save_json
        self.save_dataframe = save_dataframe
        self.signal_store_factory = signal_store_factory
        self.primary_history_store = primary_history_store

    def save_watchlist_pool_snapshot(self, signal_summaries: list[dict[str, Any]]) -> None:
        """Persist the strongest watchlist names for more aggressive live monitoring."""

        if not bool(getattr(self.config, "signal_watchlist_pool_enabled", True)):
            return

        max_products = int(getattr(self.config, "signal_watchlist_pool_max_products", 12) or 12)
        watchlist_pool_snapshot = build_watchlist_pool_snapshot(
            signal_summaries=signal_summaries,
            max_products=max_products,
        )
        self.save_json(
            watchlist_pool_snapshot,
            Path(self.config.signal_watchlist_pool_path),
        )

    def publish(
        self,
        *,
        model_type: str,
        historical_prediction_df: pd.DataFrame,
        latest_signals: list[dict[str, Any]],
        actionable_signals: list[dict[str, Any]],
        primary_signal: dict[str, Any] | None,
        trader_brain_snapshot: dict[str, Any],
        signal_source: str,
        signal_metadata: dict[str, Any],
        market_data_refresh: dict[str, Any] | None,
        market_data_refreshed_at: str | None,
        signal_inference_summary: dict[str, Any],
        portfolio_store: TradingPortfolioStore | None = None,
    ) -> SignalPublicationArtifacts:
        """Persist published signals plus their secondary snapshots and histories."""

        published_signals = self._apply_signal_metadata(latest_signals, signal_metadata)
        published_actionable_signals = self._apply_signal_metadata(actionable_signals, signal_metadata)
        published_primary_signal = (
            {
                **primary_signal,
                **signal_metadata,
            }
            if primary_signal is not None
            else None
        )

        frontend_signal_snapshot = build_frontend_signal_snapshot(
            model_type=model_type,
            primary_signal=published_primary_signal,
            latest_signals=published_signals,
            actionable_signals=published_actionable_signals,
            trader_brain=trader_brain_snapshot,
        )
        frontend_signal_snapshot.update(
            {
                "mode": signal_source,
                "marketDataSource": str(self.config.market_data_source),
                "marketDataPath": str(self.config.data_file),
                "marketDataRefresh": market_data_refresh or {},
                "marketDataRefreshedAt": market_data_refreshed_at,
                "marketDataLastTimestamp": signal_metadata.get("marketDataLastTimestamp"),
                "marketDataFirstTimestamp": signal_metadata.get("marketDataFirstTimestamp"),
                "signalInference": dict(signal_inference_summary),
            }
        )

        signal_store_summary = self._persist_live_signal_store(
            latest_signals=published_signals,
            primary_signal=published_primary_signal,
            generated_at=str(frontend_signal_snapshot.get("generatedAt") or datetime.now(timezone.utc).isoformat()),
        )
        frontend_signal_snapshot["currentSignalStore"] = {
            "status": "ready" if int(signal_store_summary.get("signalCount", 0)) > 0 else "empty",
            "signalCount": int(signal_store_summary.get("signalCount", 0)),
            "actionableCount": int(signal_store_summary.get("actionableCount", 0)),
            "primaryProductId": signal_store_summary.get("primaryProductId"),
            "generatedAt": signal_store_summary.get("generatedAt"),
            "persistedAt": signal_store_summary.get("persistedAt"),
            "storageBackend": signal_store_summary.get("storageBackend"),
            "databaseTarget": signal_store_summary.get("databaseTarget"),
        }

        self.save_dataframe(historical_prediction_df, OUTPUTS_DIR / "historicalSignals.csv")
        self.save_json(published_primary_signal or {}, OUTPUTS_DIR / "latestSignal.json")
        self.save_json({"signals": published_signals}, OUTPUTS_DIR / "latestSignals.json")
        self.save_json({"signals": published_actionable_signals}, OUTPUTS_DIR / "actionableSignals.json")
        self.save_json(frontend_signal_snapshot, OUTPUTS_DIR / "frontendSignalSnapshot.json")
        if published_primary_signal is not None and self.primary_history_store is not None:
            self.primary_history_store.save_primary_signal(published_primary_signal, signal_source)

        tracked_trade_sync = self._sync_generated_signal_trades(
            latest_signals=published_signals,
            signal_source=signal_source,
            portfolio_store=portfolio_store,
        )

        return SignalPublicationArtifacts(
            latest_signals=published_signals,
            actionable_signals=published_actionable_signals,
            primary_signal=published_primary_signal,
            frontend_snapshot=frontend_signal_snapshot,
            signal_store_summary=signal_store_summary,
            tracked_trade_sync=tracked_trade_sync,
        )

    @staticmethod
    def _apply_signal_metadata(
        signal_summaries: list[dict[str, Any]],
        signal_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Attach common publication metadata to every signal row."""

        return [
            {
                **signal_summary,
                **signal_metadata,
            }
            for signal_summary in signal_summaries
        ]

    def _build_signal_store(self) -> TradingSignalStore:
        """Create the persistent live-signal store for current and historical rows."""

        return self.signal_store_factory()

    def _log_live_signal_rows(self, signal_summaries: list[dict[str, Any]]) -> None:
        """Write one structured log line for every published live signal."""

        for signal_summary in signal_summaries:
            product_id = str(signal_summary.get("productId", "")).strip().upper() or str(
                signal_summary.get("pairSymbol", "")
            ).strip().upper()
            LOGGER.info(
                (
                    "Live signal created: symbol=%s signal=%s action=%s confidence=%.4f "
                    "timestamp=%s price=%.8f watchlistFallback=%s"
                ),
                product_id or str(signal_summary.get("symbol", "")).strip().upper() or "UNKNOWN",
                str(signal_summary.get("signal_name", "UNKNOWN")).strip().upper(),
                str(signal_summary.get("spotAction", "wait")).strip().lower(),
                float(signal_summary.get("confidence", 0.0) or 0.0),
                str(signal_summary.get("timestamp", "")).strip() or "unknown",
                float(signal_summary.get("close", 0.0) or 0.0),
                bool(signal_summary.get("watchlistFallback", False)),
            )

    def _persist_live_signal_store(
        self,
        latest_signals: list[dict[str, Any]],
        primary_signal: dict[str, Any] | None,
        generated_at: str,
    ) -> dict[str, Any]:
        """Persist the current live signal set to the database and log each row."""

        signal_store = self._build_signal_store()
        persistence_summary = signal_store.replace_current_signals(
            signal_summaries=latest_signals,
            primary_signal=primary_signal,
            generated_at=generated_at,
        )
        self._log_live_signal_rows(latest_signals)
        LOGGER.info(
            "Persisted %s live signal(s) to %s via %s.",
            int(persistence_summary["signalCount"]),
            persistence_summary["databaseTarget"],
            persistence_summary["storageBackend"],
        )
        return persistence_summary

    @staticmethod
    def _build_signal_tracking_metadata(
        signal_summary: dict[str, Any],
        signal_source: str,
    ) -> dict[str, Any]:
        """Build one compact metadata payload for an auto-tracked signal trade."""

        brain = signal_summary.get("brain", {}) if isinstance(signal_summary.get("brain"), dict) else {}
        return {
            "autoTrackedFromSignalGeneration": True,
            "signalSource": signal_source,
            "signalTimestamp": signal_summary.get("timestamp"),
            "signalName": signal_summary.get("signal_name"),
            "confidence": signal_summary.get("confidence"),
            "productId": signal_summary.get("productId"),
            "marketDataSource": signal_summary.get("marketDataSource"),
            "brainDecision": brain.get("decision"),
            "brainSummary": brain.get("summaryLine"),
        }

    def _sync_generated_signal_trades(
        self,
        latest_signals: list[dict[str, Any]],
        signal_source: str,
        portfolio_store: TradingPortfolioStore | None = None,
    ) -> dict[str, Any]:
        """Create tracked trade records for fresh BUY signals and refresh active records."""

        if not bool(getattr(self.config, "signal_track_generated_trades", False)):
            return {
                "enabled": False,
                "createdCount": 0,
                "refreshedCount": 0,
                "skippedCount": 0,
                "trackedTradeIds": [],
            }

        portfolio_store = portfolio_store or TradingPortfolioStore(
            db_path=self.config.portfolio_store_path,
            default_capital=self.config.portfolio_default_capital,
            database_url=self.config.portfolio_store_url,
        )
        created_count = 0
        refreshed_count = 0
        skipped_count = 0
        tracked_trade_ids: list[int] = []

        for signal_summary in latest_signals:
            product_id = str(signal_summary.get("productId", "")).strip().upper()
            if not product_id:
                skipped_count += 1
                continue

            signal_name = str(signal_summary.get("signal_name", "")).strip().upper()
            brain = signal_summary.get("brain", {}) if isinstance(signal_summary.get("brain"), dict) else {}
            signal_metadata = self._build_signal_tracking_metadata(signal_summary, signal_source)
            active_trade = portfolio_store.get_active_trade_for_product(product_id)

            if active_trade is not None:
                refreshed_trade = portfolio_store.refresh_trade(
                    trade_id=int(active_trade["tradeId"]),
                    current_price=float(signal_summary.get("close") or 0.0) or None,
                    stop_loss_price=brain.get("stopLossPrice"),
                    take_profit_price=brain.get("takeProfitPrice"),
                    signal_name=signal_name or None,
                    metadata=signal_metadata,
                )
                refreshed_count += 1
                tracked_trade_ids.append(int(refreshed_trade["tradeId"]))
                continue

            if signal_name != "BUY":
                skipped_count += 1
                continue
            if not bool(signal_summary.get("actionable", False)):
                skipped_count += 1
                continue

            entry_price = float(signal_summary.get("close") or 0.0)
            if entry_price <= 0:
                skipped_count += 1
                continue

            tracked_trade = portfolio_store.create_trade(
                product_id=product_id,
                entry_price=entry_price,
                take_profit_price=brain.get("takeProfitPrice"),
                stop_loss_price=brain.get("stopLossPrice"),
                quantity=0.0,
                current_price=entry_price,
                signal_name=signal_name,
                status=str(getattr(self.config, "signal_generated_trade_status", "planned") or "planned"),
                opened_at=str(signal_summary.get("timestamp") or "") or None,
                metadata=signal_metadata,
            )
            created_count += 1
            tracked_trade_ids.append(int(tracked_trade["tradeId"]))

        return {
            "enabled": True,
            "createdCount": created_count,
            "refreshedCount": refreshed_count,
            "skippedCount": skipped_count,
            "trackedTradeIds": tracked_trade_ids,
            "storageBackend": portfolio_store.database.storage_backend,
            "databaseTarget": portfolio_store.database.database_target,
        }


class PublishedSignalViewService:
    """Serve authoritative published signals, with narrow fallback paths for tool consumers."""

    def __init__(
        self,
        *,
        signal_store: TradingSignalStore,
        snapshot_store: SignalSnapshotStore | None = None,
        cached_snapshot_provider: Callable[[], dict[str, Any] | None] | None = None,
        fallback_snapshot_provider: Callable[..., dict[str, Any] | None] | None = None,
        refresh_callback: Callable[[], None] | None = None,
    ) -> None:
        self.signal_store = signal_store
        self.snapshot_store = snapshot_store
        self.cached_snapshot_provider = cached_snapshot_provider
        self.fallback_snapshot_provider = fallback_snapshot_provider
        self.refresh_callback = refresh_callback

    def get_current_signal(self) -> dict[str, Any] | None:
        """Return the primary current signal from the authoritative signal store."""

        return self.signal_store.get_current_signal()

    def list_current_signals(
        self,
        *,
        action: str = "all",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return the authoritative current signal set filtered for API serving."""

        signal_rows = self.signal_store.list_current_signals(limit=500)
        filtered_rows = _filter_signal_rows(signal_rows, action=action)
        return filtered_rows[: max(0, int(limit))]

    def build_current_signals_response(
        self,
        *,
        action: str = "all",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Build the stable API payload for the authoritative current signal set."""

        signal_rows = self.list_current_signals(action=action, limit=limit)
        store_status = self.signal_store.get_status()
        return {
            "action": action,
            "count": len(signal_rows),
            "signals": signal_rows,
            "generatedAt": store_status.get("generatedAt"),
            "primaryProductId": store_status.get("primaryProductId"),
            "storageBackend": store_status.get("storageBackend"),
            "databaseTarget": store_status.get("databaseTarget"),
        }

    def build_signal_history_response(
        self,
        *,
        limit: int = 100,
        product_id: str | None = None,
    ) -> dict[str, Any]:
        """Build the stable API payload for persisted signal history."""

        signal_rows = self.signal_store.list_signal_history(
            limit=limit,
            product_id=product_id,
        )
        return {
            "count": len(signal_rows),
            "productId": str(product_id).strip().upper() or None if product_id is not None else None,
            "signals": signal_rows,
            "storageBackend": self.signal_store.database.storage_backend,
            "databaseTarget": self.signal_store.database.database_target,
        }

    def load_tool_signal_state(
        self,
        *,
        force_refresh: bool = False,
        product_id: str | None = None,
    ) -> dict[str, Any]:
        """Return one tool-facing signal state bundle with published-first semantics."""

        if force_refresh and self.refresh_callback is not None:
            self.refresh_callback()

        if force_refresh and self.fallback_snapshot_provider is not None:
            fallback_snapshot = self._load_fallback_snapshot(
                force_refresh=force_refresh,
                product_id=product_id,
            )
            if fallback_snapshot is not None:
                return {
                    "status": "ok",
                    "source": "live",
                    "warning": "",
                    "error": "",
                    "signal": self._resolve_signal_from_snapshot(fallback_snapshot, product_id),
                    "signals": list(fallback_snapshot.get("signals", [])),
                    "overview": _build_snapshot_overview(fallback_snapshot),
                }

        current_signals = self.signal_store.list_current_signals(limit=500)
        if current_signals:
            current_signal = self.signal_store.get_current_signal()
            snapshot = self._load_cached_snapshot()
            return {
                "status": "ok",
                "source": "published",
                "warning": "",
                "error": "",
                "signal": (
                    self._find_signal_by_product(current_signals, product_id)
                    if product_id is not None
                    else current_signal
                ),
                "signals": current_signals,
                "overview": self._build_published_overview(
                    current_signals=current_signals,
                    current_signal=current_signal,
                    snapshot=snapshot,
                ),
            }

        snapshot = self._load_cached_snapshot()
        if snapshot is not None:
            return {
                "status": "ok",
                "source": "snapshot",
                "warning": "",
                "error": "",
                "signal": self._resolve_signal_from_snapshot(snapshot, product_id),
                "signals": list(snapshot.get("signals", [])),
                "overview": _build_snapshot_overview(snapshot),
            }

        fallback_snapshot = self._load_fallback_snapshot(
            force_refresh=force_refresh,
            product_id=product_id,
        )
        if fallback_snapshot is not None:
            return {
                "status": "ok",
                "source": "live",
                "warning": "",
                "error": "",
                "signal": self._resolve_signal_from_snapshot(fallback_snapshot, product_id),
                "signals": list(fallback_snapshot.get("signals", [])),
                "overview": _build_snapshot_overview(fallback_snapshot),
            }

        return {
            "status": "error",
            "source": "unavailable",
            "warning": "",
            "error": "No published signal state is available yet.",
            "signal": None,
            "signals": [],
            "overview": {},
        }

    def _build_published_overview(
        self,
        *,
        current_signals: list[dict[str, Any]],
        current_signal: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build one market overview from authoritative current signals."""

        snapshot_overview = _build_snapshot_overview(snapshot)
        actionable_signals = [
            signal_summary
            for signal_summary in current_signals
            if bool(signal_summary.get("actionable", False))
        ]
        signal_counts = {
            "buy": sum(
                1
                for signal_summary in current_signals
                if str(signal_summary.get("signal_name", "")).strip().upper() == "BUY"
            ),
            "take_profit": sum(
                1
                for signal_summary in current_signals
                if str(signal_summary.get("signal_name", "")).strip().upper() == "TAKE_PROFIT"
            ),
            "loss": sum(
                1
                for signal_summary in current_signals
                if str(signal_summary.get("signal_name", "")).strip().upper() == "LOSS"
            ),
            "wait": sum(
                1
                for signal_summary in current_signals
                if str(signal_summary.get("signal_name", "")).strip().upper() == "HOLD"
            ),
        }
        store_status = self.signal_store.get_status()
        top_signals = actionable_signals[:3] if actionable_signals else current_signals[:3]

        return {
            "generatedAt": store_status.get("generatedAt") or snapshot_overview.get("generatedAt"),
            "mode": snapshot_overview.get("mode") or "published",
            "marketDataSource": snapshot_overview.get("marketDataSource"),
            "requestMode": snapshot_overview.get("requestMode") or "published-current-store",
            "productsCovered": len(current_signals),
            "granularitySeconds": snapshot_overview.get("granularitySeconds"),
            "primarySignal": current_signal,
            "marketSummary": {
                "totalSignals": len(current_signals),
                "actionableSignals": len(actionable_signals),
                "signalCounts": signal_counts,
            },
            "marketState": snapshot_overview.get("marketState", {}),
            "marketIntelligence": snapshot_overview.get("marketIntelligence", {}),
            "traderBrain": snapshot_overview.get("traderBrain", {}),
            "topSignals": top_signals,
            "currentSignalStore": {
                "status": store_status.get("status"),
                "signalCount": len(current_signals),
                "generatedAt": store_status.get("generatedAt"),
                "primaryProductId": store_status.get("primaryProductId"),
                "storageBackend": store_status.get("storageBackend"),
                "databaseTarget": store_status.get("databaseTarget"),
            },
        }

    def _load_cached_snapshot(self) -> dict[str, Any] | None:
        """Return the cached published snapshot when one is available."""

        if self.snapshot_store is not None:
            try:
                return self.snapshot_store.get_snapshot()
            except FileNotFoundError:
                pass

        if self.cached_snapshot_provider is None:
            return None

        try:
            return self.cached_snapshot_provider()
        except Exception:
            return None

    def _load_fallback_snapshot(
        self,
        *,
        force_refresh: bool,
        product_id: str | None,
    ) -> dict[str, Any] | None:
        """Load a last-resort fallback snapshot for tools before publication exists."""

        if self.fallback_snapshot_provider is None:
            return None

        try:
            return self.fallback_snapshot_provider(
                force_refresh=force_refresh,
                product_id=product_id,
            )
        except TypeError:
            try:
                return self.fallback_snapshot_provider(force_refresh=force_refresh)
            except Exception:
                return None
        except Exception:
            return None

    @staticmethod
    def _find_signal_by_product(
        signal_rows: list[dict[str, Any]],
        product_id: str | None,
    ) -> dict[str, Any] | None:
        """Resolve one signal row from the current published signal set."""

        normalized_product_id = str(product_id or "").strip().upper()
        if not normalized_product_id:
            return None

        for signal_summary in signal_rows:
            if str(signal_summary.get("productId", "")).strip().upper() == normalized_product_id:
                return signal_summary

        return None

    @staticmethod
    def _resolve_signal_from_snapshot(
        snapshot: dict[str, Any],
        product_id: str | None,
    ) -> dict[str, Any] | None:
        """Resolve one signal row from a cached or fallback snapshot."""

        if product_id is None:
            primary_signal = snapshot.get("primarySignal")
            return primary_signal if isinstance(primary_signal, dict) else None

        normalized_product_id = str(product_id).strip().upper()
        signal_by_product = snapshot.get("signalsByProduct", {})
        if isinstance(signal_by_product, dict):
            signal_summary = signal_by_product.get(normalized_product_id)
            if isinstance(signal_summary, dict):
                return signal_summary

        for signal_summary in list(snapshot.get("signals", [])):
            if str(signal_summary.get("productId", "")).strip().upper() == normalized_product_id:
                return signal_summary

        return None
