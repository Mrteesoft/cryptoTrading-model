"""Realtime-style market inference helpers for the trading assistant backend."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from .config import MODELS_DIR, TrainingConfig, apply_runtime_market_data_settings
from .data import create_market_data_loader
from .frontend import WatchlistPoolStore, build_frontend_signal_snapshot
from .modeling import BaseSignalModel, get_model_class
from .pipeline import CryptoDatasetBuilder
from .trading.portfolio import TradingPortfolioStore
from .trading.signals import (
    apply_signal_trade_context,
    build_actionable_signal_summaries,
    build_latest_signal_summaries,
    filter_published_signal_summaries,
    is_signal_product_excluded,
    select_primary_signal,
)
from .trading.trader_brain import TraderBrain


class LiveSignalEngine:
    """
    Run on-demand inference from fresh market data using the deployed model.

    The current project uses public Coinbase REST candles rather than an
    exchange websocket feed, so this is "live" in the practical sense that
    every refresh pulls the latest finished candles before scoring them.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        self.model_dir = Path(model_dir or MODELS_DIR)
        self.config = config or TrainingConfig()
        self._cached_snapshot: Optional[Dict[str, Any]] = None
        self._cached_snapshot_generated_at: Optional[datetime] = None
        self._cached_request_key: Optional[str] = None

    def get_status(self) -> Dict[str, Any]:
        """Return the current cache and model readiness without forcing a refresh."""

        model_path = self._resolve_model_path()
        cache_age_seconds = None
        watchlist_pool_product_ids = self._load_watchlist_pool_product_ids()
        if self._cached_snapshot_generated_at is not None:
            cache_age_seconds = max(
                int((datetime.now(timezone.utc) - self._cached_snapshot_generated_at).total_seconds()),
                0,
            )

        return {
            "status": "ready" if model_path.exists() else "missing_model",
            "modelPath": str(model_path),
            "cacheAgeSeconds": cache_age_seconds,
            "cacheTtlSeconds": self._resolve_effective_cache_ttl_seconds(watchlist_pool_product_ids),
            "lastGeneratedAt": (
                self._cached_snapshot.get("generatedAt")
                if self._cached_snapshot is not None
                else None
            ),
            "watchlistPoolCount": len(watchlist_pool_product_ids),
        }

    def _load_watchlist_pool_product_ids(self) -> list[str]:
        """Load the persisted watchlist pool that should be monitored more aggressively."""

        if not bool(getattr(self.config, "signal_watchlist_pool_enabled", True)):
            return []

        pool_store = WatchlistPoolStore(Path(self.config.signal_watchlist_pool_path))
        max_products = int(getattr(self.config, "signal_watchlist_pool_max_products", 12) or 12)
        return pool_store.get_monitored_product_ids(limit=max_products)

    def _build_unbatched_runtime_config(self, runtime_config: TrainingConfig) -> TrainingConfig:
        """Clone one runtime config with market batch slicing disabled."""

        if str(runtime_config.market_data_source).strip() in {"coinmarketcap", "coinmarketcapLatestQuotes"}:
            return replace(
                runtime_config,
                coinmarketcap_product_batch_size=None,
                coinmarketcap_product_batch_number=1,
            )

        return replace(
            runtime_config,
            coinbase_product_batch_size=None,
            coinbase_product_batch_number=1,
        )

    def _score_prediction_frame(
        self,
        runtime_config: TrainingConfig,
        model: BaseSignalModel,
        *,
        product_ids: Sequence[str],
        use_quote_universe: bool,
    ) -> tuple[Any, dict[str, int]]:
        """Build one prediction frame for either the top universe or an explicit watchlist pool."""

        loader_config = self._build_unbatched_runtime_config(runtime_config)
        data_loader = create_market_data_loader(
            config=loader_config,
            data_path=loader_config.data_file,
            should_save_downloaded_data=False,
            product_ids=tuple(product_ids),
            fetch_all_quote_products=use_quote_universe,
            max_products=loader_config.live_max_products if use_quote_universe else None,
            granularity_seconds=loader_config.live_granularity_seconds,
            total_candles=loader_config.live_total_candles,
            request_pause_seconds=loader_config.live_request_pause_seconds,
            save_progress_every_products=0,
            log_progress=False,
        )
        dataset_builder = CryptoDatasetBuilder(
            config=loader_config,
            feature_columns=model.feature_columns,
            data_loader=data_loader,
        )
        feature_df = dataset_builder.build_feature_table()
        prediction_df = model.predict(feature_df)
        return prediction_df, {
            "rowsScored": int(len(prediction_df)),
            "productsScored": (
                int(prediction_df["product_id"].nunique())
                if "product_id" in prediction_df.columns
                else int(len(prediction_df))
            ),
        }

    def _resolve_effective_cache_ttl_seconds(
        self,
        watchlist_pool_product_ids: Sequence[str],
    ) -> int:
        """Use a shorter cache window while the monitored watchlist pool is active."""

        base_cache_ttl_seconds = int(self.config.live_signal_cache_seconds)
        if not watchlist_pool_product_ids:
            return base_cache_ttl_seconds

        aggressive_cache_ttl_seconds = int(
            getattr(self.config, "live_watchlist_pool_cache_seconds", base_cache_ttl_seconds)
            or base_cache_ttl_seconds
        )
        return max(1, min(base_cache_ttl_seconds, aggressive_cache_ttl_seconds))

    def get_live_snapshot(
        self,
        force_refresh: bool = False,
        product_id: str | None = None,
        product_ids: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        """Return a fresh or cached live signal snapshot for the requested universe."""

        use_quote_universe = self._should_use_quote_universe(
            product_id=product_id,
            product_ids=product_ids,
        )
        resolved_product_ids = self._resolve_requested_products(
            product_id=product_id,
            product_ids=product_ids,
        )
        watchlist_pool_product_ids: list[str] = []
        if not product_id and not product_ids:
            watchlist_pool_product_ids = self._load_watchlist_pool_product_ids()
        effective_cache_ttl_seconds = self._resolve_effective_cache_ttl_seconds(watchlist_pool_product_ids)
        request_key = self._build_request_key(
            product_id=product_id,
            product_ids=resolved_product_ids,
            monitored_product_ids=watchlist_pool_product_ids,
        )

        if not force_refresh and self._can_use_cached_snapshot(
            request_key=request_key,
            cache_ttl_seconds=effective_cache_ttl_seconds,
        ):
            return dict(self._cached_snapshot or {})

        model = self._load_model()
        runtime_config = replace(
            apply_runtime_market_data_settings(
                base_config=model.config,
                runtime_config=self.config,
            ),
            live_product_ids=tuple(resolved_product_ids),
            live_fetch_all_quote_products=use_quote_universe,
            live_max_products=self.config.live_max_products,
            live_granularity_seconds=self.config.live_granularity_seconds,
            live_total_candles=self.config.live_total_candles,
            live_request_pause_seconds=self.config.live_request_pause_seconds,
            live_signal_cache_seconds=effective_cache_ttl_seconds,
            assistant_system_name=self.config.assistant_system_name,
            assistant_enable_retrieval=self.config.assistant_enable_retrieval,
            assistant_memory_message_limit=self.config.assistant_memory_message_limit,
            assistant_retrieval_item_limit=self.config.assistant_retrieval_item_limit,
        )

        prediction_frames = []
        prediction_df, prediction_summary = self._score_prediction_frame(
            runtime_config,
            model,
            product_ids=resolved_product_ids,
            use_quote_universe=runtime_config.live_fetch_all_quote_products,
        )
        prediction_frames.append(prediction_df)
        watchlist_pool_summary = {
            "active": False,
            "count": 0,
            "productIds": [],
            "rowsScored": 0,
            "productsScored": 0,
        }
        if watchlist_pool_product_ids:
            watchlist_prediction_df, explicit_summary = self._score_prediction_frame(
                runtime_config,
                model,
                product_ids=watchlist_pool_product_ids,
                use_quote_universe=False,
            )
            if not watchlist_prediction_df.empty:
                prediction_frames.append(watchlist_prediction_df)
            watchlist_pool_summary = {
                "active": True,
                "count": len(watchlist_pool_product_ids),
                "productIds": list(watchlist_pool_product_ids),
                "rowsScored": int(explicit_summary["rowsScored"]),
                "productsScored": int(explicit_summary["productsScored"]),
            }

        prediction_df = prediction_frames[0]
        if len(prediction_frames) > 1:
            prediction_df = pd.concat(prediction_frames, ignore_index=True)
            duplicate_subset = [
                column_name
                for column_name in ("product_id", "timestamp", "time_step")
                if column_name in prediction_df.columns
            ]
            if duplicate_subset:
                prediction_df = prediction_df.drop_duplicates(
                    subset=duplicate_subset,
                    keep="last",
                )
        latest_signals = build_latest_signal_summaries(
            prediction_df,
            minimum_action_confidence=runtime_config.backtest_min_confidence,
            config=runtime_config,
        )
        if not latest_signals:
            raise ValueError(
                "No live signals remained after applying the configured signal-universe exclusions."
            )
        portfolio_store = TradingPortfolioStore(
            db_path=runtime_config.portfolio_store_path,
            default_capital=runtime_config.portfolio_default_capital,
            database_url=runtime_config.portfolio_store_url,
        )
        portfolio = portfolio_store.get_portfolio()
        positions_by_product = {
            str(position.get("productId", "")).strip().upper(): position
            for position in list(portfolio.get("positions", []))
            if str(position.get("productId", "")).strip()
        }
        active_signal_context_by_product: dict[str, dict[str, Any]] = {}
        for signal_summary in latest_signals:
            signal_product_id = str(signal_summary.get("productId", "")).strip().upper()
            if not signal_product_id:
                continue
            active_trade = portfolio_store.get_active_trade_for_product(signal_product_id)
            position = positions_by_product.get(signal_product_id)
            if active_trade is None and position is None:
                continue
            active_signal_context_by_product[signal_product_id] = {
                "entryPrice": (
                    position.get("entryPrice")
                    if position is not None and position.get("entryPrice") is not None
                    else (active_trade.get("entryPrice") if active_trade is not None else None)
                ),
                "currentPrice": (
                    position.get("currentPrice")
                    if position is not None and position.get("currentPrice") is not None
                    else (active_trade.get("currentPrice") if active_trade is not None else None)
                ),
                "stopLossPrice": active_trade.get("stopLossPrice") if active_trade is not None else None,
                "takeProfitPrice": active_trade.get("takeProfitPrice") if active_trade is not None else None,
                "positionFraction": position.get("positionFraction") if position is not None else None,
                "quantity": position.get("quantity") if position is not None else None,
                "openedAt": (
                    position.get("openedAt")
                    if position is not None and position.get("openedAt") is not None
                    else (active_trade.get("openedAt") if active_trade is not None else None)
                ),
                "status": active_trade.get("status") if active_trade is not None else None,
            }
        latest_signals = apply_signal_trade_context(
            latest_signals,
            active_trade_product_ids=portfolio_store.get_active_signal_product_ids(),
            active_signal_context_by_product=active_signal_context_by_product,
            config=runtime_config,
        )
        trader_brain_plan = TraderBrain(config=runtime_config).build_plan(
            signal_summaries=latest_signals,
            positions=list(portfolio.get("positions", [])),
            capital=float(portfolio["capital"]),
            trade_memory_by_product=portfolio_store.build_trade_learning_map(latest_signals),
        )
        latest_signals = trader_brain_plan["signals"]
        trader_brain_snapshot = {
            key: value
            for key, value in trader_brain_plan.items()
            if key != "signals"
        }
        latest_signals = filter_published_signal_summaries(latest_signals)
        actionable_signals = build_actionable_signal_summaries(latest_signals)
        primary_signal = select_primary_signal(latest_signals) if latest_signals else None

        live_snapshot = build_frontend_signal_snapshot(
            model_type=model.model_type,
            primary_signal=primary_signal,
            latest_signals=latest_signals,
            actionable_signals=actionable_signals,
            trader_brain=trader_brain_snapshot,
        )
        live_snapshot.update(
            {
                "mode": "live",
                "marketDataSource": str(runtime_config.market_data_source),
                "requestMode": (
                    "requested-product"
                    if product_id
                    else "requested-products"
                    if product_ids
                    else "quote-universe-plus-watchlist-pool"
                    if use_quote_universe and watchlist_pool_product_ids
                    else "quote-universe"
                    if use_quote_universe
                    else "configured-watchlist"
                ),
                "requestedProducts": resolved_product_ids,
                "productsCovered": len(latest_signals),
                "featureRowsScored": int(len(prediction_df)),
                "granularitySeconds": int(runtime_config.live_granularity_seconds),
                "liveSignalCacheSeconds": int(effective_cache_ttl_seconds),
                "minimumActionConfidence": float(runtime_config.backtest_min_confidence),
                "modelPath": str(self._resolve_model_path()),
                "watchlistPool": watchlist_pool_summary,
                "signalInference": {
                    "rowsScored": int(prediction_summary["rowsScored"]),
                    "productsScored": int(prediction_summary["productsScored"]),
                    "watchlistPoolRowsScored": int(watchlist_pool_summary["rowsScored"]),
                    "watchlistPoolProductsScored": int(watchlist_pool_summary["productsScored"]),
                },
            }
        )

        self._cached_snapshot = dict(live_snapshot)
        self._cached_snapshot_generated_at = datetime.now(timezone.utc)
        self._cached_request_key = request_key

        return dict(live_snapshot)

    def list_signals(
        self,
        action: str = "all",
        limit: Optional[int] = None,
        force_refresh: bool = False,
    ) -> list[Dict[str, Any]]:
        """Return live signals filtered by action."""

        snapshot = self.get_live_snapshot(force_refresh=force_refresh)
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
            signal_rows = signal_rows[: max(int(limit), 0)]

        return signal_rows

    def get_signal_by_product(
        self,
        product_id: str,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Return one live signal summary for the requested product when available."""

        snapshot = self.get_live_snapshot(
            force_refresh=force_refresh,
            product_id=product_id,
        )
        return snapshot["signalsByProduct"].get(str(product_id).strip().upper())

    def _load_model(self) -> BaseSignalModel:
        """Load the current deployed model artifact."""

        model_path = self._resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model artifact found for live inference: {model_path}. "
                "Run `python model-service/scripts/trainModel.py` first."
            )

        return BaseSignalModel.load(model_path)

    def _resolve_model_path(self) -> Path:
        """Resolve the active model path using the configured default filename first."""

        default_model_class = get_model_class(self.config.model_type)
        default_model_path = self.model_dir / default_model_class.default_model_filename
        if default_model_path.exists():
            return default_model_path

        model_candidates = sorted(
            (path for path in self.model_dir.glob("*.pkl") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if model_candidates:
            return model_candidates[0]

        return default_model_path

    def _resolve_requested_products(
        self,
        product_id: str | None,
        product_ids: Sequence[str] | None,
    ) -> list[str]:
        """Normalize the requested product universe for one live inference call."""

        if product_id:
            normalized_product_id = self._normalize_product_id(product_id)
            if is_signal_product_excluded(
                product_id=normalized_product_id,
                config=self.config,
            ):
                return []
            return [normalized_product_id]

        if product_ids:
            return [
                normalized_product_id
                for normalized_product_id in (
                    self._normalize_product_id(value)
                    for value in product_ids
                    if str(value).strip()
                )
                if not is_signal_product_excluded(
                    product_id=normalized_product_id,
                    config=self.config,
                )
            ]

        if self.config.live_fetch_all_quote_products:
            return []

        return [
            normalized_product_id
            for normalized_product_id in (
                self._normalize_product_id(value)
                for value in self.config.live_product_ids
                if str(value).strip()
            )
            if not is_signal_product_excluded(
                product_id=normalized_product_id,
                config=self.config,
            )
        ]

    def _should_use_quote_universe(
        self,
        product_id: str | None,
        product_ids: Sequence[str] | None,
    ) -> bool:
        """Return whether one live request should score the broader quote universe."""

        if product_id or product_ids:
            return False

        return bool(self.config.live_fetch_all_quote_products)

    def _build_request_key(
        self,
        product_id: str | None,
        product_ids: Sequence[str],
        monitored_product_ids: Sequence[str] | None = None,
    ) -> str:
        """Build a stable cache key for the current live request."""

        if product_id:
            return f"product:{self._normalize_product_id(product_id)}"

        if product_ids:
            return "products:" + ",".join(sorted(self._normalize_product_id(value) for value in product_ids))

        if monitored_product_ids:
            return "universe:all|watchlist:" + ",".join(sorted(self._normalize_product_id(value) for value in monitored_product_ids))

        return "universe:all"

    def _can_use_cached_snapshot(self, request_key: str, cache_ttl_seconds: int) -> bool:
        """Return whether the in-memory live snapshot is still valid for this request."""

        if self._cached_snapshot is None or self._cached_snapshot_generated_at is None:
            return False

        if self._cached_request_key != request_key:
            return False

        expires_at = self._cached_snapshot_generated_at + timedelta(seconds=int(cache_ttl_seconds))
        return datetime.now(timezone.utc) < expires_at

    @staticmethod
    def _normalize_product_id(product_id: str) -> str:
        """Normalize Coinbase-style product ids."""

        return str(product_id).strip().upper()
