"""Realtime-style market inference helpers for the trading assistant backend."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from .config import (
    MODELS_DIR,
    TrainingConfig,
    apply_runtime_market_data_settings,
    is_coinmarketcap_market_data_source,
)
from .data import create_market_data_loader
from .frontend import WatchlistPoolStore, build_frontend_signal_snapshot
from .modeling import BaseSignalModel, get_model_class
from .pipeline import CryptoDatasetBuilder
from .source_refresh import ActiveUniversePlan, SignalUniverseCoordinator
from .trading.portfolio import TradingPortfolioStore
from .trading.signals import is_signal_product_excluded, select_primary_signal
from .application import (
    SignalContextEnrichmentStage,
    SignalDecisionStage,
    SignalEnrichmentStage,
    SignalGenerationCoordinator,
    SignalInferenceStage,
)


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

    def _should_use_prioritized_active_universe(
        self,
        *,
        product_id: str | None,
        product_ids: Sequence[str] | None,
    ) -> bool:
        """Use the cache-first CMC universe path only for the default no-argument live request."""

        return bool(
            not product_id
            and not product_ids
            and is_coinmarketcap_market_data_source(self.config.market_data_source)
            and self.config.live_fetch_all_quote_products
        )

    def _load_prioritized_active_universe_plan(self) -> ActiveUniversePlan:
        """Build the active live-analysis universe from cached ranking plus follow-up state."""

        return SignalUniverseCoordinator(self.config).resolve_active_universe(
            max_products=self.config.live_max_products,
        )

    def get_live_snapshot(
        self,
        force_refresh: bool = False,
        product_id: str | None = None,
        product_ids: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        """Return a fresh or cached live signal snapshot for the requested universe."""

        active_universe_plan: ActiveUniversePlan | None = None
        use_prioritized_active_universe = self._should_use_prioritized_active_universe(
            product_id=product_id,
            product_ids=product_ids,
        )
        use_quote_universe = self._should_use_quote_universe(
            product_id=product_id,
            product_ids=product_ids,
        )
        resolved_product_ids = self._resolve_requested_products(
            product_id=product_id,
            product_ids=product_ids,
        )
        watchlist_pool_product_ids: list[str] = []
        if use_prioritized_active_universe:
            active_universe_plan = self._load_prioritized_active_universe_plan()
            resolved_product_ids = list(active_universe_plan.product_ids)
            watchlist_pool_product_ids = list(
                active_universe_plan.summary.get("watchlist", {}).get("productIds", [])
            )
            use_quote_universe = False
            if not resolved_product_ids:
                raise ValueError(
                    "No active live-analysis products were available after applying the cache-first "
                    "CoinMarketCap universe policy."
                )
        elif not product_id and not product_ids:
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
        if active_universe_plan is not None:
            scored_watchlist_product_ids = set()
            watchlist_pool_row_count = 0
            if "product_id" in prediction_df.columns:
                normalized_prediction_product_ids = prediction_df["product_id"].astype(str).str.upper()
                watched_product_id_set = {str(product).strip().upper() for product in watchlist_pool_product_ids}
                watchlist_pool_row_count = int(normalized_prediction_product_ids.isin(watched_product_id_set).sum())
                scored_watchlist_product_ids = {
                    product_value
                    for product_value in normalized_prediction_product_ids.unique()
                    if product_value in watched_product_id_set
                }

            watchlist_pool_summary = {
                "active": bool(watchlist_pool_product_ids),
                "count": len(watchlist_pool_product_ids),
                "productIds": list(watchlist_pool_product_ids),
                "rowsScored": int(watchlist_pool_row_count),
                "productsScored": int(len(scored_watchlist_product_ids)),
            }
        elif watchlist_pool_product_ids:
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
        inference_stage = SignalInferenceStage(runtime_config)
        inference_summary = {
            "rowsScored": int(prediction_summary["rowsScored"]),
            "productsScored": int(prediction_summary["productsScored"]),
            "watchlistPoolRowsScored": int(watchlist_pool_summary["rowsScored"]),
            "watchlistPoolProductsScored": int(watchlist_pool_summary["productsScored"]),
        }
        if active_universe_plan is not None:
            inference_summary.update(
                {
                    "mode": "prioritized-active-universe",
                    "warning": "",
                    "productsRequested": len(resolved_product_ids),
                    "totalAvailableProducts": int(
                        active_universe_plan.source_refresh.get("productCount", len(resolved_product_ids)) or 0
                    ),
                    "protectedProductIds": list(active_universe_plan.protected_product_ids),
                    "activeUniverse": dict(active_universe_plan.summary),
                    "sourceRefresh": dict(active_universe_plan.source_refresh),
                }
            )
        inference_artifacts = inference_stage.build_from_prediction_frame(
            prediction_df,
            summary=inference_summary,
            empty_message=(
                "No live signals remained after applying the configured signal-universe exclusions."
            ),
            protected_product_ids=(
                active_universe_plan.protected_product_ids
                if active_universe_plan is not None
                else None
            ),
        )
        portfolio_store = TradingPortfolioStore(
            db_path=runtime_config.portfolio_store_path,
            default_capital=runtime_config.portfolio_default_capital,
            database_url=runtime_config.portfolio_store_url,
        )
        coordinator = SignalGenerationCoordinator(
            inference_stage=inference_stage,
            context_stage=SignalContextEnrichmentStage(runtime_config),
            enrichment_stage=SignalEnrichmentStage(runtime_config),
            decision_stage=SignalDecisionStage(runtime_config),
        )
        pipeline_artifacts = coordinator.run_pipeline(
            inference_artifacts=inference_artifacts,
            portfolio_store=portfolio_store,
        )
        live_actions = self._apply_live_execution_policy(
            signal_summaries=pipeline_artifacts.decision.signal_summaries,
            portfolio_store=portfolio_store,
        )
        visible_signals = self._select_visible_live_signals(
            signal_summaries=pipeline_artifacts.decision.published_signals,
            config=runtime_config,
        )
        actionable_visible_signals = self._select_visible_live_signals(
            signal_summaries=pipeline_artifacts.decision.actionable_signals,
            config=runtime_config,
        )
        primary_visible_signal = self._select_visible_primary_signal(
            signal_summaries=visible_signals,
            config=runtime_config,
        )

        live_snapshot = build_frontend_signal_snapshot(
            model_type=model.model_type,
            primary_signal=primary_visible_signal,
            latest_signals=visible_signals,
            actionable_signals=actionable_visible_signals,
            trader_brain=pipeline_artifacts.enrichment.trader_brain_snapshot,
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
                    else "prioritized-active-universe"
                    if active_universe_plan is not None
                    else "quote-universe-plus-watchlist-pool"
                    if use_quote_universe and watchlist_pool_product_ids
                    else "quote-universe"
                    if use_quote_universe
                    else "configured-watchlist"
                ),
                "requestedProducts": resolved_product_ids,
                "productsCovered": len(visible_signals),
                "featureRowsScored": int(len(prediction_df)),
                "granularitySeconds": int(runtime_config.live_granularity_seconds),
                "liveSignalCacheSeconds": int(effective_cache_ttl_seconds),
                "minimumActionConfidence": float(runtime_config.backtest_min_confidence),
                "modelPath": str(self._resolve_model_path()),
                "watchlistPool": watchlist_pool_summary,
                "signalInference": dict(inference_artifacts.summary),
                "livePolicy": live_actions,
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

    def _apply_live_execution_policy(
        self,
        *,
        signal_summaries: Sequence[Dict[str, Any]],
        portfolio_store: TradingPortfolioStore,
    ) -> dict[str, Any]:
        """Auto-clear loss signals and summarize the live-engine execution policy."""

        policy_summary = {
            "buySignalsOnly": bool(getattr(self.config, "live_buy_signals_only", True)),
            "autoClearLossSignals": bool(getattr(self.config, "live_auto_clear_loss_signals", True)),
            "lossSignalsDetected": 0,
            "lossSignalsCleared": 0,
            "lossSignalsSkipped": 0,
            "actions": [],
            "storageBackend": portfolio_store.database.storage_backend,
            "databaseTarget": portfolio_store.database.database_target,
        }
        if not bool(getattr(self.config, "live_auto_clear_loss_signals", True)):
            return policy_summary

        seen_product_ids: set[str] = set()
        for signal_summary in signal_summaries:
            product_id = str(signal_summary.get("productId", "")).strip().upper()
            signal_name = str(signal_summary.get("signal_name", "")).strip().upper()
            if signal_name != "LOSS" or not product_id or product_id in seen_product_ids:
                continue

            seen_product_ids.add(product_id)
            policy_summary["lossSignalsDetected"] += 1
            action_row = self._clear_live_loss_signal(
                signal_summary=dict(signal_summary),
                portfolio_store=portfolio_store,
            )
            policy_summary["actions"].append(action_row)
            if bool(action_row.get("cleared", False)):
                policy_summary["lossSignalsCleared"] += 1
            else:
                policy_summary["lossSignalsSkipped"] += 1

        return policy_summary

    def _clear_live_loss_signal(
        self,
        *,
        signal_summary: Dict[str, Any],
        portfolio_store: TradingPortfolioStore,
    ) -> dict[str, Any]:
        """Close tracked exposure for one live LOSS signal and journal the result."""

        product_id = str(signal_summary.get("productId", "")).strip().upper()
        active_trade = portfolio_store.get_active_trade_for_product(product_id)
        position = portfolio_store.get_position(product_id)
        if active_trade is None and position is None:
            return {
                "productId": product_id,
                "signalName": "LOSS",
                "cleared": False,
                "reason": "no_open_exposure",
                "executionId": None,
                "tradeId": None,
            }

        exit_price = self._resolve_loss_exit_price(
            signal_summary=signal_summary,
            active_trade=active_trade,
            position=position,
        )
        executed_at = str(signal_summary.get("timestamp") or "").strip() or None
        close_reason = (
            str(signal_summary.get("tradeLifecycleReason") or "").strip()
            or str(signal_summary.get("reasonSummary") or "").strip()
            or "Live pipeline loss signal triggered an automatic exit."
        )
        metadata = {
            "source": "live_pipeline_loss_autoclear",
            "signalName": str(signal_summary.get("signal_name") or "").strip().upper(),
            "spotAction": str(signal_summary.get("spotAction") or "").strip().lower(),
            "reasonSummary": str(signal_summary.get("reasonSummary") or "").strip(),
            "tradeLifecycleReason": str(signal_summary.get("tradeLifecycleReason") or "").strip(),
            "timestamp": signal_summary.get("timestamp"),
        }

        execution_id = None
        trade_id = None
        if position is not None:
            quantity = float(position.get("quantity") or 0.0)
            if quantity > 0:
                execution_result = portfolio_store.record_execution(
                    product_id=product_id,
                    side="sell",
                    quantity=quantity,
                    price=exit_price,
                    current_price=exit_price,
                    executed_at=executed_at,
                    metadata={
                        **metadata,
                        "positionFractionBefore": position.get("positionFraction"),
                    },
                )
                execution_payload = execution_result.get("execution") if isinstance(execution_result, dict) else None
                if isinstance(execution_payload, dict):
                    execution_id = execution_payload.get("executionId")

        if active_trade is not None:
            closed_trade = portfolio_store.close_trade(
                trade_id=int(active_trade["tradeId"]),
                exit_price=exit_price,
                closed_at=executed_at,
                close_reason="loss_signal_auto_clear",
                current_price=exit_price,
                metadata={
                    **metadata,
                    "autoClosedBy": "live_pipeline",
                },
            )
            trade_id = closed_trade.get("tradeId") if isinstance(closed_trade, dict) else active_trade.get("tradeId")

        return {
            "productId": product_id,
            "signalName": "LOSS",
            "cleared": True,
            "reason": "loss_signal_auto_clear",
            "executionId": execution_id,
            "tradeId": trade_id,
            "exitPrice": exit_price,
        }

    @staticmethod
    def _select_visible_live_signals(
        *,
        signal_summaries: Sequence[Dict[str, Any]],
        config: TrainingConfig,
    ) -> list[Dict[str, Any]]:
        """Apply the response-side live visibility policy to the signal list."""

        visible_signals = [dict(signal_summary) for signal_summary in signal_summaries]
        if bool(getattr(config, "live_buy_signals_only", True)):
            visible_signals = [
                signal_summary
                for signal_summary in visible_signals
                if str(signal_summary.get("signal_name", "")).strip().upper() == "BUY"
            ]
        return visible_signals

    @staticmethod
    def _select_visible_primary_signal(
        *,
        signal_summaries: Sequence[Dict[str, Any]],
        config: TrainingConfig,
    ) -> Dict[str, Any] | None:
        """Resolve the featured live signal after visibility filtering."""

        if not signal_summaries:
            return None
        if len(signal_summaries) == 1:
            return dict(signal_summaries[0])
        return dict(select_primary_signal(list(signal_summaries), config=config))

    @staticmethod
    def _resolve_loss_exit_price(
        *,
        signal_summary: Dict[str, Any],
        active_trade: Dict[str, Any] | None,
        position: Dict[str, Any] | None,
    ) -> float:
        """Choose the best available price for an automatic loss-triggered exit."""

        price_candidates = [
            signal_summary.get("close"),
            ((signal_summary.get("tradeContext") or {}).get("currentPrice") if isinstance(signal_summary.get("tradeContext"), dict) else None),
            (active_trade.get("currentPrice") if active_trade is not None else None),
            (position.get("currentPrice") if position is not None else None),
            (active_trade.get("entryPrice") if active_trade is not None else None),
            (position.get("entryPrice") if position is not None else None),
        ]
        for raw_price in price_candidates:
            try:
                normalized_price = float(raw_price)
            except (TypeError, ValueError):
                continue
            if normalized_price > 0:
                return normalized_price

        raise ValueError(
            f"Could not resolve a valid exit price for live LOSS auto-clear on {signal_summary.get('productId')!r}."
        )

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
