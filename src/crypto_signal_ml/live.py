"""Realtime-style market inference helpers for the trading assistant backend."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .config import MODELS_DIR, TrainingConfig
from .data import CoinbaseExchangePriceDataLoader
from .frontend import build_frontend_signal_snapshot
from .modeling import BaseSignalModel, get_model_class
from .pipeline import CryptoDatasetBuilder
from .signals import (
    build_actionable_signal_summaries,
    build_latest_signal_summaries,
    select_primary_signal,
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
        if self._cached_snapshot_generated_at is not None:
            cache_age_seconds = max(
                int((datetime.now(timezone.utc) - self._cached_snapshot_generated_at).total_seconds()),
                0,
            )

        return {
            "status": "ready" if model_path.exists() else "missing_model",
            "modelPath": str(model_path),
            "cacheAgeSeconds": cache_age_seconds,
            "cacheTtlSeconds": int(self.config.live_signal_cache_seconds),
            "lastGeneratedAt": (
                self._cached_snapshot.get("generatedAt")
                if self._cached_snapshot is not None
                else None
            ),
        }

    def get_live_snapshot(
        self,
        force_refresh: bool = False,
        product_id: str | None = None,
        product_ids: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        """Return a fresh or cached live signal snapshot for the requested universe."""

        resolved_product_ids = self._resolve_requested_products(
            product_id=product_id,
            product_ids=product_ids,
        )
        request_key = self._build_request_key(
            product_id=product_id,
            product_ids=resolved_product_ids,
        )

        if not force_refresh and self._can_use_cached_snapshot(request_key=request_key):
            return dict(self._cached_snapshot or {})

        model = self._load_model()
        runtime_config = replace(
            model.config,
            live_product_ids=tuple(resolved_product_ids),
            live_fetch_all_quote_products=(
                self.config.live_fetch_all_quote_products and not resolved_product_ids
            ),
            live_max_products=self.config.live_max_products,
            live_granularity_seconds=self.config.live_granularity_seconds,
            live_total_candles=self.config.live_total_candles,
            live_request_pause_seconds=self.config.live_request_pause_seconds,
            live_signal_cache_seconds=self.config.live_signal_cache_seconds,
            assistant_system_name=self.config.assistant_system_name,
            assistant_enable_retrieval=self.config.assistant_enable_retrieval,
            assistant_memory_message_limit=self.config.assistant_memory_message_limit,
            assistant_retrieval_item_limit=self.config.assistant_retrieval_item_limit,
        )

        data_loader = CoinbaseExchangePriceDataLoader(
            data_path=runtime_config.data_file,
            product_ids=tuple(resolved_product_ids),
            fetch_all_quote_products=(
                runtime_config.live_fetch_all_quote_products and not resolved_product_ids
            ),
            quote_currency=runtime_config.coinbase_quote_currency,
            excluded_base_currencies=runtime_config.coinbase_excluded_base_currencies,
            max_products=runtime_config.live_max_products,
            granularity_seconds=runtime_config.live_granularity_seconds,
            total_candles=runtime_config.live_total_candles,
            request_pause_seconds=runtime_config.live_request_pause_seconds,
            should_save_downloaded_data=False,
            product_batch_size=None,
            save_progress_every_products=0,
            log_progress=False,
        )

        dataset_builder = CryptoDatasetBuilder(
            config=runtime_config,
            feature_columns=model.feature_columns,
            data_loader=data_loader,
        )
        feature_df = dataset_builder.build_feature_table()
        prediction_df = model.predict(feature_df)
        latest_signals = build_latest_signal_summaries(prediction_df)
        actionable_signals = build_actionable_signal_summaries(latest_signals)
        primary_signal = select_primary_signal(latest_signals)

        live_snapshot = build_frontend_signal_snapshot(
            model_type=model.model_type,
            primary_signal=primary_signal,
            latest_signals=latest_signals,
            actionable_signals=actionable_signals,
        )
        live_snapshot.update(
            {
                "mode": "live",
                "marketDataSource": "coinbaseExchangeRest",
                "requestMode": (
                    "requested-product"
                    if product_id
                    else "configured-watchlist"
                    if resolved_product_ids
                    else "quote-universe"
                ),
                "requestedProducts": resolved_product_ids,
                "productsCovered": len(latest_signals),
                "featureRowsScored": int(len(prediction_df)),
                "granularitySeconds": int(runtime_config.live_granularity_seconds),
                "liveSignalCacheSeconds": int(runtime_config.live_signal_cache_seconds),
                "modelPath": str(self._resolve_model_path()),
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
            return [self._normalize_product_id(product_id)]

        if product_ids:
            return [self._normalize_product_id(value) for value in product_ids if str(value).strip()]

        return [self._normalize_product_id(value) for value in self.config.live_product_ids if str(value).strip()]

    def _build_request_key(
        self,
        product_id: str | None,
        product_ids: Sequence[str],
    ) -> str:
        """Build a stable cache key for the current live request."""

        if product_id:
            return f"product:{self._normalize_product_id(product_id)}"

        if product_ids:
            return "products:" + ",".join(sorted(self._normalize_product_id(value) for value in product_ids))

        return "universe:all"

    def _can_use_cached_snapshot(self, request_key: str) -> bool:
        """Return whether the in-memory live snapshot is still valid for this request."""

        if self._cached_snapshot is None or self._cached_snapshot_generated_at is None:
            return False

        if self._cached_request_key != request_key:
            return False

        expires_at = self._cached_snapshot_generated_at + timedelta(
            seconds=int(self.config.live_signal_cache_seconds)
        )
        return datetime.now(timezone.utc) < expires_at

    @staticmethod
    def _normalize_product_id(product_id: str) -> str:
        """Normalize Coinbase-style product ids."""

        return str(product_id).strip().upper()
