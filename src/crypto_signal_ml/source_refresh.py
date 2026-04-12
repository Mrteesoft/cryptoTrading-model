"""Cache-first CoinMarketCap refresh state and prioritized active-universe selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .config import TrainingConfig, is_coinmarketcap_market_data_source
from .data import CoinMarketCapRateLimitError, create_market_data_loader
from .frontend import WatchlistPoolStore
from .trading.portfolio import TradingPortfolioStore
from .trading.signal_store import TradingSignalStore
from .trading.signals import is_signal_product_excluded
from .trading.watchlist_state import WatchlistStateStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso_timestamp(value: Any) -> datetime | None:
    normalized_value = str(value or "").strip().replace("Z", "+00:00")
    if not normalized_value:
        return None

    try:
        parsed_value = datetime.fromisoformat(normalized_value)
    except ValueError:
        return None

    if parsed_value.tzinfo is None:
        return parsed_value.replace(tzinfo=timezone.utc)

    return parsed_value.astimezone(timezone.utc)


def _normalize_product_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_product_ids(values: Sequence[Any]) -> list[str]:
    seen_product_ids: set[str] = set()
    normalized_product_ids: list[str] = []
    for value in values:
        product_id = _normalize_product_id(value)
        if not product_id or product_id in seen_product_ids:
            continue
        seen_product_ids.add(product_id)
        normalized_product_ids.append(product_id)
    return normalized_product_ids


def _read_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def _write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


@dataclass
class ActiveUniversePlan:
    """Selected products plus the refresh metadata that drove the choice."""

    product_ids: list[str]
    protected_product_ids: list[str]
    summary: dict[str, Any]
    source_refresh: dict[str, Any]


class CoinMarketCapUniverseRefreshService:
    """Load and refresh a ranked CoinMarketCap universe through one persisted cache file."""

    source_name = "coinmarketcapUniverse"

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.cache_path = Path(config.coinmarketcap_universe_cache_file)
        self.ttl_seconds = max(int(getattr(config, "coinmarketcap_universe_ttl_seconds", 21600) or 21600), 60)
        self.rate_limit_cooldown_seconds = max(
            int(getattr(config, "coinmarketcap_universe_rate_limit_cooldown_seconds", 1800) or 1800),
            60,
        )

    def resolve(self) -> dict[str, Any]:
        """Return the ranked tracked universe, refreshing only when the cache is stale."""

        cached_payload = self._load_cached_payload()
        cached_products = list(cached_payload.get("products", []))
        now = _utc_now()

        if not is_coinmarketcap_market_data_source(self.config.market_data_source):
            return self._decorate_runtime_fields(
                cached_payload,
                source_status="not_applicable",
                refresh_attempted=False,
                used_cached_snapshot=bool(cached_products),
            )

        if not bool(getattr(self.config, "coinmarketcap_fetch_all_quote_products", True)):
            explicit_product_ids = self._configured_discovery_product_ids()
            payload = self._build_payload(
                products=[
                    self._build_product_row(product_id=product_id, market_cap_rank=index + 1)
                    for index, product_id in enumerate(explicit_product_ids)
                ],
                source_status="explicit_products_only",
                last_fetched_at=cached_payload.get("lastFetchedAt"),
                last_error="",
                rate_limited_until=None,
            )
            return self._decorate_runtime_fields(
                payload,
                source_status="explicit_products_only",
                refresh_attempted=False,
                used_cached_snapshot=False,
            )

        expires_at = _parse_iso_timestamp(cached_payload.get("expiresAt"))
        if cached_products and expires_at is not None and now < expires_at:
            return self._decorate_runtime_fields(
                cached_payload,
                source_status="cache_valid",
                refresh_attempted=False,
                used_cached_snapshot=True,
            )

        rate_limited_until = _parse_iso_timestamp(cached_payload.get("rateLimitedUntil"))
        if cached_products and rate_limited_until is not None and now < rate_limited_until:
            payload = dict(cached_payload)
            payload["sourceStatus"] = "cached_rate_limited"
            payload.setdefault(
                "lastError",
                "CoinMarketCap universe refresh is cooling down after a prior rate-limit response.",
            )
            self._save_payload(payload)
            return self._decorate_runtime_fields(
                payload,
                source_status="cached_rate_limited",
                refresh_attempted=False,
                used_cached_snapshot=True,
            )

        api_key = os.getenv(str(self.config.coinmarketcap_api_key_env_var), "").strip()
        if not api_key:
            source_status = "missing_api_key_cached_fallback" if cached_products else "missing_api_key"
            payload = self._build_fallback_payload(
                cached_payload=cached_payload,
                source_status=source_status,
                last_error=(
                    "CoinMarketCap universe refresh requires an API key in "
                    f"`{self.config.coinmarketcap_api_key_env_var}`."
                ),
                rate_limited_until=cached_payload.get("rateLimitedUntil"),
            )
            self._save_payload(payload)
            return self._decorate_runtime_fields(
                payload,
                source_status=source_status,
                refresh_attempted=False,
                used_cached_snapshot=bool(cached_products),
            )

        try:
            ranked_products = self._fetch_ranked_products()
            payload = self._build_payload(
                products=ranked_products,
                source_status="refreshed",
                last_fetched_at=_utc_now_iso(),
                last_error="",
                rate_limited_until=None,
            )
            self._save_payload(payload)
            return self._decorate_runtime_fields(
                payload,
                source_status="refreshed",
                refresh_attempted=True,
                used_cached_snapshot=False,
            )
        except CoinMarketCapRateLimitError as error:
            rate_limited_until_value = (now + timedelta(seconds=self.rate_limit_cooldown_seconds)).isoformat()
            source_status = "cached_rate_limited" if cached_products else "rate_limited"
            payload = self._build_fallback_payload(
                cached_payload=cached_payload,
                source_status=source_status,
                last_error=str(error),
                rate_limited_until=rate_limited_until_value,
            )
            self._save_payload(payload)
            return self._decorate_runtime_fields(
                payload,
                source_status=source_status,
                refresh_attempted=True,
                used_cached_snapshot=bool(cached_products),
            )
        except Exception as error:
            source_status = "cached_refresh_failed" if cached_products else "refresh_failed"
            payload = self._build_fallback_payload(
                cached_payload=cached_payload,
                source_status=source_status,
                last_error=str(error),
                rate_limited_until=cached_payload.get("rateLimitedUntil"),
            )
            self._save_payload(payload)
            return self._decorate_runtime_fields(
                payload,
                source_status=source_status,
                refresh_attempted=True,
                used_cached_snapshot=bool(cached_products),
            )

    def summarize(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Trim the tracked-universe payload into the metadata the signal loop needs."""

        products = list(payload.get("products", []))
        return {
            "source": self.source_name,
            "cachePath": str(self.cache_path),
            "lastFetchedAt": payload.get("lastFetchedAt"),
            "expiresAt": payload.get("expiresAt"),
            "ttlSeconds": int(payload.get("ttlSeconds", self.ttl_seconds) or self.ttl_seconds),
            "sourceStatus": str(payload.get("sourceStatus", "unknown") or "unknown"),
            "lastError": str(payload.get("lastError", "") or ""),
            "rateLimitedUntil": payload.get("rateLimitedUntil"),
            "refreshAttempted": bool(payload.get("refreshAttempted", False)),
            "usedCachedSnapshot": bool(payload.get("usedCachedSnapshot", False)),
            "productCount": int(payload.get("productCount", len(products)) or len(products)),
            "marketDataSource": str(self.config.market_data_source),
            "quoteCurrency": str(self.config.coinmarketcap_quote_currency).upper(),
        }

    def _load_cached_payload(self) -> dict[str, Any]:
        payload = _read_json_payload(self.cache_path)
        if "ttlSeconds" not in payload:
            payload["ttlSeconds"] = self.ttl_seconds
        return payload

    def _save_payload(self, payload: dict[str, Any]) -> None:
        persisted_payload = dict(payload)
        persisted_payload.pop("refreshAttempted", None)
        persisted_payload.pop("usedCachedSnapshot", None)
        _write_json_payload(self.cache_path, persisted_payload)

    def _build_product_row(
        self,
        *,
        product_id: str,
        market_cap_rank: int,
    ) -> dict[str, Any]:
        base_currency, quote_currency = self._split_product_id(product_id)
        return {
            "productId": product_id,
            "baseCurrency": base_currency,
            "quoteCurrency": quote_currency,
            "marketCapRank": int(market_cap_rank),
        }

    def _build_payload(
        self,
        *,
        products: Sequence[dict[str, Any]],
        source_status: str,
        last_fetched_at: str | None,
        last_error: str,
        rate_limited_until: str | None,
    ) -> dict[str, Any]:
        fetched_at = str(last_fetched_at or _utc_now_iso())
        expires_at = (_parse_iso_timestamp(fetched_at) or _utc_now()) + timedelta(seconds=self.ttl_seconds)
        normalized_products = [
            {
                "productId": _normalize_product_id(product.get("productId")),
                "baseCurrency": str(product.get("baseCurrency", "")).strip().upper(),
                "quoteCurrency": str(product.get("quoteCurrency", "")).strip().upper(),
                "marketCapRank": int(product.get("marketCapRank") or 0),
            }
            for product in products
            if _normalize_product_id(product.get("productId"))
        ]
        product_ids = [product["productId"] for product in normalized_products]

        return {
            "source": self.source_name,
            "marketDataSource": str(self.config.market_data_source),
            "quoteCurrency": str(self.config.coinmarketcap_quote_currency).upper(),
            "lastFetchedAt": fetched_at,
            "expiresAt": expires_at.isoformat(),
            "ttlSeconds": int(self.ttl_seconds),
            "sourceStatus": str(source_status),
            "lastError": str(last_error or ""),
            "rateLimitedUntil": rate_limited_until,
            "productCount": len(normalized_products),
            "productIds": product_ids,
            "products": normalized_products,
        }

    def _build_fallback_payload(
        self,
        *,
        cached_payload: dict[str, Any],
        source_status: str,
        last_error: str,
        rate_limited_until: str | None,
    ) -> dict[str, Any]:
        fallback_payload = dict(cached_payload)
        fallback_payload.setdefault("source", self.source_name)
        fallback_payload.setdefault("marketDataSource", str(self.config.market_data_source))
        fallback_payload.setdefault("quoteCurrency", str(self.config.coinmarketcap_quote_currency).upper())
        fallback_payload.setdefault("ttlSeconds", int(self.ttl_seconds))
        fallback_payload.setdefault("products", [])
        fallback_payload.setdefault("productIds", [product.get("productId") for product in fallback_payload["products"]])
        fallback_payload["productCount"] = int(fallback_payload.get("productCount", len(fallback_payload["products"])))
        fallback_payload["sourceStatus"] = str(source_status)
        fallback_payload["lastError"] = str(last_error or "")
        fallback_payload["rateLimitedUntil"] = rate_limited_until
        return fallback_payload

    @staticmethod
    def _decorate_runtime_fields(
        payload: dict[str, Any],
        *,
        source_status: str,
        refresh_attempted: bool,
        used_cached_snapshot: bool,
    ) -> dict[str, Any]:
        runtime_payload = dict(payload)
        runtime_payload["sourceStatus"] = str(source_status)
        runtime_payload["refreshAttempted"] = bool(refresh_attempted)
        runtime_payload["usedCachedSnapshot"] = bool(used_cached_snapshot)
        return runtime_payload

    def _configured_discovery_product_ids(self) -> list[str]:
        return _normalize_product_ids(
            [
                *getattr(self.config, "coinmarketcap_product_ids", ()),
                getattr(self.config, "coinmarketcap_product_id", ""),
                *getattr(self.config, "live_product_ids", ()),
            ]
        )

    def _fetch_ranked_products(self) -> list[dict[str, Any]]:
        loader = create_market_data_loader(
            config=self.config,
            data_path=self.config.data_file,
            should_save_downloaded_data=False,
            fetch_all_quote_products=True,
            product_batch_size=None,
            product_batch_number=1,
            save_progress_every_products=0,
            log_progress=False,
        )
        available_products = list(getattr(loader, "get_available_products")())
        return [
            {
                "productId": _normalize_product_id(product.get("product_id")),
                "baseCurrency": str(product.get("base_currency", "")).strip().upper(),
                "quoteCurrency": str(product.get("quote_currency", "")).strip().upper(),
                "marketCapRank": index + 1,
            }
            for index, product in enumerate(available_products)
            if _normalize_product_id(product.get("product_id"))
        ]

    @staticmethod
    def _split_product_id(product_id: str) -> tuple[str, str]:
        normalized_product_id = _normalize_product_id(product_id)
        if "-" not in normalized_product_id:
            return normalized_product_id, ""
        base_currency, quote_currency = normalized_product_id.split("-", 1)
        return base_currency, quote_currency


class SignalUniverseCoordinator:
    """Assemble the active analysis universe in a priority order that protects follow-up."""

    priority_order = (
        "openPositions",
        "watchlist",
        "publishedSignals",
        "trackedUniverse",
        "discovery",
    )

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.refresh_service = CoinMarketCapUniverseRefreshService(config)

    def resolve_active_universe(self, *, max_products: int | None = None) -> ActiveUniversePlan:
        """Build the reduced live-analysis universe around positions, watchlist, and cached ranking."""

        open_position_ids = self._load_open_position_product_ids()
        watchlist_ids = self._load_watchlist_product_ids(limit=self._resolve_base_limit(max_products))
        published_signal_ids = self._load_published_signal_product_ids()
        protected_product_ids = _normalize_product_ids(
            [
                *open_position_ids,
                *watchlist_ids,
                *published_signal_ids,
            ]
        )

        base_limit = self._resolve_base_limit(max_products)
        effective_limit = max(base_limit, len(protected_product_ids))

        tracked_universe_payload = self.refresh_service.resolve()
        tracked_universe_ids = _normalize_product_ids(
            tracked_universe_payload.get("productIds")
            or [
                product.get("productId")
                for product in tracked_universe_payload.get("products", [])
            ]
        )
        discovery_ids = self._load_discovery_product_ids()

        selected_product_ids, selected_counts = self._select_product_ids(
            effective_limit=effective_limit,
            open_position_ids=open_position_ids,
            watchlist_ids=watchlist_ids,
            published_signal_ids=published_signal_ids,
            tracked_universe_ids=tracked_universe_ids,
            discovery_ids=discovery_ids,
        )

        summary = {
            "mode": "prioritized-active-universe",
            "priorityOrder": list(self.priority_order),
            "baseLimit": int(base_limit),
            "effectiveLimit": int(effective_limit),
            "selectedCount": len(selected_product_ids),
            "selectedProductIds": list(selected_product_ids),
            "protectedCount": len(protected_product_ids),
            "protectedProductIds": list(protected_product_ids),
            "selectedCounts": selected_counts,
            "openPositions": {
                "count": len(open_position_ids),
                "productIds": list(open_position_ids),
            },
            "watchlist": {
                "count": len(watchlist_ids),
                "productIds": list(watchlist_ids),
            },
            "publishedSignals": {
                "count": len(published_signal_ids),
                "productIds": list(published_signal_ids),
            },
            "trackedUniverse": {
                "count": len(tracked_universe_ids),
            },
            "discovery": {
                "count": len(discovery_ids),
                "productIds": list(discovery_ids[: min(len(discovery_ids), 25)]),
            },
        }

        return ActiveUniversePlan(
            product_ids=selected_product_ids,
            protected_product_ids=protected_product_ids,
            summary=summary,
            source_refresh=self.refresh_service.summarize(tracked_universe_payload),
        )

    def _resolve_base_limit(self, requested_limit: int | None) -> int:
        if requested_limit is not None:
            return max(int(requested_limit), 1)

        configured_limit = getattr(self.config, "live_max_products", None)
        if configured_limit is not None:
            return max(int(configured_limit), 1)

        configured_source_limit = getattr(self.config, "coinmarketcap_max_products", None)
        if configured_source_limit is not None:
            return max(int(configured_source_limit), 1)

        return 25

    def _load_open_position_product_ids(self) -> list[str]:
        portfolio_store = TradingPortfolioStore(
            db_path=self.config.portfolio_store_path,
            default_capital=self.config.portfolio_default_capital,
            database_url=self.config.portfolio_store_url,
        )
        return _normalize_product_ids(
            position.get("product_id") or position.get("productId")
            for position in portfolio_store.list_positions()
        )

    def _load_watchlist_product_ids(self, *, limit: int) -> list[str]:
        pool_limit = int(getattr(self.config, "signal_watchlist_pool_max_products", limit) or limit)
        pool_store = WatchlistPoolStore(Path(self.config.signal_watchlist_pool_path))
        watchlist_state_store = WatchlistStateStore(self.config)
        return _normalize_product_ids(
            [
                *pool_store.get_monitored_product_ids(limit=pool_limit),
                *watchlist_state_store.list_active_product_ids(limit=limit),
            ]
        )

    def _load_published_signal_product_ids(self) -> list[str]:
        signal_store = TradingSignalStore(
            db_path=self.config.signal_store_path,
            database_url=self.config.signal_store_url,
        )
        return _normalize_product_ids(
            signal_summary.get("productId")
            for signal_summary in signal_store.list_current_signals(limit=500)
        )

    def _load_discovery_product_ids(self) -> list[str]:
        discovery_product_ids = _normalize_product_ids(
            [
                *getattr(self.config, "live_product_ids", ()),
                *getattr(self.config, "coinmarketcap_product_ids", ()),
                getattr(self.config, "coinmarketcap_product_id", ""),
                *self._load_cached_market_data_product_ids(),
            ]
        )

        return [
            product_id
            for product_id in discovery_product_ids
            if not is_signal_product_excluded(product_id=product_id, config=self.config)
        ]

    def _load_cached_market_data_product_ids(self) -> list[str]:
        data_path = Path(self.config.data_file)
        if not data_path.exists():
            return []

        try:
            price_df = pd.read_csv(data_path, usecols=["product_id"])
        except (OSError, ValueError):
            return []

        if "product_id" not in price_df.columns:
            return []

        return _normalize_product_ids(price_df["product_id"].tolist())

    def _select_product_ids(
        self,
        *,
        effective_limit: int,
        open_position_ids: Sequence[str],
        watchlist_ids: Sequence[str],
        published_signal_ids: Sequence[str],
        tracked_universe_ids: Sequence[str],
        discovery_ids: Sequence[str],
    ) -> tuple[list[str], dict[str, int]]:
        selected_product_ids: list[str] = []
        seen_product_ids: set[str] = set()
        selected_counts = {priority_name: 0 for priority_name in self.priority_order}

        priority_sources = (
            ("openPositions", open_position_ids, True),
            ("watchlist", watchlist_ids, True),
            ("publishedSignals", published_signal_ids, True),
            ("trackedUniverse", tracked_universe_ids, False),
            ("discovery", discovery_ids, False),
        )

        for priority_name, product_ids, protect_from_exclusion in priority_sources:
            for product_id in product_ids:
                normalized_product_id = _normalize_product_id(product_id)
                if not normalized_product_id or normalized_product_id in seen_product_ids:
                    continue
                if (
                    not protect_from_exclusion
                    and is_signal_product_excluded(product_id=normalized_product_id, config=self.config)
                ):
                    continue
                seen_product_ids.add(normalized_product_id)
                selected_product_ids.append(normalized_product_id)
                selected_counts[priority_name] += 1
                if len(selected_product_ids) >= effective_limit:
                    return selected_product_ids, selected_counts

        return selected_product_ids, selected_counts
