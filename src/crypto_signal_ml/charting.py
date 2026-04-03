"""Chart-history and event services for frontend and TradingView-style consumers."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Sequence

import pandas as pd

from .config import TrainingConfig
from .data import CsvPriceDataLoader, create_market_data_loader


PriceLoaderFactory = Callable[..., object]


class MarketChartService:
    """Serve normalized OHLCV history, symbol metadata, and cached event marks."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        data_loader_factory: PriceLoaderFactory = create_market_data_loader,
    ) -> None:
        self.config = config or TrainingConfig()
        self.data_loader_factory = data_loader_factory

    def get_udf_config(self) -> dict[str, Any]:
        """Return the supported capabilities for TradingView's UDF adapter."""

        supported_resolutions = self._supported_resolutions()
        return {
            "supports_search": True,
            "supports_group_request": False,
            "supports_marks": True,
            "supports_timescale_marks": True,
            "supports_time": True,
            "supports_quotes": True,
            "supported_resolutions": supported_resolutions,
        }

    def get_server_time(self) -> int:
        """Return the current UTC time as UNIX seconds."""

        return int(datetime.now(timezone.utc).timestamp())

    def search_symbols(
        self,
        query: str = "",
        exchange: str | None = None,
        symbol_type: str | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Search the known symbol catalog for TradingView consumers."""

        normalized_query = str(query).strip().upper()
        normalized_exchange = str(exchange or "").strip().upper()
        normalized_type = str(symbol_type or "").strip().lower()

        matches: list[dict[str, Any]] = []
        for symbol_row in self._build_symbol_catalog():
            if normalized_exchange and symbol_row["exchange"].upper() != normalized_exchange:
                continue

            if normalized_type and symbol_row["type"].lower() != normalized_type:
                continue

            haystack = " ".join(
                [
                    str(symbol_row["symbol"]),
                    str(symbol_row["full_name"]),
                    str(symbol_row["description"]),
                ]
            ).upper()
            if normalized_query and normalized_query not in haystack:
                continue

            matches.append(symbol_row)

        return matches[: max(int(limit), 0)]

    def resolve_symbol(self, symbol: str) -> dict[str, Any]:
        """Return TradingView-compatible metadata for one symbol."""

        normalized_symbol = self._normalize_symbol(symbol)
        catalog = self._build_symbol_catalog()
        matched_row = next(
            (row for row in catalog if row["ticker"] == normalized_symbol),
            None,
        )
        if matched_row is None:
            raise ValueError(f"Unknown chart symbol: {symbol}")

        quote_currency = matched_row["quote_currency"]
        price_df = self._load_price_frame(
            symbol=normalized_symbol,
            force_refresh=False,
            total_candles=max(self.config.live_total_candles, 48),
        )
        last_close = float(price_df.iloc[-1]["close"]) if not price_df.empty else 1.0
        price_scale = self._price_scale_from_close(last_close)

        return {
            "name": normalized_symbol,
            "ticker": normalized_symbol,
            "full_name": matched_row["full_name"],
            "description": matched_row["description"],
            "type": matched_row["type"],
            "session": "24x7",
            "timezone": "Etc/UTC",
            "exchange": matched_row["exchange"],
            "listed_exchange": matched_row["exchange"],
            "minmov": 1,
            "pricescale": price_scale,
            "has_intraday": True,
            "has_daily": True,
            "has_weekly_and_monthly": True,
            "supported_resolutions": self._supported_resolutions(),
            "intraday_multipliers": [resolution for resolution in self._supported_resolutions() if resolution.isdigit()],
            "volume_precision": 8,
            "data_status": "streaming",
            "format": "price",
        }

    def get_history(
        self,
        symbol: str,
        resolution: str,
        from_seconds: int | None,
        to_seconds: int | None,
        countback: int | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Return bar history in TradingView UDF response format."""

        normalized_symbol = self._normalize_symbol(symbol)
        max_required_bars = self._estimate_required_candles(
            resolution=resolution,
            from_seconds=from_seconds,
            to_seconds=to_seconds,
            countback=countback,
        )
        price_df = self._load_price_frame(
            symbol=normalized_symbol,
            force_refresh=force_refresh,
            total_candles=max_required_bars,
        )
        bars_df = self._build_bars_for_resolution(
            price_df=price_df,
            resolution=resolution,
        )

        if from_seconds is not None:
            bars_df = bars_df.loc[bars_df["timestamp_seconds"] >= int(from_seconds)]
        if to_seconds is not None:
            bars_df = bars_df.loc[bars_df["timestamp_seconds"] <= int(to_seconds)]
        if countback is not None and countback > 0:
            bars_df = bars_df.tail(int(countback))

        if bars_df.empty:
            response: dict[str, Any] = {"s": "no_data"}
            return response

        return {
            "s": "ok",
            "t": bars_df["timestamp_seconds"].astype(int).tolist(),
            "o": bars_df["open"].astype(float).tolist(),
            "h": bars_df["high"].astype(float).tolist(),
            "l": bars_df["low"].astype(float).tolist(),
            "c": bars_df["close"].astype(float).tolist(),
            "v": bars_df["volume"].astype(float).tolist(),
        }

    def get_quotes(
        self,
        symbols: Sequence[str],
    ) -> dict[str, Any]:
        """Return the latest quote block in TradingView UDF format."""

        quote_rows: list[dict[str, Any]] = []
        for raw_symbol in symbols:
            normalized_symbol = self._normalize_symbol(raw_symbol)
            price_df = self._load_price_frame(
                symbol=normalized_symbol,
                force_refresh=False,
                total_candles=max(self.config.live_total_candles, 48),
            )
            if price_df.empty:
                continue

            latest_row = price_df.iloc[-1]
            previous_close = float(price_df.iloc[-2]["close"]) if len(price_df) > 1 else float(latest_row["close"])
            current_close = float(latest_row["close"])
            change_value = current_close - previous_close
            change_percent = (change_value / previous_close) if previous_close else 0.0

            quote_rows.append(
                {
                    "s": normalized_symbol,
                    "n": normalized_symbol,
                    "v": {
                        "lp": current_close,
                        "ch": change_value,
                        "chp": change_percent * 100.0,
                        "volume": float(latest_row["volume"]),
                        "description": normalized_symbol,
                    },
                }
            )

        return {
            "s": "ok" if quote_rows else "no_data",
            "d": quote_rows,
        }

    def get_marks(
        self,
        symbol: str,
        from_seconds: int | None,
        to_seconds: int | None,
    ) -> dict[str, Any]:
        """Return CoinMarketCal events as TradingView chart marks."""

        event_rows = self.list_events(
            symbol=symbol,
            from_seconds=from_seconds,
            to_seconds=to_seconds,
        )
        if not event_rows:
            return {"id": []}

        labels = [self._mark_label_from_category(row.get("eventCategory")) for row in event_rows]
        return {
            "id": [str(row["eventId"]) for row in event_rows],
            "time": [int(row["eventTimestampSeconds"]) for row in event_rows],
            "color": [self._mark_color_from_category(row.get("eventCategory")) for row in event_rows],
            "text": [str(row["eventTitle"]) for row in event_rows],
            "label": labels,
            "labelFontColor": ["#0b1412" for _ in event_rows],
            "minSize": [18 for _ in event_rows],
        }

    def get_timescale_marks(
        self,
        symbol: str,
        from_seconds: int | None,
        to_seconds: int | None,
    ) -> list[dict[str, Any]]:
        """Return CoinMarketCal events as TradingView timescale marks."""

        event_rows = self.list_events(
            symbol=symbol,
            from_seconds=from_seconds,
            to_seconds=to_seconds,
        )
        return [
            {
                "id": str(row["eventId"]),
                "time": int(row["eventTimestampSeconds"]),
                "color": self._mark_color_from_category(row.get("eventCategory")),
                "label": self._mark_label_from_category(row.get("eventCategory")),
                "tooltip": [str(row["eventTitle"])],
            }
            for row in event_rows
        ]

    def list_events(
        self,
        symbol: str | None = None,
        from_seconds: int | None = None,
        to_seconds: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return cached CoinMarketCal events in a frontend-friendly shape."""

        event_df = self._load_event_frame()
        if event_df.empty:
            return []

        filtered_df = event_df.copy()
        if symbol:
            normalized_symbol = self._normalize_symbol(symbol)
            base_currency = normalized_symbol.split("-")[0]
            filtered_df = filtered_df.loc[filtered_df["base_currency"] == base_currency]

        if from_seconds is not None:
            filtered_df = filtered_df.loc[
                filtered_df["event_timestamp_seconds"] >= int(from_seconds)
            ]
        if to_seconds is not None:
            filtered_df = filtered_df.loc[
                filtered_df["event_timestamp_seconds"] <= int(to_seconds)
            ]

        filtered_df = filtered_df.sort_values("event_start").head(max(int(limit), 0))

        return [
            {
                "eventId": str(event_row["event_id"]),
                "eventTitle": str(event_row["event_title"]),
                "eventCategory": str(event_row.get("event_category") or ""),
                "baseCurrency": str(event_row["base_currency"]),
                "productId": self._product_id_from_base(str(event_row["base_currency"])),
                "eventStart": pd.Timestamp(event_row["event_start"]).isoformat(),
                "eventTimestampSeconds": int(event_row["event_timestamp_seconds"]),
            }
            for _, event_row in filtered_df.iterrows()
        ]

    def _build_symbol_catalog(self) -> list[dict[str, Any]]:
        """Build the known product list for search and symbol resolution."""

        price_df = self._load_cached_price_frame()
        if price_df.empty:
            return [
                self._build_symbol_row_from_identifiers(product_id)
                for product_id in self._fallback_product_ids()
            ]

        if "product_id" not in price_df.columns:
            return []

        catalog_rows: list[dict[str, Any]] = []
        latest_by_product = (
            price_df.sort_values("timestamp")
            .groupby("product_id", as_index=False)
            .tail(1)
            .sort_values("product_id")
        )
        for _, row in latest_by_product.iterrows():
            product_id = str(row["product_id"]).strip().upper()
            symbol_row = self._build_symbol_row_from_identifiers(
                product_id=product_id,
                base_currency=str(row.get("base_currency") or product_id.split("-")[0]).upper(),
                quote_currency=str(row.get("quote_currency") or product_id.split("-")[1]).upper(),
                coin_name=str(row.get("cmc_name") or row.get("base_currency") or product_id.split("-")[0]),
            )
            catalog_rows.append(symbol_row)

        return catalog_rows

    def _build_symbol_row_from_identifiers(
        self,
        product_id: str,
        base_currency: str | None = None,
        quote_currency: str | None = None,
        coin_name: str | None = None,
    ) -> dict[str, Any]:
        """Build one frontend/tradingview symbol descriptor."""

        normalized_product_id = self._normalize_symbol(product_id)
        base_currency = str(base_currency or normalized_product_id.split("-")[0]).upper()
        quote_currency = str(quote_currency or normalized_product_id.split("-")[1]).upper()
        exchange_name = self._exchange_name()
        display_name = coin_name or base_currency

        return {
            "symbol": normalized_product_id,
            "ticker": normalized_product_id,
            "full_name": f"{exchange_name}:{normalized_product_id}",
            "description": f"{display_name} / {quote_currency}",
            "exchange": exchange_name,
            "type": "crypto",
            "base_currency": base_currency,
            "quote_currency": quote_currency,
        }

    def _load_cached_price_frame(self) -> pd.DataFrame:
        """Load the saved raw market file when it exists."""

        if not self.config.data_file.exists():
            return pd.DataFrame()

        return CsvPriceDataLoader(self.config.data_file).load()

    def _load_price_frame(
        self,
        symbol: str,
        force_refresh: bool,
        total_candles: int,
    ) -> pd.DataFrame:
        """Load one symbol's price history from cache, falling back to live fetches."""

        normalized_symbol = self._normalize_symbol(symbol)
        if not force_refresh:
            cached_df = self._load_cached_price_frame()
            if not cached_df.empty:
                filtered_cached_df = self._filter_price_df_to_symbol(cached_df, normalized_symbol)
                if not filtered_cached_df.empty:
                    return filtered_cached_df

        runtime_config = replace(
            self.config,
            live_product_ids=(normalized_symbol,),
            live_fetch_all_quote_products=False,
            live_total_candles=max(int(total_candles), 1),
        )
        api_loader = self.data_loader_factory(
            config=runtime_config,
            data_path=runtime_config.data_file,
            should_save_downloaded_data=False,
            product_ids=(normalized_symbol,),
            fetch_all_quote_products=False,
            max_products=1,
            granularity_seconds=self._base_granularity_seconds(),
            total_candles=max(int(total_candles), 1),
            request_pause_seconds=runtime_config.live_request_pause_seconds,
            product_batch_size=None,
            save_progress_every_products=0,
            log_progress=False,
        )
        fresh_df = api_loader.load()
        return self._filter_price_df_to_symbol(fresh_df, normalized_symbol)

    def _filter_price_df_to_symbol(
        self,
        price_df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Keep only one product's rows and normalize timestamp ordering."""

        normalized_symbol = self._normalize_symbol(symbol)
        filtered_df = price_df.copy()
        if "product_id" in filtered_df.columns:
            filtered_df["product_id"] = filtered_df["product_id"].astype(str).str.upper()
            filtered_df = filtered_df.loc[filtered_df["product_id"] == normalized_symbol]

        if "timestamp" in filtered_df.columns:
            filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"], errors="coerce", utc=True)
            filtered_df = filtered_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        return filtered_df.reset_index(drop=True)

    def _build_bars_for_resolution(
        self,
        price_df: pd.DataFrame,
        resolution: str,
    ) -> pd.DataFrame:
        """Build one OHLCV bar table for the requested resolution."""

        if price_df.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "timestamp_seconds",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )

        normalized_resolution = self._normalize_resolution(resolution)
        price_df = price_df.copy()
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], errors="coerce", utc=True)
        price_df = price_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        if normalized_resolution == self._base_resolution_key():
            output_df = price_df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        else:
            resample_rule = self._resolution_to_pandas_rule(normalized_resolution)
            output_df = (
                price_df.set_index("timestamp")
                .resample(resample_rule)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna(subset=["open", "high", "low", "close"])
                .reset_index()
            )

        output_df["timestamp_seconds"] = (
            pd.to_datetime(output_df["timestamp"], errors="coerce", utc=True)
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
            // 10**9
        ).astype(int)
        return output_df.reset_index(drop=True)

    def _load_event_frame(self) -> pd.DataFrame:
        """Load the cached CoinMarketCal event file when available."""

        if not self.config.coinmarketcal_events_file.exists():
            return pd.DataFrame()

        event_df = pd.read_csv(self.config.coinmarketcal_events_file)
        if event_df.empty:
            return event_df

        event_df["base_currency"] = event_df["base_currency"].astype(str).str.upper()
        event_df["event_start"] = pd.to_datetime(event_df["event_start"], errors="coerce", utc=True)
        event_df = event_df.dropna(subset=["base_currency", "event_start"]).sort_values(
            ["event_start", "base_currency", "event_id"]
        )
        event_df["event_timestamp_seconds"] = (
            pd.to_datetime(event_df["event_start"], errors="coerce", utc=True)
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
            // 10**9
        ).astype(int)
        return event_df.reset_index(drop=True)

    def _supported_resolutions(self) -> list[str]:
        """Return the supported chart resolutions for the active base data cadence."""

        base_seconds = self._base_granularity_seconds()
        if base_seconds >= 86400:
            return ["1D", "1W"]

        return ["60", "240", "1D", "1W"]

    def _estimate_required_candles(
        self,
        resolution: str,
        from_seconds: int | None,
        to_seconds: int | None,
        countback: int | None,
    ) -> int:
        """Estimate how many base candles are needed for the history query."""

        if countback is not None and countback > 0:
            target_bars = int(countback)
        elif from_seconds is not None and to_seconds is not None and to_seconds > from_seconds:
            resolution_seconds = self._resolution_to_seconds(resolution)
            target_bars = max(int((to_seconds - from_seconds) / resolution_seconds) + 8, 32)
        else:
            target_bars = max(self.config.live_total_candles, 120)

        resolution_multiplier = max(
            int(self._resolution_to_seconds(resolution) / self._base_granularity_seconds()),
            1,
        )
        return max(target_bars * resolution_multiplier, self.config.live_total_candles)

    def _base_granularity_seconds(self) -> int:
        """Resolve the active raw-candle size from the configured data source."""

        if self.config.market_data_source == "coinmarketcap":
            return int(self.config.coinmarketcap_granularity_seconds)

        return int(self.config.coinbase_granularity_seconds)

    def _base_resolution_key(self) -> str:
        """Return the TradingView resolution key matching the base candle size."""

        base_seconds = self._base_granularity_seconds()
        if base_seconds >= 86400:
            return "1D"

        return str(int(base_seconds / 60))

    def _normalize_resolution(self, resolution: str) -> str:
        """Normalize a requested resolution into one of the supported keys."""

        normalized_resolution = str(resolution).strip().upper()
        if normalized_resolution == "1H":
            normalized_resolution = "60"
        if normalized_resolution == "4H":
            normalized_resolution = "240"
        if normalized_resolution == "D":
            normalized_resolution = "1D"
        if normalized_resolution == "W":
            normalized_resolution = "1W"

        if normalized_resolution not in self._supported_resolutions():
            raise ValueError(
                f"Unsupported chart resolution: {resolution}. "
                f"Supported resolutions: {', '.join(self._supported_resolutions())}"
            )

        return normalized_resolution

    def _resolution_to_seconds(self, resolution: str) -> int:
        """Convert one TradingView resolution key into seconds."""

        normalized_resolution = self._normalize_resolution(resolution)
        if normalized_resolution.isdigit():
            return int(normalized_resolution) * 60

        if normalized_resolution == "1D":
            return 86400

        if normalized_resolution == "1W":
            return 7 * 86400

        raise ValueError(f"Unsupported chart resolution: {resolution}")

    def _resolution_to_pandas_rule(self, resolution: str) -> str:
        """Convert one TradingView resolution key into a pandas resample rule."""

        normalized_resolution = self._normalize_resolution(resolution)
        if normalized_resolution.isdigit():
            return f"{int(normalized_resolution)}min"

        if normalized_resolution == "1D":
            return "1D"

        if normalized_resolution == "1W":
            return "1W"

        raise ValueError(f"Unsupported chart resolution: {resolution}")

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize chart symbols into the project's product-id format."""

        raw_symbol = str(symbol).strip().upper()
        if ":" in raw_symbol:
            raw_symbol = raw_symbol.split(":", 1)[1]

        if "-" in raw_symbol:
            return raw_symbol

        quote_currency = self._quote_currency().upper()
        if raw_symbol.endswith(quote_currency) and len(raw_symbol) > len(quote_currency):
            base_currency = raw_symbol[: -len(quote_currency)]
            return f"{base_currency}-{quote_currency}"

        raise ValueError(
            f"Unsupported chart symbol format: {symbol}. "
            "Use product ids like BTC-USD."
        )

    def _fallback_product_ids(self) -> list[str]:
        """Return the configured fallback product ids when no cached file exists."""

        candidate_product_ids: list[str] = []
        if self.config.market_data_source == "coinmarketcap":
            candidate_product_ids.extend(self.config.coinmarketcap_product_ids)
        else:
            candidate_product_ids.extend(self.config.coinbase_product_ids)
        candidate_product_ids.extend(self.config.live_product_ids)

        deduped_product_ids: list[str] = []
        seen_product_ids: set[str] = set()
        for product_id in candidate_product_ids:
            normalized_product_id = self._normalize_symbol(product_id)
            if normalized_product_id in seen_product_ids:
                continue
            seen_product_ids.add(normalized_product_id)
            deduped_product_ids.append(normalized_product_id)

        return deduped_product_ids

    def _exchange_name(self) -> str:
        """Return the display exchange name for the active source."""

        if self.config.market_data_source == "coinmarketcap":
            return "CMC"

        return "COINBASE"

    def _quote_currency(self) -> str:
        """Return the active quote currency for the current source."""

        if self.config.market_data_source == "coinmarketcap":
            return self.config.coinmarketcap_quote_currency

        return self.config.coinbase_quote_currency

    def _product_id_from_base(self, base_currency: str) -> str:
        """Build one product id from a base currency symbol."""

        return f"{str(base_currency).upper()}-{self._quote_currency().upper()}"

    @staticmethod
    def _price_scale_from_close(last_close: float) -> int:
        """Derive a reasonable TradingView price scale from a recent close."""

        absolute_close = abs(float(last_close))
        if absolute_close >= 1000:
            return 100
        if absolute_close >= 1:
            return 10000
        if absolute_close >= 0.01:
            return 1000000
        return 100000000

    @staticmethod
    def _mark_label_from_category(category: str | None) -> str:
        """Build a compact one-character label for an event mark."""

        normalized_category = str(category or "").strip().upper()
        return normalized_category[:1] or "E"

    @staticmethod
    def _mark_color_from_category(category: str | None) -> str:
        """Choose a stable event-mark color from the CoinMarketCal category."""

        normalized_category = str(category or "").strip().lower()
        if "protocol" in normalized_category or "upgrade" in normalized_category:
            return "#6bd9a8"
        if "listing" in normalized_category or "exchange" in normalized_category:
            return "#ffd166"
        if "conference" in normalized_category or "community" in normalized_category:
            return "#8ec5ff"
        return "#f78c6b"
