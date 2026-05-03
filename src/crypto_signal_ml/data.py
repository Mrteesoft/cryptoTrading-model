"""Classes and helpers for loading, validating, and downloading market data."""

from abc import ABC, abstractmethod
import csv
import io
import math
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd
import numpy as np

from .config import TrainingConfig
from .trading.symbols import is_signal_eligible_base_currency, normalize_base_currency


REQUIRED_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


class CoinMarketCapApiError(ValueError):
    """Base error for CoinMarketCap request failures."""


class CoinMarketCapRateLimitError(CoinMarketCapApiError):
    """Raised when CoinMarketCap rejects a request with HTTP 429."""


def _is_coinmarketcap_rate_limit_error(error: Exception) -> bool:
    """Return whether the error represents a CoinMarketCap rate-limit response."""

    return isinstance(error, CoinMarketCapRateLimitError)


def _request_coinmarketcap_json(
    api_base_url: str,
    endpoint_path: str,
    query_params: Dict[str, object],
    api_key: str,
    request_pause_seconds: float,
) -> Dict[str, object]:
    """Send one CoinMarketCap request and decode the JSON payload."""

    request_url = api_base_url.rstrip("/") + endpoint_path
    if query_params:
        request_url += "?" + urlencode(query_params, doseq=True)

    request = Request(
        request_url,
        headers={
            "Accept": "application/json",
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": api_key,
            "User-Agent": "crypto-signal-ml/0.1",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        error_body = ""
        try:
            error_body = error.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = ""

        query_summary = urlencode(query_params, doseq=True)[:200]
        body_summary = error_body.strip().replace("\n", " ")[:300]
        message = (
            "CoinMarketCap request failed "
            f"with status {error.code} for {endpoint_path}. "
            f"Query: {query_summary or '<none>'}."
        )
        if body_summary:
            message += f" Response: {body_summary}"

        if error.code == 429:
            raise CoinMarketCapRateLimitError(message) from error

        raise CoinMarketCapApiError(message) from error
    except URLError as error:
        raise CoinMarketCapApiError(
            "CoinMarketCap request failed "
            f"for {endpoint_path}. Reason: {error.reason}"
        ) from error

    if request_pause_seconds > 0:
        time.sleep(request_pause_seconds)

    if not isinstance(response_payload, dict):
        raise CoinMarketCapApiError(
            "Unexpected response received from CoinMarketCap. "
            f"Response type: {type(response_payload)}"
        )

    return response_payload


class BasePriceDataLoader(ABC):
    """
    Base class for all market-data loaders.

    The template method pattern is useful here:
    - the base class handles shared validation and cleaning
    - the subclass only decides how raw data is actually read

    That keeps future loaders DRY. For example, a CSV loader and an API loader
    can share the same validation rules without rewriting them.
    """

    required_columns: List[str] = REQUIRED_PRICE_COLUMNS

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def load(self) -> pd.DataFrame:
        """
        Load raw OHLCV data and run the common cleaning pipeline.

        OHLCV means:
        - open
        - high
        - low
        - close
        - volume
        """

        price_df = self._read_data()
        self._validate_required_columns(price_df)
        price_df = self._sort_rows(price_df)
        price_df = self._convert_market_columns_to_numeric(price_df)
        self._validate_market_values(price_df)
        price_df = self._ensure_time_step(price_df)

        return price_df.reset_index(drop=True)

    @abstractmethod
    def _read_data(self) -> pd.DataFrame:
        """Read the raw source into a DataFrame."""

    def _validate_required_columns(self, price_df: pd.DataFrame) -> None:
        """Make sure the raw data has the columns the model needs."""

        missing_columns = [column for column in self.required_columns if column not in price_df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {self.data_path}: {missing_columns}. "
                "The model needs open, high, low, close, and volume."
            )

    def _sort_rows(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort rows in a way that works for both single-asset and multi-asset data.

        We sort by timestamp first so time-based train/test splits still make sense.
        If `product_id` exists, it is used as a secondary key only to keep rows
        deterministic when multiple assets share the same candle timestamp.
        """

        if "timestamp" not in price_df.columns:
            return price_df

        sorted_df = price_df.copy()
        sorted_df["timestamp"] = pd.to_datetime(sorted_df["timestamp"], errors="coerce", utc=True)

        sort_columns = ["timestamp"]
        if "product_id" in sorted_df.columns:
            sort_columns.append("product_id")

        return sorted_df.sort_values(sort_columns).reset_index(drop=True)

    def _convert_market_columns_to_numeric(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert OHLCV columns to numeric values.

        This protects the project from bad text values in the raw file.
        """

        cleaned_df = price_df.copy()
        for column in self.required_columns:
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")

        return cleaned_df

    def _validate_market_values(self, price_df: pd.DataFrame) -> None:
        """Reject rows that still contain invalid market values after conversion."""

        if price_df[self.required_columns].isna().any().any():
            raise ValueError(
                "The raw price file contains empty or invalid numeric values. "
                "Clean the CSV before training the model."
            )

    def _ensure_time_step(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a simple row counter so the rest of the project has a stable index.
        """

        enriched_df = price_df.copy()
        if "time_step" in enriched_df.columns:
            enriched_df = enriched_df.drop(columns=["time_step"])
        enriched_df.insert(0, "time_step", range(1, len(enriched_df) + 1))
        return enriched_df


class CsvPriceDataLoader(BasePriceDataLoader):
    """
    Concrete loader for CSV files.

    The subclass only implements how to read the source.
    Everything else comes from BasePriceDataLoader.
    """

    def _read_data(self) -> pd.DataFrame:
        """Read raw price data from a CSV file."""

        return pd.read_csv(self.data_path)


class BasePriceDataEnricher(ABC):
    """
    Base class for optional dataset enrichers.

    A loader gives us the raw market candles.
    An enricher can then merge in extra context such as:
    - CoinMarketCap metadata
    - market-cap features
    - fundamental or taxonomy data

    Keeping this separate avoids rewriting CSV-loading logic just to add
    one more source of information to the same candle table.
    """

    @abstractmethod
    def enrich(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Merge extra columns into the price DataFrame."""


class EnrichedCsvPriceDataLoader(CsvPriceDataLoader):
    """
    CSV loader that applies one or more enrichers after the base read/clean step.

    This lets the training pipeline stay DRY:
    - read the candle file once
    - run the normal validation once
    - then apply any number of reusable enrichment classes
    """

    def __init__(
        self,
        data_path: Path,
        enrichers: Sequence[BasePriceDataEnricher] = None,
    ) -> None:
        super().__init__(data_path)
        self.enrichers = tuple(enrichers or ())

    def load(self) -> pd.DataFrame:
        """Load the CSV and then attach all configured enrichment sources."""

        price_df = super().load()
        for enricher in self.enrichers:
            price_df = enricher.enrich(price_df)
        return price_df


class EnrichedPriceDataLoader(BasePriceDataLoader):
    """
    Wrap any base loader with one or more enrichment passes.

    The existing `EnrichedCsvPriceDataLoader` only handled CSV sources.
    Live inference now needs the same enrichment behavior when the base
    candles come directly from an API loader instead of a saved CSV.
    """

    def __init__(
        self,
        base_loader: BasePriceDataLoader,
        enrichers: Sequence[BasePriceDataEnricher] = None,
    ) -> None:
        super().__init__(base_loader.data_path)
        self.base_loader = base_loader
        self.enrichers = tuple(enrichers or ())

    def _read_data(self) -> pd.DataFrame:
        """Delegate raw reading to the wrapped base loader."""

        return self.base_loader.load()

    def load(self) -> pd.DataFrame:
        """Load the wrapped source and attach every configured enricher."""

        price_df = self.base_loader.load()
        for enricher in self.enrichers:
            price_df = enricher.enrich(price_df)
        return price_df


class BaseApiPriceDataLoader(BasePriceDataLoader):
    """
    Base class for loaders that download market data from an API.

    The subclass decides:
    - which endpoint to call
    - how to build the request windows
    - how to normalize the API response

    The base API class handles saving the downloaded dataset to disk so the
    rest of the project can train from a normal CSV file afterward.
    """

    def __init__(self, data_path: Path, should_save_downloaded_data: bool = True) -> None:
        super().__init__(data_path)
        self.should_save_downloaded_data = should_save_downloaded_data
        self.last_refresh_summary: Dict[str, object] = {}

    def refresh_data(self) -> pd.DataFrame:
        """
        Download fresh market data, validate it, and optionally save it to CSV.
        """

        price_df = self.load()
        if self.should_save_downloaded_data:
            self._save_downloaded_data(price_df)
        self._cleanup_partial_download()
        return price_df

    def _save_downloaded_data(self, price_df: pd.DataFrame) -> None:
        """Persist the downloaded candles so training can reuse them offline."""

        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_df = price_df.copy()
        if "time_step" in prepared_df.columns:
            prepared_df = prepared_df.drop(columns=["time_step"])
        prepared_df.to_csv(self.data_path, index=False)

    @property
    def partial_data_path(self) -> Path:
        """
        Path used for in-progress market downloads.

        Keeping partial work in a separate file makes it obvious whether the
        refresh finished cleanly before the training pipeline uses the data.
        """

        return self.data_path.with_name(f"{self.data_path.stem}.partial.csv")

    def _save_partial_download(self, price_df: pd.DataFrame) -> None:
        """Persist a partial refresh snapshot for long-running jobs."""

        self.partial_data_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_df = price_df.copy()
        if "time_step" in prepared_df.columns:
            prepared_df = prepared_df.drop(columns=["time_step"])
        prepared_df.to_csv(self.partial_data_path, index=False)

    def _cleanup_partial_download(self) -> None:
        """Remove the temporary partial file after a successful refresh."""

        if self.partial_data_path.exists():
            self.partial_data_path.unlink()


class CoinMarketCapContextEnricher(BasePriceDataEnricher):
    """
    Enrich candle rows with CoinMarketCap metadata and performance context.

    We use CoinMarketCap here for "what the coin is" style information, such as:
    - market-cap rank
    - market-cap and 24h volume
    - 24h / 7d / 30d performance
    - tags and category clues that describe the project's technology/theme

    This is intentionally cached to CSV because model training should not depend
    on a live API call every time the dataset is built.
    """

    map_endpoint = "/v1/cryptocurrency/map"
    info_endpoint = "/v2/cryptocurrency/info"
    latest_quotes_endpoint = "/v2/cryptocurrency/quotes/latest"
    map_request_symbol_batch_size = 25
    keyed_lookup_batch_size = 100

    def __init__(
        self,
        context_path: Path,
        api_base_url: str,
        api_key_env_var: str,
        quote_currency: str = "USD",
        request_pause_seconds: float = 0.2,
        should_refresh_context: bool = False,
        log_progress: bool = True,
    ) -> None:
        self.context_path = context_path
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key_env_var = api_key_env_var
        self.quote_currency = quote_currency
        self.request_pause_seconds = request_pause_seconds
        self.should_refresh_context = should_refresh_context
        self.log_progress = log_progress
        self.last_context_summary: Dict[str, object] = {}

    @property
    def api_key(self) -> str:
        """Read the API key lazily from the configured environment variable."""

        return os.getenv(self.api_key_env_var, "")

    def enrich(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach the latest saved or freshly downloaded CoinMarketCap context.

        If no cache exists and no API key is available, we return the candle
        data unchanged instead of failing the whole training pipeline.
        """

        price_with_keys = price_df.copy()
        if "base_currency" not in price_with_keys.columns and "product_id" in price_with_keys.columns:
            price_with_keys["base_currency"] = (
                price_with_keys["product_id"].astype(str).str.split("-").str[0]
            )

        context_df = self._load_or_refresh_context(price_with_keys)
        if context_df.empty:
            return price_with_keys

        merge_columns = ["base_currency"]
        if "product_id" in price_with_keys.columns and "product_id" in context_df.columns:
            merge_columns = ["product_id", "base_currency"]

        return price_with_keys.merge(context_df, on=merge_columns, how="left")

    def refresh_context(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Download a fresh CoinMarketCap context snapshot and save it to disk.

        This is the method to call when you explicitly want new market-cap,
        performance, and project-technology context for the current coin set.
        """

        if not self.api_key:
            raise ValueError(
                "CoinMarketCap context refresh requires an API key in the "
                f"`{self.api_key_env_var}` environment variable."
            )

        symbol_frame = self._build_symbol_frame(price_df)
        if symbol_frame.empty:
            return pd.DataFrame()

        try:
            map_lookup = self._fetch_symbol_map_lookup(symbol_frame["base_currency"].tolist())
            context_df = self._build_context_frame(
                symbol_frame=symbol_frame,
                map_lookup=map_lookup,
            )
        except CoinMarketCapRateLimitError as error:
            if not self.context_path.exists():
                raise

            cached_context_df = pd.read_csv(self.context_path)
            cached_rows = len(cached_context_df)
            self.last_context_summary = {
                "contextPath": str(self.context_path),
                "coinsRequested": int(len(symbol_frame)),
                "coinsMatched": int(
                    cached_context_df.get("cmc_context_available", pd.Series(dtype=float)).fillna(0).sum()
                )
                if not cached_context_df.empty
                else 0,
                "rowsSaved": 0,
                "existingRowsMerged": cached_rows,
                "finalRowsSaved": cached_rows,
                "usedCachedSnapshot": True,
                "warning": str(error),
            }
            self._log_progress(
                "CoinMarketCap context refresh hit the rate limit. "
                f"Reusing cached snapshot from {self.context_path}."
            )
            return cached_context_df

        existing_row_count = 0
        final_context_df = context_df.copy()

        if self.context_path.exists():
            existing_context_df = pd.read_csv(self.context_path)
            existing_row_count = len(existing_context_df)
            final_context_df = pd.concat([existing_context_df, context_df], ignore_index=True)
            final_context_df = final_context_df.drop_duplicates(
                subset=["product_id", "base_currency"],
                keep="last",
            ).reset_index(drop=True)

        self.context_path.parent.mkdir(parents=True, exist_ok=True)
        final_context_df.to_csv(self.context_path, index=False)
        self.last_context_summary = {
            "contextPath": str(self.context_path),
            "coinsRequested": int(len(symbol_frame)),
            "coinsMatched": int(context_df["cmc_context_available"].sum()) if not context_df.empty else 0,
            "rowsSaved": int(len(context_df)),
            "existingRowsMerged": existing_row_count,
            "finalRowsSaved": int(len(final_context_df)),
        }
        self._log_progress(
            "Saved CoinMarketCap context snapshot "
            f"with {len(final_context_df)} total rows to {self.context_path}"
        )

        return final_context_df

    def _load_or_refresh_context(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Load a cached context snapshot or refresh it when allowed."""

        if self.should_refresh_context and self.api_key:
            return self.refresh_context(price_df)

        if self.context_path.exists():
            return pd.read_csv(self.context_path)

        if self.api_key:
            return self.refresh_context(price_df)

        self._log_progress(
            "CoinMarketCap context skipped because no cached snapshot was found "
            f"and `{self.api_key_env_var}` is not set."
        )
        return pd.DataFrame()

    def _build_symbol_frame(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the unique list of coins that need CoinMarketCap context.

        We use `base_currency` when available, or derive it from `product_id`.
        """

        symbol_df = price_df.copy()

        if "base_currency" not in symbol_df.columns and "product_id" in symbol_df.columns:
            symbol_df["base_currency"] = symbol_df["product_id"].astype(str).str.split("-").str[0]

        required_columns = ["base_currency"]
        if "product_id" in symbol_df.columns:
            required_columns.append("product_id")
        else:
            symbol_df["product_id"] = symbol_df["base_currency"].astype(str) + f"-{self.quote_currency}"
            required_columns.append("product_id")

        symbol_df = symbol_df[required_columns].dropna().drop_duplicates().reset_index(drop=True)
        symbol_df["base_currency"] = symbol_df["base_currency"].astype(str).str.upper()

        return symbol_df.sort_values(["base_currency", "product_id"]).reset_index(drop=True)

    def _fetch_symbol_map_lookup(self, symbols: Sequence[str]) -> Dict[str, Dict[str, object]]:
        """
        Map ticker symbols such as `BTC` or `ADA` to CoinMarketCap ids.

        We keep the best exact symbol match for each requested coin.
        """

        requested_symbols = self._normalize_unique_values(symbols, uppercase=True)
        raw_rows = self._request_symbol_map_rows(requested_symbols)
        matches_by_symbol: Dict[str, List[Dict[str, object]]] = {}

        for raw_row in raw_rows:
            row_symbol = str(raw_row.get("symbol", "")).upper()
            if not row_symbol:
                continue
            matches_by_symbol.setdefault(row_symbol, []).append(raw_row)

        map_lookup: Dict[str, Dict[str, object]] = {}
        for symbol in requested_symbols:
            symbol_matches = matches_by_symbol.get(str(symbol).upper(), [])
            if not symbol_matches:
                continue

            symbol_matches = sorted(
                symbol_matches,
                key=lambda row: (
                    int(row.get("rank") or 10**9),
                    int(row.get("id") or 10**9),
                ),
            )
            map_lookup[str(symbol).upper()] = symbol_matches[0]

        return map_lookup

    def _request_symbol_map_rows(self, symbols: Sequence[str]) -> List[Dict[str, object]]:
        """
        Request CoinMarketCap symbol-map rows in smaller chunks.

        Some symbol combinations can trigger bad-request responses. Chunking and
        one-by-one fallback let a single problematic symbol fail gracefully
        instead of aborting the entire context refresh.
        """

        all_rows: List[Dict[str, object]] = []

        for symbol_chunk in self._chunk_values(symbols, self.map_request_symbol_batch_size):
            try:
                response_payload = self._request_json(
                    endpoint_path=self.map_endpoint,
                    query_params={
                        "symbol": ",".join(symbol_chunk),
                        "listing_status": "active",
                    },
                )
                all_rows.extend(self._extract_list_payload(response_payload))
            except Exception as error:
                if _is_coinmarketcap_rate_limit_error(error):
                    self._log_progress(
                        "CoinMarketCap symbol-map chunk hit the rate limit. "
                        "Skipping one-by-one retries so the cached context snapshot can be reused."
                    )
                    raise
                self._log_progress(
                    "CoinMarketCap symbol-map chunk failed for "
                    f"{len(symbol_chunk)} symbols. Retrying one by one. Error: {error}"
                )

                for symbol in symbol_chunk:
                    try:
                        response_payload = self._request_json(
                            endpoint_path=self.map_endpoint,
                            query_params={
                                "symbol": symbol,
                                "listing_status": "active",
                            },
                        )
                        all_rows.extend(self._extract_list_payload(response_payload))
                    except Exception as single_error:
                        self._log_progress(
                            f"Skipping CoinMarketCap symbol {symbol} after map lookup error: "
                            f"{single_error}"
                        )

        return all_rows

    def _build_context_frame(
        self,
        symbol_frame: pd.DataFrame,
        map_lookup: Dict[str, Dict[str, object]],
    ) -> pd.DataFrame:
        """
        Combine the CMC map, info, and latest quote endpoints into one table.
        """

        matched_ids = [
            int(map_row["id"])
            for map_row in map_lookup.values()
            if map_row.get("id") is not None
        ]

        info_lookup = self._fetch_keyed_lookup(
            endpoint_path=self.info_endpoint,
            ids=matched_ids,
        ) if matched_ids else {}
        quotes_lookup = self._fetch_keyed_lookup(
            endpoint_path=self.latest_quotes_endpoint,
            ids=matched_ids,
            extra_query_params={"convert": self.quote_currency},
        ) if matched_ids else {}

        context_rows = []
        for _, symbol_row in symbol_frame.iterrows():
            base_currency = str(symbol_row["base_currency"]).upper()
            product_id = str(symbol_row["product_id"])
            map_row = map_lookup.get(base_currency)

            if map_row is None:
                context_rows.append(
                    self._build_default_context_row(
                        product_id=product_id,
                        base_currency=base_currency,
                    )
                )
                continue

            cmc_id = str(map_row.get("id"))
            info_row = info_lookup.get(cmc_id, {})
            quote_row = quotes_lookup.get(cmc_id, {})
            quote_details = (quote_row.get("quote") or {}).get(self.quote_currency, {})
            tag_details = self._extract_tag_features(info_row.get("tags") or [])

            circulating_supply = quote_row.get("circulating_supply") or 0.0
            max_supply = quote_row.get("max_supply") or 0.0
            max_supply = max_supply if max_supply not in (None, 0) else 0.0

            context_rows.append(
                {
                    "product_id": product_id,
                    "base_currency": base_currency,
                    "cmc_context_available": 1,
                    "cmc_id": int(map_row.get("id")),
                    "cmc_name": str(map_row.get("name", "")),
                    "cmc_symbol": base_currency,
                    "cmc_slug": str(map_row.get("slug", "")),
                    "cmc_rank": float(map_row.get("rank") or 0),
                    "cmc_category": str(info_row.get("category", "")),
                    "cmc_market_cap": float(quote_details.get("market_cap") or 0.0),
                    "cmc_volume_24h": float(quote_details.get("volume_24h") or 0.0),
                    "cmc_percent_change_1h": float(quote_details.get("percent_change_1h") or 0.0),
                    "cmc_percent_change_24h": float(quote_details.get("percent_change_24h") or 0.0),
                    "cmc_percent_change_7d": float(quote_details.get("percent_change_7d") or 0.0),
                    "cmc_percent_change_30d": float(quote_details.get("percent_change_30d") or 0.0),
                    "cmc_num_market_pairs": float(quote_row.get("num_market_pairs") or 0.0),
                    "cmc_circulating_supply": float(circulating_supply or 0.0),
                    "cmc_max_supply": float(max_supply),
                    "cmc_platform_present": int(info_row.get("platform") is not None),
                    **tag_details,
                }
            )

        return pd.DataFrame(context_rows)

    def _fetch_keyed_lookup(
        self,
        endpoint_path: str,
        ids: Sequence[int],
        extra_query_params: Dict[str, object] = None,
    ) -> Dict[str, Dict[str, object]]:
        """
        Fetch a CMC endpoint whose payload is keyed by coin id.

        The API sometimes returns:
        - `{ "1": {...} }`
        - `{ "1": [{...}] }`

        We normalize both shapes into `{ "1": {...} }`.
        """

        normalized_lookup: Dict[str, Dict[str, object]] = {}
        unique_ids = [int(value) for value in self._normalize_unique_values(ids)]

        for id_chunk in self._chunk_values(unique_ids, self.keyed_lookup_batch_size):
            try:
                normalized_lookup.update(
                    self._request_keyed_lookup_chunk(
                        endpoint_path=endpoint_path,
                        ids=id_chunk,
                        extra_query_params=extra_query_params,
                    )
                )
            except Exception as error:
                if _is_coinmarketcap_rate_limit_error(error):
                    self._log_progress(
                        "CoinMarketCap keyed lookup hit the rate limit. "
                        "Skipping one-by-one retries so the cached context snapshot can be reused."
                    )
                    raise
                self._log_progress(
                    "CoinMarketCap keyed lookup failed for "
                    f"{endpoint_path} on {len(id_chunk)} ids. Retrying one by one. Error: {error}"
                )

                for cmc_id in id_chunk:
                    try:
                        normalized_lookup.update(
                            self._request_keyed_lookup_chunk(
                                endpoint_path=endpoint_path,
                                ids=[cmc_id],
                                extra_query_params=extra_query_params,
                            )
                        )
                    except Exception as single_error:
                        self._log_progress(
                            f"Skipping CoinMarketCap id {cmc_id} for {endpoint_path}: "
                            f"{single_error}"
                        )

        return normalized_lookup

    def _request_keyed_lookup_chunk(
        self,
        endpoint_path: str,
        ids: Sequence[int],
        extra_query_params: Dict[str, object] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Request one keyed CoinMarketCap payload chunk and normalize it."""

        query_params = {
            "id": ",".join(str(cmc_id) for cmc_id in ids),
        }
        if extra_query_params:
            query_params.update(extra_query_params)

        response_payload = self._request_json(
            endpoint_path=endpoint_path,
            query_params=query_params,
        )
        raw_lookup = response_payload.get("data", {}) if isinstance(response_payload, dict) else {}
        normalized_lookup: Dict[str, Dict[str, object]] = {}

        if isinstance(raw_lookup, dict):
            for lookup_key, lookup_value in raw_lookup.items():
                normalized_lookup[str(lookup_key)] = self._normalize_lookup_value(lookup_value)

        return normalized_lookup

    def _extract_list_payload(self, response_payload: Dict[str, object]) -> List[Dict[str, object]]:
        """Normalize list-style payloads into a list of dictionaries."""

        raw_rows = response_payload.get("data", []) if isinstance(response_payload, dict) else []
        if not isinstance(raw_rows, list):
            return []

        return [row for row in raw_rows if isinstance(row, dict)]

    def _normalize_lookup_value(self, lookup_value: object) -> Dict[str, object]:
        """Normalize one keyed API item into a plain dictionary."""

        if isinstance(lookup_value, list):
            if not lookup_value:
                return {}
            first_row = lookup_value[0]
            return first_row if isinstance(first_row, dict) else {}

        return lookup_value if isinstance(lookup_value, dict) else {}

    def _extract_tag_features(self, tags: Sequence[object]) -> Dict[str, object]:
        """
        Turn CoinMarketCap tag lists into simple numeric model features.

        The point is not to encode every tag.
        We only keep a few broad signals that often describe the "tech" or
        market identity of a project.
        """

        lowered_tags = [str(tag).lower() for tag in tags]

        return {
            "cmc_tags_count": float(len(lowered_tags)),
            "cmc_is_mineable": int(any("mineable" in tag for tag in lowered_tags)),
            "cmc_has_defi_tag": int(any("defi" in tag for tag in lowered_tags)),
            "cmc_has_ai_tag": int(any("ai" in tag or "artificial-intelligence" in tag for tag in lowered_tags)),
            "cmc_has_layer1_tag": int(any("layer-1" in tag or "layer 1" in tag for tag in lowered_tags)),
            "cmc_has_gaming_tag": int(any("gaming" in tag or "game" in tag for tag in lowered_tags)),
            "cmc_has_meme_tag": int(any("meme" in tag for tag in lowered_tags)),
        }

    def _build_default_context_row(
        self,
        product_id: str,
        base_currency: str,
    ) -> Dict[str, object]:
        """Return a zero-filled context row when no CMC match is found."""

        return {
            "product_id": product_id,
            "base_currency": base_currency,
            "cmc_context_available": 0,
            "cmc_id": 0,
            "cmc_name": "",
            "cmc_symbol": base_currency,
            "cmc_slug": "",
            "cmc_rank": 0.0,
            "cmc_category": "",
            "cmc_market_cap": 0.0,
            "cmc_volume_24h": 0.0,
            "cmc_percent_change_1h": 0.0,
            "cmc_percent_change_24h": 0.0,
            "cmc_percent_change_7d": 0.0,
            "cmc_percent_change_30d": 0.0,
            "cmc_num_market_pairs": 0.0,
            "cmc_circulating_supply": 0.0,
            "cmc_max_supply": 0.0,
            "cmc_platform_present": 0,
            "cmc_tags_count": 0.0,
            "cmc_is_mineable": 0,
            "cmc_has_defi_tag": 0,
            "cmc_has_ai_tag": 0,
            "cmc_has_layer1_tag": 0,
            "cmc_has_gaming_tag": 0,
            "cmc_has_meme_tag": 0,
        }

    def _normalize_unique_values(
        self,
        values: Sequence[object],
        uppercase: bool = False,
    ) -> List[str]:
        """Clean a sequence into unique string values while preserving order."""

        normalized_values: List[str] = []
        seen_values = set()

        for value in values:
            text_value = str(value).strip()
            if not text_value:
                continue

            if uppercase:
                text_value = text_value.upper()

            if text_value in seen_values:
                continue

            seen_values.add(text_value)
            normalized_values.append(text_value)

        return normalized_values

    def _chunk_values(
        self,
        values: Sequence[object],
        chunk_size: int,
    ) -> List[List[object]]:
        """Split a list of values into smaller request-sized chunks."""

        value_list = list(values)
        if not value_list:
            return []

        if chunk_size <= 0:
            return [value_list]

        return [
            value_list[start_index:start_index + chunk_size]
            for start_index in range(0, len(value_list), chunk_size)
        ]

    def _request_json(
        self,
        endpoint_path: str,
        query_params: Dict[str, object],
    ) -> Dict[str, object]:
        """Send one CoinMarketCap request and decode the JSON payload."""

        return _request_coinmarketcap_json(
            api_base_url=self.api_base_url,
            endpoint_path=endpoint_path,
            query_params=query_params,
            api_key=self.api_key,
            request_pause_seconds=self.request_pause_seconds,
        )

    def _log_progress(self, message: str) -> None:
        """Print enrichment progress when verbose mode is enabled."""

        if self.log_progress:
            print(message)


class CoinMarketCapMarketIntelligenceEnricher(BasePriceDataEnricher):
    """
    Attach cached market-wide CoinMarketCap intelligence to every candle row.

    Unlike the per-coin CMC context cache, this snapshot is global and reflects
    the broader crypto tape at refresh time. The goal is to give the model and
    trader brain a shared view of:
    - market-wide capitalization and volume
    - BTC / ETH dominance
    - stablecoin and DeFi participation
    - the current Fear & Greed reading
    """

    global_metrics_endpoint = "/v1/global-metrics/quotes/latest"
    fear_greed_latest_endpoint = "/v3/fear-and-greed/latest"

    def __init__(
        self,
        intelligence_path: Path,
        api_base_url: str,
        api_key_env_var: str,
        quote_currency: str = "USD",
        request_pause_seconds: float = 0.2,
        should_refresh_market_intelligence: bool = False,
        log_progress: bool = True,
        global_metrics_endpoint: str | None = None,
        fear_greed_latest_endpoint: str | None = None,
    ) -> None:
        self.intelligence_path = intelligence_path
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key_env_var = api_key_env_var
        self.quote_currency = str(quote_currency).strip().upper() or "USD"
        self.request_pause_seconds = request_pause_seconds
        self.should_refresh_market_intelligence = should_refresh_market_intelligence
        self.log_progress = log_progress
        self.last_market_intelligence_summary: Dict[str, object] = {}
        if global_metrics_endpoint:
            self.global_metrics_endpoint = str(global_metrics_endpoint)
        if fear_greed_latest_endpoint:
            self.fear_greed_latest_endpoint = str(fear_greed_latest_endpoint)

    @property
    def api_key(self) -> str:
        """Read the API key lazily from the configured environment variable."""

        return os.getenv(self.api_key_env_var, "")

    def enrich(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Broadcast the cached market snapshot across every row in the table."""

        intelligence_df = self._load_or_refresh_market_intelligence()
        if intelligence_df.empty:
            return self._attach_default_market_intelligence_columns(price_df)

        intelligence_row = intelligence_df.iloc[0].to_dict()
        return self._attach_market_intelligence_snapshot(price_df, intelligence_row)

    def refresh_market_intelligence(self) -> pd.DataFrame:
        """Download a fresh market-intelligence snapshot and save it to disk."""

        if not self.api_key:
            raise ValueError(
                "CoinMarketCap market-intelligence refresh requires an API key in the "
                f"`{self.api_key_env_var}` environment variable."
            )

        try:
            global_metrics_payload = self._request_json(
                endpoint_path=self.global_metrics_endpoint,
                query_params={"convert": self.quote_currency},
            )
            fear_greed_payload = self._request_json(
                endpoint_path=self.fear_greed_latest_endpoint,
                query_params={},
            )
            intelligence_row = self._build_market_intelligence_row(
                global_metrics_payload=global_metrics_payload,
                fear_greed_payload=fear_greed_payload,
            )
            intelligence_df = pd.DataFrame([intelligence_row])
        except CoinMarketCapRateLimitError as error:
            if not self.intelligence_path.exists():
                raise

            cached_intelligence_df = pd.read_csv(self.intelligence_path)
            cached_row = cached_intelligence_df.iloc[0].to_dict() if not cached_intelligence_df.empty else {}
            self.last_market_intelligence_summary = {
                "intelligencePath": str(self.intelligence_path),
                "lastUpdated": str(cached_row.get("cmc_market_last_updated", "")),
                "fearGreedValue": float(cached_row.get("cmc_market_fear_greed_value", 0.0) or 0.0),
                "fearGreedClassification": str(
                    cached_row.get("cmc_market_fear_greed_classification", "")
                ),
                "usedCachedSnapshot": True,
                "warning": str(error),
            }
            self._log_progress(
                "CoinMarketCap market-intelligence refresh hit the rate limit. "
                f"Reusing cached snapshot from {self.intelligence_path}."
            )
            return cached_intelligence_df

        self.intelligence_path.parent.mkdir(parents=True, exist_ok=True)
        intelligence_df.to_csv(self.intelligence_path, index=False)
        self.last_market_intelligence_summary = {
            "intelligencePath": str(self.intelligence_path),
            "lastUpdated": str(intelligence_row["cmc_market_last_updated"]),
            "fearGreedValue": float(intelligence_row["cmc_market_fear_greed_value"]),
            "fearGreedClassification": str(intelligence_row["cmc_market_fear_greed_classification"]),
        }
        self._log_progress(
            "Saved CoinMarketCap market intelligence snapshot "
            f"to {self.intelligence_path}"
        )

        return intelligence_df

    def _load_or_refresh_market_intelligence(self) -> pd.DataFrame:
        """Load the cached market snapshot or refresh it when allowed."""

        if self.should_refresh_market_intelligence and self.api_key:
            return self.refresh_market_intelligence()

        if self.intelligence_path.exists():
            return pd.read_csv(self.intelligence_path)

        if self.api_key:
            return self.refresh_market_intelligence()

        self._log_progress(
            "CoinMarketCap market intelligence skipped because no cached snapshot was found "
            f"and `{self.api_key_env_var}` is not set."
        )
        return pd.DataFrame()

    def _build_market_intelligence_row(
        self,
        global_metrics_payload: Dict[str, object],
        fear_greed_payload: Dict[str, object],
    ) -> Dict[str, object]:
        """Normalize the two CMC payloads into one flat snapshot row."""

        global_data = global_metrics_payload.get("data", {})
        if not isinstance(global_data, dict):
            global_data = {}

        quote_rows = global_data.get("quote", {})
        if not isinstance(quote_rows, dict):
            quote_rows = {}
        quote_details = quote_rows.get(self.quote_currency, {})
        if not isinstance(quote_details, dict):
            quote_details = {}

        fear_greed_data = fear_greed_payload.get("data", {})
        if not isinstance(fear_greed_data, dict):
            fear_greed_data = {}

        total_market_cap = self._safe_float(quote_details.get("total_market_cap"))
        altcoin_market_cap = self._safe_float(quote_details.get("altcoin_market_cap"))
        stablecoin_market_cap = self._safe_float(quote_details.get("stablecoin_market_cap"))

        return {
            "cmc_market_intelligence_available": 1,
            "cmc_market_quote_currency": self.quote_currency,
            "cmc_market_last_updated": (
                str(quote_details.get("last_updated") or global_data.get("last_updated") or fear_greed_data.get("update_time") or "")
            ),
            "cmc_market_total_market_cap": total_market_cap,
            "cmc_market_total_volume_24h": self._safe_float(quote_details.get("total_volume_24h")),
            "cmc_market_total_market_cap_change_24h": self._safe_float(
                quote_details.get("total_market_cap_yesterday_percentage_change")
            ),
            "cmc_market_total_volume_change_24h": self._safe_float(
                quote_details.get("total_volume_24h_yesterday_percentage_change")
            ),
            "cmc_market_altcoin_market_cap": altcoin_market_cap,
            "cmc_market_altcoin_share": self._safe_ratio(altcoin_market_cap, total_market_cap),
            "cmc_market_btc_dominance": self._safe_float(global_data.get("btc_dominance")),
            "cmc_market_btc_dominance_change_24h": self._safe_float(
                global_data.get("btc_dominance_24h_percentage_change")
            ),
            "cmc_market_eth_dominance": self._safe_float(global_data.get("eth_dominance")),
            "cmc_market_stablecoin_market_cap": stablecoin_market_cap,
            "cmc_market_stablecoin_share": self._safe_ratio(stablecoin_market_cap, total_market_cap),
            "cmc_market_defi_market_cap": self._safe_float(quote_details.get("defi_market_cap")),
            "cmc_market_defi_volume_24h": self._safe_float(quote_details.get("defi_volume_24h")),
            "cmc_market_derivatives_volume_24h": self._safe_float(
                quote_details.get("derivatives_volume_24h")
            ),
            "cmc_market_fear_greed_value": self._safe_float(fear_greed_data.get("value")),
            "cmc_market_fear_greed_classification": str(
                fear_greed_data.get("value_classification") or ""
            ).strip(),
        }

    def _attach_market_intelligence_snapshot(
        self,
        price_df: pd.DataFrame,
        intelligence_row: Dict[str, object],
    ) -> pd.DataFrame:
        """Attach one market-intelligence snapshot to every row in the price table."""

        output_df = self._attach_default_market_intelligence_columns(price_df)
        for column_name in self._default_market_intelligence_columns():
            output_df[column_name] = intelligence_row.get(column_name, output_df[column_name])
        return output_df

    @staticmethod
    def _default_market_intelligence_columns() -> Dict[str, object]:
        """Return the default values for every market-intelligence column."""

        return {
            "cmc_market_intelligence_available": 0.0,
            "cmc_market_quote_currency": "",
            "cmc_market_last_updated": "",
            "cmc_market_total_market_cap": 0.0,
            "cmc_market_total_volume_24h": 0.0,
            "cmc_market_total_market_cap_change_24h": 0.0,
            "cmc_market_total_volume_change_24h": 0.0,
            "cmc_market_altcoin_market_cap": 0.0,
            "cmc_market_altcoin_share": 0.0,
            "cmc_market_btc_dominance": 0.0,
            "cmc_market_btc_dominance_change_24h": 0.0,
            "cmc_market_eth_dominance": 0.0,
            "cmc_market_stablecoin_market_cap": 0.0,
            "cmc_market_stablecoin_share": 0.0,
            "cmc_market_defi_market_cap": 0.0,
            "cmc_market_defi_volume_24h": 0.0,
            "cmc_market_derivatives_volume_24h": 0.0,
            "cmc_market_fear_greed_value": 0.0,
            "cmc_market_fear_greed_classification": "",
        }

    def _attach_default_market_intelligence_columns(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Return the input table with zero-filled market-intelligence columns."""

        output_df = price_df.copy()
        for column_name, default_value in self._default_market_intelligence_columns().items():
            if column_name not in output_df.columns:
                output_df[column_name] = default_value
        return output_df

    @staticmethod
    def _safe_float(raw_value: object) -> float:
        """Convert an optional payload value into a float without raising."""

        try:
            if raw_value in {None, ""}:
                return 0.0
            return float(raw_value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        """Divide two floats while preventing divide-by-zero errors."""

        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    def _request_json(
        self,
        endpoint_path: str,
        query_params: Dict[str, object],
    ) -> Dict[str, object]:
        """Send one CoinMarketCap request and decode the JSON payload."""

        return _request_coinmarketcap_json(
            api_base_url=self.api_base_url,
            endpoint_path=endpoint_path,
            query_params=query_params,
            api_key=self.api_key,
            request_pause_seconds=self.request_pause_seconds,
        )

    def _log_progress(self, message: str) -> None:
        """Print enrichment progress when verbose mode is enabled."""

        if self.log_progress:
            print(message)


class CoinMarketCalEventEnricher(BasePriceDataEnricher):
    """
    Attach cached upcoming-event context from CoinMarketCal to candle rows.

    The first objective here is pipeline support:
    - load a saved event cache when it exists
    - optionally refresh that cache from the API
    - derive row-level event proximity columns without making them mandatory

    Event-derived columns are intentionally not part of the core ML feature set
    yet because leakage control depends on event announcement timestamps.
    """

    events_endpoint = "/events"

    def __init__(
        self,
        events_path: Path,
        api_base_url: str,
        api_key_env_var: str,
        lookahead_days: int = 30,
        request_pause_seconds: float = 0.2,
        should_refresh_events: bool = False,
        log_progress: bool = True,
    ) -> None:
        self.events_path = events_path
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key_env_var = api_key_env_var
        self.lookahead_days = max(int(lookahead_days), 1)
        self.request_pause_seconds = request_pause_seconds
        self.should_refresh_events = should_refresh_events
        self.log_progress = log_progress
        self.last_events_summary: Dict[str, object] = {}

    @property
    def api_key(self) -> str:
        """Read the API key lazily from the configured environment variable."""

        return os.getenv(self.api_key_env_var, "")

    def enrich(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Attach upcoming-event proximity columns to the current price table."""

        price_with_keys = price_df.copy()
        if "base_currency" not in price_with_keys.columns and "product_id" in price_with_keys.columns:
            price_with_keys["base_currency"] = (
                price_with_keys["product_id"].astype(str).str.split("-").str[0]
            )

        event_df = self._load_or_refresh_events(price_with_keys)
        if event_df.empty:
            return self._attach_default_event_columns(price_with_keys)

        return self._merge_event_features(price_with_keys, event_df)

    def refresh_events(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch a fresh event snapshot and save it to the configured cache file."""

        if not self.api_key:
            raise ValueError(
                "CoinMarketCal event refresh requires an API key in the "
                f"`{self.api_key_env_var}` environment variable."
            )

        symbol_frame = self._build_symbol_frame(price_df)
        if symbol_frame.empty:
            return pd.DataFrame()

        event_rows = self._request_event_rows(symbol_frame["base_currency"].tolist())
        events_df = pd.DataFrame(event_rows)
        if events_df.empty:
            return pd.DataFrame()

        events_df["base_currency"] = events_df["base_currency"].astype(str).str.upper()
        events_df["event_start"] = pd.to_datetime(events_df["event_start"], errors="coerce", utc=True)
        events_df = events_df.dropna(subset=["base_currency", "event_start"]).sort_values(
            ["base_currency", "event_start", "event_id"]
        ).reset_index(drop=True)

        existing_row_count = 0
        final_events_df = events_df.copy()

        if self.events_path.exists():
            existing_events_df = pd.read_csv(self.events_path)
            if not existing_events_df.empty:
                existing_events_df["event_start"] = pd.to_datetime(
                    existing_events_df["event_start"],
                    errors="coerce",
                    utc=True,
                )
                existing_row_count = len(existing_events_df)
                final_events_df = pd.concat([existing_events_df, events_df], ignore_index=True)
                final_events_df = final_events_df.drop_duplicates(
                    subset=["base_currency", "event_start", "event_title"],
                    keep="last",
                ).sort_values(["base_currency", "event_start", "event_id"]).reset_index(drop=True)

        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        final_events_df.to_csv(self.events_path, index=False)
        self.last_events_summary = {
            "eventsPath": str(self.events_path),
            "coinsRequested": int(len(symbol_frame)),
            "rowsSaved": int(len(events_df)),
            "existingRowsMerged": existing_row_count,
            "finalRowsSaved": int(len(final_events_df)),
        }
        self._log_progress(
            "Saved CoinMarketCal events snapshot "
            f"with {len(final_events_df)} total rows to {self.events_path}"
        )

        return final_events_df

    def _load_or_refresh_events(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Load a cached event snapshot or refresh it when allowed."""

        if self.should_refresh_events and self.api_key:
            return self.refresh_events(price_df)

        if self.events_path.exists():
            return pd.read_csv(self.events_path)

        if self.api_key:
            return self.refresh_events(price_df)

        self._log_progress(
            "CoinMarketCal events skipped because no cached snapshot was found "
            f"and `{self.api_key_env_var}` is not set."
        )
        return pd.DataFrame()

    def _build_symbol_frame(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Build the unique list of assets that need event context."""

        symbol_df = price_df.copy()
        if "base_currency" not in symbol_df.columns and "product_id" in symbol_df.columns:
            symbol_df["base_currency"] = symbol_df["product_id"].astype(str).str.split("-").str[0]

        symbol_df = symbol_df[["base_currency"]].dropna().drop_duplicates().reset_index(drop=True)
        symbol_df["base_currency"] = symbol_df["base_currency"].astype(str).str.upper()
        return symbol_df.sort_values("base_currency").reset_index(drop=True)

    def _request_event_rows(self, base_currencies: Sequence[str]) -> List[Dict[str, object]]:
        """
        Request upcoming events for the current asset set.

        The API contract is intentionally parsed defensively because CoinMarketCal
        payloads can vary by plan and endpoint version.
        """

        request_symbols = self._normalize_unique_values(base_currencies, uppercase=True)
        if not request_symbols:
            return []

        response_payload = self._request_json(
            endpoint_path=self.events_endpoint,
            query_params={
                "symbols": ",".join(request_symbols),
                "max": 1000,
            },
        )
        raw_events = self._extract_event_payload(response_payload)
        normalized_rows: List[Dict[str, object]] = []

        for raw_event in raw_events:
            normalized_rows.extend(self._normalize_event_row(raw_event))

        return normalized_rows

    def _extract_event_payload(self, response_payload: Dict[str, object]) -> List[Dict[str, object]]:
        """Normalize list-style CoinMarketCal responses into dictionaries."""

        for payload_key in ("body", "data", "events"):
            raw_rows = response_payload.get(payload_key, [])
            if isinstance(raw_rows, list):
                return [row for row in raw_rows if isinstance(row, dict)]

        return []

    def _normalize_event_row(self, raw_event: Dict[str, object]) -> List[Dict[str, object]]:
        """Convert one raw event payload into one or more asset-linked rows."""

        event_id = raw_event.get("id") or raw_event.get("event_id") or ""
        event_title = raw_event.get("title") or raw_event.get("name") or ""
        event_category = raw_event.get("category") or raw_event.get("type") or ""
        event_start = (
            raw_event.get("date_event")
            or raw_event.get("event_date")
            or raw_event.get("start_date")
            or raw_event.get("created_date")
            or raw_event.get("date")
        )

        coin_rows = raw_event.get("coins") or raw_event.get("currencies") or []
        if isinstance(coin_rows, dict):
            coin_rows = [coin_rows]

        normalized_rows: List[Dict[str, object]] = []
        for coin_row in coin_rows:
            if not isinstance(coin_row, dict):
                continue

            base_currency = (
                coin_row.get("symbol")
                or coin_row.get("code")
                or coin_row.get("short")
                or ""
            )
            base_currency = str(base_currency).strip().upper()
            if not base_currency:
                continue

            normalized_rows.append(
                {
                    "event_id": str(event_id),
                    "event_title": str(event_title),
                    "event_category": str(event_category),
                    "event_start": event_start,
                    "base_currency": base_currency,
                }
            )

        return normalized_rows

    def _merge_event_features(
        self,
        price_df: pd.DataFrame,
        event_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Derive simple upcoming-event proximity columns for each candle row."""

        output_df = self._attach_default_event_columns(price_df)
        output_df["timestamp"] = pd.to_datetime(output_df["timestamp"], errors="coerce", utc=True)

        working_event_df = event_df.copy()
        working_event_df["base_currency"] = working_event_df["base_currency"].astype(str).str.upper()
        working_event_df["event_start"] = pd.to_datetime(working_event_df["event_start"], errors="coerce", utc=True)
        working_event_df = working_event_df.dropna(subset=["base_currency", "event_start"])

        seven_day_ns = int(pd.Timedelta(days=7).value)
        thirty_day_ns = int(pd.Timedelta(days=30).value)
        hour_ns = float(pd.Timedelta(hours=1).value)

        for base_currency, row_index in output_df.groupby("base_currency", sort=False).groups.items():
            asset_event_df = working_event_df.loc[
                working_event_df["base_currency"] == str(base_currency).upper()
            ].sort_values("event_start")
            if asset_event_df.empty:
                continue

            row_positions = list(row_index)
            event_start_ns = pd.to_datetime(
                asset_event_df["event_start"],
                errors="coerce",
                utc=True,
            ).to_numpy(dtype="datetime64[ns]").astype("int64")
            row_timestamp_ns = pd.to_datetime(
                output_df.loc[row_positions, "timestamp"],
                errors="coerce",
                utc=True,
            ).to_numpy(dtype="datetime64[ns]").astype("int64")
            delta_ns = event_start_ns[np.newaxis, :] - row_timestamp_ns[:, np.newaxis]
            future_mask = delta_ns >= 0

            event_count_next_7d = np.sum(
                future_mask & (delta_ns <= seven_day_ns),
                axis=1,
            )
            event_count_next_30d = np.sum(
                future_mask & (delta_ns <= thirty_day_ns),
                axis=1,
            )
            next_delta_ns = np.where(
                future_mask,
                delta_ns,
                np.iinfo(np.int64).max,
            )
            next_event_exists = future_mask.any(axis=1)
            hours_to_next_event = np.where(
                next_event_exists,
                next_delta_ns.min(axis=1) / hour_ns,
                np.nan,
            )

            output_df.loc[row_positions, "cmcal_event_count_next_7d"] = event_count_next_7d
            output_df.loc[row_positions, "cmcal_event_count_next_30d"] = event_count_next_30d
            output_df.loc[row_positions, "cmcal_has_event_next_7d"] = (event_count_next_7d > 0).astype(float)
            output_df.loc[row_positions, "cmcal_hours_to_next_event"] = hours_to_next_event
            output_df.loc[row_positions, "cmcal_days_to_next_event"] = hours_to_next_event / 24.0

        return output_df

    def _attach_default_event_columns(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Return the input table with zero-filled event context columns."""

        output_df = price_df.copy()
        for column_name, default_value in {
            "cmcal_event_count_next_7d": 0.0,
            "cmcal_event_count_next_30d": 0.0,
            "cmcal_has_event_next_7d": 0.0,
            "cmcal_hours_to_next_event": np.nan,
            "cmcal_days_to_next_event": np.nan,
        }.items():
            if column_name not in output_df.columns:
                output_df[column_name] = default_value

        return output_df

    def _normalize_unique_values(
        self,
        values: Sequence[object],
        uppercase: bool = False,
    ) -> List[str]:
        """Clean a sequence into unique string values while preserving order."""

        normalized_values: List[str] = []
        seen_values = set()

        for value in values:
            text_value = str(value).strip()
            if not text_value:
                continue

            if uppercase:
                text_value = text_value.upper()

            if text_value in seen_values:
                continue

            seen_values.add(text_value)
            normalized_values.append(text_value)

        return normalized_values

    def _request_json(
        self,
        endpoint_path: str,
        query_params: Dict[str, object],
    ) -> Dict[str, object]:
        """Send one CoinMarketCal request and decode the JSON payload."""

        request_url = self.api_base_url + endpoint_path
        if query_params:
            request_url += "?" + urlencode(query_params, doseq=True)

        request = Request(
            request_url,
            headers={
                "Accept": "application/json",
                "x-api-key": self.api_key,
                "User-Agent": "crypto-signal-ml/0.1",
            },
        )

        try:
            with urlopen(request, timeout=30) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            error_body = ""
            try:
                error_body = error.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""

            query_summary = urlencode(query_params, doseq=True)[:200]
            body_summary = error_body.strip().replace("\n", " ")[:300]
            message = (
                "CoinMarketCal request failed "
                f"with status {error.code} for {endpoint_path}. "
                f"Query: {query_summary or '<none>'}."
            )
            if body_summary:
                message += f" Response: {body_summary}"
            raise ValueError(message) from error
        except URLError as error:
            raise ValueError(
                "CoinMarketCal request failed "
                f"for {endpoint_path}. Reason: {error.reason}"
            ) from error

        if self.request_pause_seconds > 0:
            time.sleep(self.request_pause_seconds)

        if not isinstance(response_payload, dict):
            raise ValueError(
                "Unexpected response received from CoinMarketCal. "
                f"Response type: {type(response_payload)}"
            )

        return response_payload

    def _log_progress(self, message: str) -> None:
        """Print event-enrichment progress when verbose mode is enabled."""

        if self.log_progress:
            print(message)


class CoinMarketCapOhlcvPriceDataLoader(BaseApiPriceDataLoader):
    """
    Download historical OHLCV candles from the CoinMarketCap API.

    This loader is meant to replace Coinbase as the primary backend price
    source for the model, while still keeping the rest of the project's
    expectations unchanged:
    - one combined OHLCV table
    - multi-asset support
    - batch-aware refreshes
    - the same saved CSV schema used by training and live inference
    """

    api_name = "coinmarketcap"
    valid_granularities = {3600, 86400}

    def __init__(
        self,
        data_path: Path,
        api_base_url: str,
        api_key_env_var: str,
        quote_currency: str,
        granularity_seconds: int,
        total_candles: int,
        request_pause_seconds: float = 0.2,
        should_save_downloaded_data: bool = True,
        product_id: Optional[str] = None,
        product_ids: Sequence[str] = None,
        fetch_all_quote_products: bool = False,
        excluded_base_currencies: Sequence[str] = None,
        max_products: Optional[int] = None,
        product_batch_size: Optional[int] = None,
        product_batch_number: int = 1,
        save_progress_every_products: int = 5,
        log_progress: bool = True,
        historical_endpoint: str = "/v2/cryptocurrency/ohlcv/historical",
        map_endpoint: str = "/v1/cryptocurrency/map",
    ) -> None:
        super().__init__(data_path=data_path, should_save_downloaded_data=should_save_downloaded_data)
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key_env_var = api_key_env_var
        self.quote_currency = quote_currency
        self.granularity_seconds = granularity_seconds
        self.total_candles = total_candles
        self.request_pause_seconds = request_pause_seconds
        self.product_id = product_id
        self.product_ids = tuple(product_ids or ())
        self.fetch_all_quote_products = fetch_all_quote_products
        self.excluded_base_currencies = tuple(excluded_base_currencies or ())
        self.max_products = max_products
        self.product_batch_size = product_batch_size
        self.product_batch_number = product_batch_number
        self.save_progress_every_products = save_progress_every_products
        self.log_progress = log_progress
        self.historical_endpoint = historical_endpoint
        self.map_endpoint = map_endpoint

    @property
    def api_key(self) -> str:
        """Read the API key lazily from the configured environment variable."""

        return os.getenv(self.api_key_env_var, "")

    def _read_data(self) -> pd.DataFrame:
        """Download fresh candles from CoinMarketCap and normalize them."""

        self._validate_download_settings()

        available_products = self._resolve_products_to_download()
        selected_products, batch_summary = self._slice_products_for_batch(available_products)
        all_candle_rows: List[Dict[str, object]] = []

        self._log_progress(
            "Starting CoinMarketCap market refresh: "
            f"{len(selected_products)} products in this batch, "
            f"{self.total_candles} candles per product."
        )

        for product_index, product_details in enumerate(selected_products):
            self._log_progress(
                f"[{product_index + 1}/{len(selected_products)}] "
                f"Downloading {product_details['product_id']}"
            )
            candle_rows = self._request_historical_ohlcv_rows(
                base_currency=product_details["base_currency"],
            )
            normalized_rows = self._normalize_candle_rows(
                product_id=product_details["product_id"],
                base_currency=product_details["base_currency"],
                quote_currency=product_details["quote_currency"],
                candle_rows=candle_rows,
            )
            all_candle_rows.extend(normalized_rows)
            self._maybe_save_partial_progress(
                all_candle_rows=all_candle_rows,
                completed_products=product_index + 1,
            )

        if not all_candle_rows:
            raise ValueError("The CoinMarketCap API returned no candle data for the selected products.")

        price_df = self._build_price_frame(all_candle_rows)
        self.last_refresh_summary = {
            "totalAvailableProducts": len(available_products),
            "productsDownloaded": len(selected_products),
            "requestWindowsPerProduct": 1,
            "batchSize": batch_summary["batchSize"],
            "batchNumber": batch_summary["batchNumber"],
            "batchStartIndex": batch_summary["batchStartIndex"],
            "batchEndIndex": batch_summary["batchEndIndex"],
            "partialSavePath": str(self.partial_data_path),
            "rowsDownloaded": len(price_df),
        }
        self._log_progress(
            f"Completed CoinMarketCap refresh batch with {len(price_df)} rows "
            f"across {len(selected_products)} products."
        )
        return price_df

    def _save_downloaded_data(self, price_df: pd.DataFrame) -> None:
        """Save the downloaded batch, merging with the existing market file when needed."""

        final_df = price_df.copy()
        final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors="coerce", utc=True)
        existing_row_count = 0

        if self.product_batch_size is not None and self.data_path.exists():
            existing_df = pd.read_csv(self.data_path)
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], errors="coerce", utc=True)
            existing_row_count = len(existing_df)
            final_df = pd.concat([existing_df, price_df], ignore_index=True)
            final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors="coerce", utc=True)
            final_df = final_df.drop_duplicates(subset=["timestamp", "product_id"], keep="last")
            final_df = self._sort_rows(final_df)

        super()._save_downloaded_data(final_df)
        self.last_refresh_summary["existingRowsMerged"] = existing_row_count
        self.last_refresh_summary["finalRowsSaved"] = len(final_df)

    def _validate_download_settings(self) -> None:
        """Validate the requested market-data settings before hitting the API."""

        if not self.api_key:
            raise ValueError(
                "CoinMarketCap market refresh requires an API key in the "
                f"`{self.api_key_env_var}` environment variable."
            )

        if self.granularity_seconds not in self.valid_granularities:
            raise ValueError(
                "coinmarketcap_granularity_seconds must be one of "
                f"{sorted(self.valid_granularities)}."
            )

        if self.total_candles <= 0:
            raise ValueError("coinmarketcap_total_candles must be greater than zero.")

        if self.product_batch_size is not None and self.product_batch_size <= 0:
            raise ValueError("coinmarketcap_product_batch_size must be greater than zero when provided.")

        if self.product_batch_number <= 0:
            raise ValueError("coinmarketcap_product_batch_number must be greater than zero.")

        if self.save_progress_every_products < 0:
            raise ValueError("coinmarketcap_save_progress_every_products cannot be negative.")

    def _resolve_products_to_download(self) -> List[Dict[str, str]]:
        """Decide which assets belong in the CoinMarketCap download universe."""

        if self.fetch_all_quote_products:
            return self._fetch_filtered_products()

        explicit_product_ids = list(self.product_ids)
        if self.product_id:
            explicit_product_ids.append(self.product_id)

        if not explicit_product_ids:
            raise ValueError("No CoinMarketCap products were provided for download.")

        return [
            {
                "product_id": product_id,
                "base_currency": product_id.split("-")[0].upper(),
                "quote_currency": product_id.split("-")[1].upper() if "-" in product_id else self.quote_currency.upper(),
            }
            for product_id in explicit_product_ids
        ]

    def _fetch_filtered_products(self) -> List[Dict[str, str]]:
        """Fetch the CoinMarketCap asset catalog and keep the desired universe."""

        response_payload = self._request_json(
            endpoint_path=self.map_endpoint,
            query_params={
                "listing_status": "active",
            },
        )
        raw_rows = response_payload.get("data", []) if isinstance(response_payload, dict) else []
        if not isinstance(raw_rows, list):
            raise ValueError(
                "Unexpected response received from CoinMarketCap map endpoint. "
                f"Response type: {type(raw_rows)}"
            )

        excluded_bases = {currency.upper() for currency in self.excluded_base_currencies}
        best_row_by_symbol: Dict[str, Dict[str, object]] = {}

        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                continue

            base_currency = normalize_base_currency(raw_row.get("symbol"))
            if not base_currency:
                continue
            if base_currency in excluded_bases:
                continue
            if not is_signal_eligible_base_currency(base_currency):
                continue

            current_row = best_row_by_symbol.get(base_currency)
            if current_row is None:
                best_row_by_symbol[base_currency] = raw_row
                continue

            if int(raw_row.get("rank") or 10**9) < int(current_row.get("rank") or 10**9):
                best_row_by_symbol[base_currency] = raw_row

        filtered_products = [
            {
                "product_id": f"{normalize_base_currency(raw_row.get('symbol'))}-{self.quote_currency.upper()}",
                "base_currency": normalize_base_currency(raw_row.get("symbol")),
                "quote_currency": self.quote_currency.upper(),
            }
            for raw_row in sorted(
                best_row_by_symbol.values(),
                key=lambda row: (
                    int(row.get("rank") or 10**9),
                    normalize_base_currency(row.get("symbol")),
                ),
            )
        ]

        if self.max_products is not None:
            filtered_products = filtered_products[: self.max_products]

        if not filtered_products:
            raise ValueError("No CoinMarketCap assets matched the configured multi-coin filters.")

        return filtered_products

    def get_available_products(self) -> List[Dict[str, str]]:
        """Return the filtered asset universe for the current loader settings."""

        self._validate_download_settings()
        return self._resolve_products_to_download()

    def get_total_batches(self) -> int:
        """Return how many product batches are needed for the current universe."""

        available_products = self.get_available_products()
        if self.product_batch_size is None:
            return 1

        return max(1, math.ceil(len(available_products) / self.product_batch_size))

    def _slice_products_for_batch(
        self,
        selected_products: List[Dict[str, str]],
    ) -> tuple[List[Dict[str, str]], Dict[str, int]]:
        """Limit a large asset universe to one configured batch."""

        if self.product_batch_size is None:
            batch_summary = {
                "batchSize": len(selected_products),
                "batchNumber": 1,
                "batchStartIndex": 1,
                "batchEndIndex": len(selected_products),
            }
            return selected_products, batch_summary

        total_products = len(selected_products)
        total_batches = max(1, math.ceil(total_products / self.product_batch_size))

        if self.product_batch_number > total_batches:
            raise ValueError(
                "Requested CoinMarketCap product batch is out of range. "
                f"Requested batch {self.product_batch_number}, available batches {total_batches}."
            )

        batch_start_index = (self.product_batch_number - 1) * self.product_batch_size
        batch_end_index = min(batch_start_index + self.product_batch_size, total_products)
        batch_products = selected_products[batch_start_index:batch_end_index]

        batch_summary = {
            "batchSize": len(batch_products),
            "batchNumber": self.product_batch_number,
            "batchStartIndex": batch_start_index + 1,
            "batchEndIndex": batch_end_index,
        }

        self._log_progress(
            "Using CoinMarketCap product batch "
            f"{self.product_batch_number}/{total_batches}: "
            f"products {batch_summary['batchStartIndex']} to {batch_summary['batchEndIndex']}."
        )

        return batch_products, batch_summary

    def _maybe_save_partial_progress(
        self,
        all_candle_rows: Sequence[Dict[str, object]],
        completed_products: int,
    ) -> None:
        """Save a partial CSV snapshot during a long-running market refresh."""

        if not self.should_save_downloaded_data:
            return

        if self.save_progress_every_products == 0:
            return

        if completed_products % self.save_progress_every_products != 0:
            return

        partial_df = self._build_price_frame(all_candle_rows)
        self._save_partial_download(partial_df)
        self._log_progress(
            f"Saved partial market data after {completed_products} products "
            f"to {self.partial_data_path}"
        )

    def _request_historical_ohlcv_rows(
        self,
        base_currency: str,
    ) -> List[Dict[str, object]]:
        """Request one historical OHLCV series and extract the nested quote rows."""

        response_payload = self._request_json(
            endpoint_path=self.historical_endpoint,
            query_params={
                "symbol": base_currency,
                "convert": self.quote_currency,
                "count": self.total_candles,
                "interval": self._resolve_interval_alias(),
                "time_period": self._resolve_interval_alias(),
            },
        )
        return self._extract_historical_quote_rows(response_payload, base_currency=base_currency)

    def _extract_historical_quote_rows(
        self,
        response_payload: Dict[str, object],
        base_currency: str,
    ) -> List[Dict[str, object]]:
        """Extract one flat list of OHLCV quote rows from a CoinMarketCap payload."""

        raw_data = response_payload.get("data", {}) if isinstance(response_payload, dict) else {}
        candidate_assets: List[Dict[str, object]] = []

        if isinstance(raw_data, list):
            candidate_assets.extend([row for row in raw_data if isinstance(row, dict)])
        elif isinstance(raw_data, dict):
            if any(key in raw_data for key in ("quotes", "quote", "symbol", "id")):
                candidate_assets.append(raw_data)
            else:
                for asset_value in raw_data.values():
                    if isinstance(asset_value, dict):
                        candidate_assets.append(asset_value)
                    elif isinstance(asset_value, list):
                        candidate_assets.extend([row for row in asset_value if isinstance(row, dict)])

        normalized_rows: List[Dict[str, object]] = []
        for asset_payload in candidate_assets:
            payload_symbol = str(asset_payload.get("symbol", base_currency)).strip().upper()
            if payload_symbol and payload_symbol != str(base_currency).upper():
                continue

            quote_rows = self._extract_quote_rows_from_asset(asset_payload)
            normalized_rows.extend(quote_rows)

        return normalized_rows

    def _extract_quote_rows_from_asset(self, asset_payload: Dict[str, object]) -> List[Dict[str, object]]:
        """Extract flat historical quote rows from one asset payload."""

        raw_quote_rows: Any = asset_payload.get("quotes")
        if raw_quote_rows is None:
            raw_quote_rows = asset_payload.get("quote")
        if raw_quote_rows is None:
            raw_quote_rows = asset_payload.get("ohlcv")
        if raw_quote_rows is None:
            raw_quote_rows = asset_payload.get("data")

        if isinstance(raw_quote_rows, dict) and self.quote_currency in raw_quote_rows:
            raw_quote_rows = [raw_quote_rows]
        elif isinstance(raw_quote_rows, dict):
            raw_quote_rows = [raw_quote_rows]

        if not isinstance(raw_quote_rows, list):
            return []

        normalized_rows: List[Dict[str, object]] = []
        for raw_quote_row in raw_quote_rows:
            if not isinstance(raw_quote_row, dict):
                continue

            quote_details = raw_quote_row.get("quote", raw_quote_row)
            if isinstance(quote_details, dict) and self.quote_currency in quote_details:
                quote_details = quote_details[self.quote_currency]

            if not isinstance(quote_details, dict):
                continue

            timestamp_value = (
                raw_quote_row.get("time_close")
                or raw_quote_row.get("timestamp")
                or raw_quote_row.get("time_open")
                or quote_details.get("timestamp")
            )
            parsed_timestamp = pd.to_datetime(timestamp_value, errors="coerce", utc=True)
            if pd.isna(parsed_timestamp):
                continue

            if any(
                quote_details.get(field_name) is None
                for field_name in ("open", "high", "low", "close", "volume")
            ):
                continue

            normalized_rows.append(
                {
                    "timestamp": parsed_timestamp,
                    "open": quote_details["open"],
                    "high": quote_details["high"],
                    "low": quote_details["low"],
                    "close": quote_details["close"],
                    "volume": quote_details["volume"],
                }
            )

        return normalized_rows

    def _normalize_candle_rows(
        self,
        product_id: str,
        base_currency: str,
        quote_currency: str,
        candle_rows: Sequence[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        """Convert flat CoinMarketCap quote rows into the project schema."""

        normalized_rows = []

        for candle_row in candle_rows:
            normalized_rows.append(
                {
                    "timestamp": candle_row["timestamp"],
                    "open": candle_row["open"],
                    "high": candle_row["high"],
                    "low": candle_row["low"],
                    "close": candle_row["close"],
                    "volume": candle_row["volume"],
                    "product_id": product_id,
                    "base_currency": base_currency,
                    "quote_currency": quote_currency,
                    "granularity_seconds": self.granularity_seconds,
                    "source": self.api_name,
                }
            )

        return normalized_rows

    def _build_price_frame(self, normalized_rows: Sequence[Dict[str, object]]) -> pd.DataFrame:
        """Normalize the full multi-asset candle set into the project's schema."""

        price_df = pd.DataFrame(normalized_rows)
        price_df = price_df.drop_duplicates(subset=["product_id", "timestamp"])
        return price_df.sort_values(["timestamp", "product_id"]).reset_index(drop=True)

    def _resolve_interval_alias(self) -> str:
        """Map project granularity settings into CoinMarketCap interval aliases."""

        if self.granularity_seconds == 3600:
            return "hourly"

        return "daily"

    def _request_json(
        self,
        endpoint_path: str,
        query_params: Dict[str, object],
    ) -> Dict[str, object]:
        """Send one CoinMarketCap request and decode the JSON payload."""

        return _request_coinmarketcap_json(
            api_base_url=self.api_base_url,
            endpoint_path=endpoint_path,
            query_params=query_params,
            api_key=self.api_key,
            request_pause_seconds=self.request_pause_seconds,
        )

    def _log_progress(self, message: str) -> None:
        """Print refresh progress when verbose mode is enabled."""

        if self.log_progress:
            print(message)


class CoinMarketCapLatestQuotesPriceDataLoader(BaseApiPriceDataLoader):
    """
    Build a local time series from supported CoinMarketCap latest-quote snapshots.

    Some CoinMarketCap plans expose `quotes/latest` and `map` but not the
    historical OHLCV endpoints. This loader records the current quote snapshot
    on every refresh, appends it to the existing market CSV, and keeps the rest
    of the pipeline working on an accumulated price history.
    """

    api_name = "coinmarketcapLatestQuotes"
    default_quote_request_symbol_batch_size = 100

    def __init__(
        self,
        data_path: Path,
        api_base_url: str,
        api_key_env_var: str,
        quote_currency: str,
        granularity_seconds: int,
        total_candles: int,
        request_pause_seconds: float = 0.2,
        should_save_downloaded_data: bool = True,
        product_id: Optional[str] = None,
        product_ids: Sequence[str] = None,
        fetch_all_quote_products: bool = False,
        excluded_base_currencies: Sequence[str] = None,
        max_products: Optional[int] = None,
        product_batch_size: Optional[int] = None,
        product_batch_number: int = 1,
        save_progress_every_products: int = 0,
        log_progress: bool = True,
        quotes_latest_endpoint: str = "/v2/cryptocurrency/quotes/latest",
        map_endpoint: str = "/v1/cryptocurrency/map",
    ) -> None:
        super().__init__(data_path=data_path, should_save_downloaded_data=should_save_downloaded_data)
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key_env_var = api_key_env_var
        self.quote_currency = quote_currency
        self.granularity_seconds = granularity_seconds
        self.total_candles = total_candles
        self.request_pause_seconds = request_pause_seconds
        self.product_id = product_id
        self.product_ids = tuple(product_ids or ())
        self.fetch_all_quote_products = fetch_all_quote_products
        self.excluded_base_currencies = tuple(excluded_base_currencies or ())
        self.max_products = max_products
        self.product_batch_size = product_batch_size
        self.product_batch_number = product_batch_number
        self.save_progress_every_products = save_progress_every_products
        self.log_progress = log_progress
        self.quotes_latest_endpoint = quotes_latest_endpoint
        self.map_endpoint = map_endpoint

    @property
    def api_key(self) -> str:
        """Read the API key lazily from the configured environment variable."""

        return os.getenv(self.api_key_env_var, "")

    def _read_data(self) -> pd.DataFrame:
        """Download one latest-quote snapshot and merge it with any existing local history."""

        self._validate_download_settings()

        available_products = self._resolve_products_to_download()
        selected_products, batch_summary = self._slice_products_for_batch(available_products)
        self._log_progress(
            "Starting CoinMarketCap latest-quote refresh: "
            f"{len(selected_products)} products in this batch."
        )

        snapshot_rows = self._request_latest_quote_rows(selected_products)
        if not snapshot_rows:
            raise ValueError(
                "The CoinMarketCap latest-quotes endpoint returned no rows for the selected products."
            )

        snapshot_df = self._build_price_frame(snapshot_rows)
        final_df = self._merge_with_existing_history(snapshot_df)

        self.last_refresh_summary = {
            "totalAvailableProducts": len(available_products),
            "productsDownloaded": len(selected_products),
            "requestWindowsPerProduct": 1,
            "batchSize": batch_summary["batchSize"],
            "batchNumber": batch_summary["batchNumber"],
            "batchStartIndex": batch_summary["batchStartIndex"],
            "batchEndIndex": batch_summary["batchEndIndex"],
            "snapshotRowsDownloaded": len(snapshot_df),
            "rowsDownloaded": len(final_df),
            "partialSavePath": str(self.partial_data_path),
        }
        self._log_progress(
            f"Completed CoinMarketCap latest-quote refresh with {len(snapshot_df)} fresh rows "
            f"and {len(final_df)} total rows in the prepared dataset."
        )
        return final_df

    def _save_downloaded_data(self, price_df: pd.DataFrame) -> None:
        """Persist the merged snapshot history produced by `_read_data`."""

        super()._save_downloaded_data(price_df)
        self.last_refresh_summary["finalRowsSaved"] = len(price_df)

    def _validate_download_settings(self) -> None:
        """Validate the snapshot-refresh settings before hitting the API."""

        if not self.api_key:
            raise ValueError(
                "CoinMarketCap latest-quote refresh requires an API key in the "
                f"`{self.api_key_env_var}` environment variable."
            )

        if self.granularity_seconds <= 0:
            raise ValueError("coinmarketcap_granularity_seconds must be greater than zero.")

        if self.total_candles <= 0:
            raise ValueError("coinmarketcap_total_candles must be greater than zero.")

        if self.product_batch_size is not None and self.product_batch_size <= 0:
            raise ValueError("coinmarketcap_product_batch_size must be greater than zero when provided.")

        if self.product_batch_number <= 0:
            raise ValueError("coinmarketcap_product_batch_number must be greater than zero.")

    def _resolve_products_to_download(self) -> List[Dict[str, str]]:
        """Decide which assets belong in the CoinMarketCap snapshot universe."""

        if self.fetch_all_quote_products:
            return self._fetch_filtered_products()

        explicit_product_ids = list(self.product_ids)
        if self.product_id:
            explicit_product_ids.append(self.product_id)

        if not explicit_product_ids:
            raise ValueError("No CoinMarketCap products were provided for download.")

        return [
            {
                "product_id": product_id,
                "base_currency": product_id.split("-")[0].upper(),
                "quote_currency": product_id.split("-")[1].upper() if "-" in product_id else self.quote_currency.upper(),
            }
            for product_id in explicit_product_ids
        ]

    def _fetch_filtered_products(self) -> List[Dict[str, str]]:
        """Fetch the CoinMarketCap asset catalog and keep the configured quote universe."""

        response_payload = self._request_json(
            endpoint_path=self.map_endpoint,
            query_params={
                "listing_status": "active",
            },
        )
        raw_rows = response_payload.get("data", []) if isinstance(response_payload, dict) else []
        if not isinstance(raw_rows, list):
            raise ValueError(
                "Unexpected response received from CoinMarketCap map endpoint. "
                f"Response type: {type(raw_rows)}"
            )

        excluded_bases = {currency.upper() for currency in self.excluded_base_currencies}
        best_row_by_symbol: Dict[str, Dict[str, object]] = {}

        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                continue

            base_currency = normalize_base_currency(raw_row.get("symbol"))
            if not base_currency:
                continue
            if base_currency in excluded_bases:
                continue
            if not is_signal_eligible_base_currency(base_currency):
                continue

            current_row = best_row_by_symbol.get(base_currency)
            if current_row is None or int(raw_row.get("rank") or 10**9) < int(current_row.get("rank") or 10**9):
                best_row_by_symbol[base_currency] = raw_row

        filtered_products = [
            {
                "product_id": f"{normalize_base_currency(raw_row.get('symbol'))}-{self.quote_currency.upper()}",
                "base_currency": normalize_base_currency(raw_row.get("symbol")),
                "quote_currency": self.quote_currency.upper(),
            }
            for raw_row in sorted(
                best_row_by_symbol.values(),
                key=lambda row: (
                    int(row.get("rank") or 10**9),
                    normalize_base_currency(row.get("symbol")),
                ),
            )
        ]

        if self.max_products is not None:
            filtered_products = filtered_products[: self.max_products]

        if not filtered_products:
            raise ValueError("No CoinMarketCap assets matched the configured multi-coin filters.")

        return filtered_products

    def get_available_products(self) -> List[Dict[str, str]]:
        """Return the filtered asset universe for the current loader settings."""

        self._validate_download_settings()
        return self._resolve_products_to_download()

    def get_total_batches(self) -> int:
        """Return how many product batches are needed for the current universe."""

        available_products = self.get_available_products()
        if self.product_batch_size is None:
            return 1

        return max(1, math.ceil(len(available_products) / self.product_batch_size))

    def _slice_products_for_batch(
        self,
        selected_products: List[Dict[str, str]],
    ) -> tuple[List[Dict[str, str]], Dict[str, int]]:
        """Limit a large asset universe to one configured batch."""

        if self.product_batch_size is None:
            batch_summary = {
                "batchSize": len(selected_products),
                "batchNumber": 1,
                "batchStartIndex": 1,
                "batchEndIndex": len(selected_products),
            }
            return selected_products, batch_summary

        total_products = len(selected_products)
        total_batches = max(1, math.ceil(total_products / self.product_batch_size))

        if self.product_batch_number > total_batches:
            raise ValueError(
                "Requested CoinMarketCap product batch is out of range. "
                f"Requested batch {self.product_batch_number}, available batches {total_batches}."
            )

        batch_start_index = (self.product_batch_number - 1) * self.product_batch_size
        batch_end_index = min(batch_start_index + self.product_batch_size, total_products)
        batch_products = selected_products[batch_start_index:batch_end_index]

        batch_summary = {
            "batchSize": len(batch_products),
            "batchNumber": self.product_batch_number,
            "batchStartIndex": batch_start_index + 1,
            "batchEndIndex": batch_end_index,
        }

        self._log_progress(
            "Using CoinMarketCap latest-quote product batch "
            f"{self.product_batch_number}/{total_batches}: "
            f"products {batch_summary['batchStartIndex']} to {batch_summary['batchEndIndex']}."
        )
        return batch_products, batch_summary

    def _request_latest_quote_rows(
        self,
        selected_products: Sequence[Dict[str, str]],
    ) -> List[Dict[str, object]]:
        """Fetch and normalize the current quote snapshot for the requested symbols."""

        if not selected_products:
            return []

        product_lookup = {
            str(product["base_currency"]).upper(): product
            for product in selected_products
        }
        symbol_batches = [
            list(product_lookup.keys())[start_index : start_index + self.default_quote_request_symbol_batch_size]
            for start_index in range(0, len(product_lookup), self.default_quote_request_symbol_batch_size)
        ]

        normalized_rows: List[Dict[str, object]] = []
        for batch_index, symbol_batch in enumerate(symbol_batches):
            response_payload = self._request_json(
                endpoint_path=self.quotes_latest_endpoint,
                query_params={
                    "symbol": ",".join(symbol_batch),
                    "convert": self.quote_currency,
                },
            )
            normalized_rows.extend(
                self._extract_latest_quote_rows(
                    response_payload=response_payload,
                    product_lookup=product_lookup,
                )
            )

            if batch_index < len(symbol_batches) - 1 and self.request_pause_seconds > 0:
                time.sleep(self.request_pause_seconds)

        return normalized_rows

    def _extract_latest_quote_rows(
        self,
        response_payload: Dict[str, object],
        product_lookup: Dict[str, Dict[str, str]],
    ) -> List[Dict[str, object]]:
        """Flatten one latest-quote payload into normalized project rows."""

        raw_data = response_payload.get("data", {}) if isinstance(response_payload, dict) else {}
        if not isinstance(raw_data, dict):
            raise ValueError(
                "Unexpected response received from CoinMarketCap latest quotes endpoint. "
                f"Response type: {type(raw_data)}"
            )

        normalized_rows: List[Dict[str, object]] = []
        for asset_rows in raw_data.values():
            if isinstance(asset_rows, dict):
                candidate_assets = [asset_rows]
            elif isinstance(asset_rows, list):
                candidate_assets = [asset_row for asset_row in asset_rows if isinstance(asset_row, dict)]
            else:
                continue

            for asset_payload in candidate_assets:
                base_currency = str(asset_payload.get("symbol", "")).strip().upper()
                product_details = product_lookup.get(base_currency)
                if product_details is None:
                    continue

                quote_payload = asset_payload.get("quote", {})
                if not isinstance(quote_payload, dict):
                    continue

                quote_details = quote_payload.get(self.quote_currency)
                if not isinstance(quote_details, dict):
                    continue

                timestamp_value = quote_details.get("last_updated") or response_payload.get("status", {}).get("timestamp")
                parsed_timestamp = pd.to_datetime(timestamp_value, errors="coerce", utc=True)
                if pd.isna(parsed_timestamp):
                    continue

                latest_price = quote_details.get("price")
                if latest_price is None:
                    continue

                price_value = float(latest_price)
                normalized_rows.append(
                    {
                        "timestamp": parsed_timestamp,
                        "open": price_value,
                        "high": price_value,
                        "low": price_value,
                        "close": price_value,
                        "volume": float(quote_details.get("volume_24h") or 0.0),
                        "product_id": product_details["product_id"],
                        "base_currency": product_details["base_currency"],
                        "quote_currency": product_details["quote_currency"],
                        "granularity_seconds": self.granularity_seconds,
                        "source": self.api_name,
                        "cmc_market_cap": float(quote_details.get("market_cap") or 0.0),
                        "cmc_volume_change_24h": float(quote_details.get("volume_change_24h") or 0.0),
                        "cmc_percent_change_1h": float(quote_details.get("percent_change_1h") or 0.0),
                        "cmc_percent_change_24h": float(quote_details.get("percent_change_24h") or 0.0),
                        "cmc_percent_change_7d": float(quote_details.get("percent_change_7d") or 0.0),
                        "cmc_percent_change_30d": float(quote_details.get("percent_change_30d") or 0.0),
                        "cmc_fully_diluted_market_cap": float(
                            quote_details.get("fully_diluted_market_cap") or 0.0
                        ),
                        "cmc_circulating_supply": float(asset_payload.get("circulating_supply") or 0.0),
                        "cmc_total_supply": float(asset_payload.get("total_supply") or 0.0),
                        "cmc_max_supply": float(asset_payload.get("max_supply") or 0.0),
                    }
                )

        return normalized_rows

    def _merge_with_existing_history(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        """Append the newest quote snapshot to any existing local market history."""

        final_df = snapshot_df.copy()
        final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors="coerce", utc=True)
        existing_row_count = 0

        if self.data_path.exists():
            existing_df = pd.read_csv(self.data_path)
            existing_row_count = len(existing_df)
            if "timestamp" in existing_df.columns:
                existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], errors="coerce", utc=True)
            final_df = pd.concat([existing_df, final_df], ignore_index=True)
            final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors="coerce", utc=True)
            final_df = final_df.drop_duplicates(subset=["timestamp", "product_id"], keep="last")

        if "base_currency" in final_df.columns:
            final_df = final_df.loc[
                final_df["base_currency"].map(is_signal_eligible_base_currency)
            ].copy()
        final_df = self._sort_rows(final_df)

        self.last_refresh_summary["existingRowsMerged"] = existing_row_count
        return final_df

    def _build_price_frame(self, normalized_rows: Sequence[Dict[str, object]]) -> pd.DataFrame:
        """Normalize the full multi-asset snapshot set into the project's schema."""

        price_df = pd.DataFrame(normalized_rows)
        price_df = price_df.drop_duplicates(subset=["product_id", "timestamp"])
        return price_df.sort_values(["timestamp", "product_id"]).reset_index(drop=True)

    def _request_json(
        self,
        endpoint_path: str,
        query_params: Dict[str, object],
    ) -> Dict[str, object]:
        """Send one CoinMarketCap request and decode the JSON payload."""

        return _request_coinmarketcap_json(
            api_base_url=self.api_base_url,
            endpoint_path=endpoint_path,
            query_params=query_params,
            api_key=self.api_key,
            request_pause_seconds=self.request_pause_seconds,
        )

    def _log_progress(self, message: str) -> None:
        """Print refresh progress when verbose mode is enabled."""

        if self.log_progress:
            print(message)


class CoinbaseExchangePriceDataLoader(BaseApiPriceDataLoader):
    """
    Download public candle data from the Coinbase Exchange REST API.

    This loader can work in two modes:
    - single product mode, for example `BTC-USD`
    - multi-product universe mode, for example all USD-quoted assets

    The universe mode is what lets the project study many coins together
    instead of hardcoding a single BTC series.
    """

    api_base_url = "https://api.exchange.coinbase.com/products/{product_id}/candles"
    api_products_url = "https://api.exchange.coinbase.com/products"
    api_name = "coinbaseExchange"
    max_candles_per_request = 300
    valid_granularities = {60, 300, 900, 3600, 21600, 86400}

    def __init__(
        self,
        data_path: Path,
        granularity_seconds: int,
        total_candles: int,
        request_pause_seconds: float = 0.2,
        should_save_downloaded_data: bool = True,
        product_id: Optional[str] = None,
        product_ids: Sequence[str] = None,
        fetch_all_quote_products: bool = False,
        quote_currency: str = "USD",
        excluded_base_currencies: Sequence[str] = None,
        max_products: Optional[int] = None,
        product_batch_size: Optional[int] = None,
        product_batch_number: int = 1,
        save_progress_every_products: int = 5,
        log_progress: bool = True,
    ) -> None:
        super().__init__(data_path=data_path, should_save_downloaded_data=should_save_downloaded_data)
        self.product_id = product_id
        self.product_ids = tuple(product_ids or ())
        self.fetch_all_quote_products = fetch_all_quote_products
        self.quote_currency = quote_currency
        self.excluded_base_currencies = tuple(excluded_base_currencies or ())
        self.max_products = max_products
        self.product_batch_size = product_batch_size
        self.product_batch_number = product_batch_number
        self.granularity_seconds = granularity_seconds
        self.total_candles = total_candles
        self.request_pause_seconds = request_pause_seconds
        self.save_progress_every_products = save_progress_every_products
        self.log_progress = log_progress

    def _read_data(self) -> pd.DataFrame:
        """Download fresh candles from Coinbase and normalize them."""

        self._validate_download_settings()

        available_products = self._resolve_products_to_download()
        selected_products, batch_summary = self._slice_products_for_batch(available_products)
        request_windows = self._build_request_windows()
        all_candle_rows = []

        self._log_progress(
            "Starting Coinbase market refresh: "
            f"{len(selected_products)} products in this batch, "
            f"{len(request_windows)} candle windows per product."
        )

        for product_index, product_details in enumerate(selected_products):
            self._log_progress(
                f"[{product_index + 1}/{len(selected_products)}] "
                f"Downloading {product_details['product_id']}"
            )

            for request_index, (start_epoch, end_epoch) in enumerate(request_windows):
                self._log_progress(
                    f"  window {request_index + 1}/{len(request_windows)} "
                    f"{self._epoch_to_iso8601(start_epoch)} -> {self._epoch_to_iso8601(end_epoch)}"
                )
                candle_rows = self._request_candles(
                    product_id=product_details["product_id"],
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                )

                normalized_rows = self._normalize_candle_rows(
                    product_id=product_details["product_id"],
                    base_currency=product_details["base_currency"],
                    quote_currency=product_details["quote_currency"],
                    candle_rows=candle_rows,
                )
                all_candle_rows.extend(normalized_rows)

                if request_index < len(request_windows) - 1 and self.request_pause_seconds > 0:
                    time.sleep(self.request_pause_seconds)

            self._maybe_save_partial_progress(
                all_candle_rows=all_candle_rows,
                completed_products=product_index + 1,
            )

            if product_index < len(selected_products) - 1 and self.request_pause_seconds > 0:
                time.sleep(self.request_pause_seconds)

        if not all_candle_rows:
            raise ValueError("The Coinbase API returned no candle data for the selected products.")

        price_df = self._build_price_frame(all_candle_rows)
        self.last_refresh_summary = {
            "totalAvailableProducts": len(available_products),
            "productsDownloaded": len(selected_products),
            "requestWindowsPerProduct": len(request_windows),
            "batchSize": batch_summary["batchSize"],
            "batchNumber": batch_summary["batchNumber"],
            "batchStartIndex": batch_summary["batchStartIndex"],
            "batchEndIndex": batch_summary["batchEndIndex"],
            "partialSavePath": str(self.partial_data_path),
            "rowsDownloaded": len(price_df),
        }
        self._log_progress(
            f"Completed Coinbase refresh batch with {len(price_df)} rows "
            f"across {len(selected_products)} products."
        )
        return price_df

    def _save_downloaded_data(self, price_df: pd.DataFrame) -> None:
        """
        Save the downloaded batch, merging with the existing market file when needed.

        Batch mode exists so the project can study many coins without one very
        long refresh job. When multiple batches are downloaded across runs, we
        merge them into one combined raw-data file instead of overwriting the
        earlier batches.
        """

        final_df = price_df.copy()
        final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors="coerce", utc=True)
        existing_row_count = 0

        if self.product_batch_size is not None and self.data_path.exists():
            existing_df = pd.read_csv(self.data_path)
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], errors="coerce", utc=True)
            existing_row_count = len(existing_df)
            final_df = pd.concat([existing_df, price_df], ignore_index=True)
            final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors="coerce", utc=True)

            dedupe_columns = ["timestamp"]
            if "product_id" in final_df.columns:
                dedupe_columns.append("product_id")

            final_df = final_df.drop_duplicates(subset=dedupe_columns, keep="last")
            final_df = self._sort_rows(final_df)

        super()._save_downloaded_data(final_df)
        self.last_refresh_summary["existingRowsMerged"] = existing_row_count
        self.last_refresh_summary["finalRowsSaved"] = len(final_df)

    def _validate_download_settings(self) -> None:
        """Validate the requested market-data settings before hitting the API."""

        if self.granularity_seconds not in self.valid_granularities:
            raise ValueError(
                "coinbase_granularity_seconds must be one of "
                f"{sorted(self.valid_granularities)}."
            )

        if self.total_candles <= 0:
            raise ValueError("total_candles must be greater than zero.")

        if self.product_batch_size is not None and self.product_batch_size <= 0:
            raise ValueError("product_batch_size must be greater than zero when provided.")

        if self.product_batch_number <= 0:
            raise ValueError("product_batch_number must be greater than zero.")

        if self.save_progress_every_products < 0:
            raise ValueError("save_progress_every_products cannot be negative.")

    def _resolve_products_to_download(self) -> List[Dict[str, str]]:
        """
        Decide which products belong in the download universe.

        If `fetch_all_quote_products` is enabled, we discover all products from
        Coinbase and filter them. Otherwise we use the explicit product ids.
        """

        if self.fetch_all_quote_products:
            return self._fetch_filtered_products()

        explicit_product_ids = list(self.product_ids)
        if self.product_id:
            explicit_product_ids.append(self.product_id)

        if not explicit_product_ids:
            raise ValueError("No Coinbase products were provided for download.")

        return [
            {
                "product_id": product_id,
                "base_currency": product_id.split("-")[0],
                "quote_currency": product_id.split("-")[1] if "-" in product_id else self.quote_currency,
            }
            for product_id in explicit_product_ids
        ]

    def _fetch_filtered_products(self) -> List[Dict[str, str]]:
        """
        Fetch the Coinbase product catalog and keep only the desired universe.

        Current filter rules:
        - quote currency must match the configured quote, usually USD
        - base currency must not be in the excluded stablecoin list
        - product status must be online
        """

        request = Request(
            self.api_products_url,
            headers={
                "User-Agent": "crypto-signal-ml/0.1",
                "Accept": "application/json",
            },
        )

        with urlopen(request, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        if not isinstance(response_payload, list):
            raise ValueError(
                "Unexpected response received from Coinbase products endpoint. "
                f"Response type: {type(response_payload)}"
            )

        excluded_bases = {currency.upper() for currency in self.excluded_base_currencies}
        filtered_products = []

        for product in response_payload:
            product_id = product.get("id")
            base_currency = product.get("base_currency")
            quote_currency = product.get("quote_currency")
            status = str(product.get("status", "")).lower()

            if not product_id or not base_currency or not quote_currency:
                continue

            if quote_currency.upper() != self.quote_currency.upper():
                continue

            normalized_base_currency = normalize_base_currency(base_currency)
            if normalized_base_currency in excluded_bases:
                continue

            if not is_signal_eligible_base_currency(normalized_base_currency):
                continue

            if status != "online":
                continue

            filtered_products.append(
                {
                    "product_id": product_id,
                    "base_currency": normalized_base_currency,
                    "quote_currency": quote_currency,
                }
            )

        filtered_products = sorted(filtered_products, key=lambda product: product["product_id"])

        if self.max_products is not None:
            filtered_products = filtered_products[: self.max_products]

        if not filtered_products:
            raise ValueError("No Coinbase products matched the configured multi-coin filters.")

        return filtered_products

    def _build_request_windows(self) -> List[tuple[int, int]]:
        """
        Split a large history request into Coinbase-compatible chunks.

        Coinbase only allows up to 300 candles per request, so we request the
        newest window first and then keep stepping backwards in time.
        """

        windows = []
        remaining_candles = self.total_candles
        current_end_epoch = self._floor_epoch_to_granularity(int(time.time()))

        while remaining_candles > 0:
            candle_count_for_window = min(remaining_candles, self.max_candles_per_request)
            current_start_epoch = current_end_epoch - (candle_count_for_window * self.granularity_seconds)

            windows.append((current_start_epoch, current_end_epoch))

            current_end_epoch = current_start_epoch
            remaining_candles -= candle_count_for_window

        return windows

    def get_available_products(self) -> List[Dict[str, str]]:
        """
        Return the filtered product universe for the current loader settings.

        This is used by higher-level apps that need to understand how many
        batches are required before starting a long multi-batch refresh.
        """

        self._validate_download_settings()
        return self._resolve_products_to_download()

    def get_total_batches(self) -> int:
        """
        Return how many product batches are needed for the current universe.
        """

        available_products = self.get_available_products()
        if self.product_batch_size is None:
            return 1

        return max(1, math.ceil(len(available_products) / self.product_batch_size))

    def _slice_products_for_batch(
        self,
        selected_products: List[Dict[str, str]],
    ) -> tuple[List[Dict[str, str]], Dict[str, int]]:
        """
        Limit a large product universe to one batch.

        This keeps the downloader practical when you want to study many coins
        without waiting for the entire exchange universe every single run.
        """

        if self.product_batch_size is None:
            batch_summary = {
                "batchSize": len(selected_products),
                "batchNumber": 1,
                "batchStartIndex": 1,
                "batchEndIndex": len(selected_products),
            }
            return selected_products, batch_summary

        total_products = len(selected_products)
        total_batches = max(1, math.ceil(total_products / self.product_batch_size))

        if self.product_batch_number > total_batches:
            raise ValueError(
                "Requested product batch is out of range. "
                f"Requested batch {self.product_batch_number}, available batches {total_batches}."
            )

        batch_start_index = (self.product_batch_number - 1) * self.product_batch_size
        batch_end_index = min(batch_start_index + self.product_batch_size, total_products)
        batch_products = selected_products[batch_start_index:batch_end_index]

        batch_summary = {
            "batchSize": len(batch_products),
            "batchNumber": self.product_batch_number,
            "batchStartIndex": batch_start_index + 1,
            "batchEndIndex": batch_end_index,
        }

        self._log_progress(
            "Using Coinbase product batch "
            f"{self.product_batch_number}/{total_batches}: "
            f"products {batch_summary['batchStartIndex']} to {batch_summary['batchEndIndex']}."
        )

        return batch_products, batch_summary

    def _maybe_save_partial_progress(
        self,
        all_candle_rows: Sequence[Dict[str, object]],
        completed_products: int,
    ) -> None:
        """
        Save a partial CSV snapshot during a long-running market refresh.

        That way, if the process is interrupted, you still keep a checkpoint
        of the completed products instead of losing all download progress.
        """

        if not self.should_save_downloaded_data:
            return

        if self.save_progress_every_products == 0:
            return

        if completed_products % self.save_progress_every_products != 0:
            return

        partial_df = self._build_price_frame(all_candle_rows)
        self._save_partial_download(partial_df)
        self._log_progress(
            f"Saved partial market data after {completed_products} products "
            f"to {self.partial_data_path}"
        )

    def _request_candles(
        self,
        product_id: str,
        start_epoch: int,
        end_epoch: int,
    ) -> Sequence[Sequence[float]]:
        """
        Send one Coinbase candles request and return the raw response rows.

        Coinbase returns candle rows in the form:
        [time, low, high, open, close, volume]
        """

        query_params = urlencode(
            {
                "start": self._epoch_to_iso8601(start_epoch),
                "end": self._epoch_to_iso8601(end_epoch),
                "granularity": self.granularity_seconds,
            }
        )

        request_url = self.api_base_url.format(product_id=product_id) + f"?{query_params}"
        request = Request(
            request_url,
            headers={
                "User-Agent": "crypto-signal-ml/0.1",
                "Accept": "application/json",
            },
        )

        with urlopen(request, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        if not isinstance(response_payload, list):
            raise ValueError(
                "Unexpected response received from Coinbase candles endpoint. "
                f"Response type: {type(response_payload)}"
            )

        return response_payload

    def _normalize_candle_rows(
        self,
        product_id: str,
        base_currency: str,
        quote_currency: str,
        candle_rows: Sequence[Sequence[float]],
    ) -> List[Dict[str, object]]:
        """
        Convert raw Coinbase candle arrays into named dictionaries.
        """

        normalized_rows = []

        for candle_row in candle_rows:
            if len(candle_row) < 6:
                continue

            timestamp_epoch, low_price, high_price, open_price, close_price, volume = candle_row[:6]
            normalized_rows.append(
                {
                    "timestamp": datetime.fromtimestamp(float(timestamp_epoch), tz=timezone.utc),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "product_id": product_id,
                    "base_currency": base_currency,
                    "quote_currency": quote_currency,
                    "granularity_seconds": self.granularity_seconds,
                    "source": self.api_name,
                }
            )

        return normalized_rows

    def _build_price_frame(self, normalized_rows: Sequence[Dict[str, object]]) -> pd.DataFrame:
        """
        Normalize the full multi-product candle set into the project's schema.
        """

        price_df = pd.DataFrame(normalized_rows)
        price_df = price_df.drop_duplicates(subset=["product_id", "timestamp"])

        sort_columns = ["timestamp", "product_id"]
        return price_df.sort_values(sort_columns).reset_index(drop=True)

    def _floor_epoch_to_granularity(self, epoch_seconds: int) -> int:
        """Align the current time to the nearest finished candle boundary."""

        return epoch_seconds - (epoch_seconds % self.granularity_seconds)

    def _epoch_to_iso8601(self, epoch_seconds: int) -> str:
        """Convert UNIX time to the ISO-8601 format expected by the API."""

        return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    def _log_progress(self, message: str) -> None:
        """Print refresh progress when verbose mode is enabled."""

        if self.log_progress:
            print(message)


class KrakenOhlcPriceDataLoader(BaseApiPriceDataLoader):
    """Download recent OHLCV candles from Kraken's public REST API."""

    api_name = "kraken"
    api_base_url = "https://api.kraken.com/0/public/OHLC"
    valid_granularities = {60: 1, 300: 5, 900: 15, 1800: 30, 3600: 60, 14400: 240, 86400: 1440}

    def __init__(
        self,
        data_path: Path,
        product_id: Optional[str] = None,
        product_ids: Sequence[str] = None,
        quote_currency: str = "USD",
        granularity_seconds: int = 3600,
        total_candles: int = 720,
        request_pause_seconds: float = 0.2,
        should_save_downloaded_data: bool = True,
        product_batch_size: Optional[int] = None,
        product_batch_number: int = 1,
        save_progress_every_products: int = 5,
        log_progress: bool = True,
    ) -> None:
        super().__init__(data_path=data_path, should_save_downloaded_data=should_save_downloaded_data)
        self.product_id = product_id
        self.product_ids = tuple(product_ids or ())
        self.quote_currency = quote_currency
        self.granularity_seconds = granularity_seconds
        self.total_candles = total_candles
        self.request_pause_seconds = request_pause_seconds
        self.product_batch_size = product_batch_size
        self.product_batch_number = product_batch_number
        self.save_progress_every_products = save_progress_every_products
        self.log_progress = log_progress

    def _read_data(self) -> pd.DataFrame:
        """Download and normalize Kraken OHLC rows."""

        self._validate_download_settings()
        available_products = self._resolve_products_to_download()
        selected_products, batch_summary = self._slice_products_for_batch(available_products)
        all_candle_rows: List[Dict[str, object]] = []

        self._log_progress(
            "Starting Kraken market refresh: "
            f"{len(selected_products)} products, up to {self.total_candles} candles each."
        )
        for product_index, product_details in enumerate(selected_products):
            self._log_progress(f"[{product_index + 1}/{len(selected_products)}] Downloading {product_details['product_id']}")
            raw_rows = self._request_ohlc_rows(product_details["exchange_pair"])
            normalized_rows = self._normalize_candle_rows(product_details, raw_rows[-self.total_candles :])
            all_candle_rows.extend(normalized_rows)
            self._maybe_save_partial_progress(all_candle_rows, product_index + 1)
            if product_index < len(selected_products) - 1 and self.request_pause_seconds > 0:
                time.sleep(self.request_pause_seconds)

        if not all_candle_rows:
            raise ValueError("The Kraken API returned no candle data for the selected products.")

        price_df = self._build_price_frame(all_candle_rows)
        self.last_refresh_summary = {
            "totalAvailableProducts": len(available_products),
            "productsDownloaded": len(selected_products),
            "requestWindowsPerProduct": 1,
            "batchSize": batch_summary["batchSize"],
            "batchNumber": batch_summary["batchNumber"],
            "batchStartIndex": batch_summary["batchStartIndex"],
            "batchEndIndex": batch_summary["batchEndIndex"],
            "partialSavePath": str(self.partial_data_path),
            "rowsDownloaded": len(price_df),
        }
        return price_df

    def _save_downloaded_data(self, price_df: pd.DataFrame) -> None:
        final_df = self._merge_existing_rows(price_df)
        super()._save_downloaded_data(final_df)
        self.last_refresh_summary["finalRowsSaved"] = len(final_df)

    def _validate_download_settings(self) -> None:
        if self.granularity_seconds not in self.valid_granularities:
            raise ValueError(f"kraken_granularity_seconds must be one of {sorted(self.valid_granularities)}.")
        if self.total_candles <= 0:
            raise ValueError("kraken_total_candles must be greater than zero.")
        if self.total_candles > 720:
            self._log_progress("Kraken REST OHLC returns at most 720 recent committed candles per product.")
        if self.product_batch_size is not None and self.product_batch_size <= 0:
            raise ValueError("kraken_product_batch_size must be greater than zero when provided.")
        if self.product_batch_number <= 0:
            raise ValueError("kraken_product_batch_number must be greater than zero.")

    def _resolve_products_to_download(self) -> List[Dict[str, str]]:
        explicit_product_ids = list(self.product_ids)
        if self.product_id:
            explicit_product_ids.append(self.product_id)
        if not explicit_product_ids:
            raise ValueError("No Kraken products were provided for download.")

        products = []
        seen_ids = set()
        for product_id in explicit_product_ids:
            base_currency, quote_currency = self._split_product_id(product_id)
            normalized_product_id = f"{base_currency}-{quote_currency}"
            if normalized_product_id in seen_ids:
                continue
            seen_ids.add(normalized_product_id)
            products.append(
                {
                    "product_id": normalized_product_id,
                    "base_currency": base_currency,
                    "quote_currency": quote_currency,
                    "exchange_pair": self._to_kraken_pair(base_currency, quote_currency),
                }
            )
        return products

    def get_available_products(self) -> List[Dict[str, str]]:
        self._validate_download_settings()
        return self._resolve_products_to_download()

    def get_total_batches(self) -> int:
        available_products = self.get_available_products()
        if self.product_batch_size is None:
            return 1
        return max(1, math.ceil(len(available_products) / self.product_batch_size))

    def _slice_products_for_batch(
        self,
        selected_products: List[Dict[str, str]],
    ) -> tuple[List[Dict[str, str]], Dict[str, int]]:
        if self.product_batch_size is None:
            return selected_products, {
                "batchSize": len(selected_products),
                "batchNumber": 1,
                "batchStartIndex": 1,
                "batchEndIndex": len(selected_products),
            }

        total_products = len(selected_products)
        total_batches = max(1, math.ceil(total_products / self.product_batch_size))
        if self.product_batch_number > total_batches:
            raise ValueError(
                "Requested Kraken product batch is out of range. "
                f"Requested batch {self.product_batch_number}, available batches {total_batches}."
            )
        batch_start_index = (self.product_batch_number - 1) * self.product_batch_size
        batch_end_index = min(batch_start_index + self.product_batch_size, total_products)
        return selected_products[batch_start_index:batch_end_index], {
            "batchSize": batch_end_index - batch_start_index,
            "batchNumber": self.product_batch_number,
            "batchStartIndex": batch_start_index + 1,
            "batchEndIndex": batch_end_index,
        }

    def _request_ohlc_rows(self, exchange_pair: str) -> List[Sequence[object]]:
        query_params = urlencode(
            {
                "pair": exchange_pair,
                "interval": self.valid_granularities[self.granularity_seconds],
            }
        )
        request = Request(
            f"{self.api_base_url}?{query_params}",
            headers={"User-Agent": "crypto-signal-ml/0.1", "Accept": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        errors = response_payload.get("error", []) if isinstance(response_payload, dict) else []
        if errors:
            raise ValueError(f"Kraken OHLC request failed for {exchange_pair}: {errors}")

        result_payload = response_payload.get("result", {}) if isinstance(response_payload, dict) else {}
        result_keys = [key for key in result_payload.keys() if key != "last"]
        raw_rows = result_payload.get(result_keys[0], []) if result_keys else []
        if not isinstance(raw_rows, list):
            raise ValueError(f"Unexpected response received from Kraken OHLC endpoint for {exchange_pair}.")

        # Kraken includes the current not-yet-committed candle as the final row.
        return raw_rows[:-1] if len(raw_rows) > 1 else raw_rows

    def _normalize_candle_rows(
        self,
        product_details: Dict[str, str],
        candle_rows: Sequence[Sequence[object]],
    ) -> List[Dict[str, object]]:
        normalized_rows: List[Dict[str, object]] = []
        for candle_row in candle_rows:
            if len(candle_row) < 7:
                continue
            timestamp_epoch, open_price, high_price, low_price, close_price, _vwap, volume = candle_row[:7]
            normalized_rows.append(
                {
                    "timestamp": datetime.fromtimestamp(float(timestamp_epoch), tz=timezone.utc),
                    "open": float(open_price),
                    "high": float(high_price),
                    "low": float(low_price),
                    "close": float(close_price),
                    "volume": float(volume),
                    "product_id": product_details["product_id"],
                    "base_currency": product_details["base_currency"],
                    "quote_currency": product_details["quote_currency"],
                    "granularity_seconds": self.granularity_seconds,
                    "source": self.api_name,
                }
            )
        return normalized_rows

    def _maybe_save_partial_progress(self, all_candle_rows: Sequence[Dict[str, object]], completed_products: int) -> None:
        if not self.should_save_downloaded_data or self.save_progress_every_products == 0:
            return
        if completed_products % self.save_progress_every_products != 0:
            return
        self._save_partial_download(self._build_price_frame(all_candle_rows))

    def _merge_existing_rows(self, price_df: pd.DataFrame) -> pd.DataFrame:
        final_df = price_df.copy()
        final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors="coerce", utc=True)
        if self.data_path.exists():
            existing_df = pd.read_csv(self.data_path)
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], errors="coerce", utc=True)
            final_df = pd.concat([existing_df, final_df], ignore_index=True)
            final_df = final_df.drop_duplicates(subset=["timestamp", "product_id"], keep="last")
        return self._sort_rows(final_df)

    def _build_price_frame(self, normalized_rows: Sequence[Dict[str, object]]) -> pd.DataFrame:
        price_df = pd.DataFrame(normalized_rows)
        price_df = price_df.drop_duplicates(subset=["product_id", "timestamp"])
        return price_df.sort_values(["timestamp", "product_id"]).reset_index(drop=True)

    def _split_product_id(self, product_id: str) -> tuple[str, str]:
        normalized_product_id = str(product_id).strip().upper()
        if "-" in normalized_product_id:
            base_currency, quote_currency = normalized_product_id.split("-", 1)
        else:
            base_currency, quote_currency = normalized_product_id, self.quote_currency
        return normalize_base_currency(base_currency), quote_currency.upper()

    @staticmethod
    def _to_kraken_pair(base_currency: str, quote_currency: str) -> str:
        kraken_base = "XBT" if base_currency.upper() == "BTC" else base_currency.upper()
        return f"{kraken_base}{quote_currency.upper()}"

    def _log_progress(self, message: str) -> None:
        if self.log_progress:
            print(message)


class BinancePublicDataPriceDataLoader(BaseApiPriceDataLoader):
    """Download OHLCV candles from Binance's public historical data archives."""

    api_name = "binancePublicData"
    api_base_url = "https://data.binance.vision/data/spot/monthly/klines"
    api_exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"
    api_ticker_24hr_url = "https://api.binance.com/api/v3/ticker/24hr"

    def __init__(
        self,
        data_path: Path,
        product_id: Optional[str] = None,
        product_ids: Sequence[str] = None,
        fetch_all_quote_products: bool = False,
        quote_currency: str = "USDT",
        excluded_base_currencies: Sequence[str] = None,
        max_products: Optional[int] = None,
        interval: str = "1h",
        granularity_seconds: int = 3600,
        total_candles: int = 4320,
        archive_lookback_months: int = 36,
        request_pause_seconds: float = 0.1,
        should_save_downloaded_data: bool = True,
        product_batch_size: Optional[int] = None,
        product_batch_number: int = 1,
        save_progress_every_products: int = 5,
        log_progress: bool = True,
    ) -> None:
        super().__init__(data_path=data_path, should_save_downloaded_data=should_save_downloaded_data)
        self.product_id = product_id
        self.product_ids = tuple(product_ids or ())
        self.fetch_all_quote_products = fetch_all_quote_products
        self.quote_currency = quote_currency
        self.excluded_base_currencies = tuple(excluded_base_currencies or ())
        self.max_products = max_products
        self.interval = interval
        self.granularity_seconds = granularity_seconds
        self.total_candles = total_candles
        self.archive_lookback_months = archive_lookback_months
        self.request_pause_seconds = request_pause_seconds
        self.product_batch_size = product_batch_size
        self.product_batch_number = product_batch_number
        self.save_progress_every_products = save_progress_every_products
        self.log_progress = log_progress

    def _read_data(self) -> pd.DataFrame:
        self._validate_download_settings()
        available_products = self._resolve_products_to_download()
        selected_products, batch_summary = self._slice_products_for_batch(available_products)
        year_months = self._resolve_months_to_download()
        all_candle_rows: List[Dict[str, object]] = []

        self._log_progress(
            "Starting Binance public-data refresh: "
            f"{len(selected_products)} products, {len(year_months)} monthly archive(s) each."
        )
        for product_index, product_details in enumerate(selected_products):
            product_rows: List[Dict[str, object]] = []
            for year_month in year_months:
                product_rows.extend(self._request_month_rows(product_details, year_month))
                if len(product_rows) >= self.total_candles:
                    break
                if self.request_pause_seconds > 0:
                    time.sleep(self.request_pause_seconds)
            product_rows = sorted(product_rows, key=lambda row: row["timestamp"])[-self.total_candles :]
            all_candle_rows.extend(product_rows)
            self._maybe_save_partial_progress(all_candle_rows, product_index + 1)

        if not all_candle_rows:
            raise ValueError("Binance public data returned no candle rows for the selected products.")

        price_df = self._build_price_frame(all_candle_rows)
        self.last_refresh_summary = {
            "totalAvailableProducts": len(available_products),
            "productsDownloaded": len(selected_products),
            "requestWindowsPerProduct": len(year_months),
            "batchSize": batch_summary["batchSize"],
            "batchNumber": batch_summary["batchNumber"],
            "batchStartIndex": batch_summary["batchStartIndex"],
            "batchEndIndex": batch_summary["batchEndIndex"],
            "partialSavePath": str(self.partial_data_path),
            "rowsDownloaded": len(price_df),
        }
        return price_df

    def _save_downloaded_data(self, price_df: pd.DataFrame) -> None:
        final_df = self._merge_existing_rows(price_df)
        super()._save_downloaded_data(final_df)
        self.last_refresh_summary["finalRowsSaved"] = len(final_df)

    def _validate_download_settings(self) -> None:
        if self.total_candles <= 0:
            raise ValueError("binance_total_candles must be greater than zero.")
        if self.granularity_seconds <= 0:
            raise ValueError("binance_granularity_seconds must be greater than zero.")
        if self.archive_lookback_months <= 0:
            raise ValueError("binance_archive_lookback_months must be greater than zero.")
        if not self.interval:
            raise ValueError("binance_interval must not be empty.")
        if self.product_batch_size is not None and self.product_batch_size <= 0:
            raise ValueError("binance_product_batch_size must be greater than zero when provided.")
        if self.product_batch_number <= 0:
            raise ValueError("binance_product_batch_number must be greater than zero.")

    def _resolve_products_to_download(self) -> List[Dict[str, str]]:
        if self.fetch_all_quote_products:
            return self._fetch_filtered_products()

        explicit_product_ids = list(self.product_ids)
        if self.product_id:
            explicit_product_ids.append(self.product_id)
        if not explicit_product_ids:
            raise ValueError("No Binance products were provided for download.")

        products = []
        seen_ids = set()
        for product_id in explicit_product_ids:
            base_currency, _quote_currency = self._split_product_id(product_id)
            quote_currency = self.quote_currency.upper()
            normalized_product_id = f"{base_currency}-{quote_currency}"
            if normalized_product_id in seen_ids:
                continue
            seen_ids.add(normalized_product_id)
            products.append(
                {
                    "product_id": normalized_product_id,
                    "base_currency": base_currency,
                    "quote_currency": quote_currency,
                    "exchange_symbol": f"{base_currency}{quote_currency}",
                }
            )
        return products

    def _fetch_filtered_products(self) -> List[Dict[str, str]]:
        """Fetch Binance spot symbols and keep eligible quote-currency products."""

        query_params = urlencode({"symbolStatus": "TRADING"})
        request = Request(
            f"{self.api_exchange_info_url}?{query_params}",
            headers={"User-Agent": "crypto-signal-ml/0.1", "Accept": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        raw_symbols = response_payload.get("symbols", []) if isinstance(response_payload, dict) else []
        if not isinstance(raw_symbols, list):
            raise ValueError(
                "Unexpected response received from Binance exchangeInfo endpoint. "
                f"Response type: {type(raw_symbols)}"
            )

        excluded_bases = {currency.upper() for currency in self.excluded_base_currencies}
        filtered_products: List[Dict[str, str]] = []

        for symbol_details in raw_symbols:
            if not isinstance(symbol_details, dict):
                continue

            base_currency = normalize_base_currency(symbol_details.get("baseAsset"))
            quote_currency = str(symbol_details.get("quoteAsset", "")).strip().upper()
            exchange_symbol = str(symbol_details.get("symbol", "")).strip().upper()
            status = str(symbol_details.get("status", "")).strip().upper()
            is_spot_allowed = bool(symbol_details.get("isSpotTradingAllowed", True))

            if not base_currency or not quote_currency or not exchange_symbol:
                continue
            if quote_currency != self.quote_currency.upper():
                continue
            if base_currency in excluded_bases:
                continue
            if not is_signal_eligible_base_currency(base_currency):
                continue
            if status != "TRADING" or not is_spot_allowed:
                continue

            filtered_products.append(
                {
                    "product_id": f"{base_currency}-{quote_currency}",
                    "base_currency": base_currency,
                    "quote_currency": quote_currency,
                    "exchange_symbol": exchange_symbol,
                }
            )

        quote_volume_by_symbol = self._fetch_quote_volume_by_symbol()
        filtered_products = sorted(
            filtered_products,
            key=lambda product: (
                -quote_volume_by_symbol.get(product["exchange_symbol"], 0.0),
                product["exchange_symbol"],
            ),
        )

        if self.max_products is not None:
            filtered_products = filtered_products[: self.max_products]

        if not filtered_products:
            raise ValueError("No Binance products matched the configured multi-coin filters.")

        return filtered_products

    def _fetch_quote_volume_by_symbol(self) -> Dict[str, float]:
        """Fetch Binance 24h quote volumes for liquidity-first universe ordering."""

        request = Request(
            self.api_ticker_24hr_url,
            headers={"User-Agent": "crypto-signal-ml/0.1", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return {}

        if not isinstance(response_payload, list):
            return {}

        quote_volume_by_symbol: Dict[str, float] = {}
        for ticker_row in response_payload:
            if not isinstance(ticker_row, dict):
                continue
            symbol = str(ticker_row.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            try:
                quote_volume_by_symbol[symbol] = float(ticker_row.get("quoteVolume") or 0.0)
            except (TypeError, ValueError):
                quote_volume_by_symbol[symbol] = 0.0
        return quote_volume_by_symbol

    def get_available_products(self) -> List[Dict[str, str]]:
        self._validate_download_settings()
        return self._resolve_products_to_download()

    def get_total_batches(self) -> int:
        available_products = self.get_available_products()
        if self.product_batch_size is None:
            return 1
        return max(1, math.ceil(len(available_products) / self.product_batch_size))

    def _slice_products_for_batch(
        self,
        selected_products: List[Dict[str, str]],
    ) -> tuple[List[Dict[str, str]], Dict[str, int]]:
        if self.product_batch_size is None:
            return selected_products, {
                "batchSize": len(selected_products),
                "batchNumber": 1,
                "batchStartIndex": 1,
                "batchEndIndex": len(selected_products),
            }
        total_products = len(selected_products)
        total_batches = max(1, math.ceil(total_products / self.product_batch_size))
        if self.product_batch_number > total_batches:
            raise ValueError(
                "Requested Binance product batch is out of range. "
                f"Requested batch {self.product_batch_number}, available batches {total_batches}."
            )
        batch_start_index = (self.product_batch_number - 1) * self.product_batch_size
        batch_end_index = min(batch_start_index + self.product_batch_size, total_products)
        return selected_products[batch_start_index:batch_end_index], {
            "batchSize": batch_end_index - batch_start_index,
            "batchNumber": self.product_batch_number,
            "batchStartIndex": batch_start_index + 1,
            "batchEndIndex": batch_end_index,
        }

    def _resolve_months_to_download(self) -> List[str]:
        candles_per_31_day_month = max(1, int((31 * 24 * 3600) / self.granularity_seconds))
        month_count = max(
            1,
            min(
                int(self.archive_lookback_months),
                math.ceil(self.total_candles / candles_per_31_day_month) + 18,
            ),
        )
        current_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        months = []
        cursor = current_month
        for _ in range(month_count):
            months.append(cursor.strftime("%Y-%m"))
            cursor = (cursor - timedelta(days=1)).replace(day=1)
        return months

    def _request_month_rows(self, product_details: Dict[str, str], year_month: str) -> List[Dict[str, object]]:
        exchange_symbol = product_details["exchange_symbol"]
        request_url = (
            f"{self.api_base_url}/{exchange_symbol}/{self.interval}/"
            f"{exchange_symbol}-{self.interval}-{year_month}.zip"
        )
        request = Request(request_url, headers={"User-Agent": "crypto-signal-ml/0.1"})
        try:
            with urlopen(request, timeout=60) as response:
                zip_payload = response.read()
        except HTTPError as error:
            if error.code == 404:
                self._log_progress(f"Missing Binance archive: {exchange_symbol} {self.interval} {year_month}")
                return []
            raise

        with ZipFile(io.BytesIO(zip_payload)) as archive:
            csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
            if not csv_names:
                return []
            with archive.open(csv_names[0]) as csv_file:
                decoded_file = io.TextIOWrapper(csv_file, encoding="utf-8")
                normalized_rows = []
                for row in csv.reader(decoded_file):
                    if len(row) < 6 or row[0].lower().startswith("open"):
                        continue
                    normalized_row = self._normalize_kline_row(product_details, row)
                    if normalized_row is not None:
                        normalized_rows.append(normalized_row)
                return normalized_rows

    def _normalize_kline_row(self, product_details: Dict[str, str], row: Sequence[str]) -> Dict[str, object] | None:
        open_time_ms, open_price, high_price, low_price, close_price, volume = row[:6]
        try:
            timestamp_epoch = self._normalize_binance_timestamp(float(open_time_ms))
            open_value = float(open_price)
            high_value = float(high_price)
            low_value = float(low_price)
            close_value = float(close_price)
            volume_value = float(volume)
        except (TypeError, ValueError, OSError):
            return None

        return {
            "timestamp": datetime.fromtimestamp(timestamp_epoch, tz=timezone.utc),
            "open": open_value,
            "high": high_value,
            "low": low_value,
            "close": close_value,
            "volume": volume_value,
            "product_id": product_details["product_id"],
            "base_currency": product_details["base_currency"],
            "quote_currency": product_details["quote_currency"],
            "granularity_seconds": self.granularity_seconds,
            "source": self.api_name,
        }

    @staticmethod
    def _normalize_binance_timestamp(raw_timestamp: float) -> float:
        """Normalize Binance timestamps that may arrive as seconds, ms, us, or ns."""

        if raw_timestamp >= 1e17:
            return raw_timestamp / 1_000_000_000.0
        if raw_timestamp >= 1e14:
            return raw_timestamp / 1_000_000.0
        if raw_timestamp >= 1e11:
            return raw_timestamp / 1_000.0
        return raw_timestamp

    def _maybe_save_partial_progress(self, all_candle_rows: Sequence[Dict[str, object]], completed_products: int) -> None:
        if not self.should_save_downloaded_data or self.save_progress_every_products == 0:
            return
        if completed_products % self.save_progress_every_products != 0:
            return
        self._save_partial_download(self._build_price_frame(all_candle_rows))

    def _merge_existing_rows(self, price_df: pd.DataFrame) -> pd.DataFrame:
        final_df = price_df.copy()
        final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors="coerce", utc=True)
        if self.data_path.exists():
            existing_df = pd.read_csv(self.data_path)
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], errors="coerce", utc=True)
            final_df = pd.concat([existing_df, final_df], ignore_index=True)
            final_df = final_df.drop_duplicates(subset=["timestamp", "product_id"], keep="last")
        return self._sort_rows(final_df)

    def _build_price_frame(self, normalized_rows: Sequence[Dict[str, object]]) -> pd.DataFrame:
        price_df = pd.DataFrame(normalized_rows)
        price_df = price_df.drop_duplicates(subset=["product_id", "timestamp"])
        return price_df.sort_values(["timestamp", "product_id"]).reset_index(drop=True)

    def _split_product_id(self, product_id: str) -> tuple[str, str]:
        normalized_product_id = str(product_id).strip().upper()
        if "-" in normalized_product_id:
            base_currency, quote_currency = normalized_product_id.split("-", 1)
        else:
            base_currency, quote_currency = normalized_product_id, self.quote_currency
        return normalize_base_currency(base_currency), quote_currency.upper()

    def _log_progress(self, message: str) -> None:
        if self.log_progress:
            print(message)


def create_market_data_loader(
    config: TrainingConfig,
    data_path: Path | None = None,
    should_save_downloaded_data: bool = True,
    product_id: Optional[str] = None,
    product_ids: Sequence[str] | None = None,
    fetch_all_quote_products: Optional[bool] = None,
    max_products: Optional[int] = None,
    granularity_seconds: Optional[int] = None,
    total_candles: Optional[int] = None,
    request_pause_seconds: Optional[float] = None,
    product_batch_size: Optional[int] = None,
    product_batch_number: Optional[int] = None,
    save_progress_every_products: Optional[int] = None,
    log_progress: Optional[bool] = None,
) -> BaseApiPriceDataLoader:
    """Build the configured market-data loader for the requested source."""

    market_data_source = str(config.market_data_source).strip()

    if market_data_source == "coinmarketcap":
        return CoinMarketCapOhlcvPriceDataLoader(
            data_path=data_path or config.data_file,
            api_base_url=config.coinmarketcap_api_base_url,
            api_key_env_var=config.coinmarketcap_api_key_env_var,
            quote_currency=config.coinmarketcap_quote_currency,
            granularity_seconds=int(
                granularity_seconds
                if granularity_seconds is not None
                else config.coinmarketcap_granularity_seconds
            ),
            total_candles=int(
                total_candles
                if total_candles is not None
                else config.coinmarketcap_total_candles
            ),
            request_pause_seconds=(
                request_pause_seconds
                if request_pause_seconds is not None
                else config.coinmarketcap_request_pause_seconds
            ),
            should_save_downloaded_data=should_save_downloaded_data,
            product_id=product_id if product_id is not None else config.coinmarketcap_product_id,
            product_ids=(
                tuple(product_ids)
                if product_ids is not None
                else config.coinmarketcap_product_ids
            ),
            fetch_all_quote_products=(
                fetch_all_quote_products
                if fetch_all_quote_products is not None
                else config.coinmarketcap_fetch_all_quote_products
            ),
            excluded_base_currencies=config.coinmarketcap_excluded_base_currencies,
            max_products=max_products if max_products is not None else config.coinmarketcap_max_products,
            product_batch_size=(
                product_batch_size
                if product_batch_size is not None
                else config.coinmarketcap_product_batch_size
            ),
            product_batch_number=(
                int(product_batch_number)
                if product_batch_number is not None
                else int(config.coinmarketcap_product_batch_number)
            ),
            save_progress_every_products=(
                int(save_progress_every_products)
                if save_progress_every_products is not None
                else int(config.coinmarketcap_save_progress_every_products)
            ),
            log_progress=log_progress if log_progress is not None else config.coinmarketcap_log_progress,
            historical_endpoint=config.coinmarketcap_ohlcv_historical_endpoint,
            map_endpoint=config.coinmarketcap_map_endpoint,
        )

    if market_data_source == "coinmarketcapLatestQuotes":
        return CoinMarketCapLatestQuotesPriceDataLoader(
            data_path=data_path or config.data_file,
            api_base_url=config.coinmarketcap_api_base_url,
            api_key_env_var=config.coinmarketcap_api_key_env_var,
            quote_currency=config.coinmarketcap_quote_currency,
            granularity_seconds=int(
                granularity_seconds
                if granularity_seconds is not None
                else config.coinmarketcap_granularity_seconds
            ),
            total_candles=int(
                total_candles
                if total_candles is not None
                else config.coinmarketcap_total_candles
            ),
            request_pause_seconds=(
                request_pause_seconds
                if request_pause_seconds is not None
                else config.coinmarketcap_request_pause_seconds
            ),
            should_save_downloaded_data=should_save_downloaded_data,
            product_id=product_id if product_id is not None else config.coinmarketcap_product_id,
            product_ids=(
                tuple(product_ids)
                if product_ids is not None
                else config.coinmarketcap_product_ids
            ),
            fetch_all_quote_products=(
                fetch_all_quote_products
                if fetch_all_quote_products is not None
                else config.coinmarketcap_fetch_all_quote_products
            ),
            excluded_base_currencies=config.coinmarketcap_excluded_base_currencies,
            max_products=max_products if max_products is not None else config.coinmarketcap_max_products,
            product_batch_size=(
                product_batch_size
                if product_batch_size is not None
                else config.coinmarketcap_product_batch_size
            ),
            product_batch_number=(
                int(product_batch_number)
                if product_batch_number is not None
                else int(config.coinmarketcap_product_batch_number)
            ),
            save_progress_every_products=(
                int(save_progress_every_products)
                if save_progress_every_products is not None
                else int(config.coinmarketcap_save_progress_every_products)
            ),
            log_progress=log_progress if log_progress is not None else config.coinmarketcap_log_progress,
            quotes_latest_endpoint=config.coinmarketcap_quotes_latest_endpoint,
            map_endpoint=config.coinmarketcap_map_endpoint,
        )

    if market_data_source == "coinbaseExchange":
        return CoinbaseExchangePriceDataLoader(
            data_path=data_path or config.data_file,
            product_id=product_id if product_id is not None else config.coinbase_product_id,
            product_ids=tuple(product_ids) if product_ids is not None else config.coinbase_product_ids,
            fetch_all_quote_products=(
                fetch_all_quote_products
                if fetch_all_quote_products is not None
                else config.coinbase_fetch_all_quote_products
            ),
            quote_currency=config.coinbase_quote_currency,
            excluded_base_currencies=config.coinbase_excluded_base_currencies,
            max_products=max_products if max_products is not None else config.coinbase_max_products,
            product_batch_size=(
                product_batch_size
                if product_batch_size is not None
                else config.coinbase_product_batch_size
            ),
            product_batch_number=(
                int(product_batch_number)
                if product_batch_number is not None
                else int(config.coinbase_product_batch_number)
            ),
            granularity_seconds=int(
                granularity_seconds
                if granularity_seconds is not None
                else config.coinbase_granularity_seconds
            ),
            total_candles=int(
                total_candles
                if total_candles is not None
                else config.coinbase_total_candles
            ),
            request_pause_seconds=(
                request_pause_seconds
                if request_pause_seconds is not None
                else config.coinbase_request_pause_seconds
            ),
            should_save_downloaded_data=should_save_downloaded_data,
            save_progress_every_products=(
                int(save_progress_every_products)
                if save_progress_every_products is not None
                else int(config.coinbase_save_progress_every_products)
            ),
            log_progress=log_progress if log_progress is not None else config.coinbase_log_progress,
        )

    if market_data_source == "kraken":
        return KrakenOhlcPriceDataLoader(
            data_path=data_path or config.data_file,
            product_id=product_id if product_id is not None else config.kraken_product_id,
            product_ids=tuple(product_ids) if product_ids is not None else config.kraken_product_ids,
            quote_currency=config.kraken_quote_currency,
            product_batch_size=(
                product_batch_size
                if product_batch_size is not None
                else config.kraken_product_batch_size
            ),
            product_batch_number=(
                int(product_batch_number)
                if product_batch_number is not None
                else int(config.kraken_product_batch_number)
            ),
            granularity_seconds=int(
                granularity_seconds
                if granularity_seconds is not None
                else config.kraken_granularity_seconds
            ),
            total_candles=int(
                total_candles
                if total_candles is not None
                else config.kraken_total_candles
            ),
            request_pause_seconds=(
                request_pause_seconds
                if request_pause_seconds is not None
                else config.kraken_request_pause_seconds
            ),
            should_save_downloaded_data=should_save_downloaded_data,
            save_progress_every_products=(
                int(save_progress_every_products)
                if save_progress_every_products is not None
                else int(config.kraken_save_progress_every_products)
            ),
            log_progress=log_progress if log_progress is not None else config.kraken_log_progress,
        )

    if market_data_source == "binancePublicData":
        return BinancePublicDataPriceDataLoader(
            data_path=data_path or config.data_file,
            product_id=product_id if product_id is not None else config.binance_product_id,
            product_ids=tuple(product_ids) if product_ids is not None else config.binance_product_ids,
            fetch_all_quote_products=(
                fetch_all_quote_products
                if fetch_all_quote_products is not None
                else config.binance_fetch_all_quote_products
            ),
            quote_currency=config.binance_quote_currency,
            excluded_base_currencies=config.binance_excluded_base_currencies,
            max_products=max_products if max_products is not None else config.binance_max_products,
            interval=config.binance_interval,
            product_batch_size=(
                product_batch_size
                if product_batch_size is not None
                else config.binance_product_batch_size
            ),
            product_batch_number=(
                int(product_batch_number)
                if product_batch_number is not None
                else int(config.binance_product_batch_number)
            ),
            granularity_seconds=int(
                granularity_seconds
                if granularity_seconds is not None
                else config.binance_granularity_seconds
            ),
            total_candles=int(
                total_candles
                if total_candles is not None
                else config.binance_total_candles
            ),
            archive_lookback_months=int(config.binance_archive_lookback_months),
            request_pause_seconds=(
                request_pause_seconds
                if request_pause_seconds is not None
                else config.binance_request_pause_seconds
            ),
            should_save_downloaded_data=should_save_downloaded_data,
            save_progress_every_products=(
                int(save_progress_every_products)
                if save_progress_every_products is not None
                else int(config.binance_save_progress_every_products)
            ),
            log_progress=log_progress if log_progress is not None else config.binance_log_progress,
        )

    raise ValueError(
        "Unsupported market_data_source. "
        "Currently supported: coinmarketcap, coinmarketcapLatestQuotes, coinbaseExchange, kraken, binancePublicData"
    )


def _normalize_timeframe_alias(timeframe: str) -> str:
    """Normalize a timeframe alias into a compact lowercase key such as `4h` or `1d`."""

    normalized_value = str(timeframe).strip().lower()
    if not normalized_value:
        raise ValueError("Timeframe aliases must not be empty.")

    pandas_offset_input = normalized_value
    if normalized_value.endswith("d"):
        pandas_offset_input = normalized_value[:-1] + "D"

    pandas_offset = pd.tseries.frequencies.to_offset(pandas_offset_input)
    if pandas_offset.n <= 0:
        raise ValueError(f"Unsupported timeframe alias: {timeframe}")

    rule_code = str(pandas_offset.name).lower()
    if rule_code.endswith("min"):
        return f"{pandas_offset.n}min"

    if rule_code.endswith("h"):
        return f"{pandas_offset.n}h"

    if rule_code.endswith("d"):
        return f"{pandas_offset.n}d"

    raise ValueError(
        "Unsupported timeframe alias. Supported families are minutes, hours, and days. "
        f"Received: {timeframe}"
    )


def _timeframe_alias_to_pandas_rule(timeframe: str) -> str:
    """Convert a compact timeframe alias into a pandas resample rule."""

    normalized_alias = _normalize_timeframe_alias(timeframe)
    if normalized_alias.endswith("d"):
        return normalized_alias[:-1] + "D"

    return normalized_alias


def _calculate_higher_timeframe_features(resampled_df: pd.DataFrame) -> pd.DataFrame:
    """Build a small higher-timeframe feature set before aligning it back to base rows."""

    feature_df = resampled_df.copy()
    feature_df["return_1"] = feature_df["close"].pct_change(1)
    feature_df["range_pct"] = (feature_df["high"] - feature_df["low"]) / feature_df["close"].replace(0, pd.NA)
    feature_df["volume_change_1"] = feature_df["volume"].pct_change(1)
    feature_df["sma_3"] = feature_df["close"].rolling(window=3).mean()
    feature_df["ema_3"] = feature_df["close"].ewm(span=3, adjust=False).mean()
    feature_df["close_vs_sma_3"] = (feature_df["close"] / feature_df["sma_3"]) - 1
    feature_df["close_vs_ema_3"] = (feature_df["close"] / feature_df["ema_3"]) - 1
    feature_df["volatility_3"] = feature_df["return_1"].rolling(window=3).std()
    return feature_df


def align_multi_timeframe_context(
    price_df: pd.DataFrame,
    timeframes: Sequence[str],
    base_granularity_seconds: int,
) -> pd.DataFrame:
    """
    Align completed higher-timeframe bars back onto the base candle table.

    The alignment uses candle close timestamps, not raw start timestamps, so a
    base row only sees information from higher-timeframe candles that would
    already be complete at that moment.
    """

    if not timeframes:
        return price_df.copy()

    if "timestamp" not in price_df.columns:
        return price_df.copy()

    aligned_df = price_df.copy()
    aligned_df["timestamp"] = pd.to_datetime(aligned_df["timestamp"], errors="coerce", utc=True)
    aligned_df = aligned_df.dropna(subset=["timestamp"]).reset_index(drop=True)

    product_key_column = "product_id"
    synthetic_product_key = "__GLOBAL__"
    if product_key_column not in aligned_df.columns:
        product_key_column = "__multi_timeframe_product_id"
        aligned_df[product_key_column] = synthetic_product_key

    base_granularity_seconds = max(int(base_granularity_seconds), 60)
    base_delta = pd.to_timedelta(base_granularity_seconds, unit="s")
    aligned_df["_base_close_timestamp"] = aligned_df["timestamp"] + base_delta
    aligned_df = aligned_df.sort_values([product_key_column, "_base_close_timestamp"]).reset_index(drop=True)

    aligned_output_df = aligned_df.copy()

    for timeframe in timeframes:
        normalized_alias = _normalize_timeframe_alias(timeframe)
        resample_rule = _timeframe_alias_to_pandas_rule(normalized_alias)
        aligned_frames: List[pd.DataFrame] = []

        for product_id, asset_df in aligned_df.groupby(product_key_column, sort=False):
            asset_base_df = asset_df.sort_values("_base_close_timestamp").reset_index(drop=True)
            resampled_df = (
                asset_base_df
                .resample(
                    resample_rule,
                    on="_base_close_timestamp",
                    label="right",
                    closed="right",
                )
                .agg(
                    open=("open", "first"),
                    high=("high", "max"),
                    low=("low", "min"),
                    close=("close", "last"),
                    volume=("volume", "sum"),
                )
                .dropna(subset=["open", "high", "low", "close", "volume"])
                .reset_index()
            )

            if resampled_df.empty:
                aligned_frames.append(asset_base_df.copy())
                continue

            resampled_feature_df = _calculate_higher_timeframe_features(resampled_df)
            rename_map = {
                "_base_close_timestamp": f"htf_{normalized_alias}_close_timestamp",
                "close": f"htf_{normalized_alias}_close",
                "return_1": f"htf_{normalized_alias}_return_1",
                "range_pct": f"htf_{normalized_alias}_range_pct",
                "volume_change_1": f"htf_{normalized_alias}_volume_change_1",
                "close_vs_sma_3": f"htf_{normalized_alias}_close_vs_sma_3",
                "close_vs_ema_3": f"htf_{normalized_alias}_close_vs_ema_3",
                "volatility_3": f"htf_{normalized_alias}_volatility_3",
            }
            resampled_feature_df = resampled_feature_df.rename(columns=rename_map)
            resampled_feature_df[product_key_column] = product_id

            aligned_asset_df = pd.merge_asof(
                asset_base_df,
                resampled_feature_df[
                    [
                        product_key_column,
                        f"htf_{normalized_alias}_close_timestamp",
                        f"htf_{normalized_alias}_close",
                        f"htf_{normalized_alias}_return_1",
                        f"htf_{normalized_alias}_range_pct",
                        f"htf_{normalized_alias}_volume_change_1",
                        f"htf_{normalized_alias}_close_vs_sma_3",
                        f"htf_{normalized_alias}_close_vs_ema_3",
                        f"htf_{normalized_alias}_volatility_3",
                    ]
                ],
                left_on="_base_close_timestamp",
                right_on=f"htf_{normalized_alias}_close_timestamp",
                by=product_key_column,
                direction="backward",
                allow_exact_matches=True,
            )
            aligned_frames.append(aligned_asset_df)

        aligned_output_df = pd.concat(aligned_frames, ignore_index=True)
        aligned_output_df = aligned_output_df.sort_values(
            [product_key_column, "_base_close_timestamp"]
        ).reset_index(drop=True)

    if "__multi_timeframe_product_id" in aligned_output_df.columns:
        aligned_output_df = aligned_output_df.drop(columns=["__multi_timeframe_product_id"])

    return aligned_output_df.drop(columns=["_base_close_timestamp"])


def load_price_data(csv_path: Path) -> pd.DataFrame:
    """
    Backward-compatible helper that delegates to the CSV loader class.
    """

    return CsvPriceDataLoader(csv_path).load()
