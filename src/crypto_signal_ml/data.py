"""Classes and helpers for loading, validating, and downloading market data."""

from abc import ABC, abstractmethod
import math
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


REQUIRED_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


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

        map_lookup = self._fetch_symbol_map_lookup(symbol_frame["base_currency"].tolist())
        context_df = self._build_context_frame(
            symbol_frame=symbol_frame,
            map_lookup=map_lookup,
        )

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

        request_url = self.api_base_url + endpoint_path
        if query_params:
            request_url += "?" + urlencode(query_params, doseq=True)

        request = Request(
            request_url,
            headers={
                "Accept": "application/json",
                "Accepts": "application/json",
                "X-CMC_PRO_API_KEY": self.api_key,
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
            raise ValueError(message) from error
        except URLError as error:
            raise ValueError(
                "CoinMarketCap request failed "
                f"for {endpoint_path}. Reason: {error.reason}"
            ) from error

        if self.request_pause_seconds > 0:
            time.sleep(self.request_pause_seconds)

        if not isinstance(response_payload, dict):
            raise ValueError(
                "Unexpected response received from CoinMarketCap. "
                f"Response type: {type(response_payload)}"
            )

        return response_payload

    def _log_progress(self, message: str) -> None:
        """Print enrichment progress when verbose mode is enabled."""

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

            if base_currency.upper() in excluded_bases:
                continue

            if status != "online":
                continue

            filtered_products.append(
                {
                    "product_id": product_id,
                    "base_currency": base_currency,
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


def load_price_data(csv_path: Path) -> pd.DataFrame:
    """
    Backward-compatible helper that delegates to the CSV loader class.
    """

    return CsvPriceDataLoader(csv_path).load()
