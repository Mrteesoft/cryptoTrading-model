"""Persistent storage for current live signals and their publication history."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
from uuid import uuid4

from ..storage.database import DatabaseConnection, DatabaseHandle


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


class TradingSignalStore:
    """Persist the latest published signals plus a rolling history in SQLite or PostgreSQL."""

    def __init__(
        self,
        db_path: Path,
        database_url: str | None = None,
    ) -> None:
        self.database = DatabaseHandle(
            sqlite_path=db_path,
            database_url=database_url,
        )
        self.db_path = self.database.sqlite_path
        self._initialize_schema()

    def replace_current_signals(
        self,
        signal_summaries: Sequence[Mapping[str, Any]],
        primary_signal: Mapping[str, Any] | None = None,
        generated_at: str | None = None,
    ) -> Dict[str, Any]:
        """Replace the current published signal set and append the new rows to history."""

        generation_id = str(uuid4())
        generated_at_value = str(generated_at or _utc_now_iso())
        persisted_at = _utc_now_iso()
        primary_identity = self._signal_identity(primary_signal) if primary_signal is not None else None

        normalized_rows = [
            self._normalize_signal_row(
                signal_summary=signal_summary,
                generation_id=generation_id,
                signal_rank=signal_rank,
                generated_at=generated_at_value,
                persisted_at=persisted_at,
                is_primary=(
                    self._signal_identity(signal_summary) == primary_identity
                    if primary_identity is not None
                    else signal_rank == 1
                ),
            )
            for signal_rank, signal_summary in enumerate(signal_summaries, start=1)
        ]

        with self._connect() as connection:
            connection.execute("DELETE FROM signal_current")
            for normalized_row in normalized_rows:
                params = self._row_to_sql_params(normalized_row)
                connection.execute(
                    """
                    INSERT INTO signal_current (
                        product_id,
                        generation_id,
                        signal_rank,
                        is_primary,
                        signal_name,
                        spot_action,
                        actionable,
                        confidence,
                        price,
                        signal_timestamp,
                        generated_at,
                        persisted_at,
                        signal_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params,
                )
                connection.execute(
                    """
                    INSERT INTO signal_history (
                        product_id,
                        generation_id,
                        signal_rank,
                        is_primary,
                        signal_name,
                        spot_action,
                        actionable,
                        confidence,
                        price,
                        signal_timestamp,
                        generated_at,
                        persisted_at,
                        signal_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params,
                )

        primary_product_id = ""
        if primary_signal is not None:
            primary_product_id = self._normalize_product_id(primary_signal.get("productId"))
        if not primary_product_id and normalized_rows:
            primary_product_id = str(normalized_rows[0]["product_id"])

        actionable_count = sum(int(row["actionable"]) for row in normalized_rows)
        return {
            "generationId": generation_id,
            "generatedAt": generated_at_value,
            "persistedAt": persisted_at,
            "signalCount": len(normalized_rows),
            "actionableCount": actionable_count,
            "primaryProductId": primary_product_id or None,
            "storageBackend": self.database.storage_backend,
            "databaseTarget": self.database.database_target,
        }

    def get_current_signal(self) -> Optional[Dict[str, Any]]:
        """Return the primary current signal when one is stored."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    product_id,
                    generation_id,
                    signal_rank,
                    is_primary,
                    signal_name,
                    spot_action,
                    actionable,
                    confidence,
                    price,
                    signal_timestamp,
                    generated_at,
                    persisted_at,
                    signal_json
                FROM signal_current
                ORDER BY is_primary DESC, signal_rank ASC, product_id ASC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return None

        return self._row_to_signal_dict(row)

    def list_current_signals(
        self,
        limit: int = 50,
    ) -> list[Dict[str, Any]]:
        """Return the currently published live signals."""

        query_limit = max(int(limit), 1)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    product_id,
                    generation_id,
                    signal_rank,
                    is_primary,
                    signal_name,
                    spot_action,
                    actionable,
                    confidence,
                    price,
                    signal_timestamp,
                    generated_at,
                    persisted_at,
                    signal_json
                FROM signal_current
                ORDER BY is_primary DESC, signal_rank ASC, product_id ASC
                LIMIT ?
                """,
                (query_limit,),
            ).fetchall()

        return [self._row_to_signal_dict(row) for row in rows]

    def list_signal_history(
        self,
        limit: int = 100,
        product_id: str | None = None,
    ) -> list[Dict[str, Any]]:
        """Return the newest persisted signal rows ordered from latest to oldest."""

        query_limit = max(int(limit), 1)
        normalized_product_id = self._normalize_product_id(product_id) if product_id is not None else None
        query = """
            SELECT
                product_id,
                generation_id,
                signal_rank,
                is_primary,
                signal_name,
                spot_action,
                actionable,
                confidence,
                price,
                signal_timestamp,
                generated_at,
                persisted_at,
                signal_json
            FROM signal_history
        """
        params: list[Any] = []
        if normalized_product_id is not None:
            query += " WHERE product_id = ?"
            params.append(normalized_product_id)
        query += " ORDER BY persisted_at DESC, signal_rank ASC LIMIT ?"
        params.append(query_limit)

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()

        return [self._row_to_signal_dict(row) for row in rows]

    def get_status(self) -> Dict[str, Any]:
        """Return a compact storage summary for diagnostics and health checks."""

        current_signal = self.get_current_signal()
        current_signals = self.list_current_signals(limit=500)
        return {
            "status": "ready" if current_signal is not None else "empty",
            "currentSignalCount": len(current_signals),
            "generatedAt": current_signal.get("generatedAt") if current_signal is not None else None,
            "primaryProductId": current_signal.get("productId") if current_signal is not None else None,
            "storageBackend": self.database.storage_backend,
            "databaseTarget": self.database.database_target,
        }

    def _initialize_schema(self) -> None:
        """Create the signal tables when they do not exist yet."""

        with self._connect() as connection:
            if self.database.storage_backend == "postgresql":
                number_type = "DOUBLE PRECISION"
                history_id_sql = "BIGSERIAL PRIMARY KEY"
            else:
                number_type = "REAL"
                history_id_sql = "INTEGER PRIMARY KEY AUTOINCREMENT"

            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS signal_current (
                    product_id TEXT PRIMARY KEY,
                    generation_id TEXT NOT NULL,
                    signal_rank INTEGER NOT NULL,
                    is_primary INTEGER NOT NULL DEFAULT 0,
                    signal_name TEXT NOT NULL,
                    spot_action TEXT NOT NULL,
                    actionable INTEGER NOT NULL DEFAULT 0,
                    confidence {number_type} NOT NULL DEFAULT 0,
                    price {number_type},
                    signal_timestamp TEXT,
                    generated_at TEXT NOT NULL,
                    persisted_at TEXT NOT NULL,
                    signal_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS signal_history (
                    signal_event_id {history_id_sql},
                    product_id TEXT NOT NULL,
                    generation_id TEXT NOT NULL,
                    signal_rank INTEGER NOT NULL,
                    is_primary INTEGER NOT NULL DEFAULT 0,
                    signal_name TEXT NOT NULL,
                    spot_action TEXT NOT NULL,
                    actionable INTEGER NOT NULL DEFAULT 0,
                    confidence {number_type} NOT NULL DEFAULT 0,
                    price {number_type},
                    signal_timestamp TEXT,
                    generated_at TEXT NOT NULL,
                    persisted_at TEXT NOT NULL,
                    signal_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_signal_history_product_time ON signal_history(product_id, persisted_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_signal_history_generation_rank ON signal_history(generation_id, signal_rank)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_signal_current_primary_rank ON signal_current(is_primary, signal_rank)"
            )

    def _connect(self) -> DatabaseConnection:
        """Open one normalized database connection."""

        return self.database.connect()

    @staticmethod
    def _normalize_product_id(product_id: Any) -> str:
        """Normalize one product id for storage and lookups."""

        return str(product_id or "").strip().upper()

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Convert one numeric-looking value into a float with a defensive fallback."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _signal_identity(cls, signal_summary: Mapping[str, Any] | None) -> tuple[str, str, str]:
        """Return one stable identity tuple for selecting the primary current signal."""

        if signal_summary is None:
            return ("", "", "")

        return (
            cls._normalize_product_id(signal_summary.get("productId")),
            str(signal_summary.get("timestamp", "") or "").strip(),
            str(signal_summary.get("signal_name", "") or "").strip().upper(),
        )

    @classmethod
    def _normalize_signal_row(
        cls,
        signal_summary: Mapping[str, Any],
        generation_id: str,
        signal_rank: int,
        generated_at: str,
        persisted_at: str,
        is_primary: bool,
    ) -> Dict[str, Any]:
        """Extract the indexed fields plus the raw JSON payload for one signal row."""

        product_id = cls._normalize_product_id(signal_summary.get("productId"))
        signal_name = str(signal_summary.get("signal_name", "") or "").strip().upper() or "UNKNOWN"
        spot_action = str(signal_summary.get("spotAction", "") or "").strip().lower() or "wait"
        signal_timestamp = str(signal_summary.get("timestamp", "") or "").strip()
        signal_payload = dict(signal_summary)
        signal_payload_json = json.dumps(signal_payload, ensure_ascii=True)

        return {
            "product_id": product_id,
            "generation_id": generation_id,
            "signal_rank": int(signal_rank),
            "is_primary": 1 if is_primary else 0,
            "signal_name": signal_name,
            "spot_action": spot_action,
            "actionable": 1 if bool(signal_summary.get("actionable", False)) else 0,
            "confidence": cls._safe_float(signal_summary.get("confidence")),
            "price": cls._safe_float(signal_summary.get("close")) if signal_summary.get("close") is not None else None,
            "signal_timestamp": signal_timestamp or None,
            "generated_at": generated_at,
            "persisted_at": persisted_at,
            "signal_json": signal_payload_json,
        }

    @staticmethod
    def _row_to_sql_params(normalized_row: Mapping[str, Any]) -> tuple[Any, ...]:
        """Return the SQL insert tuple for one normalized signal row."""

        return (
            normalized_row["product_id"],
            normalized_row["generation_id"],
            normalized_row["signal_rank"],
            normalized_row["is_primary"],
            normalized_row["signal_name"],
            normalized_row["spot_action"],
            normalized_row["actionable"],
            normalized_row["confidence"],
            normalized_row["price"],
            normalized_row["signal_timestamp"],
            normalized_row["generated_at"],
            normalized_row["persisted_at"],
            normalized_row["signal_json"],
        )

    @staticmethod
    def _row_to_signal_dict(row: Mapping[str, Any]) -> Dict[str, Any]:
        """Expand one stored row back into the public signal payload shape."""

        try:
            signal_payload = json.loads(str(row["signal_json"]) or "{}")
        except json.JSONDecodeError:
            signal_payload = {}

        if not isinstance(signal_payload, dict):
            signal_payload = {}

        signal_payload["productId"] = str(signal_payload.get("productId") or row["product_id"])
        signal_payload["signal_name"] = str(signal_payload.get("signal_name") or row["signal_name"])
        signal_payload["spotAction"] = str(signal_payload.get("spotAction") or row["spot_action"])
        signal_payload["actionable"] = bool(
            signal_payload.get("actionable")
            if "actionable" in signal_payload
            else row["actionable"]
        )
        signal_payload["confidence"] = float(
            signal_payload.get("confidence")
            if signal_payload.get("confidence") is not None
            else row["confidence"]
        )
        signal_payload["close"] = (
            float(signal_payload.get("close"))
            if signal_payload.get("close") is not None
            else (float(row["price"]) if row["price"] is not None else None)
        )
        signal_payload["timestamp"] = str(signal_payload.get("timestamp") or row["signal_timestamp"] or "")
        signal_payload["generationId"] = str(row["generation_id"])
        signal_payload["signalRank"] = int(row["signal_rank"])
        signal_payload["isPrimary"] = bool(row["is_primary"])
        signal_payload["generatedAt"] = str(row["generated_at"])
        signal_payload["persistedAt"] = str(row["persisted_at"])

        return signal_payload
