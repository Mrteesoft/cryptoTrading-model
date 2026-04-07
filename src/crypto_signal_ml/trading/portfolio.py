"""Persistent portfolio state, trade tracking, execution journal, and performance accounting."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ..storage.database import DatabaseConnection, DatabaseHandle

EPSILON = 1e-12
TRACKED_TRADE_STATUSES = {"planned", "open", "closed", "cancelled"}
ACTIVE_TRACKED_TRADE_STATUSES = {"planned", "open"}


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


def _parse_iso_timestamp(timestamp_value: str | None) -> datetime | None:
    """Parse one optional ISO timestamp into a timezone-aware datetime."""

    if not timestamp_value:
        return None

    normalized_value = str(timestamp_value).replace("Z", "+00:00")
    try:
        parsed_value = datetime.fromisoformat(normalized_value)
    except ValueError:
        return None

    if parsed_value.tzinfo is None:
        return parsed_value.replace(tzinfo=timezone.utc)

    return parsed_value.astimezone(timezone.utc)


class TradingPortfolioStore:
    """Persist capital, open spot positions, and executed trades in SQLite or PostgreSQL."""

    def __init__(
        self,
        db_path: Path,
        default_capital: float = 10000.0,
        database_url: str | None = None,
    ) -> None:
        self.database = DatabaseHandle(
            sqlite_path=db_path,
            database_url=database_url,
        )
        self.db_path = self.database.sqlite_path
        self.default_capital = float(default_capital)
        self._initialize_schema()
        self._initialize_default_capital()

    def get_portfolio(self) -> Dict[str, Any]:
        """Return the current capital, open positions, and performance summary."""

        capital = self.get_capital()
        positions = self.list_positions()
        current_exposure_fraction = sum(float(position.get("positionFraction") or 0.0) for position in positions)
        performance = self.get_performance_summary(capital=capital, positions=positions)
        tracked_trade_summary = self.get_tracked_trade_summary()

        return {
            "capital": capital,
            "positions": positions,
            "positionCount": len(positions),
            "currentExposureFraction": current_exposure_fraction,
            "performance": performance,
            "trackedTrades": tracked_trade_summary,
            "storageBackend": self.database.storage_backend,
            "databaseTarget": self.database.database_target,
        }

    def get_capital(self) -> float:
        """Return the configured portfolio capital."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT setting_value FROM portfolio_settings WHERE setting_key = 'capital'",
            ).fetchone()

        if row is None:
            return float(self.default_capital)

        try:
            return float(row["setting_value"])
        except (TypeError, ValueError):
            return float(self.default_capital)

    def set_capital(self, capital: float) -> Dict[str, Any]:
        """Persist the tracked portfolio capital and return the latest portfolio view."""

        normalized_capital = float(capital)
        if normalized_capital <= 0:
            raise ValueError("Capital must be greater than zero.")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO portfolio_settings (setting_key, setting_value, updated_at)
                VALUES ('capital', ?, ?)
                ON CONFLICT(setting_key) DO UPDATE SET
                    setting_value = excluded.setting_value,
                    updated_at = excluded.updated_at
                """,
                (str(normalized_capital), _utc_now_iso()),
            )

        return self.get_portfolio()

    def list_positions(self) -> list[Dict[str, Any]]:
        """Return all open positions."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    product_id,
                    quantity,
                    entry_price,
                    current_price,
                    position_fraction,
                    opened_at,
                    updated_at,
                    metadata_json
                FROM portfolio_positions
                ORDER BY product_id ASC
                """
            ).fetchall()

        return [self._row_to_position_dict(row) for row in rows]

    def get_position(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Return one open position when it exists."""

        normalized_product_id = self._normalize_product_id(product_id)
        with self._connect() as connection:
            row = self._get_position_row(connection, normalized_product_id)

        if row is None:
            return None

        return self._row_to_position_dict(row)

    def upsert_position(
        self,
        product_id: str,
        quantity: float,
        entry_price: float | None = None,
        current_price: float | None = None,
        position_fraction: float | None = None,
        opened_at: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Create or replace one open position directly."""

        normalized_product_id = self._normalize_product_id(product_id)
        normalized_quantity = float(quantity)
        if normalized_quantity < 0:
            raise ValueError("Position quantity cannot be negative.")

        normalized_position_fraction = float(position_fraction or 0.0)
        if normalized_position_fraction < 0:
            raise ValueError("positionFraction cannot be negative.")

        normalized_entry_price = None if entry_price is None else float(entry_price)
        normalized_current_price = None if current_price is None else float(current_price)
        if normalized_entry_price is not None and normalized_entry_price <= 0:
            raise ValueError("entryPrice must be greater than zero when provided.")
        if normalized_current_price is not None and normalized_current_price <= 0:
            raise ValueError("currentPrice must be greater than zero when provided.")

        metadata_json = json.dumps(metadata or {})

        with self._connect() as connection:
            existing_row = self._get_position_row(connection, normalized_product_id)
            persisted_opened_at = opened_at or (str(existing_row["opened_at"]) if existing_row is not None else "") or _utc_now_iso()
            self._upsert_position_row(
                connection=connection,
                product_id=normalized_product_id,
                quantity=normalized_quantity,
                entry_price=normalized_entry_price,
                current_price=normalized_current_price,
                position_fraction=normalized_position_fraction,
                opened_at=persisted_opened_at,
                metadata_json=metadata_json,
            )
            updated_row = self._get_position_row(connection, normalized_product_id)

        return self._row_to_position_dict(updated_row) if updated_row is not None else {}

    def delete_position(self, product_id: str) -> bool:
        """Delete one open position."""

        normalized_product_id = self._normalize_product_id(product_id)
        with self._connect() as connection:
            deleted_count = int(
                connection.execute(
                    "DELETE FROM portfolio_positions WHERE product_id = ?",
                    (normalized_product_id,),
                ).rowcount
            )

        return deleted_count > 0

    def list_trades(
        self,
        limit: int = 100,
        status: str | None = None,
    ) -> list[Dict[str, Any]]:
        """Return tracked trade records ordered from newest to oldest."""

        query_limit = max(int(limit), 1)
        normalized_status = self._normalize_tracked_trade_status(status) if status is not None else None
        query = """
            SELECT
                trade_id,
                product_id,
                signal_name,
                status,
                quantity,
                entry_price,
                current_price,
                stop_loss_price,
                take_profit_price,
                exit_price,
                realized_pnl,
                realized_return,
                outcome,
                close_reason,
                opened_at,
                closed_at,
                created_at,
                updated_at,
                metadata_json
            FROM portfolio_tracked_trades
        """
        params: list[Any] = []
        if normalized_status is not None:
            query += " WHERE status = ?"
            params.append(normalized_status)
        query += " ORDER BY trade_id DESC LIMIT ?"
        params.append(query_limit)

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()

        return [self._row_to_tracked_trade_dict(row) for row in rows]

    def get_trade_learning_snapshot(
        self,
        product_id: str,
        signal_name: str | None = None,
        recent_limit: int = 5,
    ) -> Dict[str, Any]:
        """Summarize closed tracked-trade outcomes for one product and optional signal type."""

        normalized_product_id = self._normalize_product_id(product_id)
        normalized_signal_name = str(signal_name or "").strip().upper() or None
        query = """
            SELECT
                trade_id,
                product_id,
                signal_name,
                outcome,
                realized_return,
                opened_at,
                closed_at,
                close_reason
            FROM portfolio_tracked_trades
            WHERE product_id = ?
              AND status = 'closed'
            ORDER BY COALESCE(closed_at, updated_at, created_at) DESC, trade_id DESC
        """

        with self._connect() as connection:
            product_rows = connection.execute(query, (normalized_product_id,)).fetchall()
            scoped_rows = product_rows
            scope = "product"
            if normalized_signal_name is not None:
                scoped_rows = [
                    row
                    for row in product_rows
                    if str(row["signal_name"] or "").strip().upper() == normalized_signal_name
                ]
                if scoped_rows:
                    scope = "product+signal"
                else:
                    scoped_rows = product_rows

        closed_trade_count = len(scoped_rows)
        recent_rows = list(scoped_rows[: max(int(recent_limit), 1)])
        realized_returns = [
            float(row["realized_return"])
            for row in scoped_rows
            if row["realized_return"] is not None
        ]

        win_count = 0
        loss_count = 0
        flat_count = 0
        for row in scoped_rows:
            outcome = str(row["outcome"] or "").strip().lower()
            if outcome == "win":
                win_count += 1
            elif outcome == "loss":
                loss_count += 1
            elif outcome == "flat":
                flat_count += 1

        hold_hours: list[float] = []
        for row in scoped_rows:
            opened_at = _parse_iso_timestamp(str(row["opened_at"] or ""))
            closed_at = _parse_iso_timestamp(str(row["closed_at"] or ""))
            if opened_at is None or closed_at is None:
                continue
            hold_hours.append(max((closed_at - opened_at).total_seconds() / 3600.0, 0.0))

        recent_loss_streak = 0
        recent_outcomes = [
            str(row["outcome"] or "").strip().lower()
            for row in recent_rows
            if str(row["outcome"] or "").strip()
        ]
        for outcome in recent_outcomes:
            if outcome == "loss":
                recent_loss_streak += 1
            else:
                break

        last_closed_at = str(recent_rows[0]["closed_at"]) if recent_rows and recent_rows[0]["closed_at"] is not None else None
        last_outcome = recent_outcomes[0] if recent_outcomes else None

        return {
            "productId": normalized_product_id,
            "requestedSignalName": normalized_signal_name,
            "scope": scope,
            "available": closed_trade_count > 0,
            "closedTradeCount": int(closed_trade_count),
            "winCount": int(win_count),
            "lossCount": int(loss_count),
            "flatCount": int(flat_count),
            "winRate": (win_count / closed_trade_count) if closed_trade_count > 0 else None,
            "averageRealizedReturn": (
                sum(realized_returns) / len(realized_returns)
                if realized_returns
                else None
            ),
            "averageHoldHours": (
                sum(hold_hours) / len(hold_hours)
                if hold_hours
                else None
            ),
            "recentLossStreak": int(recent_loss_streak),
            "recentOutcomes": recent_outcomes,
            "lastOutcome": last_outcome,
            "lastClosedAt": last_closed_at,
            "sampleAdequate": closed_trade_count >= 3,
        }

    def build_trade_learning_map(
        self,
        signal_summaries: Sequence[Mapping[str, Any]],
    ) -> dict[str, Dict[str, Any]]:
        """Build one product-keyed learning snapshot map from the current signal universe."""

        learning_map: dict[str, Dict[str, Any]] = {}
        for signal_summary in signal_summaries:
            raw_product_id = str(signal_summary.get("productId") or signal_summary.get("product_id") or "").strip()
            if not raw_product_id:
                continue
            product_id = self._normalize_product_id(raw_product_id)
            if not product_id or product_id in learning_map:
                continue

            signal_name = str(signal_summary.get("signal_name") or signal_summary.get("signalName") or "").strip().upper()
            learning_map[product_id] = self.get_trade_learning_snapshot(
                product_id=product_id,
                signal_name=signal_name or None,
            )

        return learning_map

    def get_trade(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Return one tracked trade record when it exists."""

        normalized_trade_id = int(trade_id)
        if normalized_trade_id <= 0:
            raise ValueError("tradeId must be greater than zero.")

        with self._connect() as connection:
            row = self._get_tracked_trade_row(connection, normalized_trade_id)

        if row is None:
            return None

        return self._row_to_tracked_trade_dict(row)

    def get_active_trade_for_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Return the newest active tracked trade for one product when it exists."""

        normalized_product_id = self._normalize_product_id(product_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    trade_id,
                    product_id,
                    signal_name,
                    status,
                    quantity,
                    entry_price,
                    current_price,
                    stop_loss_price,
                    take_profit_price,
                    exit_price,
                    realized_pnl,
                    realized_return,
                    outcome,
                    close_reason,
                    opened_at,
                    closed_at,
                    created_at,
                    updated_at,
                    metadata_json
                FROM portfolio_tracked_trades
                WHERE product_id = ?
                  AND status IN ('planned', 'open')
                ORDER BY trade_id DESC
                LIMIT 1
                """,
                (normalized_product_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_tracked_trade_dict(row)

    def get_active_signal_product_ids(self) -> list[str]:
        """Return products that currently have an open position or active tracked trade."""

        active_product_ids: set[str] = set()

        with self._connect() as connection:
            position_rows = connection.execute(
                """
                SELECT product_id
                FROM portfolio_positions
                """
            ).fetchall()
            tracked_trade_rows = connection.execute(
                """
                SELECT product_id
                FROM portfolio_tracked_trades
                WHERE status IN ('planned', 'open')
                """
            ).fetchall()

        for row in list(position_rows) + list(tracked_trade_rows):
            product_id = str(row["product_id"] or "").strip().upper()
            if product_id:
                active_product_ids.add(product_id)

        return sorted(active_product_ids)

    def create_trade(
        self,
        product_id: str,
        entry_price: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
        quantity: float | None = None,
        current_price: float | None = None,
        signal_name: str | None = None,
        status: str = "planned",
        opened_at: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Create one tracked trade record for later review and outcome analysis."""

        normalized_product_id = self._normalize_product_id(product_id)
        normalized_entry_price = float(entry_price)
        if normalized_entry_price <= 0:
            raise ValueError("entryPrice must be greater than zero.")

        normalized_take_profit_price = (
            float(take_profit_price) if take_profit_price is not None else None
        )
        if normalized_take_profit_price is not None and normalized_take_profit_price <= normalized_entry_price:
            raise ValueError("takeProfitPrice must be greater than entryPrice for a spot long trade.")

        normalized_stop_loss_price = float(stop_loss_price) if stop_loss_price is not None else None
        if normalized_stop_loss_price is not None and normalized_stop_loss_price >= normalized_entry_price:
            raise ValueError("stopLossPrice must be lower than entryPrice for a spot long trade.")

        normalized_quantity = float(quantity) if quantity is not None else 0.0
        if normalized_quantity < 0:
            raise ValueError("quantity cannot be negative.")

        normalized_current_price = float(current_price) if current_price is not None else normalized_entry_price
        if normalized_current_price <= 0:
            raise ValueError("currentPrice must be greater than zero when provided.")

        normalized_status = self._normalize_tracked_trade_status(status, allow_terminal=False)
        normalized_signal_name = str(signal_name or "").strip().upper() or None
        opened_timestamp = opened_at or _utc_now_iso()
        metadata_json = json.dumps(metadata or {})

        with self._connect() as connection:
            row = connection.execute(
                """
                INSERT INTO portfolio_tracked_trades (
                    product_id,
                    signal_name,
                    status,
                    quantity,
                    entry_price,
                    current_price,
                    stop_loss_price,
                    take_profit_price,
                    exit_price,
                    realized_pnl,
                    realized_return,
                    outcome,
                    close_reason,
                    opened_at,
                    closed_at,
                    created_at,
                    updated_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, NULL, NULL, NULL, ?, NULL, ?, ?, ?)
                RETURNING
                    trade_id,
                    product_id,
                    signal_name,
                    status,
                    quantity,
                    entry_price,
                    current_price,
                    stop_loss_price,
                    take_profit_price,
                    exit_price,
                    realized_pnl,
                    realized_return,
                    outcome,
                    close_reason,
                    opened_at,
                    closed_at,
                    created_at,
                    updated_at,
                    metadata_json
                """,
                (
                    normalized_product_id,
                    normalized_signal_name,
                    normalized_status,
                    normalized_quantity,
                    normalized_entry_price,
                    normalized_current_price,
                    normalized_stop_loss_price,
                    normalized_take_profit_price,
                    opened_timestamp,
                    _utc_now_iso(),
                    _utc_now_iso(),
                    metadata_json,
                ),
            ).fetchone()

        return self._row_to_tracked_trade_dict(row)

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        closed_at: str | None = None,
        close_reason: str | None = None,
        current_price: float | None = None,
        outcome: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Close one tracked trade and compute whether it worked or failed."""

        normalized_trade_id = int(trade_id)
        if normalized_trade_id <= 0:
            raise ValueError("tradeId must be greater than zero.")

        normalized_exit_price = float(exit_price)
        if normalized_exit_price <= 0:
            raise ValueError("exitPrice must be greater than zero.")

        normalized_current_price = float(current_price) if current_price is not None else normalized_exit_price
        if normalized_current_price <= 0:
            raise ValueError("currentPrice must be greater than zero when provided.")

        closed_timestamp = closed_at or _utc_now_iso()

        with self._connect() as connection:
            existing_row = self._get_tracked_trade_row(connection, normalized_trade_id)
            if existing_row is None:
                raise ValueError(f"Tracked trade not found: {normalized_trade_id}")

            existing_trade = self._row_to_tracked_trade_dict(existing_row)
            existing_status = str(existing_trade.get("status") or "")
            if existing_status == "closed":
                raise ValueError(f"Tracked trade {normalized_trade_id} is already closed.")
            if existing_status == "cancelled":
                raise ValueError(f"Tracked trade {normalized_trade_id} was cancelled and cannot be closed.")

            entry_price = float(existing_trade.get("entryPrice") or 0.0)
            quantity = float(existing_trade.get("quantity") or 0.0)
            realized_return = ((normalized_exit_price / entry_price) - 1.0) if entry_price > 0 else 0.0
            realized_pnl = (quantity * (normalized_exit_price - entry_price)) if quantity > 0 else 0.0

            normalized_outcome = str(outcome or "").strip().lower()
            if not normalized_outcome:
                if realized_return > EPSILON:
                    normalized_outcome = "win"
                elif realized_return < -EPSILON:
                    normalized_outcome = "loss"
                else:
                    normalized_outcome = "flat"

            merged_metadata = dict(existing_trade.get("metadata") or {})
            merged_metadata.update(metadata or {})

            connection.execute(
                """
                UPDATE portfolio_tracked_trades
                SET
                    status = 'closed',
                    current_price = ?,
                    exit_price = ?,
                    realized_pnl = ?,
                    realized_return = ?,
                    outcome = ?,
                    close_reason = ?,
                    closed_at = ?,
                    updated_at = ?,
                    metadata_json = ?
                WHERE trade_id = ?
                """,
                (
                    normalized_current_price,
                    normalized_exit_price,
                    realized_pnl,
                    realized_return,
                    normalized_outcome,
                    str(close_reason or "").strip() or None,
                    closed_timestamp,
                    _utc_now_iso(),
                    json.dumps(merged_metadata),
                    normalized_trade_id,
                ),
            )
            updated_row = self._get_tracked_trade_row(connection, normalized_trade_id)

        if updated_row is None:
            raise ValueError(f"Tracked trade not found after update: {normalized_trade_id}")

        return self._row_to_tracked_trade_dict(updated_row)

    def refresh_trade(
        self,
        trade_id: int,
        *,
        current_price: float | None = None,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        signal_name: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Refresh one active tracked trade with the latest signal-time market snapshot."""

        normalized_trade_id = int(trade_id)
        if normalized_trade_id <= 0:
            raise ValueError("tradeId must be greater than zero.")

        normalized_current_price = float(current_price) if current_price is not None else None
        if normalized_current_price is not None and normalized_current_price <= 0:
            raise ValueError("currentPrice must be greater than zero when provided.")

        normalized_stop_loss_price = float(stop_loss_price) if stop_loss_price is not None else None
        normalized_take_profit_price = float(take_profit_price) if take_profit_price is not None else None
        normalized_signal_name = str(signal_name or "").strip().upper() or None

        with self._connect() as connection:
            existing_row = self._get_tracked_trade_row(connection, normalized_trade_id)
            if existing_row is None:
                raise ValueError(f"Tracked trade not found: {normalized_trade_id}")

            existing_trade = self._row_to_tracked_trade_dict(existing_row)
            existing_metadata = dict(existing_trade.get("metadata") or {})
            existing_metadata.update(metadata or {})
            connection.execute(
                """
                UPDATE portfolio_tracked_trades
                SET
                    signal_name = COALESCE(?, signal_name),
                    current_price = COALESCE(?, current_price),
                    stop_loss_price = COALESCE(?, stop_loss_price),
                    take_profit_price = COALESCE(?, take_profit_price),
                    updated_at = ?,
                    metadata_json = ?
                WHERE trade_id = ?
                """,
                (
                    normalized_signal_name,
                    normalized_current_price,
                    normalized_stop_loss_price,
                    normalized_take_profit_price,
                    _utc_now_iso(),
                    json.dumps(existing_metadata),
                    normalized_trade_id,
                ),
            )
            updated_row = self._get_tracked_trade_row(connection, normalized_trade_id)

        if updated_row is None:
            raise ValueError(f"Tracked trade not found after refresh: {normalized_trade_id}")

        return self._row_to_tracked_trade_dict(updated_row)

    def get_tracked_trade_summary(self) -> Dict[str, Any]:
        """Summarize how many tracked trades are open, closed, winning, and losing."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS trade_count,
                    SUM(CASE WHEN status IN ('planned', 'open') THEN 1 ELSE 0 END) AS active_trade_count,
                    SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) AS closed_trade_count,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) AS winning_trade_count,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) AS losing_trade_count,
                    SUM(CASE WHEN status = 'closed' THEN realized_pnl ELSE 0 END) AS closed_realized_pnl,
                    AVG(CASE WHEN status = 'closed' THEN realized_return ELSE NULL END) AS average_closed_return
                FROM portfolio_tracked_trades
                """
            ).fetchone()

        return {
            "tradeCount": int(row["trade_count"] or 0),
            "activeTradeCount": int(row["active_trade_count"] or 0),
            "closedTradeCount": int(row["closed_trade_count"] or 0),
            "winningTradeCount": int(row["winning_trade_count"] or 0),
            "losingTradeCount": int(row["losing_trade_count"] or 0),
            "closedRealizedPnl": float(row["closed_realized_pnl"] or 0.0),
            "averageClosedReturn": float(row["average_closed_return"] or 0.0),
        }

    def list_journal(self, limit: int = 100) -> list[Dict[str, Any]]:
        """Return the newest execution journal rows first."""

        query_limit = max(int(limit), 1)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    execution_id,
                    product_id,
                    side,
                    quantity,
                    price,
                    fee,
                    realized_pnl,
                    cash_flow,
                    position_quantity_after,
                    position_fraction_after,
                    executed_at,
                    metadata_json
                FROM portfolio_executions
                ORDER BY execution_id DESC
                LIMIT ?
                """,
                (query_limit,),
            ).fetchall()

        return [self._row_to_execution_dict(row) for row in rows]

    def record_execution(
        self,
        product_id: str,
        side: str,
        quantity: float,
        price: float,
        fee: float = 0.0,
        current_price: float | None = None,
        executed_at: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Apply one spot execution to the portfolio and append it to the journal."""

        normalized_product_id = self._normalize_product_id(product_id)
        normalized_side = str(side).strip().lower()
        if normalized_side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'.")

        normalized_quantity = float(quantity)
        normalized_price = float(price)
        normalized_fee = float(fee or 0.0)
        if normalized_quantity <= 0:
            raise ValueError("quantity must be greater than zero.")
        if normalized_price <= 0:
            raise ValueError("price must be greater than zero.")
        if normalized_fee < 0:
            raise ValueError("fee cannot be negative.")

        effective_current_price = float(current_price) if current_price is not None else normalized_price
        if effective_current_price <= 0:
            raise ValueError("currentPrice must be greater than zero when provided.")

        execution_timestamp = executed_at or _utc_now_iso()
        metadata_json = json.dumps(metadata or {})

        with self._connect() as connection:
            capital = self._get_capital_from_connection(connection)
            existing_row = self._get_position_row(connection, normalized_product_id)
            existing_position = self._row_to_position_dict(existing_row) if existing_row is not None else None
            existing_quantity = float(existing_position.get("quantity") or 0.0) if existing_position else 0.0
            existing_entry_price = float(existing_position.get("entryPrice") or 0.0) if existing_position else 0.0
            existing_opened_at = str(existing_position.get("openedAt") or "") if existing_position else ""

            realized_pnl = 0.0
            cash_flow = 0.0
            position_quantity_after = 0.0
            position_fraction_after = 0.0

            if normalized_side == "buy":
                total_cost_before = existing_quantity * existing_entry_price
                total_cost_after = total_cost_before + (normalized_quantity * normalized_price) + normalized_fee
                position_quantity_after = existing_quantity + normalized_quantity
                entry_price_after = total_cost_after / position_quantity_after
                position_fraction_after = (
                    (position_quantity_after * effective_current_price) / capital
                    if capital > 0
                    else 0.0
                )
                cash_flow = -((normalized_quantity * normalized_price) + normalized_fee)
                self._upsert_position_row(
                    connection=connection,
                    product_id=normalized_product_id,
                    quantity=position_quantity_after,
                    entry_price=entry_price_after,
                    current_price=effective_current_price,
                    position_fraction=position_fraction_after,
                    opened_at=existing_opened_at or execution_timestamp,
                    metadata_json=metadata_json,
                )
            else:
                if existing_position is None or existing_quantity <= EPSILON:
                    raise ValueError(
                        f"Cannot sell {normalized_product_id} because no open position is tracked."
                    )
                if normalized_quantity > (existing_quantity + EPSILON):
                    raise ValueError(
                        f"Cannot sell {normalized_quantity:.8f} {normalized_product_id}; tracked quantity is {existing_quantity:.8f}."
                    )

                realized_pnl = (normalized_quantity * (normalized_price - existing_entry_price)) - normalized_fee
                cash_flow = (normalized_quantity * normalized_price) - normalized_fee
                position_quantity_after = max(existing_quantity - normalized_quantity, 0.0)

                if position_quantity_after <= EPSILON:
                    position_quantity_after = 0.0
                    position_fraction_after = 0.0
                    connection.execute(
                        "DELETE FROM portfolio_positions WHERE product_id = ?",
                        (normalized_product_id,),
                    )
                else:
                    position_fraction_after = (
                        (position_quantity_after * effective_current_price) / capital
                        if capital > 0
                        else 0.0
                    )
                    self._upsert_position_row(
                        connection=connection,
                        product_id=normalized_product_id,
                        quantity=position_quantity_after,
                        entry_price=existing_entry_price,
                        current_price=effective_current_price,
                        position_fraction=position_fraction_after,
                        opened_at=existing_opened_at or execution_timestamp,
                        metadata_json=json.dumps(existing_position.get("metadata") or {}),
                    )

            execution_row = connection.execute(
                """
                INSERT INTO portfolio_executions (
                    product_id,
                    side,
                    quantity,
                    price,
                    fee,
                    realized_pnl,
                    cash_flow,
                    position_quantity_after,
                    position_fraction_after,
                    executed_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING
                    execution_id,
                    product_id,
                    side,
                    quantity,
                    price,
                    fee,
                    realized_pnl,
                    cash_flow,
                    position_quantity_after,
                    position_fraction_after,
                    executed_at,
                    metadata_json
                """,
                (
                    normalized_product_id,
                    normalized_side,
                    normalized_quantity,
                    normalized_price,
                    normalized_fee,
                    realized_pnl,
                    cash_flow,
                    position_quantity_after,
                    position_fraction_after,
                    execution_timestamp,
                    metadata_json,
                ),
            ).fetchone()

            position_after = self._get_position_row(connection, normalized_product_id)

        return {
            "execution": self._row_to_execution_dict(execution_row),
            "position": self._row_to_position_dict(position_after) if position_after is not None else None,
            "portfolio": self.get_portfolio(),
        }

    def get_performance_summary(
        self,
        capital: float | None = None,
        positions: list[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Return realized and unrealized performance metrics from the journal and open book."""

        resolved_capital = float(capital) if capital is not None else self.get_capital()
        resolved_positions = positions if positions is not None else self.list_positions()

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS execution_count,
                    SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) AS buy_count,
                    SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) AS sell_count,
                    SUM(fee) AS total_fees,
                    SUM(realized_pnl) AS realized_pnl,
                    SUM(CASE WHEN side = 'sell' AND realized_pnl > 0 THEN 1 ELSE 0 END) AS winning_sells,
                    SUM(CASE WHEN side = 'sell' AND realized_pnl < 0 THEN 1 ELSE 0 END) AS losing_sells
                FROM portfolio_executions
                """
            ).fetchone()

        unrealized_pnl = sum(float(position.get("unrealizedPnl") or 0.0) for position in resolved_positions)
        realized_pnl = float(row["realized_pnl"] or 0.0)
        net_pnl = realized_pnl + unrealized_pnl
        marked_equity = resolved_capital + net_pnl
        sell_count = int(row["sell_count"] or 0)

        return {
            "executionCount": int(row["execution_count"] or 0),
            "buyCount": int(row["buy_count"] or 0),
            "sellCount": sell_count,
            "winningSellCount": int(row["winning_sells"] or 0),
            "losingSellCount": int(row["losing_sells"] or 0),
            "winRate": (float(row["winning_sells"] or 0) / sell_count) if sell_count else 0.0,
            "totalFees": float(row["total_fees"] or 0.0),
            "realizedPnl": realized_pnl,
            "unrealizedPnl": unrealized_pnl,
            "netPnl": net_pnl,
            "markedEquity": marked_equity,
            "realizedReturn": (realized_pnl / resolved_capital) if resolved_capital > 0 else 0.0,
            "netReturn": (net_pnl / resolved_capital) if resolved_capital > 0 else 0.0,
        }

    def _row_to_position_dict(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """Convert one SQLite row into a JSON-friendly position dictionary."""

        quantity = float(row["quantity"])
        entry_price = float(row["entry_price"]) if row["entry_price"] is not None else None
        current_price = float(row["current_price"]) if row["current_price"] is not None else None
        market_value = (quantity * current_price) if current_price is not None else None
        cost_basis = (quantity * entry_price) if entry_price is not None else None
        unrealized_pnl = (
            (quantity * (current_price - entry_price))
            if entry_price is not None and current_price is not None
            else None
        )
        unrealized_return = (
            (current_price / entry_price) - 1.0
            if entry_price is not None and current_price is not None and entry_price > 0
            else None
        )

        opened_at = str(row["opened_at"] or "")
        opened_timestamp = _parse_iso_timestamp(opened_at)
        age_hours = None
        if opened_timestamp is not None:
            age_hours = max((datetime.now(timezone.utc) - opened_timestamp).total_seconds() / 3600, 0.0)

        return {
            "productId": str(row["product_id"]),
            "quantity": quantity,
            "entryPrice": entry_price,
            "currentPrice": current_price,
            "positionFraction": float(row["position_fraction"] or 0.0),
            "marketValue": market_value,
            "costBasis": cost_basis,
            "unrealizedPnl": unrealized_pnl,
            "unrealizedReturn": unrealized_return,
            "openedAt": opened_at or None,
            "updatedAt": str(row["updated_at"]),
            "ageHours": age_hours,
            "metadata": json.loads(str(row["metadata_json"]) or "{}"),
        }

    def _row_to_execution_dict(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """Convert one SQLite row into a JSON-friendly execution dictionary."""

        return {
            "executionId": int(row["execution_id"]),
            "productId": str(row["product_id"]),
            "side": str(row["side"]),
            "quantity": float(row["quantity"]),
            "price": float(row["price"]),
            "fee": float(row["fee"] or 0.0),
            "realizedPnl": float(row["realized_pnl"] or 0.0),
            "cashFlow": float(row["cash_flow"] or 0.0),
            "positionQuantityAfter": float(row["position_quantity_after"] or 0.0),
            "positionFractionAfter": float(row["position_fraction_after"] or 0.0),
            "executedAt": str(row["executed_at"]),
            "metadata": json.loads(str(row["metadata_json"]) or "{}"),
        }

    def _row_to_tracked_trade_dict(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """Convert one tracked-trade row into a JSON-friendly trade record."""

        entry_price = float(row["entry_price"])
        current_price = float(row["current_price"]) if row["current_price"] is not None else None
        stop_loss_price = float(row["stop_loss_price"]) if row["stop_loss_price"] is not None else None
        take_profit_price = float(row["take_profit_price"]) if row["take_profit_price"] is not None else None
        exit_price = float(row["exit_price"]) if row["exit_price"] is not None else None
        realized_return = float(row["realized_return"]) if row["realized_return"] is not None else None
        current_return = (
            ((current_price / entry_price) - 1.0)
            if current_price is not None and entry_price > 0
            else None
        )

        return {
            "tradeId": int(row["trade_id"]),
            "productId": str(row["product_id"]),
            "signalName": str(row["signal_name"]) if row["signal_name"] is not None else None,
            "status": str(row["status"]),
            "quantity": float(row["quantity"] or 0.0),
            "entryPrice": entry_price,
            "currentPrice": current_price,
            "stopLossPrice": stop_loss_price,
            "takeProfitPrice": take_profit_price,
            "exitPrice": exit_price,
            "realizedPnl": float(row["realized_pnl"] or 0.0),
            "realizedReturn": realized_return,
            "currentReturn": current_return,
            "outcome": str(row["outcome"]) if row["outcome"] is not None else None,
            "closeReason": str(row["close_reason"]) if row["close_reason"] is not None else None,
            "openedAt": str(row["opened_at"]),
            "closedAt": str(row["closed_at"]) if row["closed_at"] is not None else None,
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
            "riskRewardRatio": (
                ((take_profit_price - entry_price) / (entry_price - stop_loss_price))
                if take_profit_price is not None
                and stop_loss_price is not None
                and entry_price > stop_loss_price
                else None
            ),
            "metadata": json.loads(str(row["metadata_json"]) or "{}"),
        }

    def _initialize_schema(self) -> None:
        """Create the SQLite tables used for portfolio state and execution history."""

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    product_id TEXT PRIMARY KEY,
                    quantity REAL NOT NULL DEFAULT 0,
                    entry_price REAL,
                    current_price REAL,
                    position_fraction REAL NOT NULL DEFAULT 0,
                    opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            if self.database.storage_backend == "postgresql":
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS portfolio_executions (
                        execution_id BIGSERIAL PRIMARY KEY,
                        product_id TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity DOUBLE PRECISION NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        fee DOUBLE PRECISION NOT NULL DEFAULT 0,
                        realized_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
                        cash_flow DOUBLE PRECISION NOT NULL DEFAULT 0,
                        position_quantity_after DOUBLE PRECISION NOT NULL DEFAULT 0,
                        position_fraction_after DOUBLE PRECISION NOT NULL DEFAULT 0,
                        executed_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
            else:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS portfolio_executions (
                        execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        product_id TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        fee REAL NOT NULL DEFAULT 0,
                        realized_pnl REAL NOT NULL DEFAULT 0,
                        cash_flow REAL NOT NULL DEFAULT 0,
                        position_quantity_after REAL NOT NULL DEFAULT 0,
                        position_fraction_after REAL NOT NULL DEFAULT 0,
                        executed_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_portfolio_executions_product_time ON portfolio_executions(product_id, executed_at)"
            )
            if self.database.storage_backend == "postgresql":
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS portfolio_tracked_trades (
                        trade_id BIGSERIAL PRIMARY KEY,
                        product_id TEXT NOT NULL,
                        signal_name TEXT,
                        status TEXT NOT NULL,
                        quantity DOUBLE PRECISION NOT NULL DEFAULT 0,
                        entry_price DOUBLE PRECISION NOT NULL,
                        current_price DOUBLE PRECISION,
                        stop_loss_price DOUBLE PRECISION,
                        take_profit_price DOUBLE PRECISION,
                        exit_price DOUBLE PRECISION,
                        realized_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
                        realized_return DOUBLE PRECISION,
                        outcome TEXT,
                        close_reason TEXT,
                        opened_at TEXT NOT NULL,
                        closed_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
            else:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS portfolio_tracked_trades (
                        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        product_id TEXT NOT NULL,
                        signal_name TEXT,
                        status TEXT NOT NULL,
                        quantity REAL NOT NULL DEFAULT 0,
                        entry_price REAL NOT NULL,
                        current_price REAL,
                        stop_loss_price REAL,
                        take_profit_price REAL,
                        exit_price REAL,
                        realized_pnl REAL NOT NULL DEFAULT 0,
                        realized_return REAL,
                        outcome TEXT,
                        close_reason TEXT,
                        opened_at TEXT NOT NULL,
                        closed_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_portfolio_tracked_trades_status_time ON portfolio_tracked_trades(status, opened_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_portfolio_tracked_trades_product_time ON portfolio_tracked_trades(product_id, opened_at)"
            )

    def _initialize_default_capital(self) -> None:
        """Ensure the capital setting exists on first boot."""

        with self._connect() as connection:
            if self.database.storage_backend == "postgresql":
                connection.execute(
                    """
                    INSERT INTO portfolio_settings (setting_key, setting_value, updated_at)
                    VALUES ('capital', ?, ?)
                    ON CONFLICT(setting_key) DO NOTHING
                    """,
                    (str(self.default_capital), _utc_now_iso()),
                )
            else:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO portfolio_settings (setting_key, setting_value, updated_at)
                    VALUES ('capital', ?, ?)
                    """,
                    (str(self.default_capital), _utc_now_iso()),
                )

    def _get_capital_from_connection(self, connection: DatabaseConnection) -> float:
        """Return the tracked portfolio capital using an existing transaction."""

        row = connection.execute(
            "SELECT setting_value FROM portfolio_settings WHERE setting_key = 'capital'",
        ).fetchone()
        if row is None:
            return float(self.default_capital)

        try:
            return float(row["setting_value"])
        except (TypeError, ValueError):
            return float(self.default_capital)

    def _get_position_row(
        self,
        connection: DatabaseConnection,
        product_id: str,
    ) -> Mapping[str, Any] | None:
        """Return one raw position row inside an existing transaction."""

        return connection.execute(
            """
            SELECT
                product_id,
                quantity,
                entry_price,
                current_price,
                position_fraction,
                opened_at,
                updated_at,
                metadata_json
            FROM portfolio_positions
            WHERE product_id = ?
            """,
            (product_id,),
        ).fetchone()

    def _get_tracked_trade_row(
        self,
        connection: DatabaseConnection,
        trade_id: int,
    ) -> Mapping[str, Any] | None:
        """Return one raw tracked-trade row inside an existing transaction."""

        return connection.execute(
            """
            SELECT
                trade_id,
                product_id,
                signal_name,
                status,
                quantity,
                entry_price,
                current_price,
                stop_loss_price,
                take_profit_price,
                exit_price,
                realized_pnl,
                realized_return,
                outcome,
                close_reason,
                opened_at,
                closed_at,
                created_at,
                updated_at,
                metadata_json
            FROM portfolio_tracked_trades
            WHERE trade_id = ?
            """,
            (trade_id,),
        ).fetchone()

    def _upsert_position_row(
        self,
        connection: DatabaseConnection,
        product_id: str,
        quantity: float,
        entry_price: float | None,
        current_price: float | None,
        position_fraction: float,
        opened_at: str,
        metadata_json: str,
    ) -> None:
        """Create or replace one raw position row inside an existing transaction."""

        connection.execute(
            """
            INSERT INTO portfolio_positions (
                product_id,
                quantity,
                entry_price,
                current_price,
                position_fraction,
                opened_at,
                updated_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                quantity = excluded.quantity,
                entry_price = excluded.entry_price,
                current_price = excluded.current_price,
                position_fraction = excluded.position_fraction,
                opened_at = excluded.opened_at,
                updated_at = excluded.updated_at,
                metadata_json = excluded.metadata_json
            """,
            (
                product_id,
                quantity,
                entry_price,
                current_price,
                position_fraction,
                opened_at,
                _utc_now_iso(),
                metadata_json,
            ),
        )

    @staticmethod
    def _normalize_product_id(product_id: str) -> str:
        """Normalize product ids into uppercase exchange-style symbols."""

        normalized_product_id = str(product_id).strip().upper()
        if not normalized_product_id:
            raise ValueError("productId is empty.")

        return normalized_product_id

    @staticmethod
    def _normalize_tracked_trade_status(
        status: str | None,
        allow_terminal: bool = True,
    ) -> str:
        """Normalize one tracked-trade status into the allowed status set."""

        normalized_status = str(status or "").strip().lower()
        if not normalized_status:
            raise ValueError("status is empty.")
        if normalized_status not in TRACKED_TRADE_STATUSES:
            raise ValueError(
                "status must be one of: planned, open, closed, cancelled."
            )
        if not allow_terminal and normalized_status not in ACTIVE_TRACKED_TRADE_STATUSES:
            raise ValueError("New tracked trades must start as planned or open.")

        return normalized_status

    def _connect(self) -> DatabaseConnection:
        """Open a backend-aware connection with dict-style row access."""

        return self.database.connect()
