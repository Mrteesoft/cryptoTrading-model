"""Small database adapter that supports SQLite locally and PostgreSQL in production."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - exercised only when postgres support is requested.
    psycopg = None
    dict_row = None


POSTGRES_SCHEMES = {"postgres", "postgresql"}


class DatabaseConnection(AbstractContextManager["DatabaseConnection"]):
    """Context-managed connection proxy that normalizes SQL placeholders."""

    def __init__(self, handle: "DatabaseHandle", raw_connection: Any) -> None:
        self.handle = handle
        self.raw_connection = raw_connection

    def __enter__(self) -> "DatabaseConnection":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        if exc_type is None:
            self.raw_connection.commit()
        else:
            self.raw_connection.rollback()

        self.raw_connection.close()
        return None

    def execute(self, query: str, params: tuple[Any, ...] | list[Any] = ()) -> Any:
        """Execute one SQL statement with backend-appropriate placeholder syntax."""

        return self.raw_connection.execute(self.handle.sql(query), params)

    def __getattr__(self, attribute_name: str) -> Any:
        return getattr(self.raw_connection, attribute_name)


class DatabaseHandle:
    """Resolve one logical database target into SQLite or PostgreSQL access."""

    def __init__(
        self,
        sqlite_path: Path | str | None = None,
        database_url: str | None = None,
    ) -> None:
        normalized_database_url = str(database_url or "").strip() or None
        self.database_url = normalized_database_url
        self.backend = "sqlite"
        self.sqlite_path: Path | None = None

        if normalized_database_url:
            parsed_url = urlsplit(normalized_database_url)
            scheme = parsed_url.scheme.lower()
            if scheme in POSTGRES_SCHEMES:
                if psycopg is None or dict_row is None:
                    raise RuntimeError(
                        "PostgreSQL storage requires the `psycopg` package. Install `psycopg[binary]` first."
                    )
                self.backend = "postgresql"
            elif scheme == "sqlite":
                self.sqlite_path = self._sqlite_path_from_url(normalized_database_url)
                self.database_url = None
            else:
                raise ValueError(
                    "Unsupported database URL. Use sqlite:///... or postgresql://... style URLs."
                )

        if self.backend == "sqlite":
            if self.sqlite_path is None:
                if sqlite_path is None:
                    raise ValueError("A SQLite path or database URL must be provided.")
                self.sqlite_path = Path(sqlite_path)
            if str(self.sqlite_path) != ":memory:":
                self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def storage_backend(self) -> str:
        """Return the active storage backend label."""

        return self.backend

    @property
    def database_target(self) -> str:
        """Return a safe display string for the active storage target."""

        if self.backend == "postgresql":
            return self._redact_database_url(self.database_url or "")
        return str(self.sqlite_path)

    def connect(self) -> DatabaseConnection:
        """Open one backend-specific connection wrapped in a placeholder-aware proxy."""

        if self.backend == "postgresql":
            raw_connection = psycopg.connect(self.database_url, row_factory=dict_row)
        else:
            raw_connection = sqlite3.connect(self.sqlite_path)
            raw_connection.row_factory = sqlite3.Row

        return DatabaseConnection(handle=self, raw_connection=raw_connection)

    def sql(self, query: str) -> str:
        """Translate parameter placeholders for the active backend."""

        if self.backend == "postgresql":
            return str(query).replace("?", "%s")

        return str(query)

    @staticmethod
    def _sqlite_path_from_url(database_url: str) -> Path:
        """Resolve a sqlite:/// URL into a local filesystem path."""

        parsed_url = urlsplit(database_url)
        raw_path = unquote(parsed_url.path or "")
        if parsed_url.netloc and parsed_url.netloc not in {"", "localhost"}:
            raw_path = f"//{parsed_url.netloc}{raw_path}"
        if raw_path in {"", "/:memory:", ":memory:"}:
            return Path(":memory:")
        if raw_path.startswith("/") and len(raw_path) >= 3 and raw_path[2] == ":":
            raw_path = raw_path[1:]
        return Path(raw_path)

    @staticmethod
    def _redact_database_url(database_url: str) -> str:
        """Hide the password component before exposing a DSN in status payloads."""

        parsed_url = urlsplit(database_url)
        if not parsed_url.netloc:
            return database_url

        username = parsed_url.username or ""
        host = parsed_url.hostname or ""
        netloc = host
        if parsed_url.port is not None:
            netloc = f"{netloc}:{parsed_url.port}"
        if username:
            netloc = f"{username}@{netloc}"

        return urlunsplit(
            (
                parsed_url.scheme,
                netloc,
                parsed_url.path,
                parsed_url.query,
                parsed_url.fragment,
            )
        )
