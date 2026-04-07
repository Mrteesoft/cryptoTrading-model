"""Lightweight retrieval-augmented generation helpers for external knowledge."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from .storage.database import DatabaseConnection, DatabaseHandle


COMMON_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "with",
}


class _HtmlToTextParser(HTMLParser):
    """Extract readable text from simple HTML documents."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth > 0:
            return

        cleaned_data = _normalize_whitespace(data)
        if cleaned_data:
            self._parts.append(cleaned_data)

    def get_text(self) -> str:
        """Return the normalized extracted text."""

        return _normalize_whitespace(" ".join(self._parts))


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim the result."""

    return re.sub(r"\s+", " ", str(text)).strip()


def _tokenize(text: str) -> set[str]:
    """Turn free text into a lightweight token set for retrieval scoring."""

    return {
        token
        for token in re.findall(r"[a-z0-9\-]+", str(text).lower())
        if len(token) > 1 and token not in COMMON_STOP_WORDS
    }


class RagKnowledgeStore:
    """Store source documents and searchable chunks in SQLite or PostgreSQL."""

    def __init__(
        self,
        db_path: Path,
        database_url: str | None = None,
        chunk_size_chars: int = 900,
        chunk_overlap_chars: int = 120,
        fetch_timeout_seconds: float = 15.0,
        fetch_max_chars: int = 50000,
    ) -> None:
        self.database = DatabaseHandle(
            sqlite_path=db_path,
            database_url=database_url,
        )
        self.db_path = self.database.sqlite_path
        self.chunk_size_chars = max(int(chunk_size_chars), 200)
        self.chunk_overlap_chars = max(min(int(chunk_overlap_chars), self.chunk_size_chars - 1), 0)
        self.fetch_timeout_seconds = max(float(fetch_timeout_seconds), 1.0)
        self.fetch_max_chars = max(int(fetch_max_chars), 1000)
        self._initialize_schema()

    def get_status(self) -> Dict[str, Any]:
        """Return counts that describe the current knowledge-store state."""

        with self._connect() as connection:
            source_count = int(
                connection.execute("SELECT COUNT(*) AS source_count FROM rag_sources").fetchone()["source_count"]
            )
            chunk_count = int(
                connection.execute("SELECT COUNT(*) AS chunk_count FROM rag_chunks").fetchone()["chunk_count"]
            )

        return {
            "enabled": True,
            "dbPath": str(self.db_path) if self.db_path is not None else None,
            "storageBackend": self.database.storage_backend,
            "databaseTarget": self.database.database_target,
            "sourceCount": source_count,
            "chunkCount": chunk_count,
        }

    def list_sources(self, limit: int = 50) -> list[Dict[str, Any]]:
        """Return recently ingested knowledge sources."""

        query_limit = max(int(limit), 1)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source_id, title, source_type, source_uri, created_at, metadata_json, content_hash, chunk_count
                FROM rag_sources
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (query_limit,),
            ).fetchall()

        return [self._row_to_source_dict(row) for row in rows]

    def ingest_text(
        self,
        title: str,
        content: str,
        source_uri: str | None = None,
        metadata: Dict[str, Any] | None = None,
        source_type: str = "text",
    ) -> Dict[str, Any]:
        """Ingest one raw text document into searchable chunks."""

        normalized_title = _normalize_whitespace(title)
        normalized_content = _normalize_whitespace(content)
        if not normalized_title:
            raise ValueError("Document title is empty.")
        if not normalized_content:
            raise ValueError("Document content is empty.")

        source_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        content_hash = sha256(normalized_content.encode("utf-8")).hexdigest()
        metadata_json = json.dumps(metadata or {})
        normalized_source_uri = _normalize_whitespace(source_uri or "") or None

        if normalized_source_uri:
            self._delete_source_by_uri(normalized_source_uri)

        content_chunks = self._chunk_text(normalized_content)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rag_sources (
                    source_id,
                    title,
                    source_type,
                    source_uri,
                    created_at,
                    metadata_json,
                    content_hash,
                    chunk_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    normalized_title,
                    source_type,
                    normalized_source_uri,
                    created_at,
                    metadata_json,
                    content_hash,
                    len(content_chunks),
                ),
            )

            for chunk_index, chunk_content in enumerate(content_chunks):
                connection.execute(
                    """
                    INSERT INTO rag_chunks (
                        source_id,
                        chunk_index,
                        title,
                        source_type,
                        source_uri,
                        content,
                        snippet,
                        metadata_json,
                        token_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        chunk_index,
                        normalized_title,
                        source_type,
                        normalized_source_uri,
                        chunk_content,
                        self._build_snippet(chunk_content),
                        metadata_json,
                        json.dumps(sorted(_tokenize(chunk_content))),
                        created_at,
                    ),
                )

        return self.get_source(source_id) or {}

    def ingest_file(
        self,
        file_path: Path,
        title: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Ingest one local text-like file into the knowledge store."""

        normalized_path = Path(file_path)
        if not normalized_path.exists():
            raise FileNotFoundError(f"Knowledge source file not found: {normalized_path}")

        content = normalized_path.read_text(encoding="utf-8")
        return self.ingest_text(
            title=title or normalized_path.name,
            content=content,
            source_uri=str(normalized_path),
            metadata=metadata,
            source_type="file",
        )

    def ingest_url(
        self,
        url: str,
        title: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Fetch a URL, normalize its text, and ingest it as a knowledge source."""

        normalized_url = _normalize_whitespace(url)
        if not normalized_url:
            raise ValueError("URL is empty.")

        request = Request(
            normalized_url,
            headers={
                "User-Agent": "crypto-signal-ml-rag/0.1",
                "Accept": "text/html, text/plain, application/json;q=0.9, */*;q=0.8",
            },
        )

        try:
            with urlopen(request, timeout=self.fetch_timeout_seconds) as response:
                response_bytes = response.read(self.fetch_max_chars)
                content_type = str(response.headers.get("Content-Type", "")).lower()
        except HTTPError as error:
            raise ValueError(f"Source fetch failed with status {error.code} for {normalized_url}.") from error
        except URLError as error:
            raise ValueError(f"Source fetch failed for {normalized_url}: {error.reason}") from error

        decoded_text = response_bytes.decode("utf-8", errors="replace")
        source_title = title
        source_body = decoded_text

        if "html" in content_type or "<html" in decoded_text.lower():
            source_title = source_title or self._extract_html_title(decoded_text) or normalized_url
            source_body = self._extract_html_text(decoded_text)
        else:
            source_title = source_title or normalized_url
            source_body = _normalize_whitespace(decoded_text)

        return self.ingest_text(
            title=source_title or normalized_url,
            content=source_body,
            source_uri=normalized_url,
            metadata=metadata,
            source_type="url",
        )

    def search(
        self,
        query: str,
        limit: int = 6,
    ) -> list[Dict[str, Any]]:
        """Return the most relevant knowledge chunks for the query."""

        normalized_query = _normalize_whitespace(query)
        query_tokens = _tokenize(normalized_query)
        if not normalized_query or not query_tokens:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    source_id,
                    chunk_index,
                    title,
                    source_type,
                    source_uri,
                    content,
                    snippet,
                    metadata_json,
                    token_json,
                    created_at
                FROM rag_chunks
                """
            ).fetchall()

        scored_rows = []
        lowered_query = normalized_query.lower()
        for row in rows:
            chunk_tokens = set(json.loads(str(row["token_json"]) or "[]"))
            title_tokens = _tokenize(str(row["title"]))
            source_uri_tokens = _tokenize(str(row["source_uri"] or ""))
            overlap = len(query_tokens & chunk_tokens)
            title_overlap = len(query_tokens & title_tokens)
            uri_overlap = len(query_tokens & source_uri_tokens)
            exact_phrase_bonus = 2 if lowered_query in str(row["content"]).lower() else 0
            score = (overlap * 3) + (title_overlap * 4) + (uri_overlap * 2) + exact_phrase_bonus

            if score <= 0:
                continue

            scored_rows.append(
                (
                    score,
                    {
                        "chunkId": int(row["chunk_id"]),
                        "sourceId": str(row["source_id"]),
                        "chunkIndex": int(row["chunk_index"]),
                        "title": str(row["title"]),
                        "sourceType": str(row["source_type"]),
                        "sourceUri": str(row["source_uri"] or ""),
                        "snippet": str(row["snippet"]),
                        "content": str(row["content"]),
                        "score": float(score),
                        "createdAt": str(row["created_at"]),
                        "metadata": json.loads(str(row["metadata_json"]) or "{}"),
                    },
                )
            )

        ranked_rows = sorted(
            scored_rows,
            key=lambda item: (item[0], -item[1]["chunkIndex"]),
            reverse=True,
        )
        return [row for _, row in ranked_rows[: max(int(limit), 1)]]

    def get_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Return one ingested source by id."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT source_id, title, source_type, source_uri, created_at, metadata_json, content_hash, chunk_count
                FROM rag_sources
                WHERE source_id = ?
                """,
                (source_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_source_dict(row)

    def delete_source(self, source_id: str) -> bool:
        """Delete one source and all of its chunks."""

        with self._connect() as connection:
            connection.execute("DELETE FROM rag_chunks WHERE source_id = ?", (source_id,))
            deleted_count = int(
                connection.execute("DELETE FROM rag_sources WHERE source_id = ?", (source_id,)).rowcount
            )

        return deleted_count > 0

    def _delete_source_by_uri(self, source_uri: str) -> None:
        """Replace any existing source that uses the same URI."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source_id FROM rag_sources WHERE source_uri = ?",
                (source_uri,),
            ).fetchall()

        for row in rows:
            self.delete_source(str(row["source_id"]))

    def _chunk_text(self, text: str) -> list[str]:
        """Split one document into overlapping character chunks."""

        normalized_text = _normalize_whitespace(text)
        if not normalized_text:
            return []

        chunks = []
        start_index = 0
        text_length = len(normalized_text)
        step_size = max(self.chunk_size_chars - self.chunk_overlap_chars, 1)

        while start_index < text_length:
            end_index = min(start_index + self.chunk_size_chars, text_length)
            chunk_text = normalized_text[start_index:end_index].strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end_index >= text_length:
                break
            start_index += step_size

        return chunks

    @staticmethod
    def _build_snippet(text: str, max_length: int = 220) -> str:
        """Build a short preview snippet for one chunk."""

        normalized_text = _normalize_whitespace(text)
        if len(normalized_text) <= max_length:
            return normalized_text

        return normalized_text[: max_length - 3].rstrip() + "..."

    @staticmethod
    def _extract_html_title(html_text: str) -> str:
        """Pull a page title from basic HTML when it is available."""

        title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
        if title_match is None:
            return ""

        return _normalize_whitespace(unescape(title_match.group(1)))

    @staticmethod
    def _extract_html_text(html_text: str) -> str:
        """Convert HTML into normalized text for indexing."""

        parser = _HtmlToTextParser()
        parser.feed(html_text)
        return parser.get_text()

    def _row_to_source_dict(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """Convert one source row into a JSON-friendly dictionary."""

        return {
            "sourceId": str(row["source_id"]),
            "title": str(row["title"]),
            "sourceType": str(row["source_type"]),
            "sourceUri": str(row["source_uri"] or ""),
            "createdAt": str(row["created_at"]),
            "metadata": json.loads(str(row["metadata_json"]) or "{}"),
            "contentHash": str(row["content_hash"]),
            "chunkCount": int(row["chunk_count"]),
        }

    def _initialize_schema(self) -> None:
        """Create the SQLite tables used by the knowledge store."""

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_sources (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_uri TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    content_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            if self.database.storage_backend == "postgresql":
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        chunk_id BIGSERIAL PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_uri TEXT,
                        content TEXT NOT NULL,
                        snippet TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        token_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(source_id) REFERENCES rag_sources(source_id)
                    )
                    """
                )
            else:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_uri TEXT,
                        content TEXT NOT NULL,
                        snippet TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        token_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(source_id) REFERENCES rag_sources(source_id)
                    )
                    """
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_chunks_source_id ON rag_chunks(source_id)"
            )

    def _connect(self) -> DatabaseConnection:
        """Open a backend-aware connection with dict-style row access."""

        return self.database.connect()
