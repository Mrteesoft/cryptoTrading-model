"""Structured retrieval tools backed by the local knowledge store."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..rag import RagKnowledgeStore


RETRIEVAL_SEARCH_TOOL_SCHEMA = {
    "name": "search_knowledge",
    "description": "Search the authoritative RAG knowledge store for supporting context.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to run against the indexed knowledge store.",
            },
            "limit": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 25,
                "description": "Maximum number of matching chunks to return.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

TOOL_SCHEMAS = [RETRIEVAL_SEARCH_TOOL_SCHEMA]


def _utc_now_iso() -> str:
    """Return one UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


class RetrievalToolService:
    """Expose stable knowledge-search tooling over the local RAG store."""

    tool_schemas = TOOL_SCHEMAS

    def __init__(
        self,
        knowledge_store: RagKnowledgeStore | None,
        *,
        default_limit: int = 5,
    ) -> None:
        self.knowledge_store = knowledge_store
        self.default_limit = max(int(default_limit), 1)

    def search_knowledge(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Return one stable knowledge-search tool payload."""

        normalized_query = str(query).strip()
        if not normalized_query:
            return {
                "toolName": "search_knowledge",
                "requestedAt": _utc_now_iso(),
                "status": "error",
                "query": normalized_query,
                "limit": max(int(limit or self.default_limit), 1),
                "count": 0,
                "results": [],
                "error": "Query is empty.",
            }

        if self.knowledge_store is None:
            return {
                "toolName": "search_knowledge",
                "requestedAt": _utc_now_iso(),
                "status": "disabled",
                "query": normalized_query,
                "limit": max(int(limit or self.default_limit), 1),
                "count": 0,
                "results": [],
                "error": "",
            }

        resolved_limit = max(int(limit or self.default_limit), 1)
        try:
            results = self.knowledge_store.search(normalized_query, limit=resolved_limit)
        except Exception as error:
            return {
                "toolName": "search_knowledge",
                "requestedAt": _utc_now_iso(),
                "status": "error",
                "query": normalized_query,
                "limit": resolved_limit,
                "count": 0,
                "results": [],
                "error": str(error),
            }

        return {
            "toolName": "search_knowledge",
            "requestedAt": _utc_now_iso(),
            "status": "ok",
            "query": normalized_query,
            "limit": resolved_limit,
            "count": len(results),
            "results": results,
            "error": "",
        }
