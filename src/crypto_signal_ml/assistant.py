"""Conversational trading-assistant services for live market Q&A."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Mapping, Optional
from uuid import uuid4

from .application import PublishedSignalViewService
from .chat.flow import ToolDrivenChatFlow
from .config import TrainingConfig
from .live import LiveSignalEngine
from .rag import RagKnowledgeStore
from .storage.database import DatabaseConnection, DatabaseHandle
from .tools import ModelToolService, RetrievalToolService, SignalToolService, ToolRegistry, TraderToolService
from .trading.portfolio import TradingPortfolioStore


COMMON_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "should",
    "the",
    "this",
    "to",
    "what",
    "with",
}


class ConversationSessionStore:
    """Persist assistant sessions and chat messages in SQLite or PostgreSQL."""

    def __init__(self, db_path: Path, database_url: str | None = None) -> None:
        self.database = DatabaseHandle(
            sqlite_path=db_path,
            database_url=database_url,
        )
        self.db_path = self.database.sqlite_path
        self._initialize_schema()

    def create_session(self, title: str | None = None) -> Dict[str, Any]:
        """Create and return a new chat session record."""

        session_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO assistant_sessions (session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, title or "Trading session", timestamp, timestamp),
            )

        return self.get_session(session_id) or {}

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return one saved session plus its message count."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    session_id,
                    title,
                    created_at,
                    updated_at,
                    (
                        SELECT COUNT(*)
                        FROM assistant_messages
                        WHERE assistant_messages.session_id = assistant_sessions.session_id
                    ) AS message_count
                FROM assistant_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "sessionId": str(row["session_id"]),
            "title": str(row["title"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
            "messageCount": int(row["message_count"]),
        }

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Append one chat message to a session."""

        if self.get_session(session_id) is None:
            raise ValueError(f"Assistant session not found: {session_id}")

        timestamp = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata or {})

        with self._connect() as connection:
            row = connection.execute(
                """
                INSERT INTO assistant_messages (session_id, role, content, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                RETURNING message_id, session_id, role, content, created_at, metadata_json
                """,
                (session_id, role, content, timestamp, metadata_json),
            ).fetchone()
            connection.execute(
                """
                UPDATE assistant_sessions
                SET updated_at = ?
                WHERE session_id = ?
                """,
                (timestamp, session_id),
            )

        return self._row_to_message_dict(row) if row is not None else {}

    def get_message(self, message_id: int) -> Optional[Dict[str, Any]]:
        """Return one saved chat message."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT message_id, session_id, role, content, created_at, metadata_json
                FROM assistant_messages
                WHERE message_id = ?
                """,
                (int(message_id),),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_message_dict(row)

    def list_messages(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[Dict[str, Any]]:
        """Return the newest messages for a session in chronological order."""

        query_limit = max(int(limit), 1)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT message_id, session_id, role, content, created_at, metadata_json
                FROM (
                    SELECT *
                    FROM assistant_messages
                    WHERE session_id = ?
                    ORDER BY message_id DESC
                    LIMIT ?
                )
                ORDER BY message_id ASC
                """,
                (session_id, query_limit),
            ).fetchall()

        return [self._row_to_message_dict(row) for row in rows]

    @staticmethod
    def _row_to_message_dict(row: Mapping[str, Any]) -> Dict[str, Any]:
        """Convert one stored assistant message row into the public response shape."""

        return {
            "messageId": int(row["message_id"]),
            "sessionId": str(row["session_id"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "createdAt": str(row["created_at"]),
            "metadata": json.loads(str(row["metadata_json"]) or "{}"),
        }

    def _initialize_schema(self) -> None:
        """Create the SQLite schema if it does not exist yet."""

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS assistant_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            if self.database.storage_backend == "postgresql":
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS assistant_messages (
                        message_id BIGSERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        FOREIGN KEY(session_id) REFERENCES assistant_sessions(session_id)
                    )
                    """
                )
            else:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS assistant_messages (
                        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        FOREIGN KEY(session_id) REFERENCES assistant_sessions(session_id)
                    )
                    """
                )

    def _connect(self) -> DatabaseConnection:
        """Open a backend-aware connection with dict-like row access."""

        return self.database.connect()


class TradingAssistantService:
    """
    Answer trading questions by combining live signals, model status, and memory.

    This is intentionally provider-agnostic. The first implementation uses a
    deterministic response composer so the app can ship without an external LLM
    dependency. The service boundary is ready for an LLM adapter later.
    """

    def __init__(
        self,
        live_signal_engine: LiveSignalEngine,
        session_store: ConversationSessionStore,
        model_summary_provider: Callable[[], Dict[str, Any]],
        cached_snapshot_provider: Callable[[], Dict[str, Any] | None] | None = None,
        knowledge_store: RagKnowledgeStore | None = None,
        portfolio_store: TradingPortfolioStore | None = None,
        tool_registry: ToolRegistry | None = None,
        chat_flow: ToolDrivenChatFlow | None = None,
        published_signal_service: PublishedSignalViewService | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        self.live_signal_engine = live_signal_engine
        self.session_store = session_store
        self.model_summary_provider = model_summary_provider
        self.cached_snapshot_provider = cached_snapshot_provider
        self.knowledge_store = knowledge_store
        self.config = config or TrainingConfig()
        self.portfolio_store = portfolio_store or TradingPortfolioStore(
            db_path=self.config.portfolio_store_path,
            default_capital=self.config.portfolio_default_capital,
            database_url=self.config.portfolio_store_url,
        )
        self.signal_tools = SignalToolService(
            live_signal_engine=self.live_signal_engine,
            cached_snapshot_provider=self.cached_snapshot_provider,
            published_signal_service=published_signal_service,
        )
        self.trader_tools = TraderToolService(
            signal_tools=self.signal_tools,
            portfolio_store=self.portfolio_store,
            config=self.config,
            published_signal_service=published_signal_service,
        )
        self.model_tools = ModelToolService(model_status_provider=self.model_summary_provider)
        self.retrieval_tools = RetrievalToolService(
            knowledge_store=self.knowledge_store,
            default_limit=self.config.rag_search_limit,
        )
        self.tool_registry = tool_registry or ToolRegistry(
            signal_tools=self.signal_tools,
            trader_tools=self.trader_tools,
            model_tools=self.model_tools,
            retrieval_tools=self.retrieval_tools,
        )
        self.chat_flow = chat_flow or ToolDrivenChatFlow(
            tool_registry=self.tool_registry,
            session_store=self.session_store,
            config=self.config,
        )

    def create_session(self, title: str | None = None) -> Dict[str, Any]:
        """Create a session and seed it with a welcome message."""

        session = self.session_store.create_session(title=title or self.config.assistant_system_name)
        welcome_message = self.session_store.add_message(
            session_id=session["sessionId"],
            role="assistant",
            content=(
                f"{self.config.assistant_system_name} is ready. Ask about a coin, the live market overview, "
                "what the model is seeing in the latest candle data, or add external sources to the knowledge base "
                "for deeper retrieval."
            ),
            metadata={"type": "welcome"},
        )
        return {
            "session": self.session_store.get_session(session["sessionId"]),
            "messages": [welcome_message],
        }

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Return one chat session plus its recent messages."""

        session = self.session_store.get_session(session_id)
        if session is None:
            raise ValueError(f"Assistant session not found: {session_id}")

        return {
            "session": session,
            "messages": self.session_store.list_messages(
                session_id=session_id,
                limit=self.config.assistant_memory_message_limit,
            ),
        }

    def answer_question(
        self,
        session_id: str,
        question: str,
        product_id: str | None = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Store the user message, compose an answer, and persist the reply."""

        normalized_question = str(question).strip()
        if not normalized_question:
            raise ValueError("Question is empty.")

        user_message = self.session_store.add_message(
            session_id=session_id,
            role="user",
            content=normalized_question,
            metadata={"productId": product_id},
        )
        flow_result = self.chat_flow.run(
            session_id=session_id,
            question=normalized_question,
            explicit_product_id=product_id,
            force_refresh=force_refresh,
        )
        resolved_product_id = flow_result["resolvedProductId"]
        resolved_product_ids = list(flow_result.get("resolvedProductIds", []))
        answer_text = str(flow_result["replyText"])
        retrieval = dict(flow_result["retrieval"])
        live_context = dict(flow_result["liveContext"])
        tool_calls = list(flow_result["toolCalls"])
        tool_results = list(flow_result["toolResults"])
        routing = dict(flow_result.get("routing") or {})
        tool_telemetry = dict(flow_result.get("toolTelemetry") or {})
        assistant_message = self.session_store.add_message(
            session_id=session_id,
            role="assistant",
            content=answer_text,
            metadata={
                "productId": resolved_product_id,
                "productIds": resolved_product_ids,
                "liveSource": live_context.get("source"),
                "liveError": live_context.get("error"),
                "toolCalls": tool_calls,
                "toolNames": [tool_call["name"] for tool_call in tool_calls],
                "routing": routing,
                "toolTelemetry": tool_telemetry,
                "retrieval": {
                    "messageCount": len(retrieval.get("messages", [])),
                    "knowledgeCount": len(retrieval.get("knowledge", [])),
                },
            },
        )

        return {
            "session": self.session_store.get_session(session_id),
            "userMessage": user_message,
            "assistantMessage": assistant_message,
            "messages": self.session_store.list_messages(
                session_id=session_id,
                limit=self.config.assistant_memory_message_limit,
            ),
            "liveContext": live_context,
            "retrieval": retrieval,
            "routing": routing,
            "toolCalls": tool_calls,
            "toolTelemetry": tool_telemetry,
            "toolResults": tool_results,
        }

    def _load_cached_snapshot(self) -> Dict[str, Any] | None:
        """Return the cached frontend snapshot when it is available."""

        if self.cached_snapshot_provider is None:
            return None

        try:
            return self.cached_snapshot_provider()
        except Exception:
            return None

    def _build_retrieval_context(
        self,
        session_id: str,
        question: str,
        live_snapshot: Dict[str, Any] | None,
        resolved_product_id: str | None,
    ) -> Dict[str, Any]:
        """Collect the most relevant live signals and prior messages for one reply."""

        if not self.config.assistant_enable_retrieval:
            return {"signals": [], "messages": [], "knowledge": []}

        question_tokens = self._tokenize(question)
        signal_rows = list((live_snapshot or {}).get("signals", []))
        scored_signals = []

        for signal_summary in signal_rows:
            signal_tokens = self._tokenize(
                " ".join(
                    [
                        str(signal_summary.get("productId", "")),
                        str(signal_summary.get("coinSymbol", "")),
                        str(signal_summary.get("coinName", "")),
                        str(signal_summary.get("signal_name", "")),
                        str(signal_summary.get("signalChat", "")),
                    ]
                )
            )
            score = len(question_tokens & signal_tokens)
            if resolved_product_id and str(signal_summary.get("productId", "")).upper() == resolved_product_id.upper():
                score += 5
            if "buy" in question_tokens and str(signal_summary.get("signal_name")) == "BUY":
                score += 1
            if "sell" in question_tokens and str(signal_summary.get("signal_name")) in {"TAKE_PROFIT", "LOSS"}:
                score += 1
            if "loss" in question_tokens and str(signal_summary.get("signal_name")) == "LOSS":
                score += 1
            if "hold" in question_tokens and str(signal_summary.get("signal_name")) == "HOLD":
                score += 1
            if score > 0:
                scored_signals.append((score, signal_summary))

        top_signals = [
            signal_summary
            for _, signal_summary in sorted(
                scored_signals,
                key=lambda item: (
                    item[0],
                    float(item[1].get("confidence", 0.0)),
                ),
                reverse=True,
            )[: self.config.assistant_retrieval_item_limit]
        ]

        message_rows = self.session_store.list_messages(
            session_id=session_id,
            limit=self.config.assistant_memory_message_limit,
        )
        scored_messages = []
        for message_row in message_rows[:-1]:
            message_tokens = self._tokenize(message_row["content"])
            score = len(question_tokens & message_tokens)
            if score > 0:
                scored_messages.append((score, message_row))

        top_messages = [
            {
                "role": message_row["role"],
                "content": message_row["content"],
                "createdAt": message_row["createdAt"],
            }
            for _, message_row in sorted(scored_messages, key=lambda item: item[0], reverse=True)[
                : self.config.assistant_retrieval_item_limit
            ]
        ]

        knowledge_chunks = []
        if self.config.rag_enabled and self.knowledge_store is not None:
            knowledge_query = question
            if resolved_product_id:
                knowledge_query = f"{question} {resolved_product_id}"
            knowledge_chunks = self.knowledge_store.search(
                knowledge_query,
                limit=self.config.rag_search_limit,
            )

        return {
            "signals": top_signals,
            "messages": top_messages,
            "knowledge": knowledge_chunks,
        }

    def _compose_answer(
        self,
        question: str,
        resolved_product_id: str | None,
        live_snapshot: Dict[str, Any] | None,
        live_source: str,
        live_error: str,
        model_summary: Dict[str, Any],
        retrieval: Dict[str, Any],
    ) -> str:
        """Generate a natural-language answer from structured market context."""

        signal_by_product = (live_snapshot or {}).get("signalsByProduct", {})
        target_signal = (
            signal_by_product.get(resolved_product_id.upper())
            if resolved_product_id
            else None
        )
        primary_signal = (live_snapshot or {}).get("primarySignal") or {}
        market_summary = (live_snapshot or {}).get("marketSummary", {})
        actionable_signals = list((live_snapshot or {}).get("actionableSignals", []))

        paragraphs = []

        if live_source == "unavailable":
            paragraphs.append(
                "I could not load live market data for this answer, so I do not have a fresh market read right now."
            )
            if live_error:
                paragraphs.append(f"Live data error: {live_error}")
        elif target_signal is not None:
            paragraphs.append(
                f"{target_signal['productId']} is currently a {target_signal['signal_name']} setup at "
                f"{self._format_price(target_signal.get('close'))} with "
                f"{self._format_percent(target_signal.get('confidence'))} confidence from the latest "
                f"{int((live_snapshot or {}).get('granularitySeconds', 0) or 0) // 60} minute candle cadence."
            )
            paragraphs.append(str(target_signal.get("signalChat", "")))
            paragraphs.append(self._build_tactical_note(target_signal))
        elif live_snapshot is not None:
            actionable_count = int(market_summary.get("actionableSignals") or 0)
            total_signals = int(market_summary.get("totalSignals") or 0)
            if total_signals <= 0:
                paragraphs.append(
                    "Live market overview: there are no published trade-ready signals right now. "
                    "The engine is keeping candidates on the internal watchlist until a BUY appears "
                    "or an open trade needs HOLD, TAKE_PROFIT, or LOSS management."
                )
            else:
                lead_pair = primary_signal.get("productId", "the lead pair")
                lead_signal = primary_signal.get("signal_name", "HOLD")
                lead_confidence = self._format_percent(primary_signal.get("confidence"))
                paragraphs.append(
                    f"Live market overview: {actionable_count} actionable setups across {total_signals} tracked pairs. "
                    f"The lead signal is {lead_pair} with a {lead_signal} call at {lead_confidence} confidence."
                )

            if actionable_signals:
                top_lines = [
                    f"{signal_summary['productId']} {signal_summary['signal_name']} ({self._format_percent(signal_summary.get('confidence'))})"
                    for signal_summary in actionable_signals[:3]
                ]
                paragraphs.append("Top live setups: " + ", ".join(top_lines) + ".")
            else:
                paragraphs.append(
                    "The live engine is mostly neutral right now, so the model is not surfacing strong spot entries."
                )

        if model_summary.get("status") == "ready":
            lifecycle = model_summary.get("lifecycle", {})
            if lifecycle.get("retrainingDue"):
                paragraphs.append(
                    "Operational note: the currently deployed model artifact is due for retraining, "
                    "so treat the live read as useful but not freshly revalidated."
                )
            elif model_summary.get("trainingMetrics", {}).get("balancedAccuracy") is not None:
                paragraphs.append(
                    "The deployed model's last recorded balanced accuracy was "
                    f"{self._format_percent(model_summary['trainingMetrics']['balancedAccuracy'])}."
                )

        if retrieval.get("messages"):
            last_recalled_message = retrieval["messages"][-1]["content"]
            paragraphs.append(f"From earlier in this session, the most relevant prior thread was: {last_recalled_message}")

        if retrieval.get("signals") and target_signal is None:
            recalled_products = ", ".join(
                signal_summary["productId"]
                for signal_summary in retrieval["signals"][:3]
            )
            paragraphs.append(f"The most relevant live pairs for your question were: {recalled_products}.")

        if retrieval.get("knowledge"):
            knowledge_items = retrieval["knowledge"][:2]
            knowledge_lines = []
            for knowledge_item in knowledge_items:
                source_title = str(knowledge_item.get("title") or "External source")
                source_uri = str(knowledge_item.get("sourceUri") or "").strip()
                snippet = self._trim_text(str(knowledge_item.get("snippet") or ""), max_length=220)
                source_label = source_title if not source_uri else f"{source_title} ({source_uri})"
                if snippet:
                    knowledge_lines.append(f"{source_label}: {snippet}")
                else:
                    knowledge_lines.append(source_label)

            if knowledge_lines:
                paragraphs.append(
                    "External retrieval context: " + " ".join(knowledge_lines)
                )

        if any(keyword in question.lower() for keyword in ("risk", "safe", "danger", "protect")):
            paragraphs.append(
                "Risk framing: use the live signal as decision support, not as an execution guarantee. "
                "Position sizing, stops, and liquidity still need a separate risk rule."
            )

        if live_source == "cached":
            paragraphs.append(
                "Live refresh was unavailable for this reply, so the answer fell back to the latest cached signal snapshot."
            )

        return "\n\n".join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())

    def _build_tactical_note(self, signal_summary: Dict[str, Any]) -> str:
        """Add one concise tactical interpretation for the current live signal."""

        signal_name = str(signal_summary.get("signal_name", "HOLD"))

        if signal_name == "BUY":
            return (
                "Tactical read: momentum is constructive enough for a spot entry watch, but confirmation still "
                "depends on the next candle holding strength rather than immediately reversing."
            )

        if signal_name == "TAKE_PROFIT":
            return (
                "Tactical read: the model is not calling for a short here. It is warning that a recent long "
                "position may be better trimmed or de-risked."
            )

        if signal_name == "LOSS":
            return (
                "Tactical read: the trade has moved against the plan enough that the system prefers cutting risk "
                "instead of waiting for a rebound."
            )

        return (
            "Tactical read: the model does not currently see a strong directional edge, so patience is more "
            "consistent with the latest feature state."
        )

    def _extract_requested_product_id(self, question: str) -> str | None:
        """Infer a Coinbase-style product id from the user's question when possible."""

        explicit_match = re.search(r"\b([A-Z]{2,10}-[A-Z]{2,10})\b", question.upper())
        if explicit_match:
            return explicit_match.group(1).upper()

        upper_question = question.upper()
        for candidate in self.config.live_product_ids:
            base_currency = str(candidate).split("-")[0].upper()
            if re.search(rf"\b{re.escape(base_currency)}\b", upper_question):
                return str(candidate).upper()

        return None

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Turn free text into a normalized token set for simple retrieval."""

        return {
            token
            for token in re.findall(r"[a-z0-9\-]+", str(text).lower())
            if len(token) > 1 and token not in COMMON_STOP_WORDS
        }

    @staticmethod
    def _format_percent(value: Any) -> str:
        """Format a decimal probability or return into a readable percentage."""

        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return "-"

    @staticmethod
    def _format_price(value: Any) -> str:
        """Format numeric prices consistently."""

        try:
            return f"${float(value):,.2f}"
        except (TypeError, ValueError):
            return "-"

    @staticmethod
    def _trim_text(text: str, max_length: int = 220) -> str:
        """Trim long retrieval snippets into concise assistant-ready text."""

        normalized_text = str(text).strip()
        if len(normalized_text) <= max_length:
            return normalized_text

        return normalized_text[: max_length - 3].rstrip() + "..."
