"""
database.py
============
Lightweight SQLite persistence layer.

SQLite stores application metadata that does NOT belong in a vector
database:
  * Document records (filename, status, chunk count, which vector DB
    provider indexed it, timestamps, errors).
  * Search history (for the "Search History" dashboard panel).

The actual vector embeddings live in Chroma / Pinecone / Qdrant -- SQLite
is only the lightweight "system of record" for metadata.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from config import settings
from logger import get_logger
from models import DocumentRecord, DocumentStatus, SearchHistoryItem

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id       TEXT PRIMARY KEY,
    filename          TEXT NOT NULL,
    file_type         TEXT NOT NULL,
    file_size_bytes   INTEGER NOT NULL,
    status            TEXT NOT NULL,
    chunk_count       INTEGER NOT NULL DEFAULT 0,
    vector_db_provider TEXT NOT NULL,
    uploaded_at       TEXT NOT NULL,
    indexed_at        TEXT,
    error_message     TEXT
);

CREATE TABLE IF NOT EXISTS search_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    query         TEXT NOT NULL,
    mode          TEXT NOT NULL,
    provider      TEXT NOT NULL,
    result_count  INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_history_created_at ON search_history(created_at);
"""


class Database:
    """Thin wrapper around a single SQLite database file."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.SQLITE_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.info("SQLite schema initialized at %s", self.db_path)

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def create_document(self, record: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, filename, file_type, file_size_bytes,
                    status, chunk_count, vector_db_provider,
                    uploaded_at, indexed_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.document_id,
                    record.filename,
                    record.file_type,
                    record.file_size_bytes,
                    record.status.value,
                    record.chunk_count,
                    record.vector_db_provider,
                    record.uploaded_at.isoformat(),
                    record.indexed_at.isoformat() if record.indexed_at else None,
                    record.error_message,
                ),
            )

    def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        indexed_at = (
            datetime.now(timezone.utc).isoformat()
            if status == DocumentStatus.INDEXED
            else None
        )
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = ?,
                    chunk_count = COALESCE(?, chunk_count),
                    error_message = ?,
                    indexed_at = COALESCE(?, indexed_at)
                WHERE document_id = ?
                """,
                (status.value, chunk_count, error_message, indexed_at, document_id),
            )

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def delete_document(self, document_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
        data: dict[str, Any] = dict(row)
        return DocumentRecord(
            document_id=data["document_id"],
            filename=data["filename"],
            file_type=data["file_type"],
            file_size_bytes=data["file_size_bytes"],
            status=DocumentStatus(data["status"]),
            chunk_count=data["chunk_count"],
            vector_db_provider=data["vector_db_provider"],
            uploaded_at=datetime.fromisoformat(data["uploaded_at"]),
            indexed_at=(
                datetime.fromisoformat(data["indexed_at"])
                if data["indexed_at"]
                else None
            ),
            error_message=data["error_message"],
        )

    # ------------------------------------------------------------------
    # Search history
    # ------------------------------------------------------------------

    def log_search(
        self, query: str, mode: str, provider: str, result_count: int
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO search_history (query, mode, provider, result_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    query,
                    mode,
                    provider,
                    result_count,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_search_history(self, limit: int = 50) -> list[SearchHistoryItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM search_history ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            SearchHistoryItem(
                id=r["id"],
                query=r["query"],
                mode=r["mode"],
                provider=r["provider"],
                result_count=r["result_count"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]


# Module-level singleton, created lazily on first import use.
db = Database()
