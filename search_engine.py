"""
search_engine.py
==================
Orchestrates the full Retrieval-Augmented Generation (RAG) pipeline:

    Document Upload -> Text Extraction -> Chunking -> Embedding Generation
    -> Vector Database -> Retriever -> Similarity Search -> LLM -> Final Response

This module ties together ``document_processor``, ``chunking``,
``embeddings``, and the vector store adapters into two high-level
operations: ``index_document`` (the write path) and ``search`` /
``compare_providers`` (the read path).
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import chunking
import document_processor
import embeddings
from config import settings
from database import db
from logger import get_logger
from models import (
    CompareResultGroup,
    DocumentRecord,
    DocumentStatus,
    SearchMode,
    SearchResultItem,
)
from vector_store_factory import get_vector_store

logger = get_logger(__name__)


class SearchEngineError(Exception):
    """Raised when the search/index pipeline fails in a user-facing way."""


# ----------------------------------------------------------------------
# Indexing (write path)
# ----------------------------------------------------------------------


def index_document(
    file_path: str,
    filename: str,
    file_size_bytes: int,
    provider: Optional[str] = None,
) -> DocumentRecord:
    """Run a file through the full indexing pipeline and persist metadata.

    Steps: extract text -> chunk -> embed -> upsert into vector store.
    """
    provider = provider or settings.VECTOR_DB_PROVIDER
    document_id = uuid.uuid4().hex
    file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"

    record = DocumentRecord(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        file_size_bytes=file_size_bytes,
        status=DocumentStatus.PROCESSING,
        chunk_count=0,
        vector_db_provider=provider,
        uploaded_at=datetime.now(timezone.utc),
    )
    db.create_document(record)

    try:
        # 1. Text Extraction
        text = document_processor.extract_text(file_path, filename)

        # 2. Chunking
        chunks = chunking.chunk_text(
            text=text,
            document_id=document_id,
            base_metadata={"filename": filename, "file_type": file_type},
        )
        if not chunks:
            raise SearchEngineError("Document produced no usable text chunks.")

        # 3. Embedding Generation
        chunk_texts = [c.text for c in chunks]
        vectors = embeddings.embed_texts(chunk_texts)

        # 4. Vector Database (store)
        store = get_vector_store(provider)
        store.upsert_chunks(chunks, vectors)

        db.update_document_status(
            document_id, DocumentStatus.INDEXED, chunk_count=len(chunks)
        )
        record.status = DocumentStatus.INDEXED
        record.chunk_count = len(chunks)
        logger.info(
            "Indexed document '%s' (%s) into %s: %d chunks",
            filename,
            document_id,
            provider,
            len(chunks),
        )
        return record

    except Exception as exc:  # noqa: BLE001
        logger.exception("Indexing failed for '%s'", filename)
        db.update_document_status(
            document_id, DocumentStatus.FAILED, error_message=str(exc)
        )
        raise SearchEngineError(str(exc)) from exc


def delete_document(document_id: str) -> None:
    """Remove a document's chunks from its vector store and delete its metadata."""
    record = db.get_document(document_id)
    if not record:
        raise SearchEngineError(f"Document '{document_id}' not found.")

    store = get_vector_store(record.vector_db_provider)
    store.delete_document(document_id)
    db.delete_document(document_id)
    logger.info("Deleted document '%s'", document_id)


# ----------------------------------------------------------------------
# Retrieval (read path)
# ----------------------------------------------------------------------


def search(
    query: str,
    mode: SearchMode = SearchMode.SEMANTIC,
    top_k: int = 5,
    provider: Optional[str] = None,
    metadata_filter: Optional[dict[str, Any]] = None,
    generate_answer: bool = True,
) -> tuple[list[SearchResultItem], Optional[str], float]:
    """Run the retrieval (and optional generation) steps of the pipeline.

    Returns:
        (results, answer, latency_ms)
    """
    start = time.perf_counter()
    provider = provider or settings.VECTOR_DB_PROVIDER
    store = get_vector_store(provider)

    # 5. Retriever + 6. Similarity Search
    query_vector = embeddings.embed_query(query)

    # "metadata" mode searches with a filter but no free-text similarity bias
    # beyond the query embedding; "hybrid" combines both signals; "semantic"
    # and "similarity" both perform a pure vector search (semantic search IS
    # similarity search over embeddings -- we expose both terms because
    # learners often ask what the difference is, and the answer is: none,
    # at the mechanical level, when no keyword/BM25 signal is layered on).
    effective_filter = metadata_filter if mode in (SearchMode.METADATA, SearchMode.HYBRID) else None

    results = store.similarity_search(
        query_embedding=query_vector,
        top_k=top_k,
        metadata_filter=effective_filter,
    )

    # 7. LLM (Final Response)
    answer: Optional[str] = None
    if generate_answer:
        answer = embeddings.generate_answer(
            query=query, context_chunks=[r.text for r in results]
        )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    db.log_search(query=query, mode=mode.value, provider=provider, result_count=len(results))
    logger.info(
        "Search '%s' (mode=%s, provider=%s) -> %d results in %.2fms",
        query,
        mode.value,
        provider,
        len(results),
        latency_ms,
    )

    return results, answer, latency_ms


def compare_providers(
    query: str, top_k: int, providers: list[str]
) -> list[CompareResultGroup]:
    """Run the same query against multiple vector database providers.

    This directly demonstrates how different vector databases can return
    different results (and different latencies) for identical inputs --
    useful for the "Compare search results" feature.
    """
    query_vector = embeddings.embed_query(query)
    groups: list[CompareResultGroup] = []

    for provider in providers:
        start = time.perf_counter()
        try:
            store = get_vector_store(provider)
            results = store.similarity_search(query_embedding=query_vector, top_k=top_k)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            groups.append(
                CompareResultGroup(provider=provider, results=results, latency_ms=latency_ms)
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.exception("Compare failed for provider=%s", provider)
            groups.append(
                CompareResultGroup(
                    provider=provider, results=[], latency_ms=latency_ms, error=str(exc)
                )
            )

    return groups


def generate_answer_stream(query: str, context_chunks: list[str]):
    """Expose the streaming LLM generator for the streaming API endpoint."""
    return embeddings.generate_answer(query=query, context_chunks=context_chunks, stream=True)
