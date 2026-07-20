"""
vector_store_chroma.py
=======================
Vector store adapter for ChromaDB.

Chroma is an open-source, embedded vector database that runs locally with
zero external infrastructure -- it persists to a folder on disk
(``CHROMA_PERSIST_DIR``). This makes it the best default choice for local
development and demos, since it requires no API keys or network access.
"""

from __future__ import annotations

from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from logger import get_logger
from models import DocumentChunk, SearchResultItem
from vector_store_base import VectorStoreAdapter

logger = get_logger(__name__)


class ChromaVectorStore(VectorStoreAdapter):
    provider_name = "chroma"

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma collection '%s' ready at '%s'",
            settings.CHROMA_COLLECTION_NAME,
            settings.CHROMA_PERSIST_DIR,
        )

    def upsert_chunks(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {**c.metadata, "document_id": c.document_id} for c in chunks
            ],
        )
        logger.info("Upserted %d chunks into Chroma", len(chunks))

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResultItem]:
        where = metadata_filter if metadata_filter else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        items: list[SearchResultItem] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            # Chroma returns cosine *distance*; convert to a similarity score in [0, 1].
            score = max(0.0, 1.0 - distance)
            items.append(
                SearchResultItem(
                    chunk_id=chunk_id,
                    document_id=metadata.get("document_id", "unknown"),
                    filename=metadata.get("filename", "unknown"),
                    text=text,
                    score=round(score, 4),
                    metadata=metadata,
                )
            )
        return items

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})
        logger.info("Deleted chunks for document_id=%s from Chroma", document_id)

    def health_check(self) -> bool:
        try:
            self._collection.count()
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Chroma health check failed")
            return False
