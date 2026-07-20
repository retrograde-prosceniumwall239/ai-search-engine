"""
vector_store_pinecone.py
==========================
Vector store adapter for Pinecone.

Pinecone is a fully-managed, cloud-hosted vector database. Unlike Chroma,
it requires an API key and creates its index on Pinecone's infrastructure
rather than on local disk -- making it a good choice for production
deployments that need to scale beyond a single machine.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from pinecone import Pinecone, ServerlessSpec

from config import settings
from logger import get_logger
from models import DocumentChunk, SearchResultItem
from vector_store_base import VectorStoreAdapter

logger = get_logger(__name__)


class PineconeVectorStore(VectorStoreAdapter):
    provider_name = "pinecone"

    def __init__(self) -> None:
        if not settings.PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY is not set. Add it to your .env file to use "
                "the Pinecone provider."
            )

        self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._index_name = settings.PINECONE_INDEX_NAME
        self._ensure_index_exists()
        self._index = self._pc.Index(self._index_name)
        logger.info("Pinecone index '%s' ready", self._index_name)

    def _ensure_index_exists(self) -> None:
        existing = [idx["name"] for idx in self._pc.list_indexes()]
        if self._index_name in existing:
            return

        logger.info("Creating Pinecone index '%s'...", self._index_name)
        self._pc.create_index(
            name=self._index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION,
            ),
        )
        # Wait for the index to become ready before using it.
        while not self._pc.describe_index(self._index_name).status["ready"]:
            time.sleep(1)

    def upsert_chunks(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        if not chunks:
            return

        vectors = [
            {
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": {
                    **_flatten_metadata(chunk.metadata),
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                },
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Pinecone recommends batching upserts in groups of ~100.
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self._index.upsert(vectors=vectors[i : i + batch_size])

        logger.info("Upserted %d chunks into Pinecone", len(chunks))

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResultItem]:
        response = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter or None,
        )

        items: list[SearchResultItem] = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {}) or {}
            items.append(
                SearchResultItem(
                    chunk_id=match["id"],
                    document_id=metadata.get("document_id", "unknown"),
                    filename=metadata.get("filename", "unknown"),
                    text=metadata.get("text", ""),
                    score=round(float(match.get("score", 0.0)), 4),
                    metadata=metadata,
                )
            )
        return items

    def delete_document(self, document_id: str) -> None:
        self._index.delete(filter={"document_id": {"$eq": document_id}})
        logger.info("Deleted chunks for document_id=%s from Pinecone", document_id)

    def health_check(self) -> bool:
        try:
            self._index.describe_index_stats()
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Pinecone health check failed")
            return False


def _flatten_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Pinecone metadata values must be str, number, bool, or list of str."""
    flat: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat
