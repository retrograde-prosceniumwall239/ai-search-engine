"""
vector_store_qdrant.py
========================
Vector store adapter for Qdrant.

Qdrant is an open-source vector database that can run either locally
(via Docker) or as a managed cloud service (Qdrant Cloud). It offers
strong support for metadata filtering (Qdrant calls this "payload
filtering"), making it a good example of hybrid search: combining a
vector similarity search with structured metadata constraints in a
single query.
"""

from __future__ import annotations

from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import settings
from logger import get_logger
from models import DocumentChunk, SearchResultItem
from vector_store_base import VectorStoreAdapter

logger = get_logger(__name__)


class QdrantVectorStore(VectorStoreAdapter):
    provider_name = "qdrant"

    def __init__(self) -> None:
        self._client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )
        self._collection_name = settings.QDRANT_COLLECTION_NAME
        self._ensure_collection_exists()
        logger.info("Qdrant collection '%s' ready", self._collection_name)

    def _ensure_collection_exists(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection_name in existing:
            return

        logger.info("Creating Qdrant collection '%s'...", self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        if not chunks:
            return

        points = [
            PointStruct(
                id=_stable_point_id(chunk.chunk_id),
                vector=embedding,
                payload={
                    **chunk.metadata,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        self._client.upsert(collection_name=self._collection_name, points=points)
        logger.info("Upserted %d chunks into Qdrant", len(chunks))

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResultItem]:
        qdrant_filter = _build_filter(metadata_filter)

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        items: list[SearchResultItem] = []
        for point in results:
            payload = point.payload or {}
            items.append(
                SearchResultItem(
                    chunk_id=payload.get("chunk_id", str(point.id)),
                    document_id=payload.get("document_id", "unknown"),
                    filename=payload.get("filename", "unknown"),
                    text=payload.get("text", ""),
                    score=round(float(point.score), 4),
                    metadata=payload,
                )
            )
        return items

    def delete_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
        logger.info("Deleted chunks for document_id=%s from Qdrant", document_id)

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Qdrant health check failed")
            return False


def _build_filter(metadata_filter: Optional[dict[str, Any]]) -> Optional[Filter]:
    if not metadata_filter:
        return None
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in metadata_filter.items()
    ]
    return Filter(must=conditions)


def _stable_point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be a UUID or unsigned int; derive a UUID from the chunk_id."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
