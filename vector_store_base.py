"""
vector_store_base.py
=====================
Defines the common interface every vector database adapter must implement.

What is a vector database?
---------------------------
A vector database stores embeddings (numeric vectors) alongside their
original text and metadata, and provides fast "nearest neighbor" search:
given a query vector, it returns the stored vectors that are closest to
it (by cosine similarity, dot product, or Euclidean distance -- the
provider decides). This is the core building block of semantic search.

Why an adapter pattern?
-------------------------
Chroma, Pinecone, and Qdrant each have different SDKs and APIs. By
defining one abstract interface (``VectorStoreAdapter``) and implementing
it three times, the rest of the application (the search engine, the API
routes) never needs to know which provider is active. Switching providers
is a one-line configuration change (``VECTOR_DB_PROVIDER`` in ``.env``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from models import DocumentChunk, SearchResultItem


class VectorStoreAdapter(ABC):
    """Abstract interface implemented by every vector database adapter."""

    provider_name: str = "base"

    @abstractmethod
    def upsert_chunks(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        """Insert or update chunks and their embeddings in the vector store."""
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResultItem]:
        """Return the ``top_k`` chunks most similar to ``query_embedding``.

        Args:
            query_embedding: The embedding vector of the search query.
            top_k: Maximum number of results to return.
            metadata_filter: Optional exact-match metadata filter, e.g.
                ``{"file_type": "pdf"}``, used for metadata / hybrid search.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to ``document_id`` from the store."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the underlying vector database is reachable."""
        raise NotImplementedError
