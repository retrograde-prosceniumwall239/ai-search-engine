"""
vector_store_factory.py
=========================
Factory responsible for instantiating the correct ``VectorStoreAdapter``
based on configuration, and caching adapters so each provider is only
initialized once per process.

This is the single place in the codebase that knows about all three
concrete adapter classes -- everything else depends only on the
``VectorStoreAdapter`` interface.
"""

from __future__ import annotations

from logger import get_logger
from vector_store_base import VectorStoreAdapter

logger = get_logger(__name__)

_adapter_cache: dict[str, VectorStoreAdapter] = {}


def get_vector_store(provider: str | None = None) -> VectorStoreAdapter:
    """Return a (cached) adapter instance for the given provider.

    Args:
        provider: One of "chroma", "pinecone", "qdrant". Defaults to the
            value of ``VECTOR_DB_PROVIDER`` in the current settings.
    """
    from config import settings  # local import avoids circular import at module load

    provider = (provider or settings.VECTOR_DB_PROVIDER).lower()

    if provider in _adapter_cache:
        return _adapter_cache[provider]

    logger.info("Initializing vector store adapter: %s", provider)

    if provider == "chroma":
        from vector_store_chroma import ChromaVectorStore

        adapter: VectorStoreAdapter = ChromaVectorStore()
    elif provider == "pinecone":
        from vector_store_pinecone import PineconeVectorStore

        adapter = PineconeVectorStore()
    elif provider == "qdrant":
        from vector_store_qdrant import QdrantVectorStore

        adapter = QdrantVectorStore()
    else:
        raise ValueError(
            f"Unknown VECTOR_DB_PROVIDER '{provider}'. "
            "Expected one of: chroma, pinecone, qdrant."
        )

    _adapter_cache[provider] = adapter
    return adapter


def clear_cache() -> None:
    """Clear cached adapters. Mainly useful for tests."""
    _adapter_cache.clear()
