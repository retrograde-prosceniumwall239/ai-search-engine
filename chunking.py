"""
chunking.py
===========
Splits extracted document text into overlapping chunks suitable for
embedding.

Why chunk at all?
-----------------
Embedding models work best on short, semantically coherent passages.
Splitting a document into overlapping chunks:
  1. Keeps each vector focused on one idea, improving retrieval precision.
  2. Lets us cite the exact passage a search result came from.
  3. Avoids exceeding the embedding model's input token limit.

We use LangChain's ``RecursiveCharacterTextSplitter``, which tries to
split on paragraph breaks first, then sentences, then words -- only
falling back to a hard character cut if nothing else fits.
"""

from __future__ import annotations

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from logger import get_logger
from models import DocumentChunk

logger = get_logger(__name__)


def chunk_text(
    text: str,
    document_id: str,
    base_metadata: dict | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Split ``text`` into overlapping ``DocumentChunk`` objects.

    Args:
        text: Full extracted text of the document.
        document_id: The parent document's unique ID.
        base_metadata: Metadata to attach to every chunk (e.g. filename).
        chunk_size: Max characters per chunk (defaults to app setting).
        chunk_overlap: Overlap between consecutive chunks (defaults to app setting).

    Returns:
        A list of ``DocumentChunk`` in document order.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    base_metadata = base_metadata or {}

    if not text or not text.strip():
        logger.warning("chunk_text called with empty text for document_id=%s", document_id)
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)
    chunks: list[DocumentChunk] = []

    for index, raw_chunk in enumerate(raw_chunks):
        cleaned = raw_chunk.strip()
        if not cleaned:
            continue
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}_{uuid.uuid4().hex[:8]}",
                document_id=document_id,
                text=cleaned,
                chunk_index=index,
                metadata={**base_metadata, "chunk_index": index},
            )
        )

    logger.info(
        "Chunked document_id=%s into %d chunks (chunk_size=%d, overlap=%d)",
        document_id,
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
