"""Unit tests for chunking.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunking import chunk_text


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("", document_id="doc1") == []
    assert chunk_text("   ", document_id="doc1") == []


def test_chunk_text_short_text_returns_single_chunk():
    text = "This is a short sentence that fits in one chunk."
    chunks = chunk_text(text, document_id="doc1", chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].document_id == "doc1"
    assert chunks[0].chunk_index == 0


def test_chunk_text_long_text_splits_into_multiple_chunks():
    text = "Paragraph one. " * 200  # long enough to force multiple chunks
    chunks = chunk_text(text, document_id="doc2", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.document_id == "doc2"
        assert chunk.chunk_id.startswith("doc2_")


def test_chunk_text_includes_base_metadata():
    chunks = chunk_text(
        "Some text to embed.",
        document_id="doc3",
        base_metadata={"filename": "example.txt"},
    )
    assert chunks[0].metadata["filename"] == "example.txt"
    assert "chunk_index" in chunks[0].metadata
