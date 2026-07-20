"""
models.py
=========
Pydantic v2 data models shared across the application.

These models describe:
  * The shape of documents and chunks as they move through the pipeline.
  * The request/response schemas for the FastAPI HTTP endpoints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------


class VectorDBProvider(str, Enum):
    CHROMA = "chroma"
    PINECONE = "pinecone"
    QDRANT = "qdrant"


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    SIMILARITY = "similarity"
    METADATA = "metadata"
    HYBRID = "hybrid"


# ----------------------------------------------------------------------
# Core domain models
# ----------------------------------------------------------------------


class DocumentChunk(BaseModel):
    """A single chunk of text extracted from a source document."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    """Metadata record for an uploaded document, stored in SQLite."""

    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    status: DocumentStatus
    chunk_count: int = 0
    vector_db_provider: str
    uploaded_at: datetime
    indexed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class SearchResultItem(BaseModel):
    """A single retrieved chunk returned from a similarity search."""

    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------------------------
# API request schemas
# ----------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: SearchMode = SearchMode.SEMANTIC
    top_k: int = Field(default=5, ge=1, le=50)
    provider: Optional[VectorDBProvider] = None
    metadata_filter: Optional[dict[str, Any]] = None
    generate_answer: bool = True

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query must not be empty or whitespace-only")
        return stripped


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    providers: list[VectorDBProvider] = Field(
        default_factory=lambda: [
            VectorDBProvider.CHROMA,
            VectorDBProvider.PINECONE,
            VectorDBProvider.QDRANT,
        ]
    )


class SettingsUpdateRequest(BaseModel):
    vector_db_provider: VectorDBProvider


# ----------------------------------------------------------------------
# API response schemas
# ----------------------------------------------------------------------


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    chunk_count: int
    vector_db_provider: str
    message: str


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    provider: str
    results: list[SearchResultItem]
    answer: Optional[str] = None
    latency_ms: float


class CompareResultGroup(BaseModel):
    provider: str
    results: list[SearchResultItem]
    latency_ms: float
    error: Optional[str] = None


class CompareResponse(BaseModel):
    query: str
    groups: list[CompareResultGroup]


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]
    total: int


class HealthResponse(BaseModel):
    status: str
    vector_db_provider: str
    openai_configured: bool


class SearchHistoryItem(BaseModel):
    id: int
    query: str
    mode: str
    provider: str
    result_count: int
    created_at: datetime
