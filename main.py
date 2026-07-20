"""
main.py
=======
FastAPI application entrypoint for the AI Search Engine.

Run locally with:
    uvicorn main:app --reload

Or simply double-click "Start App.bat" (Windows) / "Start App (Mac).command"
(macOS), which handle environment setup automatically.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import document_processor
import search_engine
from config import settings
from database import db
from logger import get_logger
from models import (
    CompareRequest,
    CompareResponse,
    DocumentListResponse,
    HealthResponse,
    SearchHistoryItem,
    SearchRequest,
    SearchResponse,
    SettingsUpdateRequest,
    UploadResponse,
)

logger = get_logger(__name__)

app = FastAPI(
    title="AI Search Engine",
    description=(
        "A production-ready semantic search engine demonstrating embeddings, "
        "retrieval-augmented generation, and multi-provider vector database "
        "support (Chroma, Pinecone, Qdrant)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
async def on_startup() -> None:
    settings.ensure_directories()
    logger.info("AI Search Engine starting up (provider=%s)", settings.VECTOR_DB_PROVIDER)


# ----------------------------------------------------------------------
# Frontend
# ----------------------------------------------------------------------


@app.get("/")
async def serve_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vector_db_provider=settings.VECTOR_DB_PROVIDER,
        openai_configured=bool(settings.OPENAI_API_KEY),
    )


# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not document_processor.is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type. Supported: "
                f"{', '.join(sorted(document_processor.SUPPORTED_EXTENSIONS))}"
            ),
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    temp_name = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = upload_dir / temp_name

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = temp_path.stat().st_size

        record = search_engine.index_document(
            file_path=str(temp_path),
            filename=file.filename,
            file_size_bytes=file_size,
        )

        return UploadResponse(
            document_id=record.document_id,
            filename=record.filename,
            status=record.status,
            chunk_count=record.chunk_count,
            vector_db_provider=record.vector_db_provider,
            message=f"Document indexed successfully into {record.chunk_count} chunks.",
        )
    except search_engine.SearchEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        # Uploaded originals are intentionally not retained long-term;
        # only the extracted chunks/embeddings persist in the vector store.
        temp_path.unlink(missing_ok=True)


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    documents = db.list_documents()
    return DocumentListResponse(documents=documents, total=len(documents))


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str) -> dict:
    try:
        search_engine.delete_document(document_id)
    except search_engine.SearchEngineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"message": f"Document '{document_id}' deleted."}


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    provider = request.provider.value if request.provider else settings.VECTOR_DB_PROVIDER
    try:
        results, answer, latency_ms = search_engine.search(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            provider=provider,
            metadata_filter=request.metadata_filter,
            generate_answer=request.generate_answer,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        provider=provider,
        results=results,
        answer=answer,
        latency_ms=latency_ms,
    )


@app.post("/api/search/stream")
async def search_stream(request: SearchRequest):
    """Streaming variant: retrieves context, then streams the LLM's answer
    token-by-token using Server-Sent-Events-style chunked text."""
    provider = request.provider.value if request.provider else settings.VECTOR_DB_PROVIDER

    results, _, _ = search_engine.search(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
        provider=provider,
        metadata_filter=request.metadata_filter,
        generate_answer=False,
    )

    def token_stream():
        for token in search_engine.generate_answer_stream(
            query=request.query, context_chunks=[r.text for r in results]
        ):
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.post("/api/compare", response_model=CompareResponse)
async def compare(request: CompareRequest) -> CompareResponse:
    groups = search_engine.compare_providers(
        query=request.query,
        top_k=request.top_k,
        providers=[p.value for p in request.providers],
    )
    return CompareResponse(query=request.query, groups=groups)


@app.get("/api/search/history", response_model=list[SearchHistoryItem])
async def search_history(limit: int = 50) -> list[SearchHistoryItem]:
    return db.get_search_history(limit=limit)


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------


@app.get("/api/settings")
async def get_settings_view() -> dict:
    return {
        "vector_db_provider": settings.VECTOR_DB_PROVIDER,
        "openai_embedding_model": settings.OPENAI_EMBEDDING_MODEL,
        "openai_chat_model": settings.OPENAI_CHAT_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "top_k_results": settings.TOP_K_RESULTS,
        "available_providers": ["chroma", "pinecone", "qdrant"],
    }


@app.post("/api/settings/provider")
async def update_provider(request: SettingsUpdateRequest) -> dict:
    # Runtime provider switching is scoped per-request via the `provider`
    # field on /api/search; this endpoint validates connectivity and
    # reports back so the UI can confirm the switch succeeded.
    from vector_store_factory import get_vector_store

    try:
        store = get_vector_store(request.vector_db_provider.value)
        healthy = store.health_check()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not healthy:
        raise HTTPException(
            status_code=503,
            detail=f"Provider '{request.vector_db_provider.value}' is not reachable.",
        )

    return {
        "message": f"Provider '{request.vector_db_provider.value}' is configured and reachable.",
        "note": (
            "Set VECTOR_DB_PROVIDER in your .env file and restart the app to change "
            "the default provider permanently, or pass 'provider' per-request."
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
