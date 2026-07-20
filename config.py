"""
config.py
=========
Centralized application configuration.

All configuration values are loaded from environment variables (typically
via a local ``.env`` file). This keeps secrets out of source control and
allows the app's behavior (e.g. which vector database to use) to be changed
without touching any code.

Usage
-----
    from config import settings

    print(settings.VECTOR_DB_PROVIDER)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ---- OpenAI ----
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"

    # ---- Vector Database Selection ----
    VECTOR_DB_PROVIDER: Literal["chroma", "pinecone", "qdrant"] = "chroma"

    # ---- Chroma ----
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION_NAME: str = "documents"

    # ---- Pinecone ----
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "ai-search-engine"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # ---- Qdrant ----
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "documents"

    # ---- App Settings ----
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000
    APP_ENV: Literal["development", "production"] = "development"
    LOG_LEVEL: str = "INFO"

    # ---- Chunking ----
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # ---- Retrieval ----
    TOP_K_RESULTS: int = 5

    # ---- Storage ----
    SQLITE_DB_PATH: str = "./app_data.db"
    UPLOAD_DIR: str = "./uploads"

    # ---- Embedding Dimensions (per OpenAI model) ----
    EMBEDDING_DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    @property
    def embedding_dimension(self) -> int:
        """Return the vector dimension for the configured embedding model."""
        return self.EMBEDDING_DIMENSIONS.get(self.OPENAI_EMBEDDING_MODEL, 1536)

    def ensure_directories(self) -> None:
        """Create local directories the app depends on, if they don't exist."""
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


settings = get_settings()
