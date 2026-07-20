"""
embeddings.py
=============
Thin wrapper around the OpenAI Python SDK for:
  1. Generating embeddings (turning text into vectors).
  2. Generating the final natural-language answer from retrieved context
     (the "LLM" step at the end of the RAG pipeline).

What is an embedding?
----------------------
An embedding is a fixed-length list of numbers (a vector) that represents
the *meaning* of a piece of text. Texts with similar meaning end up with
vectors that are close together in that high-dimensional space. This is
what makes "semantic search" possible: instead of matching exact keywords,
we compare the distance/angle between vectors.
"""

from __future__ import annotations

from openai import OpenAI

from config import settings
from logger import get_logger

logger = get_logger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a lazily-instantiated, shared OpenAI client."""
    global _client
    if _client is None:
        if not settings.OPENAI_API_KEY:
            logger.warning(
                "OPENAI_API_KEY is not set. Embedding and chat calls will fail "
                "until it is configured in your .env file."
            )
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    OpenAI's embeddings endpoint accepts a list of strings in a single
    request, which is far more efficient than one request per chunk.
    """
    if not texts:
        return []

    client = get_client()
    response = client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    # The API preserves input order in the output list.
    return [item.embedding for item in response.data]


def embed_query(query: str) -> list[float]:
    """Generate a single embedding vector for a search query."""
    return embed_texts([query])[0]


def generate_answer(
    query: str,
    context_chunks: list[str],
    stream: bool = False,
):
    """Generate a final natural-language answer grounded in retrieved chunks.

    Args:
        query: The user's original search query.
        context_chunks: The text of the top retrieved chunks, used as context.
        stream: If True, returns a generator yielding text deltas instead of
            a single string (used for the streaming HTTP endpoint).

    Returns:
        Either a full answer string, or a generator of text chunks if
        ``stream`` is True.
    """
    client = get_client()

    context_block = "\n\n---\n\n".join(
        f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    system_prompt = (
        "You are a helpful search assistant. Answer the user's question "
        "using ONLY the provided sources. Cite sources inline using "
        "[Source N] notation. If the sources don't contain the answer, "
        "say so clearly instead of guessing."
    )
    user_prompt = f"Sources:\n{context_block}\n\nQuestion: {query}"

    if not context_chunks:
        system_prompt = (
            "You are a helpful search assistant. No relevant sources were "
            "found for this query. Tell the user no matching documents "
            "were found and suggest they upload relevant documents first."
        )

    if stream:
        return _stream_completion(client, system_prompt, user_prompt)

    completion = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content or ""


def _stream_completion(client: OpenAI, system_prompt: str, user_prompt: str):
    """Yield text deltas from a streaming chat completion."""
    stream = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
