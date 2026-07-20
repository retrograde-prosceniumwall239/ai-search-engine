# AI Search Engine

A production-ready semantic search engine built with FastAPI, the OpenAI API, and
your choice of three vector databases - **Chroma**, **Pinecone**, and **Qdrant**.
Upload PDF, TXT, or Markdown documents; the app extracts, chunks, and embeds
their text, stores the vectors, and lets you search them by *meaning* rather
than exact keywords, with cited, LLM-generated answers.

This project is also a hands-on teaching tool: every module is documented to
explain **why** each step of the pipeline exists, not just what it does.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Concepts](#core-concepts)
4. [Folder Structure](#folder-structure)
5. [Installation](#installation)
6. [Visual Studio Code Setup](#visual-studio-code-setup)
7. [Running the Application](#running-the-application)
8. [Examples](#examples)
9. [Screenshots](#screenshots)
10. [Deployment](#deployment)
11. [Troubleshooting](#troubleshooting)
12. [Future Improvements](#future-improvements)

---

## Project Overview

The app implements a full **Retrieval-Augmented Generation (RAG)** pipeline:

- **Upload** documents through a web dashboard (drag-and-drop or file picker).
- **Extract** text from PDF, TXT, or Markdown files.
- **Chunk** the text into overlapping passages sized for embedding.
- **Embed** each chunk into a vector using OpenAI's embeddings API.
- **Store** the vectors in a vector database of your choice.
- **Search** semantically, by similarity, by metadata filter, or hybrid.
- **Compare** the same query across all three vector databases side by side.
- **Generate** a cited, natural-language answer from the retrieved passages.
- **Track** search history and document status in a lightweight SQLite database.

## Architecture

```
 Document Upload
       |
       v
 Text Extraction        (document_processor.py)
       |
       v
 Chunking                (chunking.py)
       |
       v
 Embedding Generation    (embeddings.py)
       |
       v
 Vector Database         (vector_store_chroma.py / _pinecone.py / _qdrant.py)
       |
       v
 Retriever               (search_engine.py)
       |
       v
 Similarity Search       (vector_store_*.py -> similarity_search)
       |
       v
 LLM                     (embeddings.py -> generate_answer)
       |
       v
 Final Response           (returned via main.py -> FastAPI JSON / stream)
```

**Design pattern:** every vector database is implemented as an **adapter**
that conforms to the `VectorStoreAdapter` interface in
`vector_store_base.py`. `vector_store_factory.py` picks the right adapter at
runtime based on `VECTOR_DB_PROVIDER` in your `.env` file (or a per-request
override). The rest of the app - the FastAPI routes, the search engine,
the UI - never needs to know which provider is active.

## Core Concepts

**What are embeddings?**
An embedding is a list of numbers (a vector) that represents the *meaning*
of a piece of text. Texts with similar meaning produce vectors that sit
close together in that high-dimensional space - this is the mathematical
foundation that makes semantic search possible.

**What are vector databases?**
A vector database stores embeddings alongside their original text and
metadata, and provides fast "nearest neighbor" search: given a query
vector, it returns the stored vectors closest to it.

- **Chroma** - an open-source, embedded vector database that persists to a
  local folder on disk. No API key or external service required, which
  makes it the default for local development in this project.
- **Pinecone** - a fully-managed, cloud-hosted vector database. Requires an
  API key; a good fit for production deployments that need to scale.
- **Qdrant** - an open-source vector database that can run locally via
  Docker or as a managed cloud service, with strong payload (metadata)
  filtering support.

**Semantic search vs. similarity search.**
At the mechanical level they're the same operation: comparing the query's
embedding vector against stored vectors and returning the closest matches.
"Semantic search" is the user-facing name for the capability; "similarity
search" is the underlying vector-math operation that powers it. This app
exposes both terms in the UI because learners frequently ask what
distinguishes them.

**Metadata filtering.**
Beyond vector similarity, each chunk carries metadata (filename, file
type, chunk index). Metadata filtering narrows a search to only the
chunks matching exact criteria - e.g. "only search PDFs" - which can be
combined with vector similarity for **hybrid search**.

## Folder Structure

The repository intentionally keeps almost everything in the root directory
so it's easy to browse on GitHub - only `static/`, `templates/`, `tests/`,
and `docs/` are separate folders.

```
ai-search-engine/
├── main.py                      FastAPI app & all HTTP routes
├── config.py                    Environment-based settings
├── logger.py                    Shared logging setup
├── models.py                    Pydantic v2 schemas
├── database.py                  SQLite persistence (documents, history)
├── document_processor.py        PDF / TXT / MD text extraction
├── chunking.py                  Text splitting for embeddings
├── embeddings.py                OpenAI embeddings + chat completion
├── search_engine.py             RAG pipeline orchestration
├── vector_store_base.py         Adapter interface
├── vector_store_chroma.py       Chroma adapter
├── vector_store_pinecone.py     Pinecone adapter
├── vector_store_qdrant.py       Qdrant adapter
├── vector_store_factory.py      Picks the active adapter
├── requirements.txt
├── .env.example
├── .gitignore
├── Start App.bat                 Windows startup script
├── Start App (Mac).command       macOS startup script
├── static/
│   ├── style.css
│   └── app.js
├── templates/
│   └── index.html
├── tests/
│   ├── test_chunking.py
│   └── test_document_processor.py
└── docs/
    └── screenshots/              (placeholders - see below)
```

## Installation

### Python Installation

Install **Python 3.12+**:

- **Windows:** download from [python.org](https://www.python.org/downloads/)
  and check "Add Python to PATH" during setup.
- **macOS:** `brew install python` (or download from python.org).

Verify with:

```bash
python3 --version
```

### Clone the repository

```bash
git clone https://github.com/your-username/ai-search-engine.git
cd ai-search-engine
```

### Virtual Environment

The startup scripts create and activate this automatically. To do it
manually:

```bash
python3 -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### API Keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

At minimum, set `OPENAI_API_KEY`. If you plan to use Pinecone or Qdrant,
also set `PINECONE_API_KEY` or `QDRANT_URL` / `QDRANT_API_KEY`. Chroma
requires no keys - it runs locally out of the box.

## Visual Studio Code Setup

1. Open the project folder in VS Code (`File > Open Folder...`).
2. Install the **Python** extension (Microsoft) if prompted.
3. Select the virtual environment as your interpreter:
   `Ctrl/Cmd+Shift+P` -> `Python: Select Interpreter` -> choose
   `./venv/bin/python` (macOS) or `.\venv\Scripts\python.exe` (Windows).
4. Open a terminal in VS Code (`` Ctrl+` ``) and run the app (see below).
5. Recommended extensions: **Python**, **Pylance**, **Even Better TOML**.

## Running the Application

**Option A - Startup scripts (recommended for first run):**

- Windows: double-click `Start App.bat`
- macOS: double-click `Start App (Mac).command`
  (if macOS blocks it, right-click -> Open, or run
  `chmod +x "Start App (Mac).command"` once in Terminal)

Both scripts verify Python, create/activate a virtual environment, install
dependencies, verify your `.env` file, and launch the server.

**Option B - Manual:**

```bash
source venv/bin/activate # or venv\Scripts\activate on Windows
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

## Examples

**Upload a document** - go to the *Documents* tab, drop in a PDF/TXT/MD
file. It's extracted, chunked, embedded, and indexed automatically.

**Search semantically:**

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the termination clauses?", "mode": "semantic", "top_k": 5}'
```

**Compare providers:**

```bash
curl -X POST http://127.0.0.1:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"query": "refund policy", "providers": ["chroma", "pinecone", "qdrant"]}'
```

**Switch the default vector database:** set `VECTOR_DB_PROVIDER=qdrant` (or
`pinecone`) in `.env` and restart - or select a provider per-search in the
UI without restarting.

## Screenshots

> Screenshots are omitted from this repository to keep it lightweight.
> Add your own to `docs/screenshots/` after running the app locally:
> `docs/screenshots/search.png`, `docs/screenshots/documents.png`,
> `docs/screenshots/compare.png`.

## Deployment

This app is a standard ASGI application and can be deployed anywhere that
runs Python:

- **Render / Railway / Fly.io** - point the build at `pip install -r
  requirements.txt` and the start command at
  `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **Docker** - build a minimal image with a `python:3.12-slim` base,
  copy the repo, `pip install -r requirements.txt`, and run the same
  `uvicorn` command.
- **Vector database in production** - Chroma's local persistence directory
  does not survive ephemeral filesystems on most PaaS platforms; use
  Pinecone or a hosted Qdrant instance for production deployments.
- Always set environment variables (API keys, `VECTOR_DB_PROVIDER`)
  through your platform's secret manager rather than committing `.env`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Install Python 3.12+ and ensure it's on your PATH. |
| `ModuleNotFoundError` on startup | Activate the virtual environment, then re-run `pip install -r requirements.txt`. |
| `OPENAI_API_KEY is not set` warning | Add your key to `.env`, then restart the app. |
| PDF upload extracts no text | The PDF may be a scanned image with no text layer; OCR is not included in this project. |
| Pinecone errors on startup | Confirm `PINECONE_API_KEY` is set and the account has index-creation permission. |
| Qdrant connection refused | Start a local Qdrant instance (`docker run -p 6333:6333 qdrant/qdrant`) or point `QDRANT_URL` at a running instance. |
| Port 8000 already in use | Change `APP_PORT` in `.env`, or stop the process using that port. |
| macOS blocks the `.command` script | Right-click the file -> Open, or run `chmod +x "Start App (Mac).command"` in Terminal. |

## Future Improvements

- OCR support for scanned/image-only PDFs.
- User authentication and per-user document isolation.
- Additional file formats (DOCX, HTML, CSV).
- Reranking retrieved chunks with a cross-encoder before generation.
- BM25 keyword search fused with vector search for true hybrid retrieval.
- Background job queue for large-file indexing instead of synchronous upload.
- Automated CI (lint + tests) via GitHub Actions.

---

Built as a learning-oriented reference implementation of RAG and
multi-provider vector search. Contributions and issues are welcome.
