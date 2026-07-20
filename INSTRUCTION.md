# INSTRUCTION.md

## AI Search Engine - Complete Beginner's Installation Guide

Welcome! This guide assumes you have **never** used Python, Visual Studio
Code, Git, FastAPI, LangChain, Chroma, Pinecone, Qdrant, or the OpenAI API
before. Every step is spelled out in full - nothing is assumed.

Follow the sections in order. Each one builds on the last. By the end,
you will have a working AI Search Engine running on your own computer.

> **Time required:** roughly 30-60 minutes for a first-time setup.
> **Cost:** Python, VS Code, Git, and Chroma are all free. The OpenAI API
> requires a small amount of pay-as-you-go credit (a few cents will cover
> all the testing in this guide). Pinecone and Qdrant both offer free
> tiers that are enough for this project.

---

## Table of Contents

1. [Installing Python](#1-installing-python)
2. [Installing Visual Studio Code](#2-installing-visual-studio-code)
3. [Installing Git](#3-installing-git)
4. [Required VS Code Extensions](#4-required-vs-code-extensions)
5. [Opening the Project](#5-opening-the-project)
6. [Creating a Virtual Environment](#6-creating-a-virtual-environment)
7. [Activating the Virtual Environment](#7-activating-the-virtual-environment)
8. [Installing Dependencies](#8-installing-dependencies)
9. [Creating the .env File](#9-creating-the-env-file)
10. [Getting OpenAI API Keys](#10-getting-openai-api-keys)
11. [Creating Pinecone API Keys](#11-creating-pinecone-api-keys)
12. [Running Chroma Locally](#12-running-chroma-locally)
13. [Running Qdrant Locally (Optional)](#13-running-qdrant-locally-optional)
14. [Starting the Application](#14-starting-the-application)
15. [Uploading Documents](#15-uploading-documents)
16. [Creating Embeddings](#16-creating-embeddings)
17. [Indexing Documents](#17-indexing-documents)
18. [Running Semantic Search](#18-running-semantic-search)
19. [Switching Between Chroma, Pinecone, and Qdrant](#19-switching-between-chroma-pinecone-and-qdrant)
20. [Testing All Features](#20-testing-all-features)
21. [Common Errors](#21-common-errors)
22. [Troubleshooting](#22-troubleshooting)
23. [FAQ](#23-faq)
24. [Security Recommendations](#24-security-recommendations)
25. [Next Learning Steps](#25-next-learning-steps)

---

## 1. Installing Python

Python is the programming language this entire project is written in. You
need **Python 3.12 or newer**.

### 1.1 Check if Python is already installed

Open a terminal:

- **Windows:** Press `Windows key`, type `cmd`, press `Enter`.
- **macOS:** Press `Cmd + Space`, type `Terminal`, press `Enter`.

Type this command and press `Enter`:

```bash
python3 --version
```

If you see something like:

```
Python 3.12.4
```

...and the number is `3.12` or higher, skip to [Section 2](#2-installing-visual-studio-code).

If you see an error like `command not found` or a version below `3.12`,
continue below.

### 1.2 Download Python

| Operating System | Where to download |
|---|---|
| Windows | [python.org/downloads](https://www.python.org/downloads/) |
| macOS | [python.org/downloads](https://www.python.org/downloads/) or `brew install python` if you use Homebrew |

Click the yellow **"Download Python 3.12.x"** button on the homepage.

### 1.3 Install Python (Windows)

1. Run the downloaded `.exe` file.
2. **Very important:** on the first install screen, check the box that
   says **"Add python.exe to PATH"** at the bottom. If you skip this,
   Python will not work from the terminal.
3. Click **"Install Now"**.
4. Wait for installation to finish, then click **"Close"**.

### 1.4 Install Python (macOS)

1. Run the downloaded `.pkg` file.
2. Click through **Continue -> Continue -> Agree -> Install**.
3. Enter your Mac password if prompted.
4. Click **Close** when finished.

Alternatively, if you have [Homebrew](https://brew.sh) installed:

```bash
brew install python
```

### 1.5 Verify installation

Close and reopen your terminal, then run:

```bash
python3 --version
```

**Expected output:**

```
Python 3.12.4
```

Also verify `pip` (Python's package installer) is available:

```bash
pip3 --version
```

**Expected output (version numbers may differ):**

```
pip 24.0 from /usr/local/lib/python3.12/site-packages/pip (python 3.12)
```

---

## 2. Installing Visual Studio Code

Visual Studio Code (VS Code) is the code editor we'll use to open, edit,
and run the project.

### 2.1 Download

Go to **[code.visualstudio.com](https://code.visualstudio.com)**. The
site auto-detects your operating system - click the big blue
**Download** button.

### 2.2 Install (Windows)

1. Run the downloaded `VSCodeUserSetup-x64-*.exe` file.
2. Accept the license agreement, click **Next** through the default
   options.
3. On the "Select Additional Tasks" screen, make sure **"Add to PATH"**
   is checked.
4. Click **Install**, then **Finish**.

### 2.3 Install (macOS)

1. Open the downloaded `VSCode-darwin-universal.zip` file (it usually
   unzips automatically).
2. Drag the **Visual Studio Code** app into your **Applications** folder.
3. Open VS Code from Applications (macOS may ask you to confirm you want
   to open an app downloaded from the internet - click **Open**).

### 2.4 Verify installation

Open a terminal and run:

```bash
code --version
```

**Expected output:**

```
1.94.0
e9f6ecb...
x64
```

If `code` is not recognized, open VS Code manually, press
`Cmd/Ctrl+Shift+P`, type **"Shell Command: Install 'code' command in
PATH"**, and select it. Then reopen your terminal.

---

## 3. Installing Git

Git is used to download ("clone") the project repository and to track
code changes. Even if you're only downloading a `.zip` of this project,
Git is worth installing now since most tutorials and future projects
assume you have it.

### 3.1 Check if Git is already installed

```bash
git --version
```

**Expected output:**

```
git version 2.43.0
```

If you see a version number, skip to [Section 4](#4-required-vs-code-extensions).

### 3.2 Install Git (Windows)

1. Download the installer from **[git-scm.com/download/win](https://git-scm.com/download/win)**.
2. Run the installer. The default options are fine for beginners - keep
   clicking **Next**, then **Install**.
3. Click **Finish**.

### 3.3 Install Git (macOS)

The easiest method is via the Xcode Command Line Tools:

```bash
git --version
```

If Git isn't installed, macOS will automatically prompt you to install
the **Command Line Developer Tools**. Click **Install** and wait for it
to finish.

Alternatively, with Homebrew:

```bash
brew install git
```

### 3.4 Verify installation

Close and reopen your terminal:

```bash
git --version
```

**Expected output:**

```
git version 2.43.0
```

---

## 4. Required VS Code Extensions

Extensions add functionality to VS Code. Open VS Code, then click the
**Extensions icon** in the left sidebar (it looks like four squares, one
detached) - or press `Ctrl/Cmd+Shift+X`.

Search for and install each of the following:

| Extension Name | Publisher | Why you need it |
|---|---|---|
| **Python** | Microsoft | Core Python language support, debugging, interpreter selection |
| **Pylance** | Microsoft | Fast, smart autocomplete and error-checking for Python (usually installs automatically with the Python extension) |
| **Even Better TOML** | tamasfe | Syntax highlighting for config files like `pyproject.toml` |

To install: type the name in the search box, click the blue **Install**
button on the correct result.

You do **not** need any FastAPI-, LangChain-, or database-specific
extensions - those are Python libraries, not VS Code extensions, and
they'll be installed in Section 8.

---

## 5. Opening the Project

You should already have the project as a folder or a `.zip` file named
`ai-search-engine`.

### 5.1 If you have a .zip file

1. Extract (unzip) it to a location you'll remember, e.g.
   `Documents/ai-search-engine`.
   - **Windows:** right-click the `.zip` -> "Extract All..."
   - **macOS:** double-click the `.zip` file

### 5.2 Open the folder in VS Code

**Option A - from VS Code:**
1. Open VS Code.
2. Click **File -> Open Folder...** (macOS: **File -> Open...**).
3. Select the `ai-search-engine` folder.
4. Click **Select Folder** (Windows) / **Open** (macOS).

**Option B - from the terminal:**

```bash
cd path/to/ai-search-engine
code .
```

The `.` means "open the current folder." You should now see the file
tree on the left with files like `main.py`, `README.md`, and
`requirements.txt`.

### 5.3 Open the integrated terminal

All remaining commands in this guide are run **inside VS Code's built-in
terminal**, so you don't need to switch windows.

Open it with `` Ctrl+` `` (backtick key, top-left of most keyboards) or
via the menu: **Terminal -> New Terminal**.

Confirm you're in the right folder - the terminal prompt should end in
`ai-search-engine`:

```
PS C:\Users\you\Documents\ai-search-engine>
```

or on macOS:

```
you@MacBook ai-search-engine %
```

---

## 6. Creating a Virtual Environment

### What is a virtual environment, and why do I need one?

A **virtual environment** ("venv") is an isolated Python installation
just for this project. Without it, every Python package you install
would be installed **globally** on your computer, which can cause
version conflicts between different projects. A venv keeps this
project's dependencies separate and disposable.

### 6.1 Create the venv

In the VS Code terminal, run:

```bash
python3 -m venv venv
```

This creates a new folder named `venv/` inside your project. Nothing
will print to the screen - that's normal. It may take 10-30 seconds.

### 6.2 Confirm it was created

```bash
# Windows
dir venv

# macOS
ls venv
```

**Expected output (macOS):**

```
bin      include      lib      pyvenv.cfg
```

**Expected output (Windows):**

```
Include    Lib    Scripts    pyvenv.cfg
```

> **Tip:** the startup scripts (`Start App.bat` / `Start App (Mac).command`)
> included in this project do steps 6 and 7 automatically. This section
> teaches you the manual process so you understand what's happening
> under the hood.

---

## 7. Activating the Virtual Environment

"Activating" tells your terminal to use the isolated Python inside
`venv/` instead of your computer's global Python.

### 7.1 Activate

```bash
# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 7.2 Confirm activation

Your terminal prompt should now show `(venv)` at the beginning:

```
(venv) PS C:\Users\you\Documents\ai-search-engine>
```

```
(venv) you@MacBook ai-search-engine %
```

You must see `(venv)` before running any of the commands in the next
sections. If you close and reopen your terminal, you'll need to
reactivate.

> **PowerShell "running scripts is disabled" error?** See
> [Section 21 - Common Errors](#21-common-errors).

### 7.3 Deactivate (for later reference)

When you're done working, you can exit the virtual environment with:

```bash
deactivate
```

You don't need to do this now - just good to know it exists.

---

## 8. Installing Dependencies

**Dependencies** are the third-party Python packages this project relies
on - FastAPI, LangChain, the OpenAI SDK, and the three vector database
clients. They're all listed in `requirements.txt`.

### 8.1 Upgrade pip first (recommended)

```bash
python -m pip install --upgrade pip
```

### 8.2 Install all dependencies

With your virtual environment still activated (`(venv)` visible):

```bash
pip install -r requirements.txt
```

This will download and install everything, including:

| Package | Purpose |
|---|---|
| `fastapi` | The web framework that powers the backend API |
| `uvicorn` | The server that runs the FastAPI app |
| `openai` | Talks to OpenAI's embeddings and chat models |
| `langchain` / `langchain-text-splitters` | Splits documents into chunks |
| `chromadb` | Local vector database |
| `pinecone` | Cloud vector database client |
| `qdrant-client` | Qdrant vector database client |
| `pypdf` | Extracts text from PDF files |
| `pydantic` / `pydantic-settings` | Data validation and settings management |

This step can take **2-5 minutes** depending on your internet speed -
`chromadb` in particular is a larger download.

**Expected output (abbreviated):**

```
Collecting fastapi==0.115.6
  Downloading fastapi-0.115.6-py3-none-any.whl (94 kB)
...
Successfully installed fastapi-0.115.6 uvicorn-0.34.0 openai-1.59.6 ...
```

### 8.3 Verify installation

```bash
pip list
```

You should see a long list including `fastapi`, `openai`, `chromadb`,
`pinecone`, `qdrant-client`, and more.

---

## 9. Creating the .env File

### What is a `.env` file?

A `.env` file stores **secret configuration values** (like API keys)
outside of your source code, so you never accidentally commit secrets to
version control. This project reads all its settings from `.env` at
startup via `config.py`.

### 9.1 Copy the example file

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

### 9.2 Open .env in VS Code

In the VS Code file explorer (left sidebar), click on the new `.env`
file to open it.

You'll see settings like:

```ini
OPENAI_API_KEY=sk-your-openai-key-here
VECTOR_DB_PROVIDER=chroma
CHROMA_PERSIST_DIR=./chroma_data
...
```

Leave everything as-is for now - you'll fill in the actual API key in
the next section. **Do not delete any lines**, even ones you're not
using yet (e.g. Pinecone/Qdrant settings) - the app reads all of them at
startup and has safe defaults for the ones you leave blank.

> ⚠️ **Never share your `.env` file or commit it to GitHub.** It's
> already excluded via `.gitignore`, so a normal `git add .` will not
> accidentally include it.

---

## 10. Getting OpenAI API Keys

The OpenAI API is used for two things in this project: generating
**embeddings** (turning text into vectors) and generating the final
**answer** to your search queries.

### 10.1 Create an OpenAI account

1. Go to **[platform.openai.com/signup](https://platform.openai.com/signup)**.
2. Sign up with an email address, or continue with Google/Microsoft.
3. Verify your email and phone number if prompted.

### 10.2 Add billing (required for API access)

1. Go to **[platform.openai.com/settings/organization/billing](https://platform.openai.com/settings/organization/billing)**.
2. Click **"Add payment method"** and enter a card.
3. Optional but recommended: set a **usage limit** (e.g. $5/month) under
   **Limits** so you can't be charged more than you expect.

> This project's testing typically costs well under $1 in API usage -
> embeddings and the `gpt-4o-mini` chat model are both inexpensive.

### 10.3 Create an API key

1. Go to **[platform.openai.com/api-keys](https://platform.openai.com/api-keys)**.
2. Click **"Create new secret key"**.
3. Give it a name, e.g. `ai-search-engine-local`.
4. Click **Create secret key**.
5. **Copy the key immediately** - it starts with `sk-` and will only be
   shown once.

### 10.4 Add it to your .env file

Open `.env` in VS Code and replace the placeholder:

```ini
OPENAI_API_KEY=sk-abc123yourrealkeygoeshere
```

Save the file with `Ctrl/Cmd+S`.

### 10.5 Verify the key format

| Correct | Incorrect |
|---|---|
| `OPENAI_API_KEY=sk-proj-Ab12Cd34...` | `OPENAI_API_KEY="sk-proj-Ab12Cd34..."` (no quotes needed) |
| `OPENAI_API_KEY=sk-proj-Ab12Cd34...` | `OPENAI_API_KEY= sk-proj-Ab12Cd34...` (no leading space) |

---

## 11. Creating Pinecone API Keys

Pinecone is **optional** - the app defaults to Chroma, which needs no
API key at all. Only complete this section if you want to test the
Pinecone adapter.

### 11.1 Create a Pinecone account

1. Go to **[app.pinecone.io](https://app.pinecone.io)**.
2. Sign up with email, Google, or GitHub.
3. Complete the onboarding questions (you can select "just exploring").

### 11.2 Get your API key

1. Once logged in, go to the **API Keys** section in the left sidebar.
2. You'll see a **default key** already created - click the eye icon
   or **Copy** button to reveal/copy it.

### 11.3 Add it to your .env file

```ini
VECTOR_DB_PROVIDER=pinecone
PINECONE_API_KEY=pcsk_yourrealkeygoeshere
PINECONE_INDEX_NAME=ai-search-engine
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

> You do **not** need to manually create the index in the Pinecone
> dashboard - `vector_store_pinecone.py` creates it automatically the
> first time the app runs, using the name, cloud, and region from your
> `.env` file.

### 11.4 Free tier limits

| Plan | Indexes | Good for |
|---|---|---|
| Starter (free) | Up to 5 serverless indexes | This project, learning, small demos |
| Standard/Enterprise | More, with SLAs | Production workloads |

---

## 12. Running Chroma Locally

Chroma requires **no installation, no account, and no API key** - it's
already included in `requirements.txt` and runs as an embedded database
that saves data to a folder on your disk.

### 12.1 Confirm it's the default provider

Open `.env` and check:

```ini
VECTOR_DB_PROVIDER=chroma
CHROMA_PERSIST_DIR=./chroma_data
CHROMA_COLLECTION_NAME=documents
```

### 12.2 How it works

The first time you upload and index a document, the app automatically
creates a `chroma_data/` folder in your project directory. This is where
all your vectors are stored persistently - closing and reopening the app
will not lose your data.

**No extra steps required** - as long as `VECTOR_DB_PROVIDER=chroma`,
this happens automatically when you start the app in
[Section 14](#14-starting-the-application).

---

## 13. Running Qdrant Locally (Optional)

Qdrant can run locally via Docker, or you can use a free hosted Qdrant
Cloud instance. This section covers the local Docker method.

### 13.1 Install Docker Desktop

Download and install Docker Desktop from
**[docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)**
for your operating system, then open it once to complete setup.

Verify it's running:

```bash
docker --version
```

**Expected output:**

```
Docker version 27.3.1, build ce12230
```

### 13.2 Run the Qdrant container

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**Expected output (abbreviated):**

```
Unable to find image 'qdrant/qdrant:latest' locally
latest: Pulling from qdrant/qdrant
...
Qdrant HTTP listening on 6333
Qdrant gRPC listening on 6334
```

Leave this terminal window running - Qdrant needs to stay active while
you use the app. Open a **new** terminal tab/window for the rest of this
guide.

### 13.3 Add Qdrant settings to .env

```ini
VECTOR_DB_PROVIDER=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=documents
```

Leave `QDRANT_API_KEY` blank for a local instance - it's only required
for Qdrant Cloud.

### 13.4 Alternative: Qdrant Cloud (no Docker needed)

1. Sign up at **[cloud.qdrant.io](https://cloud.qdrant.io)**.
2. Create a free cluster.
3. Copy the **cluster URL** and **API key** from the dashboard.
4. Use those values in `.env`:

```ini
QDRANT_URL=https://your-cluster-id.qdrant.io
QDRANT_API_KEY=your-cloud-api-key
```

---

## 14. Starting the Application

You have two options: the automated startup scripts, or manual commands.

### 14.1 Option A - Startup scripts (easiest)

| OS | File to run |
|---|---|
| Windows | Double-click `Start App.bat` |
| macOS | Double-click `Start App (Mac).command` |

> **macOS security note:** the first time you double-click the
> `.command` file, macOS may block it. Right-click the file -> **Open**
> -> confirm **Open** in the dialog. This is only needed once.

The script automatically checks Python, creates/activates the virtual
environment, installs dependencies, checks your `.env` file, and starts
the server - effectively repeating Sections 6-9 for you.

### 14.2 Option B - Manual (from the VS Code terminal)

Make sure your virtual environment is activated (`(venv)` visible), then:

```bash
uvicorn main:app --reload
```

### 14.3 Expected output

```
2026-07-20 10:00:00 | INFO     | database | SQLite schema initialized at ./app_data.db
INFO:     Will watch for changes in these directories: ['.../ai-search-engine']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
2026-07-20 10:00:00 | INFO     | main | AI Search Engine starting up (provider=chroma)
INFO:     Application startup complete.
```

### 14.4 Open the app

Open your browser and go to:

```
http://127.0.0.1:8000
```

You should see the **AI Search Engine dashboard** with a sidebar
(Search, Documents, Compare, History, Settings).

### 14.5 Stopping the app

Click into the terminal running the server and press `Ctrl+C`.

---

## 15. Uploading Documents

1. In the browser, click **"Documents"** in the left sidebar.
2. Click the dashed **upload box**, or drag a file directly onto it.
3. Choose a `.pdf`, `.txt`, or `.md` file from your computer.

**Expected result:** a status message appears:

```
✓ notes.txt indexed into 4 chunks (chroma).
```

And a new row appears in the documents table with status **`indexed`**.

> **Don't have a test file handy?** Create one quickly:
>
> ```bash
> echo "The Eiffel Tower is located in Paris, France. It was completed in 1889." > sample.txt
> ```
>
> Then upload `sample.txt` through the browser.

---

## 16. Creating Embeddings

You don't need to trigger this manually - it happens automatically as
part of the upload process described in Section 15. But it's worth
understanding what happens behind the scenes:

1. Your uploaded file's text is extracted (`document_processor.py`).
2. The text is split into overlapping chunks (`chunking.py`).
3. Each chunk is sent to OpenAI's embeddings API
   (`embeddings.py` -> `embed_texts()`), which returns a vector (a list of
   1,536 numbers for `text-embedding-3-small`) representing that chunk's
   meaning.

You can watch this happen live in the terminal running the server:

```
2026-07-20 10:05:00 | INFO | chunking | Chunked document_id=... into 4 chunks
2026-07-20 10:05:01 | INFO | vector_store_chroma | Upserted 4 chunks into Chroma
```

---

## 17. Indexing Documents

"Indexing" means storing the embeddings in the vector database so they
can be searched later. Like embedding generation, this happens
automatically on upload - there is no separate manual step.

Confirm indexing succeeded by checking the **Documents** tab:

| Column | What it means |
|---|---|
| Status | `indexed` = success, `processing` = in progress, `failed` = check the error |
| Chunks | How many pieces the document was split into |
| Provider | Which vector database it was indexed into |

If status shows `failed`, see [Section 21 - Common Errors](#21-common-errors).

---

## 18. Running Semantic Search

1. Click **"Search"** in the sidebar.
2. Type a question related to something in your uploaded document(s),
   e.g.:

   ```
   Where is the Eiffel Tower located?
   ```

3. Choose a **Search Mode** (Semantic is the default and best starting
   point) and a **Provider** (Chroma, if you haven't set up the others).
4. Click **Search**.

**Expected result:**

- A **Generated Answer** card appears with an LLM-written response
  citing your source(s), e.g.: *"The Eiffel Tower is located in Paris,
  France [Source 1]."*
- Below it, one or more **result cards** show the raw matching text
  chunk, the source filename, and a similarity score (shown as a small
  glowing dot on a horizontal bar - the closer to 100%, the more
  relevant).

| Search Mode | What it does |
|---|---|
| **Semantic** | Pure meaning-based vector search (most common) |
| **Similarity** | Same underlying mechanism as Semantic |
| **Metadata** | Filters by exact metadata fields (e.g. file type) |
| **Hybrid** | Combines vector similarity with a metadata filter |

---

## 19. Switching Between Chroma, Pinecone, and Qdrant

There are two ways to switch providers:

### 19.1 Per-search (no restart needed)

On the **Search** tab, use the **provider dropdown** next to the search
bar to pick Chroma, Pinecone, or Qdrant for that specific query. This
only works for providers you've already configured with valid
credentials in `.env`.

### 19.2 Changing the default provider (requires restart)

1. Stop the server (`Ctrl+C` in its terminal).
2. Open `.env` and change:

   ```ini
   VECTOR_DB_PROVIDER=qdrant
   ```

   (or `pinecone`, or back to `chroma`)
3. Save the file.
4. Restart the app (Section 14).

> **Important:** each provider has its **own separate index** of your
> documents. Uploading a file while `VECTOR_DB_PROVIDER=chroma` does
> **not** make it searchable under Pinecone or Qdrant - you'd need to
> re-upload it after switching, or use the **Compare** tab (Section 20)
> to query multiple providers using whatever each already has indexed.

---

## 20. Testing All Features

Use this checklist to confirm your installation works end-to-end.

| # | Feature | How to test | Expected result |
|---|---|---|---|
| 1 | Health check | Visit `http://127.0.0.1:8000/api/health` | JSON with `"status": "ok"` |
| 2 | Upload | Documents tab -> upload a `.txt` file | Status becomes `indexed` |
| 3 | Embeddings | Check server terminal logs during upload | Log line `Upserted N chunks` |
| 4 | Semantic search | Search tab -> ask a relevant question | Answer + result cards appear |
| 5 | Metadata search | Search tab -> mode `Metadata` | Filtered results only |
| 6 | Compare | Compare tab -> run a query | Three columns (Chroma/Pinecone/Qdrant) |
| 7 | History | History tab | Your past queries listed |
| 8 | Settings | Settings tab | Current config values displayed |
| 9 | Delete document | Documents tab -> click Delete | Row disappears, vectors removed |
| 10 | Automated tests | `pytest` in terminal | All tests pass |

### 20.1 Run the automated test suite

```bash
pytest
```

**Expected output:**

```
tests/test_chunking.py ....                                            [ 50%]
tests/test_document_processor.py ....                                  [100%]

============================== 8 passed in 0.20s ===============================
```

---

## 21. Common Errors

| Error message | Cause | Fix |
|---|---|---|
| `'python3' is not recognized as an internal or external command` | Python not on PATH | Reinstall Python and check "Add to PATH" (Section 1.3) |
| `running scripts is disabled on this system` (PowerShell) | PowerShell execution policy blocks the venv activation script | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry activation |
| `ModuleNotFoundError: No module named 'fastapi'` | Dependencies not installed, or venv not activated | Run `source venv/bin/activate` (or Windows equivalent), then `pip install -r requirements.txt` |
| `OPENAI_API_KEY is not set` (log warning) | `.env` missing or key not filled in | Complete Section 9 and 10 |
| `401 Unauthorized` from OpenAI | Invalid or expired API key | Generate a new key (Section 10.3) and update `.env` |
| `429 Too Many Requests` from OpenAI | Rate limit or no billing set up | Add a payment method (Section 10.2) |
| `PINECONE_API_KEY is not set. Add it to your .env file...` | Trying to use Pinecone without a key | Complete Section 11, or switch `VECTOR_DB_PROVIDER` back to `chroma` |
| `Connection refused` to `localhost:6333` | Qdrant container isn't running | Run the `docker run` command from Section 13.2 |
| `Address already in use` on port 8000 | Another process is using port 8000 | Change `APP_PORT` in `.env`, or stop the other process |
| Upload fails with "Unsupported file type" | File isn't `.pdf`, `.txt`, or `.md` | Convert or choose a supported file |
| PDF uploads but 0 chunks / extraction error | PDF is a scanned image with no real text layer | This project does not include OCR - use a text-based PDF |

---

## 22. Troubleshooting

### 22.1 The server won't start

1. Confirm `(venv)` is visible in your terminal prompt.
2. Run `pip list` and confirm `fastapi` and `uvicorn` are listed.
3. Run `python -m py_compile main.py` - if this produces no output,
   the file has no syntax errors and the issue is elsewhere (likely
   missing dependencies or a bad `.env` value).

### 22.2 The page loads but search returns nothing

- Make sure you've **uploaded at least one document** under the
  currently selected provider.
- Confirm the document's status is `indexed`, not `failed`, on the
  Documents tab.

### 22.3 Everything was working, then suddenly broke

- Check whether you edited `.env` and left a typo (e.g. missing `=`).
- Restart the server after any `.env` change - it's only read at
  startup.

### 22.4 Still stuck?

Re-run the automated tests to isolate whether the problem is in your
environment or your data:

```bash
pytest -v
```

If tests pass but the live app doesn't work, the issue is most likely
your `.env` configuration or API keys. If tests fail, the issue is your
Python environment or dependency installation - revisit Sections 6-8.

---

## 23. FAQ

**Q: Do I need to know how to code to use this project?**
A: No - you can run the app, upload documents, and search entirely
through the browser. Understanding the code is only needed if you want
to modify it.

**Q: Is my data sent anywhere besides OpenAI and my chosen vector
database?**
A: No. The app itself only talks to OpenAI (for embeddings/answers) and
your chosen vector database (Chroma is entirely local; Pinecone/Qdrant
are cloud services if you configure them).

**Q: Can I use a different LLM instead of OpenAI?**
A: Not out of the box - `embeddings.py` is written specifically against
the OpenAI SDK. Swapping providers would require rewriting that module.

**Q: Why does the Documents tab show "processing" for a long time?**
A: Very large PDFs take longer to extract, chunk, and embed. If it's
stuck for several minutes, check the server terminal for an error.

**Q: Can I run this without an internet connection?**
A: No - embeddings and answer generation both require calls to the
OpenAI API, which needs internet access. Chroma itself works offline,
but the embedding step doesn't.

**Q: Do I need to restart the app every time I upload a document?**
A: No - only `.env` changes require a restart. Uploading, searching, and
deleting documents all work live.

**Q: What happens to my uploaded file after indexing?**
A: The original uploaded file is deleted from disk immediately after its
text is extracted and embedded - only the extracted chunks and vectors
persist in your vector database.

---

## 24. Security Recommendations

- **Never commit your `.env` file.** It's excluded by `.gitignore`
  already - don't remove that exclusion.
- **Never paste your API keys into chat messages, screenshots, or
  public forums.** If you accidentally expose a key, revoke it
  immediately from the provider's dashboard and generate a new one.
- **Set a usage limit** on your OpenAI account (Section 10.2) so a bug
  or leaked key can't run up an unexpected bill.
- **Use separate API keys per project.** Naming them clearly (e.g.
  `ai-search-engine-local`) makes it easy to revoke just one if needed.
- **Rotate keys periodically**, especially if you've shared your screen
  or a `.env` file with someone.
- **Don't expose this app directly to the public internet** without
  adding authentication first - as shipped, anyone who can reach
  `http://your-server:8000` can upload documents and run searches (and
  therefore consume your OpenAI credits).
- **Restrict CORS in production.** `main.py` currently allows all
  origins (`allow_origins=["*"]`) for local development convenience;
  tighten this before deploying publicly.

---

## 25. Next Learning Steps

Once you're comfortable running the app, here are good next steps for
deepening your understanding:

| Goal | Where to look |
|---|---|
| Understand the full RAG pipeline | Read the docstrings at the top of `search_engine.py` |
| Learn how chunking works | Read `chunking.py` and try changing `CHUNK_SIZE` in `.env` |
| Understand vector math | Read `vector_store_chroma.py`'s `similarity_search()` method |
| Learn FastAPI basics | [fastapi.tiangolo.com/tutorial](https://fastapi.tiangolo.com/tutorial/) |
| Learn about embeddings in depth | [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings) |
| Learn LangChain concepts | [python.langchain.com/docs/introduction](https://python.langchain.com/docs/introduction/) |
| Explore Chroma further | [docs.trychroma.com](https://docs.trychroma.com) |
| Explore Pinecone further | [docs.pinecone.io](https://docs.pinecone.io) |
| Explore Qdrant further | [qdrant.tech/documentation](https://qdrant.tech/documentation/) |
| Add a new feature | Try adding DOCX support in `document_processor.py` |
| Learn Git properly | [git-scm.com/book](https://git-scm.com/book/en/v2) - start with chapters 1-3 |
| Deploy your app | Re-read the **Deployment** section of `README.md` |

You now have a fully working, locally-run AI Search Engine - and a solid
foundation in the core concepts behind modern retrieval-augmented
generation systems. Good luck, and enjoy building!
