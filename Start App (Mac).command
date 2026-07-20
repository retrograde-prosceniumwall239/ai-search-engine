#!/usr/bin/env bash
# ==========================================================
#  AI Search Engine - macOS Startup Script
#  Double-click this file (or run it) to set up and launch the app.
# ==========================================================

set -uo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "  AI Search Engine - Starting Up"
echo "============================================================"
echo

fail() {
    echo
    echo "[ERROR] $1"
    echo
    read -n 1 -s -r -p "Press any key to close this window..."
    exit 1
}

# ---- 1. Verify Python is installed ----
echo "[1/6] Checking for Python..."
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    fail "Python was not found. Install Python 3.12+ from https://www.python.org/downloads/ (or 'brew install python')."
fi
echo "      Found $($PYTHON_BIN --version)"
echo

# ---- 2. Create virtual environment if needed ----
echo "[2/6] Checking for virtual environment..."
if [ ! -d "venv" ]; then
    echo "      No venv found. Creating one now..."
    $PYTHON_BIN -m venv venv || fail "Failed to create virtual environment."
    echo "      Virtual environment created."
else
    echo "      Virtual environment already exists."
fi
echo

# ---- 3. Activate virtual environment ----
echo "[3/6] Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate || fail "Failed to activate virtual environment."
echo "      Activated."
echo

# ---- 4. Install dependencies ----
echo "[4/6] Installing dependencies (this may take a minute on first run)..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt || fail "Failed to install dependencies. See the output above for details."
echo "      Dependencies installed."
echo

# ---- 5. Verify .env file ----
echo "[5/6] Checking for .env configuration file..."
if [ ! -f ".env" ]; then
    echo "      No .env found. Creating one from .env.example..."
    cp ".env.example" ".env"
    echo
    echo "[ACTION REQUIRED] A new .env file was created."
    echo "      Please open .env and add your OPENAI_API_KEY"
    echo "      (and any Pinecone/Qdrant keys you plan to use),"
    echo "      then re-run this script."
    echo
    read -n 1 -s -r -p "Press any key to close this window..."
    exit 0
else
    echo "      .env file found."
fi
echo

# ---- 6. Start the FastAPI application ----
echo "[6/6] Starting AI Search Engine..."
echo
echo "============================================================"
echo "  The app will be available at: http://127.0.0.1:8000"
echo "  Press CTRL+C to stop the server."
echo "============================================================"
echo

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

status=$?
if [ $status -ne 0 ]; then
    fail "The application exited with an error. See the output above."
fi
