@echo off
REM ==========================================================
REM  AI Search Engine - Windows Startup Script
REM  Double-click this file to set up and launch the app.
REM ==========================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   AI Search Engine - Starting Up
echo ============================================================
echo.

REM ---- 1. Verify Python is installed ----
echo [1/6] Checking for Python...
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Python was not found on your PATH.
    echo         Please install Python 3.12+ from https://www.python.org/downloads/
    echo         and make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VERSION=%%v
echo       Found Python %PY_VERSION%
echo.

REM ---- 2. Create virtual environment if needed ----
echo [2/6] Checking for virtual environment...
if not exist "venv\" (
    echo       No venv found. Creating one now...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Virtual environment created.
) else (
    echo       Virtual environment already exists.
)
echo.

REM ---- 3. Activate virtual environment ----
echo [3/6] Activating virtual environment...
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo       Activated.
echo.

REM ---- 4. Install dependencies ----
echo [4/6] Installing dependencies (this may take a minute on first run)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies. See the output above for details.
    pause
    exit /b 1
)
echo       Dependencies installed.
echo.

REM ---- 5. Verify .env file ----
echo [5/6] Checking for .env configuration file...
if not exist ".env" (
    echo       No .env found. Creating one from .env.example...
    copy ".env.example" ".env" >nul
    echo.
    echo [ACTION REQUIRED] A new .env file was created.
    echo       Please open .env and add your OPENAI_API_KEY
    echo       ^(and any Pinecone/Qdrant keys you plan to use^),
    echo       then re-run this script.
    echo.
    pause
    exit /b 0
) else (
    echo       .env file found.
)
echo.

REM ---- 6. Start the FastAPI application ----
echo [6/6] Starting AI Search Engine...
echo.
echo ============================================================
echo   The app will be available at: http://127.0.0.1:8000
echo   Press CTRL+C to stop the server.
echo ============================================================
echo.

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

if errorlevel 1 (
    echo.
    echo [ERROR] The application exited with an error. See the output above.
    pause
    exit /b 1
)

pause
