@echo off
REM Malnutrition Prediction Multi-Model API Windows Start Script

echo ===================================================
echo   Starting Malnutrition Prediction Multi-Model API
echo ===================================================

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH.
    echo Please install Python 3.10+ or add it to PATH.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [INFO] Installing/verifying dependencies...
pip install -r requirements.txt

REM Verify model files exist in models folder
if not exist "models\*.pkl" (
    echo [ERROR] No model files (*.pkl) found in models\ directory!
    pause
    exit /b 1
)

REM Launch FastAPI with Uvicorn
echo [INFO] Starting FastAPI server on http://127.0.0.1:8000
echo [INFO] Available Models: http://127.0.0.1:8000/models
echo [INFO] Interactive documentation: http://127.0.0.1:8000/docs
echo Press Ctrl+C to stop.
echo ===================================================

uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
