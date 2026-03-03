@echo off
setlocal

cd /d "%~dp0.."

python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>nul
if %errorlevel% neq 0 (
    echo Python 3.11+ is required.
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Starting Path-Lite development server...
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level info
