#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SERVER_DIR"

if ! python3 -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
    echo "Python 3.11+ is required. Current: $(python3 --version)"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Path-Lite development server..."
uvicorn main:app --reload --host "${SERVER_HOST:-0.0.0.0}" --port "${SERVER_PORT:-8000}" --log-level "${LOG_LEVEL:-info}"
