# Path-Lite Server

FastAPI backend for the Path-Lite patient management and AI-assisted care application.

## Requirements

- **Python 3.11+** — verify with `python --version`
- **pip** — comes with Python (verify with `pip --version`)

> If Python is below 3.11, download from [python.org](https://www.python.org/downloads/) and upgrade before proceeding.

## Project Structure

```
Server/
├── main.py                  # FastAPI application entry point
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── pytest.ini               # Pytest configuration
├── .env                     # Local environment variables (not committed)
├── .env.example             # Environment variables template
├── app/
│   ├── core/
│   │   ├── config.py        # Pydantic Settings configuration
│   │   └── middleware.py    # Custom middleware
│   ├── routers/
│   │   ├── auth.py          # Authentication endpoints
│   │   ├── patients.py      # Patient management endpoints
│   │   └── ai_processing.py # AI conversation endpoints
│   ├── models/
│   │   ├── auth.py          # Authentication Pydantic schemas
│   │   ├── patient.py       # Patient Pydantic schemas
│   │   └── ai.py            # AI processing Pydantic schemas
│   ├── services/            # Business logic layer
│   ├── utils/               # Helper functions
│   └── dependencies.py      # Dependency injection functions
├── scripts/
│   ├── dev.py               # Development server launcher
│   ├── start.sh             # Unix/macOS start script
│   └── start.bat            # Windows start script
└── tests/
    ├── conftest.py          # Shared test fixtures
    ├── test_main.py         # Main app tests
    ├── test_auth.py         # Auth router tests
    ├── test_patients.py     # Patients router tests
    └── test_ai_processing.py# AI router tests
```

## Setup

### 1. Create Virtual Environment

```bash
# Navigate to Server directory
cd Server

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Unix/macOS/WSL:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# For development (includes testing tools)
pip install -r requirements-dev.txt
```

### 3. Configure Environment

```bash
# Windows PowerShell/CMD:
copy .env.example .env
# Unix/macOS/WSL:
cp .env.example .env

# Edit .env with your local values
```

## Running the Development Server

```bash
# Make sure virtual environment is activated first!
# Then run uvicorn directly:
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or using Python module syntax:
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Using shell script (Unix/macOS)
./scripts/start.sh

# Using batch script (Windows)
scripts\start.bat
```

The server starts at `http://localhost:8000`.

### API Documentation

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | Swagger UI (interactive) |
| http://localhost:8000/redoc | ReDoc documentation |
| http://localhost:8000/openapi.json | OpenAPI JSON schema |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /api/v1/auth/login | User login |
| POST | /api/v1/auth/register | User registration |
| POST | /api/v1/auth/logout | User logout |
| GET | /api/v1/auth/me | Current user profile |
| GET | /api/v1/patients | List patients |
| POST | /api/v1/patients | Create patient |
| GET | /api/v1/patients/{id} | Get patient by ID |
| PUT | /api/v1/patients/{id} | Update patient |
| POST | /api/v1/ai/conversations | Start AI conversation |
| POST | /api/v1/ai/conversations/{id}/messages | Send message |
| GET | /api/v1/ai/conversations/{id} | Get conversation history |

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| APP_NAME | Path-Lite API | Application name |
| DEBUG_MODE | true | Enable debug mode |
| SERVER_HOST | 0.0.0.0 | Server bind host |
| SERVER_PORT | 8000 | Server port |
| LOG_LEVEL | info | Logging level |
| CORS_ORIGINS | [...] | Allowed CORS origins |
| SECRET_KEY | - | JWT signing secret |
| AZURE_OPENAI_API_KEY | - | Azure OpenAI API key |
| AZURE_OPENAI_ENDPOINT | - | Azure OpenAI endpoint |

## Testing

```bash
# Make sure virtual environment is activated and dev dependencies installed!

# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_auth.py

# Run by marker
pytest -m unit
pytest -m integration

# Verbose output
pytest -v
```

Coverage reports are generated in `htmlcov/index.html`.

### Writing New Tests

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_my_endpoint():
    response = client.get("/my-endpoint")
    assert response.status_code == 200
```

## Troubleshooting

- **Python version error**: Install Python 3.11+ from [python.org](https://www.python.org/downloads/)
- **Virtual environment not activated**: Run the activation command for your OS (see Setup section)
- **Port already in use**: Change port with `--port 8080` or kill the existing process
- **Dependencies fail**: Try `pip install --upgrade pip` then reinstall requirements
- **Module not found errors**: Ensure virtual environment is activated and dependencies installed
