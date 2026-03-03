import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from main import app
from app.core.config import settings
from app.services.schema_validation_service import SchemaValidationService
from app.services.medical_term_extraction_service import MedicalTermExtractionService


@pytest.fixture(scope="session")
def test_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture(autouse=True)
def reset_metrics(mock_medical_extraction_service):
    """Reset metrics before each test"""
    if hasattr(mock_medical_extraction_service, 'metrics'):
        mock_medical_extraction_service.metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_retries": 0,
            "validation_failures": 0,
            "confidence_scores": [],
            "by_field": {}
        }
    yield


@pytest.fixture
def mock_settings():
    return settings


@pytest.fixture
def mock_user():
    return {
        "id": "test-user-id",
        "username": "testuser",
        "email": "testuser@example.com",
    }


@pytest.fixture
def mock_azure_openai_client():
    """Mock Azure OpenAI client for testing"""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    return client


@pytest.fixture
def mock_schema_validation_service():
    """Mock schema validation service"""
    return SchemaValidationService()


@pytest.fixture
def mock_medical_extraction_service(mock_azure_openai_client, mock_schema_validation_service):
    """Mock medical extraction service"""
    return MedicalTermExtractionService(
        azure_client=mock_azure_openai_client,
        schema_validation_service=mock_schema_validation_service
    )


@pytest.fixture
def load_test_cases():
    """Load medical test cases from YAML"""
    test_data_path = Path(__file__).parent / "data" / "medical_test_cases.yaml"
    
    if not test_data_path.exists():
        return {}
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}
