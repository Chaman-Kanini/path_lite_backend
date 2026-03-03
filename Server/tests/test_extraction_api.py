import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.models.structured_outputs import FieldType


@pytest.fixture
def mock_extraction_service():
    """Mock ExtractionService for testing."""
    service = MagicMock()
    
    service.extract_field = AsyncMock(return_value={
        "value": {"mrn": "12345", "confidence_score": 0.95},
        "confidence_score": 0.95,
        "field_type": "mrn",
        "processing_time_ms": 450,
        "fallback_to_manual": False,
        "validation": {"is_valid": True, "warnings": [], "errors": []}
    })
    
    service.extract_patient_data = AsyncMock(return_value={
        "data": {
            "mrn": {"mrn": "12345", "confidence_score": 0.95},
            "name": {
                "first_name": "John",
                "last_name": "Doe",
                "confidence_score": 0.98
            }
        },
        "processing_time_ms": 850,
        "fallback_to_manual": False
    })
    
    service.client.verify_connection = AsyncMock(return_value=True)
    
    return service


@pytest.fixture
def client(mock_extraction_service):
    """Create test client with mocked dependencies."""
    from main import app
    from app.dependencies import get_extraction_service_dependency
    
    app.dependency_overrides[get_extraction_service_dependency] = lambda: mock_extraction_service
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


class TestExtractionEndpoints:
    """Test suite for extraction API endpoints."""
    
    def test_extract_field_success(self, client, mock_extraction_service):
        """Test successful single field extraction."""
        response = client.post(
            "/ai/extract",
            json={
                "text": "The patient's MRN is 12345",
                "field_type": "mrn"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["field_type"] == "mrn"
        assert data["confidence_score"] == 0.95
        assert data["processing_time_ms"] > 0
        assert data["fallback_to_manual"] is False
        
        mock_extraction_service.extract_field.assert_called_once()
    
    def test_extract_field_latency_requirement(self, client, mock_extraction_service):
        """Test that extraction meets <1000ms latency requirement."""
        response = client.post(
            "/ai/extract",
            json={
                "text": "Patient MRN is ABC123",
                "field_type": "mrn"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["processing_time_ms"] < 1000, "Processing time exceeds 1 second requirement"
    
    def test_extract_batch_success(self, client, mock_extraction_service):
        """Test successful batch field extraction."""
        response = client.post(
            "/ai/extract/batch",
            json={
                "text": "Patient John Doe, MRN 12345, born May 15, 1990",
                "field_types": ["mrn", "first_name"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "extractions" in data
        assert len(data["extractions"]) == 2
        assert data["total_processing_time_ms"] > 0
    
    def test_extract_patient_data_success(self, client, mock_extraction_service):
        """Test successful complete patient data extraction."""
        response = client.post(
            "/ai/extract/patient",
            json={
                "text": "Patient John Doe, MRN 12345, born May 15, 1990, male"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "mrn" in data["data"]
        assert "name" in data["data"]
        assert data["processing_time_ms"] > 0
        assert data["fallback_to_manual"] is False
    
    def test_extract_field_invalid_input(self, client):
        """Test extraction with invalid input."""
        response = client.post(
            "/ai/extract",
            json={
                "text": "",
                "field_type": "mrn"
            }
        )
        
        assert response.status_code == 422
    
    def test_extract_field_error_handling(self, client, mock_extraction_service):
        """Test error handling when extraction fails."""
        mock_extraction_service.extract_field.side_effect = Exception("API Error")
        
        response = client.post(
            "/ai/extract",
            json={
                "text": "Test text",
                "field_type": "mrn"
            }
        )
        
        assert response.status_code == 500
        assert "Extraction failed" in response.json()["detail"]
    
    def test_health_check_success(self, client, mock_extraction_service):
        """Test health check endpoint when Azure OpenAI is connected."""
        response = client.get("/ai/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["azure_openai_connected"] is True
        assert "message" in data
    
    def test_health_check_failure(self, client, mock_extraction_service):
        """Test health check endpoint when Azure OpenAI connection fails."""
        mock_extraction_service.client.verify_connection.side_effect = Exception("Connection failed")
        
        response = client.get("/ai/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["azure_openai_connected"] is False


class TestExtractionAccuracy:
    """Test suite for extraction accuracy requirements."""
    
    @pytest.mark.parametrize("test_case", [
        {
            "text": "The patient's medical record number is MRN12345",
            "field_type": "mrn",
            "expected_confidence": 0.90
        },
        {
            "text": "Patient name is John Doe",
            "field_type": "first_name",
            "expected_confidence": 0.90
        },
        {
            "text": "Date of birth is May 15, 1990",
            "field_type": "dob",
            "expected_confidence": 0.90
        },
    ])
    def test_extraction_accuracy_threshold(self, client, mock_extraction_service, test_case):
        """Test that extractions meet 95% accuracy requirement (confidence >= 0.90)."""
        response = client.post(
            "/ai/extract",
            json={
                "text": test_case["text"],
                "field_type": test_case["field_type"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["confidence_score"] >= test_case["expected_confidence"], \
            f"Confidence score {data['confidence_score']} below threshold for {test_case['field_type']}"


class TestFallbackMechanism:
    """Test suite for fallback to manual entry."""
    
    def test_fallback_after_retries(self, client, mock_extraction_service):
        """Test that fallback_to_manual flag is set after failed retries."""
        mock_extraction_service.extract_field.return_value = {
            "value": {},
            "confidence_score": 0.0,
            "field_type": "mrn",
            "processing_time_ms": 500,
            "fallback_to_manual": True,
            "validation": {"is_valid": False, "warnings": [], "errors": ["Extraction failed"]}
        }
        
        response = client.post(
            "/ai/extract",
            json={
                "text": "unclear text",
                "field_type": "mrn"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["fallback_to_manual"] is True
