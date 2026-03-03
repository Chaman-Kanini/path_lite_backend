"""
Medical Extraction API Endpoint Tests

Integration tests for medical extraction API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from app.models.medical_extraction import MedicalFieldType
from tests.fixtures.mock_medical_responses import (
    MOCK_HBSAG_POSITIVE,
    MOCK_LOCATION_OR,
    MOCK_GENDER_MALE
)


class TestMedicalExtractionEndpoints:
    """Test medical extraction API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from main import app
        return TestClient(app)
    
    @pytest.fixture
    def mock_service(self):
        """Create mock medical extraction service"""
        with patch('app.routers.medical_extraction.get_medical_extraction_service') as mock:
            service = Mock()
            mock.return_value = service
            yield service
    
    def test_extract_endpoint_hbsag_success(self, client, mock_service):
        """Test POST /extract endpoint with HBsAg"""
        from app.models.medical_extraction import MedicalExtractionResponse
        
        # Mock service response
        mock_response = MedicalExtractionResponse(
            extracted_value="Positive",
            confidence=0.95,
            is_valid=True,
            retry_count=0,
            validation_message=None,
            raw_text_snippet="Hep B positive",
            field_type=MedicalFieldType.HBSAG
        )
        mock_service.extract_medical_term = Mock(return_value=mock_response)
        
        # Make request
        response = client.post(
            "/api/v1/medical-extraction/extract",
            json={
                "raw_text": "Patient is Hep B positive",
                "field_type": "hbsag"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["extracted_value"] == "Positive"
        assert data["is_valid"] is True
        assert data["confidence"] == 0.95
    
    def test_extract_endpoint_treatment_location_success(self, client, mock_service):
        """Test POST /extract endpoint with treatment location"""
        from app.models.medical_extraction import MedicalExtractionResponse
        
        mock_response = MedicalExtractionResponse(
            extracted_value="OR",
            confidence=0.94,
            is_valid=True,
            retry_count=0,
            validation_message=None,
            raw_text_snippet="operating room",
            field_type=MedicalFieldType.TREATMENT_LOCATION
        )
        mock_service.extract_medical_term = Mock(return_value=mock_response)
        
        response = client.post(
            "/api/v1/medical-extraction/extract",
            json={
                "raw_text": "Patient is in the operating room",
                "field_type": "treatment_location"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["extracted_value"] == "OR"
        assert data["is_valid"] is True
    
    def test_extract_endpoint_gender_success(self, client, mock_service):
        """Test POST /extract endpoint with gender"""
        from app.models.medical_extraction import MedicalExtractionResponse
        
        mock_response = MedicalExtractionResponse(
            extracted_value="Male",
            confidence=0.97,
            is_valid=True,
            retry_count=0,
            validation_message=None,
            raw_text_snippet="patient is male",
            field_type=MedicalFieldType.GENDER
        )
        mock_service.extract_medical_term = Mock(return_value=mock_response)
        
        response = client.post(
            "/api/v1/medical-extraction/extract",
            json={
                "raw_text": "The patient is male",
                "field_type": "gender"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["extracted_value"] == "Male"
        assert data["is_valid"] is True
    
    def test_extract_endpoint_invalid_request(self, client):
        """Test POST /extract with invalid request returns 422"""
        response = client.post(
            "/api/v1/medical-extraction/extract",
            json={
                "raw_text": "",  # Empty text
                "field_type": "hbsag"
            }
        )
        
        assert response.status_code == 422
    
    def test_extract_endpoint_invalid_field_type(self, client):
        """Test POST /extract with invalid field type returns 422"""
        response = client.post(
            "/api/v1/medical-extraction/extract",
            json={
                "raw_text": "Patient is Hep B positive",
                "field_type": "invalid_type"
            }
        )
        
        assert response.status_code == 422
    
    def test_batch_extract_endpoint_success(self, client, mock_service):
        """Test POST /batch endpoint with multiple extractions"""
        from app.models.medical_extraction import MedicalExtractionResponse
        
        mock_responses = [
            MedicalExtractionResponse(
                extracted_value="Positive",
                confidence=0.95,
                is_valid=True,
                retry_count=0,
                field_type=MedicalFieldType.HBSAG
            ),
            MedicalExtractionResponse(
                extracted_value="OR",
                confidence=0.94,
                is_valid=True,
                retry_count=0,
                field_type=MedicalFieldType.TREATMENT_LOCATION
            )
        ]
        
        mock_service.extract_medical_term = Mock(side_effect=mock_responses)
        
        response = client.post(
            "/api/v1/medical-extraction/batch",
            json={
                "extractions": [
                    {
                        "raw_text": "Patient is Hep B positive",
                        "field_type": "hbsag"
                    },
                    {
                        "raw_text": "Treatment in the OR",
                        "field_type": "treatment_location"
                    }
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert data["successful_count"] == 2
        assert data["failed_count"] == 0
        assert len(data["results"]) == 2
    
    def test_batch_extract_endpoint_max_limit(self, client):
        """Test POST /batch enforces max 10 extractions"""
        extractions = [
            {
                "raw_text": f"Test {i}",
                "field_type": "hbsag"
            }
            for i in range(11)  # 11 extractions (over limit)
        ]
        
        response = client.post(
            "/api/v1/medical-extraction/batch",
            json={"extractions": extractions}
        )
        
        assert response.status_code == 422
    
    def test_batch_extract_endpoint_partial_failure(self, client, mock_service):
        """Test POST /batch handles partial failures"""
        from app.models.medical_extraction import MedicalExtractionResponse
        
        # First succeeds, second fails
        mock_service.extract_medical_term = Mock(
            side_effect=[
                MedicalExtractionResponse(
                    extracted_value="Positive",
                    confidence=0.95,
                    is_valid=True,
                    retry_count=0,
                    field_type=MedicalFieldType.HBSAG
                ),
                Exception("API Error")
            ]
        )
        
        response = client.post(
            "/api/v1/medical-extraction/batch",
            json={
                "extractions": [
                    {
                        "raw_text": "Patient is Hep B positive",
                        "field_type": "hbsag"
                    },
                    {
                        "raw_text": "Treatment location",
                        "field_type": "treatment_location"
                    }
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert data["successful_count"] == 1
        assert data["failed_count"] == 1
    
    def test_accuracy_metrics_endpoint(self, client, mock_service):
        """Test GET /accuracy-metrics endpoint"""
        from app.models.medical_extraction import AccuracyMetrics
        
        mock_metrics = AccuracyMetrics(
            total_extractions=100,
            successful_extractions=96,
            failed_extractions=4,
            accuracy_rate=0.96,
            average_confidence=0.93,
            retry_rate=0.08,
            validation_failure_rate=0.04,
            metrics_by_field={}
        )
        
        mock_service.get_accuracy_metrics = Mock(return_value=mock_metrics)
        
        response = client.get("/api/v1/medical-extraction/accuracy-metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_extractions"] == 100
        assert data["accuracy_rate"] == 0.96
        assert data["average_confidence"] == 0.93
    
    def test_extract_endpoint_response_format(self, client, mock_service):
        """Test response format compliance"""
        from app.models.medical_extraction import MedicalExtractionResponse
        
        mock_response = MedicalExtractionResponse(
            extracted_value="Positive",
            confidence=0.95,
            is_valid=True,
            retry_count=0,
            validation_message=None,
            raw_text_snippet="Hep B positive",
            field_type=MedicalFieldType.HBSAG
        )
        mock_service.extract_medical_term = Mock(return_value=mock_response)
        
        response = client.post(
            "/api/v1/medical-extraction/extract",
            json={
                "raw_text": "Patient is Hep B positive",
                "field_type": "hbsag"
            }
        )
        
        data = response.json()
        
        # Verify all required fields present
        assert "extracted_value" in data
        assert "confidence" in data
        assert "is_valid" in data
        assert "retry_count" in data
        assert "field_type" in data
    
    def test_extract_endpoint_with_context(self, client, mock_service):
        """Test extraction with additional context"""
        from app.models.medical_extraction import MedicalExtractionResponse
        
        mock_response = MedicalExtractionResponse(
            extracted_value="Positive",
            confidence=0.95,
            is_valid=True,
            retry_count=0,
            field_type=MedicalFieldType.HBSAG
        )
        mock_service.extract_medical_term = Mock(return_value=mock_response)
        
        response = client.post(
            "/api/v1/medical-extraction/extract",
            json={
                "raw_text": "Patient is Hep B positive",
                "field_type": "hbsag",
                "context": "Patient intake form"
            }
        )
        
        assert response.status_code == 200
    
    def test_extract_endpoint_with_conversation_history(self, client, mock_service):
        """Test extraction with conversation history"""
        from app.models.medical_extraction import MedicalExtractionResponse
        
        mock_response = MedicalExtractionResponse(
            extracted_value="Positive",
            confidence=0.95,
            is_valid=True,
            retry_count=0,
            field_type=MedicalFieldType.HBSAG
        )
        mock_service.extract_medical_term = Mock(return_value=mock_response)
        
        response = client.post(
            "/api/v1/medical-extraction/extract",
            json={
                "raw_text": "Patient is Hep B positive",
                "field_type": "hbsag",
                "conversation_history": [
                    {"role": "user", "content": "What is your HBsAg status?"}
                ]
            }
        )
        
        assert response.status_code == 200
