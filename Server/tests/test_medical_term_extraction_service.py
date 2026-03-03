"""
Medical Term Extraction Service Tests

Unit tests for MedicalTermExtractionService with mocked Azure OpenAI.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from app.services.medical_term_extraction_service import MedicalTermExtractionService
from app.services.schema_validation_service import SchemaValidationService
from app.models.medical_extraction import MedicalExtractionRequest, MedicalFieldType
from tests.fixtures.mock_medical_responses import (
    MockChatCompletion,
    MOCK_HBSAG_POSITIVE,
    MOCK_HBSAG_NEGATIVE,
    MOCK_LOCATION_OR,
    MOCK_GENDER_MALE
)


class TestMedicalTermExtractionService:
    """Test medical term extraction service"""
    
    @pytest.fixture
    def mock_azure_client(self):
        """Create mock Azure OpenAI client"""
        client = Mock()
        client.chat = Mock()
        client.chat.completions = Mock()
        return client
    
    @pytest.fixture
    def schema_validation_service(self):
        """Create schema validation service"""
        return SchemaValidationService()
    
    @pytest.fixture
    def service(self, mock_azure_client, schema_validation_service):
        """Create medical term extraction service with mocks"""
        return MedicalTermExtractionService(
            azure_client=mock_azure_client,
            schema_validation_service=schema_validation_service
        )
    
    @pytest.mark.asyncio
    async def test_extract_hbsag_positive(self, service, mock_azure_client):
        """Test HBsAg positive extraction"""
        # Mock Azure OpenAI response
        mock_response = MockChatCompletion(json.dumps(MOCK_HBSAG_POSITIVE))
        mock_azure_client.chat.completions.create = Mock(return_value=mock_response)
        
        request = MedicalExtractionRequest(
            raw_text="Patient is Hep B positive",
            field_type=MedicalFieldType.HBSAG
        )
        
        response = await service.extract_medical_term(request)
        
        assert response.extracted_value == "Positive"
        assert response.is_valid is True
        assert response.confidence > 0.9
    
    @pytest.mark.asyncio
    async def test_extract_hbsag_negative(self, service, mock_azure_client):
        """Test HBsAg negative extraction"""
        mock_response = MockChatCompletion(json.dumps(MOCK_HBSAG_NEGATIVE))
        mock_azure_client.chat.completions.create = Mock(return_value=mock_response)
        
        request = MedicalExtractionRequest(
            raw_text="HBsAg test came back negative",
            field_type=MedicalFieldType.HBSAG
        )
        
        response = await service.extract_medical_term(request)
        
        assert response.extracted_value == "Negative"
        assert response.is_valid is True
    
    @pytest.mark.asyncio
    async def test_extract_treatment_location_or(self, service, mock_azure_client):
        """Test treatment location OR extraction"""
        mock_response = MockChatCompletion(json.dumps(MOCK_LOCATION_OR))
        mock_azure_client.chat.completions.create = Mock(return_value=mock_response)
        
        request = MedicalExtractionRequest(
            raw_text="Patient is in the operating room",
            field_type=MedicalFieldType.TREATMENT_LOCATION
        )
        
        response = await service.extract_medical_term(request)
        
        assert response.extracted_value == "OR"
        assert response.is_valid is True
    
    @pytest.mark.asyncio
    async def test_extract_gender_male(self, service, mock_azure_client):
        """Test gender male extraction"""
        mock_response = MockChatCompletion(json.dumps(MOCK_GENDER_MALE))
        mock_azure_client.chat.completions.create = Mock(return_value=mock_response)
        
        request = MedicalExtractionRequest(
            raw_text="The patient is male",
            field_type=MedicalFieldType.GENDER
        )
        
        response = await service.extract_medical_term(request)
        
        assert response.extracted_value == "Male"
        assert response.is_valid is True
    
    @pytest.mark.asyncio
    async def test_invalid_value_triggers_retry(self, service, mock_azure_client):
        """Test that invalid values trigger re-prompting"""
        # First call returns invalid value
        invalid_response = {
            "status": "Maybe Positive",
            "confidence": 0.70,
            "raw_text": "unclear"
        }
        
        # Second call returns valid value
        valid_response = MOCK_HBSAG_POSITIVE
        
        mock_azure_client.chat.completions.create = Mock(
            side_effect=[
                MockChatCompletion(json.dumps(invalid_response)),
                MockChatCompletion(json.dumps(valid_response))
            ]
        )
        
        request = MedicalExtractionRequest(
            raw_text="Patient might have Hep B",
            field_type=MedicalFieldType.HBSAG
        )
        
        response = await service.extract_medical_term(request)
        
        # Should succeed after retry
        assert response.retry_count >= 1
    
    @pytest.mark.asyncio
    async def test_max_retries_enforced(self, service, mock_azure_client):
        """Test max retries (3) are enforced"""
        # Always return invalid value
        invalid_response = {
            "status": "Invalid Status",
            "confidence": 0.70,
            "raw_text": "unclear"
        }
        
        mock_azure_client.chat.completions.create = Mock(
            return_value=MockChatCompletion(json.dumps(invalid_response))
        )
        
        request = MedicalExtractionRequest(
            raw_text="Unclear status",
            field_type=MedicalFieldType.HBSAG
        )
        
        response = await service.extract_medical_term(request)
        
        # Should fail after max retries
        assert response.is_valid is False
        assert response.retry_count == service.MAX_RETRIES
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self, service, mock_azure_client):
        """Test confidence scores are captured"""
        mock_response = MockChatCompletion(json.dumps(MOCK_HBSAG_POSITIVE))
        mock_azure_client.chat.completions.create = Mock(return_value=mock_response)
        
        request = MedicalExtractionRequest(
            raw_text="Patient is Hep B positive",
            field_type=MedicalFieldType.HBSAG
        )
        
        response = await service.extract_medical_term(request)
        
        assert 0.0 <= response.confidence <= 1.0
        assert response.confidence == MOCK_HBSAG_POSITIVE["confidence"]
    
    def test_accuracy_metrics_tracking(self, service):
        """Test accuracy metrics are tracked"""
        metrics = service.get_accuracy_metrics()
        
        assert metrics.total_extractions >= 0
        assert metrics.successful_extractions >= 0
        assert metrics.failed_extractions >= 0
        assert 0.0 <= metrics.accuracy_rate <= 1.0
    
    def test_prompt_templates_loaded(self, service):
        """Test prompt templates are loaded on initialization"""
        assert MedicalFieldType.HBSAG in service.prompt_templates
        assert MedicalFieldType.TREATMENT_LOCATION in service.prompt_templates
        assert MedicalFieldType.GENDER in service.prompt_templates
    
    @pytest.mark.asyncio
    async def test_error_handling(self, service, mock_azure_client):
        """Test error handling for API failures"""
        # Simulate API error
        mock_azure_client.chat.completions.create = Mock(
            side_effect=Exception("API Error")
        )
        
        request = MedicalExtractionRequest(
            raw_text="Patient is Hep B positive",
            field_type=MedicalFieldType.HBSAG
        )
        
        response = await service.extract_medical_term(request)
        
        # Should return failed response
        assert response.is_valid is False
        assert response.validation_message is not None
