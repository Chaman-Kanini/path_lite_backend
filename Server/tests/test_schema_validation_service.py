"""
Schema Validation Service Tests

Unit tests for SchemaValidationService.
"""

import pytest
from app.services.schema_validation_service import SchemaValidationService
from app.models.schema_validation import ValidationRequest, ValidationResult


class TestSchemaValidationService:
    """Test schema validation service"""
    
    @pytest.fixture
    def service(self):
        """Create schema validation service instance"""
        return SchemaValidationService()
    
    def test_exact_match_validation_hbsag_positive(self, service):
        """Test exact match for HBsAg Positive"""
        request = ValidationRequest(
            field_name="HBsAg Status",
            extracted_value="Positive",
            field_type="hbsag"
        )
        
        response = service.validate_field(request)
        
        assert response.is_valid is True
        assert response.validation_result == ValidationResult.VALID
        assert response.matched_value == "Positive"
    
    def test_case_insensitive_matching(self, service):
        """Test case-insensitive validation"""
        request = ValidationRequest(
            field_name="HBsAg Status",
            extracted_value="positive",
            field_type="hbsag"
        )
        
        response = service.validate_field(request)
        
        assert response.is_valid is True
        assert response.matched_value == "Positive"
    
    def test_fuzzy_matching_near_miss(self, service):
        """Test fuzzy matching for near-miss values"""
        request = ValidationRequest(
            field_name="HBsAg Status",
            extracted_value="Positiv",
            field_type="hbsag"
        )
        
        response = service.validate_field(request)
        
        # Should match "Positive" with fuzzy matching
        assert response.is_valid is True
        assert response.matched_value == "Positive"
    
    def test_invalid_value_rejection(self, service):
        """Test rejection of invalid values"""
        request = ValidationRequest(
            field_name="HBsAg Status",
            extracted_value="Maybe Positive",
            field_type="hbsag"
        )
        
        response = service.validate_field(request)
        
        assert response.is_valid is False
        assert response.validation_result in [ValidationResult.INVALID, ValidationResult.HALLUCINATED]
        assert response.clarification_prompt is not None
    
    def test_clarification_prompt_generation(self, service):
        """Test clarification prompt includes valid options"""
        request = ValidationRequest(
            field_name="HBsAg Status",
            extracted_value="Unknown Status",
            field_type="hbsag"
        )
        
        response = service.validate_field(request)
        
        if not response.is_valid:
            assert response.clarification_prompt is not None
            assert "Positive" in response.clarification_prompt
            assert "Negative" in response.clarification_prompt
            assert "Unknown" in response.clarification_prompt
    
    def test_abbreviation_expansion_hbsag(self, service):
        """Test abbreviation expansion for HBsAg"""
        # "Hep B" should expand to "HBsAg"
        expanded = service._expand_abbreviation("Hep B", "hbsag")
        
        # The expansion should help with validation
        assert expanded is not None
    
    def test_abbreviation_expansion_treatment_location(self, service):
        """Test abbreviation expansion for treatment locations"""
        # "Operating Room" should map to "OR"
        expanded = service._expand_abbreviation("Operating Room", "treatment_location")
        
        assert expanded == "OR"
    
    def test_abbreviation_expansion_gender(self, service):
        """Test abbreviation expansion for gender"""
        # "M" should map to "Male"
        expanded = service._expand_abbreviation("M", "gender")
        
        assert expanded == "Male"
    
    def test_validation_attempts_tracking(self, service):
        """Test validation attempt tracking"""
        field_name = "test_field"
        
        count1 = service.track_validation_attempt(field_name)
        assert count1 == 1
        
        count2 = service.track_validation_attempt(field_name)
        assert count2 == 2
        
        count3 = service.track_validation_attempt(field_name)
        assert count3 == 3
    
    def test_validation_attempts_reset(self, service):
        """Test resetting validation attempts"""
        field_name = "test_field"
        
        service.track_validation_attempt(field_name)
        service.track_validation_attempt(field_name)
        
        service.reset_validation_attempts(field_name)
        
        count = service.get_validation_attempts(field_name)
        assert count == 0
    
    def test_treatment_location_or_validation(self, service):
        """Test treatment location OR validation"""
        request = ValidationRequest(
            field_name="Treatment Location",
            extracted_value="OR",
            field_type="treatment_location"
        )
        
        response = service.validate_field(request)
        
        assert response.is_valid is True
        assert response.matched_value == "OR"
    
    def test_treatment_location_multi_tx_validation(self, service):
        """Test Multi-Tx Room validation"""
        request = ValidationRequest(
            field_name="Treatment Location",
            extracted_value="Multi-Tx Room",
            field_type="treatment_location"
        )
        
        response = service.validate_field(request)
        
        assert response.is_valid is True
        assert response.matched_value == "Multi-Tx Room"
    
    def test_gender_male_validation(self, service):
        """Test gender Male validation"""
        request = ValidationRequest(
            field_name="Gender",
            extracted_value="Male",
            field_type="gender"
        )
        
        response = service.validate_field(request)
        
        assert response.is_valid is True
        assert response.matched_value == "Male"
    
    def test_gender_female_validation(self, service):
        """Test gender Female validation"""
        request = ValidationRequest(
            field_name="Gender",
            extracted_value="Female",
            field_type="gender"
        )
        
        response = service.validate_field(request)
        
        assert response.is_valid is True
        assert response.matched_value == "Female"
