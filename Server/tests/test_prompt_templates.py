"""
Prompt Template Validation Tests

Tests for validating medical term prompt templates.
"""

import pytest
from pathlib import Path
import yaml


class TestPromptTemplates:
    """Test prompt template structure and content"""
    
    @pytest.fixture
    def prompts_path(self):
        """Get path to prompts directory"""
        base_path = Path(__file__).parent.parent
        return base_path / "app" / "prompts" / "medical_terms"
    
    def test_hbsag_prompt_exists(self, prompts_path):
        """Test HBsAg prompt template exists"""
        template_path = prompts_path / "hbsag_extraction.txt"
        assert template_path.exists(), "HBsAg prompt template not found"
    
    def test_treatment_location_prompt_exists(self, prompts_path):
        """Test treatment location prompt template exists"""
        template_path = prompts_path / "treatment_location_extraction.txt"
        assert template_path.exists(), "Treatment location prompt template not found"
    
    def test_gender_prompt_exists(self, prompts_path):
        """Test gender prompt template exists"""
        template_path = prompts_path / "gender_extraction.txt"
        assert template_path.exists(), "Gender prompt template not found"
    
    def test_hbsag_prompt_has_few_shot_examples(self, prompts_path):
        """Test HBsAg prompt has at least 5 few-shot examples"""
        template_path = prompts_path / "hbsag_extraction.txt"
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count example markers
        example_count = content.count("Example ")
        assert example_count >= 5, f"HBsAg prompt has {example_count} examples, need at least 5"
    
    def test_treatment_location_prompt_has_few_shot_examples(self, prompts_path):
        """Test treatment location prompt has at least 5 few-shot examples"""
        template_path = prompts_path / "treatment_location_extraction.txt"
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        example_count = content.count("Example ")
        assert example_count >= 5, f"Treatment location prompt has {example_count} examples, need at least 5"
    
    def test_gender_prompt_has_few_shot_examples(self, prompts_path):
        """Test gender prompt has at least 5 few-shot examples"""
        template_path = prompts_path / "gender_extraction.txt"
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        example_count = content.count("Example ")
        assert example_count >= 5, f"Gender prompt has {example_count} examples, need at least 5"
    
    def test_prompts_have_structured_output_schema(self, prompts_path):
        """Test all prompts define structured output schema"""
        templates = [
            "hbsag_extraction.txt",
            "treatment_location_extraction.txt",
            "gender_extraction.txt"
        ]
        
        for template_name in templates:
            template_path = prompts_path / template_name
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "Structured Output Schema" in content, f"{template_name} missing structured output schema"
            assert "confidence" in content.lower(), f"{template_name} missing confidence in schema"
    
    def test_prompts_have_validation_rules(self, prompts_path):
        """Test all prompts include validation rules"""
        templates = [
            "hbsag_extraction.txt",
            "treatment_location_extraction.txt",
            "gender_extraction.txt"
        ]
        
        for template_name in templates:
            template_path = prompts_path / template_name
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "Validation Rules" in content, f"{template_name} missing validation rules"
    
    def test_abbreviation_mappings_valid_yaml(self, prompts_path):
        """Test abbreviation mappings YAML is valid"""
        yaml_path = prompts_path / "abbreviation_mappings.yaml"
        assert yaml_path.exists(), "Abbreviation mappings YAML not found"
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        assert data is not None, "Abbreviation mappings YAML is empty"
        assert "hbsag" in data, "Missing hbsag abbreviations"
        assert "treatment_location" in data, "Missing treatment_location abbreviations"
        assert "gender" in data, "Missing gender abbreviations"
    
    def test_field_schemas_valid_yaml(self, prompts_path):
        """Test field schemas YAML is valid"""
        yaml_path = prompts_path / "field_schemas.yaml"
        assert yaml_path.exists(), "Field schemas YAML not found"
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        assert data is not None, "Field schemas YAML is empty"
        assert "fields" in data, "Missing 'fields' key in schema"
        
        fields = data["fields"]
        assert "hbsag" in fields, "Missing hbsag field schema"
        assert "treatment_location" in fields, "Missing treatment_location field schema"
        assert "gender" in fields, "Missing gender field schema"
    
    def test_field_schemas_have_enum_values(self, prompts_path):
        """Test field schemas define enum values"""
        yaml_path = prompts_path / "field_schemas.yaml"
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        fields = data["fields"]
        
        # HBsAg should have Positive, Negative, Unknown
        hbsag_enums = fields["hbsag"]["enum_values"]
        assert "Positive" in hbsag_enums
        assert "Negative" in hbsag_enums
        assert "Unknown" in hbsag_enums
        
        # Treatment location should have OR, Bedside, ICU-CCU, ER, Multi-Tx Room
        location_enums = fields["treatment_location"]["enum_values"]
        assert "OR" in location_enums
        assert "Bedside" in location_enums
        assert "ICU-CCU" in location_enums
        assert "ER" in location_enums
        assert "Multi-Tx Room" in location_enums
        
        # Gender should have Male, Female
        gender_enums = fields["gender"]["enum_values"]
        assert "Male" in gender_enums
        assert "Female" in gender_enums
    
    def test_prompts_have_variable_placeholders(self, prompts_path):
        """Test prompts include variable placeholders for substitution"""
        templates = [
            "hbsag_extraction.txt",
            "treatment_location_extraction.txt",
            "gender_extraction.txt"
        ]
        
        for template_name in templates:
            template_path = prompts_path / template_name
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "{raw_text}" in content, f"{template_name} missing {{raw_text}} placeholder"
