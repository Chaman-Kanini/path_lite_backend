"""
Schema Validation Service

Validates LLM-extracted medical terms against predefined enum schemas.
Detects hallucinated values and generates clarification prompts for re-prompting.
"""

import yaml
from pathlib import Path
from typing import Optional, List, Tuple
from difflib import SequenceMatcher
import logging

from app.core.schema_loader import SchemaLoader
from app.models.schema_validation import (
    ValidationRequest,
    ValidationResponse,
    ValidationResult,
    FieldSchema
)

logger = logging.getLogger(__name__)


class SchemaValidationService:
    """Service for validating medical term extractions against schemas"""
    
    def __init__(self, schema_loader: Optional[SchemaLoader] = None):
        """
        Initialize schema validation service
        
        Args:
            schema_loader: SchemaLoader instance (creates new if None)
        """
        self.schema_loader = schema_loader or SchemaLoader()
        self.abbreviation_mappings = self._load_abbreviation_mappings()
        self.validation_attempts: dict = {}  # Track retry attempts per field
    
    def _load_abbreviation_mappings(self) -> dict:
        """Load abbreviation mappings from YAML"""
        try:
            base_path = Path(__file__).parent.parent
            abbrev_path = base_path / "prompts" / "medical_terms" / "abbreviation_mappings.yaml"
            
            if not abbrev_path.exists():
                logger.warning(f"Abbreviation mappings file not found: {abbrev_path}")
                return {}
            
            with open(abbrev_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load abbreviation mappings: {e}")
            return {}
    
    def validate_field(self, request: ValidationRequest) -> ValidationResponse:
        """
        Validate an extracted field value against schema
        
        Args:
            request: ValidationRequest with field details
        
        Returns:
            ValidationResponse with validation result
        """
        # Get field schema
        schema = self.schema_loader.get_field_schema(request.field_type)
        if not schema:
            logger.error(f"Schema not found for field type: {request.field_type}")
            return ValidationResponse(
                is_valid=False,
                validation_result=ValidationResult.INVALID,
                error_message=f"Schema not found for field type: {request.field_type}",
                suggested_values=[]
            )
        
        # Expand abbreviations
        expanded_value = self._expand_abbreviation(request.extracted_value, request.field_type)
        
        # Exact match validation (case-insensitive)
        exact_match = self._exact_match(expanded_value, schema.enum_values)
        if exact_match:
            return ValidationResponse(
                is_valid=True,
                validation_result=ValidationResult.VALID,
                matched_value=exact_match,
                confidence=1.0
            )
        
        # Fuzzy matching if enabled
        if schema.fuzzy_matching.get('enabled', False):
            fuzzy_threshold = schema.fuzzy_matching.get('threshold', 0.85)
            fuzzy_match, confidence = self._fuzzy_match(expanded_value, schema.enum_values, fuzzy_threshold)
            
            if fuzzy_match:
                return ValidationResponse(
                    is_valid=True,
                    validation_result=ValidationResult.VALID,
                    matched_value=fuzzy_match,
                    confidence=confidence
                )
        
        # Check if hallucinated
        is_hallucinated = self._detect_hallucination(expanded_value, request.field_type)
        
        # Generate clarification prompt
        clarification = self._generate_clarification_prompt(
            request.field_name,
            expanded_value,
            schema.enum_values
        )
        
        return ValidationResponse(
            is_valid=False,
            validation_result=ValidationResult.HALLUCINATED if is_hallucinated else ValidationResult.INVALID,
            error_message=f"Invalid value '{expanded_value}' for field '{request.field_name}'",
            suggested_values=schema.enum_values,
            clarification_prompt=clarification,
            confidence=0.0
        )
    
    def _exact_match(self, value: str, enum_values: List[str]) -> Optional[str]:
        """
        Perform case-insensitive exact match
        
        Args:
            value: Value to match
            enum_values: List of valid enum values
        
        Returns:
            Matched enum value or None
        """
        value_lower = value.lower().strip()
        for enum_value in enum_values:
            if enum_value.lower().strip() == value_lower:
                return enum_value
        return None
    
    def _fuzzy_match(self, value: str, enum_values: List[str], threshold: float) -> Tuple[Optional[str], float]:
        """
        Perform fuzzy string matching using Levenshtein distance
        
        Args:
            value: Value to match
            enum_values: List of valid enum values
            threshold: Similarity threshold (0.0-1.0)
        
        Returns:
            Tuple of (matched value, confidence) or (None, 0.0)
        """
        best_match = None
        best_ratio = 0.0
        
        value_lower = value.lower().strip()
        
        for enum_value in enum_values:
            enum_lower = enum_value.lower().strip()
            ratio = SequenceMatcher(None, value_lower, enum_lower).ratio()
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = enum_value
        
        if best_ratio >= threshold:
            return best_match, best_ratio
        
        return None, 0.0
    
    def _expand_abbreviation(self, value: str, field_type: str) -> str:
        """
        Expand abbreviations using abbreviation mappings
        
        Args:
            value: Value to expand
            field_type: Field type (hbsag, treatment_location, gender)
        
        Returns:
            Expanded value or original if no mapping found
        """
        if field_type not in self.abbreviation_mappings:
            return value
        
        field_mappings = self.abbreviation_mappings[field_type]
        value_lower = value.lower().strip()
        
        # For nested mappings (treatment_location, gender)
        if isinstance(field_mappings, dict):
            for key, mapping_data in field_mappings.items():
                if isinstance(mapping_data, dict):
                    # Check abbreviations
                    abbreviations = mapping_data.get('abbreviations', [])
                    for abbrev in abbreviations:
                        if abbrev.lower() == value_lower:
                            return key
                    
                    # Check variations
                    variations = mapping_data.get('variations', [])
                    for variation in variations:
                        if variation.lower() == value_lower:
                            return key
                    
                    # Check pronouns (for gender)
                    pronouns = mapping_data.get('pronouns', [])
                    for pronoun in pronouns:
                        if pronoun.lower() == value_lower:
                            return key
        
        # For simple mappings (hbsag)
        if 'abbreviations' in field_mappings:
            abbreviations = field_mappings.get('abbreviations', [])
            for abbrev in abbreviations:
                if abbrev.lower() == value_lower:
                    return field_mappings.get('full_term', value)
            
            variations = field_mappings.get('variations', [])
            for variation in variations:
                if variation.lower() == value_lower:
                    return field_mappings.get('full_term', value)
        
        return value
    
    def _detect_hallucination(self, value: str, field_type: str) -> bool:
        """
        Detect if value is a hallucinated medical term
        
        Args:
            value: Value to check
            field_type: Field type
        
        Returns:
            True if hallucinated, False otherwise
        """
        # Check if value exists in any schema
        all_schemas = self.schema_loader.get_all_schemas()
        
        for schema in all_schemas.values():
            if self._exact_match(value, schema.enum_values):
                # Value exists in a different schema - likely hallucinated for this field
                return True
        
        # Check if value is completely invalid (not in any schema)
        # This is a strong indicator of hallucination
        value_lower = value.lower().strip()
        common_invalid_patterns = [
            'unknown', 'n/a', 'not specified', 'unclear', 'none',
            'not mentioned', 'not stated', 'not applicable'
        ]
        
        if value_lower in common_invalid_patterns and field_type != 'hbsag':
            # 'Unknown' is valid for hbsag but not for other fields
            return False
        
        return False
    
    def _generate_clarification_prompt(
        self,
        field_name: str,
        invalid_value: str,
        valid_options: List[str]
    ) -> str:
        """
        Generate clarification prompt for re-asking the question
        
        Args:
            field_name: Name of the field
            invalid_value: Invalid value that was extracted
            valid_options: List of valid options
        
        Returns:
            Clarification prompt string
        """
        options_str = ", ".join(valid_options)
        
        prompt = (
            f"I extracted '{invalid_value}' for {field_name}, but this doesn't match our valid options. "
            f"Could you please clarify? The valid options are: {options_str}. "
            f"Which one best describes your situation?"
        )
        
        return prompt
    
    def track_validation_attempt(self, field_name: str) -> int:
        """
        Track validation attempts for a field
        
        Args:
            field_name: Name of the field
        
        Returns:
            Current attempt count
        """
        if field_name not in self.validation_attempts:
            self.validation_attempts[field_name] = 0
        
        self.validation_attempts[field_name] += 1
        return self.validation_attempts[field_name]
    
    def reset_validation_attempts(self, field_name: str) -> None:
        """Reset validation attempts for a field"""
        if field_name in self.validation_attempts:
            del self.validation_attempts[field_name]
    
    def get_validation_attempts(self, field_name: str) -> int:
        """Get current validation attempt count for a field"""
        return self.validation_attempts.get(field_name, 0)
