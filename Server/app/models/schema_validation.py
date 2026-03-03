"""
Schema Validation Models

Pydantic models for medical term schema validation, including validation requests,
responses, and field schema definitions.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class ValidationResult(str, Enum):
    """Validation result status"""
    VALID = "valid"
    INVALID = "invalid"
    HALLUCINATED = "hallucinated"
    AMBIGUOUS = "ambiguous"


class FieldSchema(BaseModel):
    """Schema definition for a medical field"""
    field_name: str = Field(..., description="Human-readable field name")
    field_type: str = Field(..., description="Field type (enum, string, etc.)")
    enum_values: List[str] = Field(..., description="Valid enum values")
    validation_rules: List[str] = Field(default_factory=list, description="Validation rules")
    fuzzy_matching: dict = Field(default_factory=dict, description="Fuzzy matching configuration")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    description: Optional[str] = Field(None, description="Field description")
    privacy_note: Optional[str] = Field(None, description="Privacy considerations")


class ValidationRequest(BaseModel):
    """Request to validate an extracted medical term"""
    field_name: str = Field(..., description="Name of the field being validated")
    extracted_value: str = Field(..., description="Value extracted by LLM")
    field_type: str = Field(..., description="Type of field (hbsag, treatment_location, gender)")
    
    @field_validator('extracted_value')
    @classmethod
    def validate_extracted_value(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("extracted_value cannot be empty")
        return v.strip()


class ValidationResponse(BaseModel):
    """Response from schema validation"""
    is_valid: bool = Field(..., description="Whether the value is valid")
    validation_result: ValidationResult = Field(..., description="Detailed validation result")
    error_message: Optional[str] = Field(None, description="Error message if invalid")
    suggested_values: List[str] = Field(default_factory=list, description="Suggested valid values")
    clarification_prompt: Optional[str] = Field(None, description="Prompt for re-asking the question")
    matched_value: Optional[str] = Field(None, description="Matched enum value if fuzzy matched")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in validation result")
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_valid": True,
                "validation_result": "valid",
                "error_message": None,
                "suggested_values": [],
                "clarification_prompt": None,
                "matched_value": "Positive",
                "confidence": 0.95
            }
        }
