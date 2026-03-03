"""
Medical Extraction Models

Pydantic models for medical term extraction requests and responses.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class MedicalFieldType(str, Enum):
    """Medical field types for extraction"""
    HBSAG = "hbsag"
    TREATMENT_LOCATION = "treatment_location"
    GENDER = "gender"


class MedicalExtractionRequest(BaseModel):
    """Request to extract a medical term from raw text"""
    raw_text: str = Field(..., description="Raw conversational text from patient")
    field_type: MedicalFieldType = Field(..., description="Type of medical field to extract")
    context: Optional[str] = Field(None, description="Additional context for extraction")
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Previous conversation turns for context"
    )
    
    @field_validator('raw_text')
    @classmethod
    def validate_raw_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("raw_text cannot be empty")
        if len(v) > 10000:
            raise ValueError("raw_text exceeds maximum length of 10000 characters")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "raw_text": "Patient is Hep B positive",
                "field_type": "hbsag",
                "context": "Extracting HBsAg status from patient intake",
                "conversation_history": []
            }
        }


class MedicalExtractionResponse(BaseModel):
    """Response from medical term extraction"""
    extracted_value: Optional[str] = Field(None, description="Extracted medical term value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of extraction")
    is_valid: bool = Field(..., description="Whether extracted value is valid per schema")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    validation_message: Optional[str] = Field(None, description="Validation error or warning message")
    raw_text_snippet: Optional[str] = Field(None, description="Text snippet that supports extraction")
    field_type: MedicalFieldType = Field(..., description="Type of medical field extracted")
    
    class Config:
        json_schema_extra = {
            "example": {
                "extracted_value": "Positive",
                "confidence": 0.95,
                "is_valid": True,
                "retry_count": 0,
                "validation_message": None,
                "raw_text_snippet": "Hep B positive",
                "field_type": "hbsag"
            }
        }


class RePromptRequest(BaseModel):
    """Request to re-prompt for clarification"""
    original_text: str = Field(..., description="Original patient response")
    invalid_value: str = Field(..., description="Invalid value that was extracted")
    valid_options: List[str] = Field(..., description="List of valid options")
    field_name: str = Field(..., description="Name of the field being clarified")
    
    class Config:
        json_schema_extra = {
            "example": {
                "original_text": "Patient has some liver condition",
                "invalid_value": "liver condition",
                "valid_options": ["Positive", "Negative", "Unknown"],
                "field_name": "HBsAg Status"
            }
        }


class BatchExtractionRequest(BaseModel):
    """Request to extract multiple medical terms"""
    extractions: List[MedicalExtractionRequest] = Field(
        ...,
        description="List of extraction requests",
        min_length=1,
        max_length=10
    )
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class BatchExtractionResponse(BaseModel):
    """Response from batch extraction"""
    results: List[MedicalExtractionResponse] = Field(..., description="List of extraction results")
    total_count: int = Field(..., description="Total number of extractions")
    successful_count: int = Field(..., description="Number of successful extractions")
    failed_count: int = Field(..., description="Number of failed extractions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "extracted_value": "Positive",
                        "confidence": 0.95,
                        "is_valid": True,
                        "retry_count": 0,
                        "field_type": "hbsag"
                    }
                ],
                "total_count": 1,
                "successful_count": 1,
                "failed_count": 0
            }
        }


class AccuracyMetrics(BaseModel):
    """Accuracy metrics for medical term extraction"""
    total_extractions: int = Field(..., description="Total number of extractions")
    successful_extractions: int = Field(..., description="Number of successful extractions")
    failed_extractions: int = Field(..., description="Number of failed extractions")
    accuracy_rate: float = Field(..., ge=0.0, le=1.0, description="Overall accuracy rate")
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence score")
    retry_rate: float = Field(..., ge=0.0, le=1.0, description="Percentage of extractions requiring retry")
    validation_failure_rate: float = Field(..., ge=0.0, le=1.0, description="Percentage of validation failures")
    metrics_by_field: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metrics broken down by field type"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_extractions": 100,
                "successful_extractions": 96,
                "failed_extractions": 4,
                "accuracy_rate": 0.96,
                "average_confidence": 0.93,
                "retry_rate": 0.08,
                "validation_failure_rate": 0.04,
                "metrics_by_field": {
                    "hbsag": {
                        "accuracy": 0.97,
                        "count": 40
                    }
                }
            }
        }
