from pydantic import BaseModel, Field
from typing import Any, List, Optional
from app.models.structured_outputs import FieldType


class ExtractionRequest(BaseModel):
    """Request model for single field extraction."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Conversational text from STT or user input"
    )
    field_type: FieldType = Field(
        ...,
        description="Type of field to extract"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "The patient's MRN is 12345",
                    "field_type": "mrn"
                }
            ]
        }
    }


class ExtractionResponse(BaseModel):
    """Response model for field extraction."""
    
    value: Any = Field(
        ...,
        description="Extracted field value"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of extraction (0.0-1.0)"
    )
    field_type: str = Field(
        ...,
        description="Type of field extracted"
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds"
    )
    fallback_to_manual: bool = Field(
        default=False,
        description="Whether to fallback to manual entry"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "value": {"mrn": "12345", "confidence_score": 0.95},
                    "confidence_score": 0.95,
                    "field_type": "mrn",
                    "processing_time_ms": 850,
                    "fallback_to_manual": False
                }
            ]
        }
    }


class BatchExtractionRequest(BaseModel):
    """Request model for batch field extraction."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Conversational text from STT or user input"
    )
    field_types: List[FieldType] = Field(
        ...,
        min_length=1,
        description="List of field types to extract"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Patient John Doe, MRN 12345, born May 15, 1990",
                    "field_types": ["mrn", "first_name", "dob"]
                }
            ]
        }
    }


class BatchExtractionResponse(BaseModel):
    """Response model for batch field extraction."""
    
    extractions: List[ExtractionResponse] = Field(
        ...,
        description="List of extraction results"
    )
    total_processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Total processing time in milliseconds"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "extractions": [
                        {
                            "value": {"mrn": "12345", "confidence_score": 0.95},
                            "confidence_score": 0.95,
                            "field_type": "mrn",
                            "processing_time_ms": 450,
                            "fallback_to_manual": False
                        }
                    ],
                    "total_processing_time_ms": 850
                }
            ]
        }
    }


class PatientDataExtractionRequest(BaseModel):
    """Request model for complete patient data extraction."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Conversational text from STT or user input"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Patient John Doe, MRN 12345, born May 15, 1990, male"
                }
            ]
        }
    }


class PatientDataExtractionResponse(BaseModel):
    """Response model for complete patient data extraction."""
    
    data: dict = Field(
        ...,
        description="Extracted patient data with all available fields"
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds"
    )
    fallback_to_manual: bool = Field(
        default=False,
        description="Whether to fallback to manual entry"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": {
                        "mrn": {"mrn": "12345", "confidence_score": 0.95},
                        "name": {
                            "first_name": "John",
                            "last_name": "Doe",
                            "confidence_score": 0.98
                        }
                    },
                    "processing_time_ms": 950,
                    "fallback_to_manual": False
                }
            ]
        }
    }


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(
        ...,
        description="Health status"
    )
    azure_openai_connected: bool = Field(
        ...,
        description="Whether Azure OpenAI is connected"
    )
    message: Optional[str] = Field(
        None,
        description="Additional status message"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "azure_openai_connected": True,
                    "message": "All systems operational"
                }
            ]
        }
    }
