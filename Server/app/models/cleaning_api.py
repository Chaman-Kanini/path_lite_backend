from pydantic import BaseModel, Field
from typing import Optional, List
from app.models.structured_outputs import FieldType
from app.models.data_cleaning import CleaningConfig


class CleanRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Raw conversational text to clean"
    )
    field_type: FieldType = Field(
        ...,
        description="Type of field to clean"
    )
    config: Optional[CleaningConfig] = Field(
        default=None,
        description="Optional cleaning configuration"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Request model for single field cleaning",
            "examples": [
                {
                    "text": "Patient name is JOHN SMITH",
                    "field_type": "first_name",
                    "config": None
                }
            ]
        }
    }


class CleanResponse(BaseModel):
    cleaned_value: str = Field(
        ...,
        description="Cleaned value after transformations"
    )
    original: str = Field(
        ...,
        description="Original value before cleaning"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the cleaning operation"
    )
    requires_review: bool = Field(
        default=False,
        description="Flag indicating if manual review is needed"
    )
    clarification_needed: bool = Field(
        default=False,
        description="Flag indicating if clarification is needed"
    )
    clarification_message: Optional[str] = Field(
        default=None,
        description="Message requesting clarification from user"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Response model for single field cleaning",
            "examples": [
                {
                    "cleaned_value": "John Smith",
                    "original": "Patient name is JOHN SMITH",
                    "confidence_score": 0.92,
                    "requires_review": False,
                    "clarification_needed": False,
                    "clarification_message": None
                }
            ]
        }
    }


class BatchCleanRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Raw conversational text to clean"
    )
    field_types: List[FieldType] = Field(
        ...,
        min_length=1,
        description="List of field types to clean"
    )
    config: Optional[CleaningConfig] = Field(
        default=None,
        description="Optional cleaning configuration"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Request model for batch field cleaning",
            "examples": [
                {
                    "text": "Patient name is JOHN SMITH, born on January 15th, 1980",
                    "field_types": ["first_name", "last_name", "dob"],
                    "config": None
                }
            ]
        }
    }


class BatchCleanResponse(BaseModel):
    fields: List[CleanResponse] = Field(
        default_factory=list,
        description="List of cleaned fields"
    )
    requires_review: bool = Field(
        default=False,
        description="Flag indicating if any field requires manual review"
    )
    clarification_needed: bool = Field(
        default=False,
        description="Flag indicating if clarification is needed"
    )
    clarification_message: Optional[str] = Field(
        default=None,
        description="Message requesting clarification from user"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Response model for batch field cleaning",
            "examples": [
                {
                    "fields": [
                        {
                            "cleaned_value": "John Smith",
                            "original": "Patient name is JOHN SMITH",
                            "confidence_score": 0.92,
                            "requires_review": False,
                            "clarification_needed": False,
                            "clarification_message": None
                        }
                    ],
                    "requires_review": False,
                    "clarification_needed": False,
                    "clarification_message": None
                }
            ]
        }
    }
