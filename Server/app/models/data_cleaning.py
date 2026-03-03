from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from enum import Enum


class CleanedField(BaseModel):
    value: str = Field(
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
        description="Confidence score of the cleaning operation (0.0 to 1.0)"
    )
    requires_review: bool = Field(
        default=False,
        description="Flag indicating if manual review is needed (confidence < threshold)"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Cleaned field with confidence scoring and review flag",
            "examples": [
                {
                    "value": "John Smith",
                    "original": "Patient name is JOHN SMITH",
                    "confidence_score": 0.92,
                    "requires_review": False
                }
            ]
        }
    }


class FillerWordConfig(BaseModel):
    filler_words: List[str] = Field(
        default=["the", "is", "was", "patient", "name", "uh", "um", "like"],
        description="List of filler words to remove from text"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Configuration for filler word removal",
            "examples": [
                {
                    "filler_words": ["the", "is", "was", "patient", "name", "uh", "um", "like"]
                }
            ]
        }
    }


class CleaningConfig(BaseModel):
    filler_words: List[str] = Field(
        default=["the", "is", "was", "patient", "name", "uh", "um", "like"],
        description="List of filler words to remove from text"
    )
    confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required to avoid manual review"
    )
    preserve_patterns: List[str] = Field(
        default=["date", "decimal"],
        description="Patterns to preserve during punctuation cleaning"
    )

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v

    model_config = {
        "json_schema_extra": {
            "description": "Configuration for data cleaning operations",
            "examples": [
                {
                    "filler_words": ["the", "is", "was", "patient", "name"],
                    "confidence_threshold": 0.85,
                    "preserve_patterns": ["date", "decimal"]
                }
            ]
        }
    }


class CleaningResult(BaseModel):
    fields: List[CleanedField] = Field(
        default_factory=list,
        description="List of cleaned fields"
    )
    requires_review: bool = Field(
        default=False,
        description="Flag indicating if any field requires manual review"
    )
    clarification_needed: bool = Field(
        default=False,
        description="Flag indicating if clarification is needed (e.g., ambiguous dates)"
    )
    clarification_message: Optional[str] = Field(
        default=None,
        description="Message requesting clarification from user"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Result of cleaning operation with all processed fields",
            "examples": [
                {
                    "fields": [
                        {
                            "value": "John Smith",
                            "original": "Patient name is JOHN SMITH",
                            "confidence_score": 0.92,
                            "requires_review": False
                        }
                    ],
                    "requires_review": False,
                    "clarification_needed": False,
                    "clarification_message": None
                }
            ]
        }
    }
