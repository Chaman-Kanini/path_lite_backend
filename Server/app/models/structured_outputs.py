from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from datetime import date
from enum import Enum


class FieldType(str, Enum):
    MRN = "mrn"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    DOB = "dob"
    GENDER = "gender"


class BaseExtraction(BaseModel):
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the extraction (0.0 to 1.0)"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Base model for all extraction results with confidence scoring"
        }
    }


class MRNExtraction(BaseExtraction):
    mrn: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Medical Record Number extracted from text"
    )

    @field_validator("mrn")
    @classmethod
    def validate_mrn(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("MRN cannot be empty or whitespace")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "description": "Medical Record Number extraction with validation",
            "examples": [
                {"mrn": "MRN12345", "confidence_score": 0.95}
            ]
        }
    }


class NameExtraction(BaseExtraction):
    first_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Patient's first name"
    )
    last_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Patient's last name"
    )

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be empty or whitespace")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "description": "Patient name extraction with validation",
            "examples": [
                {
                    "first_name": "John",
                    "last_name": "Doe",
                    "confidence_score": 0.98
                }
            ]
        }
    }


class DOBExtraction(BaseExtraction):
    dob: date = Field(
        ...,
        description="Patient's date of birth in YYYY-MM-DD format"
    )

    @field_validator("dob")
    @classmethod
    def validate_dob(cls, v: date) -> date:
        from datetime import date as date_type
        today = date_type.today()
        
        if v > today:
            raise ValueError("Date of birth cannot be in the future")
        
        min_year = today.year - 150
        if v.year < min_year:
            raise ValueError(f"Date of birth cannot be before year {min_year}")
        
        return v

    model_config = {
        "json_schema_extra": {
            "description": "Date of birth extraction with reasonable date range validation",
            "examples": [
                {"dob": "1990-05-15", "confidence_score": 0.92}
            ]
        }
    }


class GenderExtraction(BaseExtraction):
    gender: Literal["Male", "Female", "Other", "Unknown"] = Field(
        ...,
        description="Patient's gender"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Gender extraction with predefined options",
            "examples": [
                {"gender": "Male", "confidence_score": 0.99}
            ]
        }
    }


class PatientFieldExtraction(BaseModel):
    mrn: Optional[MRNExtraction] = Field(
        None,
        description="Extracted Medical Record Number"
    )
    name: Optional[NameExtraction] = Field(
        None,
        description="Extracted patient name"
    )
    dob: Optional[DOBExtraction] = Field(
        None,
        description="Extracted date of birth"
    )
    gender: Optional[GenderExtraction] = Field(
        None,
        description="Extracted gender"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Composite model for extracting all patient form fields from conversational text",
            "examples": [
                {
                    "mrn": {"mrn": "MRN12345", "confidence_score": 0.95},
                    "name": {
                        "first_name": "John",
                        "last_name": "Doe",
                        "confidence_score": 0.98
                    },
                    "dob": {"dob": "1990-05-15", "confidence_score": 0.92},
                    "gender": {"gender": "Male", "confidence_score": 0.99}
                }
            ]
        }
    }
