from fastapi import Depends, HTTPException, status
from functools import lru_cache
from app.core.config import Settings, settings
from app.services.extraction_service import ExtractionService, get_extraction_service
from app.services.data_cleaning_service import DataCleaningService, get_data_cleaning_service
from app.services.schema_validation_service import SchemaValidationService
from app.services.medical_term_extraction_service import MedicalTermExtractionService
from app.core.schema_loader import SchemaLoader


def get_settings() -> Settings:
    return settings


async def get_db():
    yield None


async def get_current_user(db=Depends(get_db)):
    return {"id": "placeholder-user-id", "username": "placeholder", "email": "user@example.com"}


def get_extraction_service_dependency() -> ExtractionService:
    """Dependency for injecting ExtractionService into endpoints."""
    return get_extraction_service()


def get_data_cleaning_service_dependency() -> DataCleaningService:
    """Dependency for injecting DataCleaningService into endpoints."""
    return get_data_cleaning_service()


@lru_cache(maxsize=1)
def get_schema_validation_service() -> SchemaValidationService:
    """Dependency for injecting SchemaValidationService into endpoints."""
    schema_loader = SchemaLoader()
    return SchemaValidationService(schema_loader=schema_loader)


@lru_cache(maxsize=1)
def get_medical_extraction_service() -> MedicalTermExtractionService:
    """Dependency for injecting MedicalTermExtractionService into endpoints."""
    schema_validation_service = get_schema_validation_service()
    return MedicalTermExtractionService(schema_validation_service=schema_validation_service)
