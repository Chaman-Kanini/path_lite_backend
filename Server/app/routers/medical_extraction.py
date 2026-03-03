"""
Medical Extraction API Router

API endpoints for medical term extraction from conversational patient responses.
"""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from uuid import uuid4

from app.models.medical_extraction import (
    MedicalExtractionRequest,
    MedicalExtractionResponse,
    BatchExtractionRequest,
    BatchExtractionResponse,
    AccuracyMetrics
)
from app.services.medical_term_extraction_service import MedicalTermExtractionService
from app.dependencies import get_schema_validation_service
from app.services.schema_validation_service import SchemaValidationService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/medical-extraction",
    tags=["Medical Extraction"]
)

# Service instance (singleton pattern)
_medical_extraction_service: MedicalTermExtractionService = None


def get_medical_extraction_service(
    schema_validation_service: SchemaValidationService = Depends(get_schema_validation_service)
) -> MedicalTermExtractionService:
    """Get or create medical extraction service instance"""
    global _medical_extraction_service
    if _medical_extraction_service is None:
        _medical_extraction_service = MedicalTermExtractionService(
            schema_validation_service=schema_validation_service
        )
    return _medical_extraction_service


@router.post(
    "/extract",
    response_model=MedicalExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract medical term from patient response",
    description="Extract a single medical term (HBsAg, Treatment Location, or Gender) from conversational patient text using Azure OpenAI with few-shot prompts and schema validation."
)
async def extract_medical_term(
    request: MedicalExtractionRequest,
    service: MedicalTermExtractionService = Depends(get_medical_extraction_service)
) -> MedicalExtractionResponse:
    """
    Extract a medical term from raw patient text.
    
    **Process:**
    1. Load appropriate prompt template for field type
    2. Call Azure OpenAI with few-shot examples
    3. Validate extracted value against schema
    4. Retry with clarification if validation fails (max 3 attempts)
    5. Return validated extraction result
    
    **Field Types:**
    - `hbsag`: HBsAg status (Positive, Negative, Unknown)
    - `treatment_location`: Treatment location (OR, Bedside, ICU-CCU, ER, Multi-Tx Room)
    - `gender`: Patient gender (Male, Female)
    
    **Example:**
    ```json
    {
      "raw_text": "Patient is Hep B positive",
      "field_type": "hbsag",
      "context": "Patient intake form"
    }
    ```
    
    **Returns:**
    - Extracted value with confidence score
    - Validation status
    - Retry count if re-prompting occurred
    """
    correlation_id = str(uuid4())
    logger.info(
        f"[{correlation_id}] Medical extraction request: field_type={request.field_type}, "
        f"text_length={len(request.raw_text)}"
    )
    
    try:
        response = await service.extract_medical_term(request)
        
        logger.info(
            f"[{correlation_id}] Extraction result: value={response.extracted_value}, "
            f"valid={response.is_valid}, confidence={response.confidence:.2f}, "
            f"retries={response.retry_count}"
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"[{correlation_id}] Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"[{correlation_id}] Extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Medical term extraction failed"
        )


@router.post(
    "/batch",
    response_model=BatchExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract multiple medical terms in batch",
    description="Extract multiple medical terms from different patient responses in a single request. Maximum 10 extractions per batch."
)
async def batch_extract_medical_terms(
    request: BatchExtractionRequest,
    service: MedicalTermExtractionService = Depends(get_medical_extraction_service)
) -> BatchExtractionResponse:
    """
    Extract multiple medical terms in batch.
    
    **Limits:**
    - Minimum: 1 extraction
    - Maximum: 10 extractions per batch
    
    **Example:**
    ```json
    {
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
    ```
    
    **Returns:**
    - List of extraction results
    - Summary statistics (total, successful, failed)
    """
    correlation_id = str(uuid4())
    logger.info(f"[{correlation_id}] Batch extraction request: count={len(request.extractions)}")
    
    results: List[MedicalExtractionResponse] = []
    successful_count = 0
    failed_count = 0
    
    for idx, extraction_request in enumerate(request.extractions):
        try:
            result = await service.extract_medical_term(extraction_request)
            results.append(result)
            
            if result.is_valid:
                successful_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            logger.error(f"[{correlation_id}] Batch item {idx} failed: {e}")
            
            # Add failed result
            results.append(
                MedicalExtractionResponse(
                    extracted_value=None,
                    confidence=0.0,
                    is_valid=False,
                    retry_count=0,
                    validation_message=f"Extraction failed: {str(e)}",
                    raw_text_snippet="",
                    field_type=extraction_request.field_type
                )
            )
            failed_count += 1
    
    logger.info(
        f"[{correlation_id}] Batch extraction complete: "
        f"total={len(results)}, successful={successful_count}, failed={failed_count}"
    )
    
    return BatchExtractionResponse(
        results=results,
        total_count=len(results),
        successful_count=successful_count,
        failed_count=failed_count
    )


@router.get(
    "/accuracy-metrics",
    response_model=AccuracyMetrics,
    status_code=status.HTTP_200_OK,
    summary="Get accuracy metrics",
    description="Retrieve accuracy metrics for medical term extraction including success rate, average confidence, retry rate, and validation failure rate."
)
async def get_accuracy_metrics(
    service: MedicalTermExtractionService = Depends(get_medical_extraction_service)
) -> AccuracyMetrics:
    """
    Get accuracy metrics for medical term extraction.
    
    **Metrics Included:**
    - Total extractions performed
    - Successful vs failed extractions
    - Overall accuracy rate (target: >95%)
    - Average confidence score
    - Retry rate (percentage requiring re-prompting)
    - Validation failure rate
    - Metrics broken down by field type
    
    **Returns:**
    - Comprehensive accuracy metrics
    """
    try:
        metrics = service.get_accuracy_metrics()
        logger.info(
            f"Accuracy metrics: total={metrics.total_extractions}, "
            f"accuracy={metrics.accuracy_rate:.2%}, "
            f"avg_confidence={metrics.average_confidence:.2f}"
        )
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to retrieve accuracy metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve accuracy metrics"
        )
