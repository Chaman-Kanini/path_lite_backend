from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
import logging
from app.models.ai import (
    ConversationCreate,
    MessageCreate,
    ConversationResponse,
    MessageResponse,
    MessageRole,
    ConversationStatus,
)
from app.models.extraction_api import (
    ExtractionRequest,
    ExtractionResponse,
    BatchExtractionRequest,
    BatchExtractionResponse,
    PatientDataExtractionRequest,
    PatientDataExtractionResponse,
    HealthCheckResponse,
)
from app.models.cleaning_api import (
    CleanRequest,
    CleanResponse,
    BatchCleanRequest,
    BatchCleanResponse,
)
from app.dependencies import (
    get_current_user,
    get_db,
    get_extraction_service_dependency,
    get_data_cleaning_service_dependency,
)
from app.services.extraction_service import ExtractionService
from app.services.data_cleaning_service import DataCleaningService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["AI Processing"])

_PLACEHOLDER_MESSAGE = MessageResponse(
    id="msg-001",
    content="Hello! How can I help you today?",
    role=MessageRole.assistant,
    timestamp=datetime.utcnow(),
)

_PLACEHOLDER_CONVERSATION = ConversationResponse(
    id="conv-001",
    patient_id="patient-001",
    messages=[_PLACEHOLDER_MESSAGE],
    status=ConversationStatus.active,
    created_at=datetime.utcnow(),
)


@router.post("/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    body: ConversationCreate,
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    return ConversationResponse(
        id="conv-new",
        patient_id=body.patient_id,
        messages=[],
        status=ConversationStatus.active,
        created_at=datetime.utcnow(),
    )


@router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse, status_code=201)
async def send_message(
    conversation_id: str,
    message: MessageCreate,
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    return MessageResponse(
        id="msg-new",
        content=message.content,
        role=message.role,
        timestamp=datetime.utcnow(),
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    return _PLACEHOLDER_CONVERSATION


@router.post("/extract", response_model=ExtractionResponse, status_code=200)
async def extract_field(
    request: ExtractionRequest,
    extraction_service: ExtractionService = Depends(get_extraction_service_dependency),
    current_user=Depends(get_current_user),
):
    """
    Extract a single field from conversational text using Azure OpenAI.
    
    - **text**: Conversational text from STT or user input
    - **field_type**: Type of field to extract (mrn, first_name, last_name, dob, gender)
    
    Returns extracted value with confidence score and processing time.
    """
    try:
        result = await extraction_service.extract_field(
            text=request.text,
            field_type=request.field_type,
        )
        return ExtractionResponse(**result)
    except Exception as e:
        logger.error(f"Field extraction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}"
        )


@router.post("/extract/batch", response_model=BatchExtractionResponse, status_code=200)
async def extract_batch(
    request: BatchExtractionRequest,
    extraction_service: ExtractionService = Depends(get_extraction_service_dependency),
    current_user=Depends(get_current_user),
):
    """
    Extract multiple fields from conversational text using Azure OpenAI.
    
    - **text**: Conversational text from STT or user input
    - **field_types**: List of field types to extract
    
    Returns list of extracted values with confidence scores.
    """
    try:
        import time
        start_time = time.perf_counter()
        
        extractions = []
        for field_type in request.field_types:
            result = await extraction_service.extract_field(
                text=request.text,
                field_type=field_type,
            )
            extractions.append(ExtractionResponse(**result))
        
        total_time_ms = int((time.perf_counter() - start_time) * 1000)
        
        return BatchExtractionResponse(
            extractions=extractions,
            total_processing_time_ms=total_time_ms
        )
    except Exception as e:
        logger.error(f"Batch extraction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch extraction failed: {str(e)}"
        )


@router.post("/extract/patient", response_model=PatientDataExtractionResponse, status_code=200)
async def extract_patient_data(
    request: PatientDataExtractionRequest,
    extraction_service: ExtractionService = Depends(get_extraction_service_dependency),
    current_user=Depends(get_current_user),
):
    """
    Extract all patient fields from conversational text using Azure OpenAI.
    
    - **text**: Conversational text from STT or user input
    
    Returns all extracted patient fields with confidence scores.
    """
    try:
        result = await extraction_service.extract_patient_data(text=request.text)
        return PatientDataExtractionResponse(**result)
    except Exception as e:
        logger.error(f"Patient data extraction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Patient data extraction failed: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse, status_code=200)
async def health_check(
    extraction_service: ExtractionService = Depends(get_extraction_service_dependency),
):
    """
    Check Azure OpenAI connectivity and service health.
    
    Returns health status and connection information.
    """
    try:
        is_connected = await extraction_service.client.verify_connection()
        return HealthCheckResponse(
            status="healthy",
            azure_openai_connected=is_connected,
            message="All systems operational"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            azure_openai_connected=False,
            message=f"Azure OpenAI connection failed: {str(e)}"
        )


@router.post("/clean", response_model=CleanResponse, status_code=200)
async def clean_field(
    request: CleanRequest,
    cleaning_service: DataCleaningService = Depends(get_data_cleaning_service_dependency),
    current_user=Depends(get_current_user),
):
    """
    Clean a single field from conversational text.
    
    - **text**: Raw conversational text from STT or user input
    - **field_type**: Type of field to clean (mrn, first_name, last_name, dob, gender)
    - **config**: Optional cleaning configuration
    
    Returns cleaned value with confidence score and review flags.
    """
    try:
        cleaned_field = await cleaning_service.clean_field(
            text=request.text,
            field_type=request.field_type,
            config=request.config,
        )
        
        return CleanResponse(
            cleaned_value=cleaned_field.value,
            original=cleaned_field.original,
            confidence_score=cleaned_field.confidence_score,
            requires_review=cleaned_field.requires_review,
            clarification_needed=False,
            clarification_message=None,
        )
    except Exception as e:
        logger.error(f"Field cleaning failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleaning failed: {str(e)}"
        )


@router.post("/clean/batch", response_model=BatchCleanResponse, status_code=200)
async def clean_batch(
    request: BatchCleanRequest,
    cleaning_service: DataCleaningService = Depends(get_data_cleaning_service_dependency),
    current_user=Depends(get_current_user),
):
    """
    Clean multiple fields from conversational text.
    
    - **text**: Raw conversational text from STT or user input
    - **field_types**: List of field types to clean
    - **config**: Optional cleaning configuration
    
    Returns list of cleaned values with confidence scores and review flags.
    """
    try:
        import time
        start_time = time.perf_counter()
        
        cleaned_fields = []
        requires_review = False
        
        for field_type in request.field_types:
            cleaned_field = await cleaning_service.clean_field(
                text=request.text,
                field_type=field_type,
                config=request.config,
            )
            
            cleaned_fields.append(
                CleanResponse(
                    cleaned_value=cleaned_field.value,
                    original=cleaned_field.original,
                    confidence_score=cleaned_field.confidence_score,
                    requires_review=cleaned_field.requires_review,
                    clarification_needed=False,
                    clarification_message=None,
                )
            )
            
            if cleaned_field.requires_review:
                requires_review = True
        
        total_time_ms = int((time.perf_counter() - start_time) * 1000)
        
        logger.info(f"Batch cleaning completed in {total_time_ms}ms for {len(cleaned_fields)} fields")
        
        return BatchCleanResponse(
            fields=cleaned_fields,
            requires_review=requires_review,
            clarification_needed=False,
            clarification_message=None,
        )
    except Exception as e:
        logger.error(f"Batch cleaning failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch cleaning failed: {str(e)}"
        )
