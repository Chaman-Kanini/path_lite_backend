import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.azure_openai import get_azure_openai_client
from app.core.config import settings
from app.core.guardrails import (
    sanitize_input,
    validate_output,
    redact_pii_for_logging,
    check_token_budget,
)
from app.models.structured_outputs import (
    FieldType,
    BaseExtraction,
    MRNExtraction,
    NameExtraction,
    DOBExtraction,
    GenderExtraction,
    PatientFieldExtraction,
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseExtraction)


class ExtractionService:
    """Service for extracting structured patient data from conversational text using Azure OpenAI."""

    def __init__(self):
        self.client = get_azure_openai_client()
        self._rate_limiter = asyncio.Semaphore(10)
        self._load_prompts()

    def _load_prompts(self):
        """Load prompt templates from files."""
        prompts_dir = Path(__file__).parent.parent / "prompts" / "extraction"
        
        system_prompt_path = prompts_dir / "system_prompt.txt"
        field_extraction_path = prompts_dir / "field_extraction.txt"
        
        try:
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
            
            with open(field_extraction_path, "r", encoding="utf-8") as f:
                self.field_extraction_template = f.read()
            
            logger.info("Prompt templates loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load prompt templates: {str(e)}")
            self.system_prompt = "You are a medical data extraction assistant."
            self.field_extraction_template = "Extract {field_type} from: {transcript}"

    def _get_extraction_model(self, field_type: FieldType) -> Type[BaseExtraction]:
        """Get the appropriate Pydantic model for the field type."""
        model_map = {
            FieldType.MRN: MRNExtraction,
            FieldType.FIRST_NAME: NameExtraction,
            FieldType.LAST_NAME: NameExtraction,
            FieldType.DOB: DOBExtraction,
            FieldType.GENDER: GenderExtraction,
        }
        return model_map.get(field_type, MRNExtraction)

    @retry(
        stop=stop_after_attempt(settings.AZURE_OPENAI_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def extract_field(
        self,
        text: str,
        field_type: FieldType,
    ) -> Dict[str, Any]:
        """
        Extract a single field from conversational text.
        
        Args:
            text: Conversational text (e.g., STT output)
            field_type: Type of field to extract
            
        Returns:
            Dictionary with extracted value, confidence score, and metadata
        """
        async with self._rate_limiter:
            start_time = time.perf_counter()
            
            sanitized_text = sanitize_input(text)
            
            if not check_token_budget(sanitized_text):
                logger.warning("Input text exceeds token budget, truncating")
                sanitized_text = sanitized_text[:8000]
            
            user_prompt = self.field_extraction_template.format(
                field_type=field_type.value,
                transcript=sanitized_text
            )
            
            logger.info(
                f"Extracting {field_type.value} from text: {redact_pii_for_logging(sanitized_text[:100])}..."
            )
            
            try:
                extraction_model = self._get_extraction_model(field_type)
                
                response = self.client.client.beta.chat.completions.parse(
                    model=settings.AZURE_OPENAI_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=extraction_model,
                    temperature=0.0,
                )
                
                extracted_data = response.choices[0].message.parsed
                
                validation_result = validate_output(extracted_data)
                
                processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                
                result = {
                    "value": extracted_data.model_dump(),
                    "confidence_score": extracted_data.confidence_score,
                    "field_type": field_type.value,
                    "processing_time_ms": processing_time_ms,
                    "fallback_to_manual": False,
                    "validation": validation_result,
                }
                
                logger.info(
                    f"Successfully extracted {field_type.value} in {processing_time_ms}ms "
                    f"(confidence: {extracted_data.confidence_score:.2f})"
                )
                
                return result
                
            except Exception as e:
                processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                logger.error(f"Extraction failed for {field_type.value}: {str(e)}")
                raise

    async def extract_patient_data(self, text: str) -> Dict[str, Any]:
        """
        Extract all patient fields from conversational text.
        
        Args:
            text: Conversational text (e.g., STT output)
            
        Returns:
            Dictionary with all extracted patient fields
        """
        start_time = time.perf_counter()
        
        sanitized_text = sanitize_input(text)
        
        if not check_token_budget(sanitized_text, max_tokens=6000):
            logger.warning("Input text exceeds token budget for batch extraction")
            sanitized_text = sanitized_text[:12000]
        
        user_prompt = f"""Extract all available patient information from the following conversation transcript:

Transcript:
{sanitized_text}

Extract any of the following fields that are mentioned:
- Medical Record Number (MRN)
- Patient name (first and last)
- Date of birth
- Gender

For each field found, provide a confidence score. If a field is not mentioned, omit it from the response.
"""
        
        logger.info(
            f"Extracting patient data from text: {redact_pii_for_logging(sanitized_text[:100])}..."
        )
        
        try:
            response = self.client.client.beta.chat.completions.parse(
                model=settings.AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=PatientFieldExtraction,
                temperature=0.0,
            )
            
            extracted_data = response.choices[0].message.parsed
            
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            
            result = {
                "data": extracted_data.model_dump(exclude_none=True),
                "processing_time_ms": processing_time_ms,
                "fallback_to_manual": False,
            }
            
            logger.info(
                f"Successfully extracted patient data in {processing_time_ms}ms"
            )
            
            return result
            
        except Exception as e:
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Patient data extraction failed: {str(e)}")
            
            return {
                "data": {},
                "processing_time_ms": processing_time_ms,
                "fallback_to_manual": True,
                "error": str(e),
            }


_extraction_service: Optional[ExtractionService] = None


def get_extraction_service() -> ExtractionService:
    """Get or create the extraction service singleton."""
    global _extraction_service
    if _extraction_service is None:
        _extraction_service = ExtractionService()
    return _extraction_service
