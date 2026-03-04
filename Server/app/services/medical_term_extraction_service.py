"""
Medical Term Extraction Service

Extracts medical terminology from conversational patient responses using
Azure OpenAI with few-shot prompts and schema validation.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.services.schema_validation_service import SchemaValidationService
from app.models.medical_extraction import (
    MedicalExtractionRequest,
    MedicalExtractionResponse,
    MedicalFieldType,
    AccuracyMetrics
)
from app.models.schema_validation import ValidationRequest

logger = logging.getLogger(__name__)


class MedicalTermExtractionService:
    """Service for extracting medical terms using Azure OpenAI"""
    
    MAX_RETRIES = 3
    
    def __init__(
        self,
        azure_client: Optional[AzureOpenAI] = None,
        schema_validation_service: Optional[SchemaValidationService] = None
    ):
        """
        Initialize medical term extraction service
        
        Args:
            azure_client: Azure OpenAI client (creates new if None)
            schema_validation_service: Schema validation service instance
        """
        self.azure_client = azure_client or self._create_azure_client()
        self.schema_validation_service = schema_validation_service or SchemaValidationService()
        self.prompt_templates: Dict[str, str] = {}
        self._load_prompt_templates()
        
        # Metrics tracking
        self.metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_retries": 0,
            "validation_failures": 0,
            "confidence_scores": [],
            "by_field": {}
        }
    
    def _create_azure_client(self) -> AzureOpenAI:
        """Create Azure OpenAI client from settings"""
        return AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
    
    def _load_prompt_templates(self) -> None:
        """Load prompt templates from medical_terms directory"""
        try:
            base_path = Path(__file__).parent.parent
            prompts_path = base_path / "prompts" / "medical_terms"
            
            templates = {
                MedicalFieldType.HBSAG: "hbsag_extraction.txt",
                MedicalFieldType.TREATMENT_LOCATION: "treatment_location_extraction.txt",
                MedicalFieldType.GENDER: "gender_extraction.txt"
            }
            
            for field_type, filename in templates.items():
                template_path = prompts_path / filename
                if template_path.exists():
                    with open(template_path, 'r', encoding='utf-8') as f:
                        self.prompt_templates[field_type] = f.read()
                    logger.info(f"Loaded prompt template for {field_type}")
                else:
                    logger.error(f"Prompt template not found: {template_path}")
            
        except Exception as e:
            logger.error(f"Failed to load prompt templates: {e}")
            raise
    
    async def extract_medical_term(
        self,
        request: MedicalExtractionRequest
    ) -> MedicalExtractionResponse:
        """
        Extract medical term from raw text with validation and retry logic
        
        Args:
            request: Medical extraction request
        
        Returns:
            Medical extraction response with validated result
        """
        retry_count = 0
        conversation_history = request.conversation_history.copy()
        
        while retry_count < self.MAX_RETRIES:
            try:
                # Route to field-specific extraction method
                extraction_result = await self._extract_by_field_type(
                    request.raw_text,
                    request.field_type,
                    request.context,
                    conversation_history
                )
                
                # Validate extracted value
                validation_request = ValidationRequest(
                    field_name=request.field_type.value,
                    extracted_value=extraction_result["extracted_value"],
                    field_type=request.field_type.value
                )
                
                validation_response = self.schema_validation_service.validate_field(
                    validation_request
                )
                
                # Update metrics
                self.metrics["total_extractions"] += 1
                self.metrics["confidence_scores"].append(extraction_result["confidence"])
                
                if validation_response.is_valid:
                    # Success - return validated result
                    self.metrics["successful_extractions"] += 1
                    
                    return MedicalExtractionResponse(
                        extracted_value=validation_response.matched_value or extraction_result["extracted_value"],
                        confidence=extraction_result["confidence"],
                        is_valid=True,
                        retry_count=retry_count,
                        validation_message=None,
                        raw_text_snippet=extraction_result.get("raw_text", ""),
                        field_type=request.field_type
                    )
                else:
                    # Validation failed - prepare for retry
                    retry_count += 1
                    self.metrics["total_retries"] += 1
                    self.metrics["validation_failures"] += 1
                    
                    if retry_count >= self.MAX_RETRIES:
                        # Max retries reached
                        self.metrics["failed_extractions"] += 1
                        
                        return MedicalExtractionResponse(
                            extracted_value=None,
                            confidence=0.0,
                            is_valid=False,
                            retry_count=retry_count,
                            validation_message=validation_response.error_message,
                            raw_text_snippet=extraction_result.get("raw_text", ""),
                            field_type=request.field_type
                        )
                    
                    # Generate clarification prompt and retry
                    clarification = validation_response.clarification_prompt
                    conversation_history.append({
                        "role": "assistant",
                        "content": clarification
                    })
                    
                    logger.info(
                        f"Retry {retry_count}/{self.MAX_RETRIES} for {request.field_type}: "
                        f"Invalid value '{extraction_result['extracted_value']}'"
                    )
                    
            except Exception as e:
                logger.error(f"Extraction error: {e}")
                retry_count += 1
                
                if retry_count >= self.MAX_RETRIES:
                    self.metrics["failed_extractions"] += 1
                    
                    return MedicalExtractionResponse(
                        extracted_value=None,
                        confidence=0.0,
                        is_valid=False,
                        retry_count=retry_count,
                        validation_message=f"Extraction failed: {str(e)}",
                        raw_text_snippet="",
                        field_type=request.field_type
                    )
        
        # Should not reach here
        self.metrics["failed_extractions"] += 1
        return MedicalExtractionResponse(
            extracted_value=None,
            confidence=0.0,
            is_valid=False,
            retry_count=retry_count,
            validation_message="Max retries exceeded",
            raw_text_snippet="",
            field_type=request.field_type
        )
    
    async def _extract_by_field_type(
        self,
        raw_text: str,
        field_type: MedicalFieldType,
        context: Optional[str],
        conversation_history: list
    ) -> Dict[str, Any]:
        """
        Route extraction to field-specific method
        
        Args:
            raw_text: Raw patient text
            field_type: Type of field to extract
            context: Additional context
            conversation_history: Previous conversation turns
        
        Returns:
            Extraction result dictionary
        """
        if field_type == MedicalFieldType.HBSAG:
            return await self._extract_hbsag(raw_text, context, conversation_history)
        elif field_type == MedicalFieldType.TREATMENT_LOCATION:
            return await self._extract_treatment_location(raw_text, context, conversation_history)
        elif field_type == MedicalFieldType.GENDER:
            return await self._extract_gender(raw_text, context, conversation_history)
        else:
            raise ValueError(f"Unsupported field type: {field_type}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _extract_hbsag(
        self,
        raw_text: str,
        context: Optional[str],
        conversation_history: list
    ) -> Dict[str, Any]:
        """Extract HBsAg status from raw text"""
        return await self._call_azure_openai(
            MedicalFieldType.HBSAG,
            raw_text,
            context,
            conversation_history
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _extract_treatment_location(
        self,
        raw_text: str,
        context: Optional[str],
        conversation_history: list
    ) -> Dict[str, Any]:
        """Extract treatment location from raw text"""
        return await self._call_azure_openai(
            MedicalFieldType.TREATMENT_LOCATION,
            raw_text,
            context,
            conversation_history
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _extract_gender(
        self,
        raw_text: str,
        context: Optional[str],
        conversation_history: list
    ) -> Dict[str, Any]:
        """Extract gender from raw text"""
        return await self._call_azure_openai(
            MedicalFieldType.GENDER,
            raw_text,
            context,
            conversation_history
        )
    
    async def _call_azure_openai(
        self,
        field_type: MedicalFieldType,
        raw_text: str,
        context: Optional[str],
        conversation_history: list
    ) -> Dict[str, Any]:
        """
        Call Azure OpenAI with prompt template
        
        Args:
            field_type: Type of field to extract
            raw_text: Raw patient text
            context: Additional context
            conversation_history: Previous conversation turns
        
        Returns:
            Extraction result dictionary
        """
        # Load prompt template
        template = self.prompt_templates.get(field_type)
        if not template:
            raise ValueError(f"Prompt template not found for {field_type}")
        
        # Substitute variables in template
        prompt = template.replace("{raw_text}", raw_text)
        if context:
            prompt = prompt.replace("{context}", context)
        
        # Build messages
        messages = [{"role": "system", "content": prompt}]
        messages.extend(conversation_history)
        
        # Call Azure OpenAI
        try:
            response = self.azure_client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=500,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Extract fields based on field type
            if field_type == MedicalFieldType.HBSAG:
                return {
                    "extracted_value": result.get("status", ""),
                    "confidence": result.get("confidence", 0.0),
                    "raw_text": result.get("raw_text", "")
                }
            elif field_type == MedicalFieldType.TREATMENT_LOCATION:
                return {
                    "extracted_value": result.get("location", ""),
                    "confidence": result.get("confidence", 0.0),
                    "raw_text": result.get("raw_text", "")
                }
            elif field_type == MedicalFieldType.GENDER:
                return {
                    "extracted_value": result.get("gender", ""),
                    "confidence": result.get("confidence", 0.0),
                    "raw_text": result.get("raw_text", "")
                }
            
        except Exception as e:
            logger.error(f"Azure OpenAI call failed: {e}")
            raise
    
    def get_accuracy_metrics(self) -> AccuracyMetrics:
        """
        Get accuracy metrics for medical term extraction
        
        Returns:
            AccuracyMetrics with current statistics
        """
        total = self.metrics["total_extractions"]
        
        if total == 0:
            return AccuracyMetrics(
                total_extractions=0,
                successful_extractions=0,
                failed_extractions=0,
                accuracy_rate=0.0,
                average_confidence=0.0,
                retry_rate=0.0,
                validation_failure_rate=0.0,
                metrics_by_field={}
            )
        
        successful = self.metrics["successful_extractions"]
        failed = self.metrics["failed_extractions"]
        accuracy_rate = successful / total if total > 0 else 0.0
        
        confidence_scores = self.metrics["confidence_scores"]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        retry_rate = self.metrics["total_retries"] / total if total > 0 else 0.0
        validation_failure_rate = self.metrics["validation_failures"] / total if total > 0 else 0.0
        
        return AccuracyMetrics(
            total_extractions=total,
            successful_extractions=successful,
            failed_extractions=failed,
            accuracy_rate=accuracy_rate,
            average_confidence=avg_confidence,
            retry_rate=retry_rate,
            validation_failure_rate=validation_failure_rate,
            metrics_by_field=self.metrics.get("by_field", {})
        )
