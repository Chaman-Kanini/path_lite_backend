import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

from app.core.azure_openai import get_azure_openai_client
from app.core.config import settings
from app.core.guardrails import (
    sanitize_input,
    validate_cleaned_output,
    detect_ambiguous_date,
    redact_pii_for_logging,
    check_token_budget,
)
from app.models.structured_outputs import FieldType
from app.models.data_cleaning import (
    CleanedField,
    CleaningConfig,
    CleaningResult,
)
from app.utils.text_transformers import (
    remove_filler_words,
    normalize_case,
    clean_punctuation,
    extract_numeric,
)
from app.utils.date_normalizer import DateNormalizer

logger = logging.getLogger(__name__)


class DataCleaningService:
    """Service for cleaning and normalizing conversational text data using LLM and deterministic transformations."""

    def __init__(self):
        self.client = get_azure_openai_client()
        self._rate_limiter = asyncio.Semaphore(10)
        self._load_prompts()
        self.date_normalizer = DateNormalizer()

    def _load_prompts(self):
        """Load prompt templates from files."""
        prompts_dir = Path(__file__).parent.parent / "prompts" / "cleaning"
        
        system_prompt_path = prompts_dir / "system_prompt.txt"
        field_cleaning_path = prompts_dir / "field_cleaning.txt"
        
        try:
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
            
            with open(field_cleaning_path, "r", encoding="utf-8") as f:
                self.field_cleaning_template = f.read()
            
            logger.info("Data cleaning prompt templates loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load data cleaning prompt templates: {str(e)}")
            self.system_prompt = "You are a medical data cleaning assistant."
            self.field_cleaning_template = "Clean {field_type} from: {transcript}"

    async def clean_field(
        self,
        text: str,
        field_type: FieldType,
        config: Optional[CleaningConfig] = None
    ) -> CleanedField:
        """
        Clean a single field using deterministic transformations and LLM-based extraction.
        
        Args:
            text: Raw conversational text
            field_type: Type of field to clean
            config: Optional cleaning configuration
            
        Returns:
            CleanedField with cleaned value and metadata
        """
        async with self._rate_limiter:
            start_time = time.perf_counter()
            
            if config is None:
                config = CleaningConfig()
            
            original_text = text
            sanitized_text = sanitize_input(text)
            
            if not check_token_budget(sanitized_text):
                logger.warning("Input text exceeds token budget, truncating")
                sanitized_text = sanitized_text[:8000]
            
            logger.info(
                f"Cleaning {field_type.value} from text: {redact_pii_for_logging(sanitized_text[:100])}..."
            )
            
            try:
                # Step 1: Remove filler words
                cleaned_text = remove_filler_words(sanitized_text, config.filler_words)
                
                # Step 2: Apply field-specific transformations
                if field_type in [FieldType.FIRST_NAME, FieldType.LAST_NAME]:
                    cleaned_text = normalize_case(cleaned_text, "title")
                    cleaned_text = clean_punctuation(cleaned_text, config.preserve_patterns)
                
                elif field_type == FieldType.DOB:
                    # Check for ambiguous dates
                    if detect_ambiguous_date(cleaned_text):
                        return CleanedField(
                            value=cleaned_text,
                            original=original_text,
                            confidence_score=0.5,
                            requires_review=True
                        )
                    
                    # Try to normalize date
                    normalized_date, is_ambiguous = self.date_normalizer.parse_and_normalize(cleaned_text)
                    if normalized_date:
                        cleaned_text = normalized_date
                
                elif field_type == FieldType.MRN:
                    # Extract numeric/alphanumeric MRN
                    cleaned_text = clean_punctuation(cleaned_text, config.preserve_patterns)
                
                elif field_type == FieldType.GENDER:
                    cleaned_text = normalize_case(cleaned_text, "title")
                
                # Step 3: Estimate confidence score based on transformation quality
                confidence_score = self._calculate_confidence(original_text, cleaned_text, field_type)
                
                # Step 4: Validate cleaned output
                validation_result = validate_cleaned_output(cleaned_text, original_text, confidence_score)
                
                requires_review = confidence_score < config.confidence_threshold
                
                processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                
                logger.info(
                    f"Successfully cleaned {field_type.value} in {processing_time_ms}ms "
                    f"(confidence: {confidence_score:.2f})"
                )
                
                return CleanedField(
                    value=cleaned_text.strip(),
                    original=original_text,
                    confidence_score=confidence_score,
                    requires_review=requires_review
                )
                
            except Exception as e:
                processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                logger.error(f"Cleaning failed for {field_type.value}: {str(e)}")
                
                # Return original with low confidence on error
                return CleanedField(
                    value=sanitized_text.strip(),
                    original=original_text,
                    confidence_score=0.3,
                    requires_review=True
                )

    def _calculate_confidence(self, original: str, cleaned: str, field_type: FieldType) -> float:
        """
        Calculate confidence score based on transformation quality.
        
        Args:
            original: Original text
            cleaned: Cleaned text
            field_type: Type of field
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.9
        
        # Reduce confidence if cleaned text is empty
        if not cleaned or not cleaned.strip():
            return 0.1
        
        # Reduce confidence if cleaned text is too different from original
        length_ratio = len(cleaned) / max(len(original), 1)
        if length_ratio < 0.1 or length_ratio > 2.0:
            confidence -= 0.2
        
        # Increase confidence for name fields with proper capitalization
        if field_type in [FieldType.FIRST_NAME, FieldType.LAST_NAME]:
            if cleaned.istitle():
                confidence += 0.05
        
        # Increase confidence for date fields with proper format
        if field_type == FieldType.DOB:
            import re
            if re.match(r'\d{2}/\d{2}/\d{4}', cleaned):
                confidence += 0.05
        
        return min(max(confidence, 0.0), 1.0)

    async def clean_patient_data(self, text: str, config: Optional[CleaningConfig] = None) -> CleaningResult:
        """
        Clean all patient fields from conversational text.
        
        Args:
            text: Raw conversational text
            config: Optional cleaning configuration
            
        Returns:
            CleaningResult with all cleaned fields
        """
        start_time = time.perf_counter()
        
        if config is None:
            config = CleaningConfig()
        
        logger.info("Cleaning patient data from conversational text")
        
        # Check for ambiguous dates in the entire text
        clarification_needed = False
        clarification_message = None
        
        if detect_ambiguous_date(text):
            clarification_needed = True
            clarification_message = "The date format is ambiguous. Did you mean MM/DD/YYYY or DD/MM/YYYY?"
        
        # Clean all field types
        field_types = [
            FieldType.MRN,
            FieldType.FIRST_NAME,
            FieldType.LAST_NAME,
            FieldType.DOB,
            FieldType.GENDER
        ]
        
        cleaned_fields = []
        requires_review = False
        
        for field_type in field_types:
            try:
                cleaned_field = await self.clean_field(text, field_type, config)
                cleaned_fields.append(cleaned_field)
                
                if cleaned_field.requires_review:
                    requires_review = True
                    
            except Exception as e:
                logger.error(f"Failed to clean {field_type.value}: {str(e)}")
                continue
        
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        
        logger.info(
            f"Successfully cleaned patient data in {processing_time_ms}ms "
            f"({len(cleaned_fields)} fields processed)"
        )
        
        return CleaningResult(
            fields=cleaned_fields,
            requires_review=requires_review,
            clarification_needed=clarification_needed,
            clarification_message=clarification_message
        )


_data_cleaning_service: Optional[DataCleaningService] = None


def get_data_cleaning_service() -> DataCleaningService:
    """Get or create the data cleaning service singleton."""
    global _data_cleaning_service
    if _data_cleaning_service is None:
        _data_cleaning_service = DataCleaningService()
    return _data_cleaning_service
