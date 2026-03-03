import re
import logging
from typing import Any, Dict, List
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def sanitize_input(text: str) -> str:
    """
    Sanitize input text before sending to Azure OpenAI.
    
    Args:
        text: Raw input text from STT or user input
        
    Returns:
        Sanitized text safe for API processing
    """
    if not text or not isinstance(text, str):
        return ""
    
    sanitized = text.strip()
    
    sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', sanitized)
    
    if len(sanitized) > 10000:
        logger.warning(f"Input text truncated from {len(sanitized)} to 10000 characters")
        sanitized = sanitized[:10000]
    
    return sanitized


def validate_output(output: BaseModel, min_confidence: float = 0.5) -> Dict[str, Any]:
    """
    Validate extraction output against quality thresholds.
    
    Args:
        output: Pydantic model instance with extraction results
        min_confidence: Minimum acceptable confidence score (default: 0.5)
        
    Returns:
        Validation result with status and details
    """
    validation_result = {
        "is_valid": True,
        "warnings": [],
        "errors": []
    }
    
    if hasattr(output, 'confidence_score'):
        if output.confidence_score < min_confidence:
            validation_result["warnings"].append(
                f"Low confidence score: {output.confidence_score:.2f} (threshold: {min_confidence})"
            )
    
    try:
        output.model_validate(output.model_dump())
    except Exception as e:
        validation_result["is_valid"] = False
        validation_result["errors"].append(f"Model validation failed: {str(e)}")
    
    return validation_result


def redact_pii_for_logging(text: str) -> str:
    """
    Redact PII from text before logging.
    
    Args:
        text: Text that may contain PII
        
    Returns:
        Text with PII redacted
    """
    redacted = text
    
    redacted = re.sub(r'\b\d{2}/\d{2}/\d{4}\b', '[DATE_REDACTED]', redacted)
    redacted = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[DATE_REDACTED]', redacted)
    
    redacted = re.sub(r'\b[A-Z]{3,}\d{3,}\b', '[MRN_REDACTED]', redacted)
    redacted = re.sub(r'\bMRN\s*[:\-]?\s*\w+\b', '[MRN_REDACTED]', redacted, flags=re.IGNORECASE)
    
    redacted = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', redacted)
    
    return redacted


def check_token_budget(text: str, max_tokens: int = 4000) -> bool:
    """
    Estimate token count and check against budget.
    
    Args:
        text: Input text to check
        max_tokens: Maximum allowed tokens
        
    Returns:
        True if within budget, False otherwise
    """
    estimated_tokens = len(text.split()) * 1.3
    
    if estimated_tokens > max_tokens:
        logger.warning(
            f"Estimated token count ({estimated_tokens:.0f}) exceeds budget ({max_tokens})"
        )
        return False
    
    return True


def validate_cleaned_output(cleaned_value: str, original_value: str, confidence_score: float) -> Dict[str, Any]:
    """
    Validate cleaned output against quality thresholds.
    
    Args:
        cleaned_value: Cleaned value after transformations
        original_value: Original value before cleaning
        confidence_score: Confidence score of the cleaning operation
        
    Returns:
        Validation result with status and warnings
    """
    validation_result = {
        "is_valid": True,
        "warnings": [],
        "errors": []
    }
    
    if not cleaned_value or not cleaned_value.strip():
        validation_result["is_valid"] = False
        validation_result["errors"].append("Cleaned value is empty")
        return validation_result
    
    if confidence_score < 0.85:
        validation_result["warnings"].append(
            f"Low confidence score: {confidence_score:.2f} - manual review recommended"
        )
    
    if len(cleaned_value) > len(original_value) * 1.5:
        validation_result["warnings"].append(
            "Cleaned value is significantly longer than original - possible hallucination"
        )
    
    return validation_result


def detect_ambiguous_date(text: str) -> bool:
    """
    Detect if a date string is ambiguous (e.g., 03/04/1980 could be March 4 or April 3).
    
    Args:
        text: Date string to check
        
    Returns:
        True if date is ambiguous, False otherwise
    """
    if not text:
        return False
    
    numeric_date_pattern = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b'
    match = re.search(numeric_date_pattern, text)
    
    if match:
        first_part = int(match.group(1))
        second_part = int(match.group(2))
        
        if 1 <= first_part <= 12 and 1 <= second_part <= 12 and first_part != second_part:
            return True
    
    return False


def validate_against_schema(value: str, valid_values: List[str]) -> bool:
    """
    Validate a value against a schema's valid enum values.
    
    Args:
        value: Value to validate
        valid_values: List of valid enum values
        
    Returns:
        True if value is valid, False otherwise
    """
    if not value or not valid_values:
        return False
    
    value_lower = value.lower().strip()
    
    for valid_value in valid_values:
        if valid_value.lower().strip() == value_lower:
            return True
    
    return False


def detect_medical_hallucination(value: str, field_type: str, all_valid_values: Dict[str, List[str]]) -> bool:
    """
    Detect if a medical term value is hallucinated.
    
    Args:
        value: Extracted value to check
        field_type: Type of field (hbsag, treatment_location, gender)
        all_valid_values: Dictionary of all valid values across all fields
        
    Returns:
        True if hallucinated, False otherwise
    """
    if not value:
        return False
    
    value_lower = value.lower().strip()
    
    # Check if value exists in a different field's schema
    for other_field_type, valid_values in all_valid_values.items():
        if other_field_type != field_type:
            for valid_value in valid_values:
                if valid_value.lower().strip() == value_lower:
                    logger.warning(
                        f"Potential hallucination: '{value}' is valid for '{other_field_type}' "
                        f"but not for '{field_type}'"
                    )
                    return True
    
    return False


def sanitize_medical_term(value: str) -> str:
    """
    Sanitize a medical term value before validation.
    
    Args:
        value: Medical term value to sanitize
        
    Returns:
        Sanitized value
    """
    if not value or not isinstance(value, str):
        return ""
    
    # Remove extra whitespace
    sanitized = value.strip()
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Remove special characters except hyphens (for ICU-CCU, Multi-Tx)
    sanitized = re.sub(r'[^a-zA-Z0-9\s\-]', '', sanitized)
    
    # Limit length
    if len(sanitized) > 100:
        logger.warning(f"Medical term truncated from {len(sanitized)} to 100 characters")
        sanitized = sanitized[:100]
    
    return sanitized
