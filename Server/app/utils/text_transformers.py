import re
from typing import List, Optional, Literal


def remove_filler_words(text: str, filler_words: List[str]) -> str:
    """
    Remove filler words from text while preserving sentence structure.
    
    Args:
        text: Input text to clean
        filler_words: List of filler words to remove
        
    Returns:
        Cleaned text with filler words removed
    """
    if not text or not filler_words:
        return text
    
    words = text.split()
    cleaned_words = [
        word for word in words 
        if word.lower() not in [fw.lower() for fw in filler_words]
    ]
    
    return ' '.join(cleaned_words).strip()


def normalize_case(text: str, case_type: Literal["title", "upper", "lower"] = "title") -> str:
    """
    Normalize text casing according to specified type.
    
    Args:
        text: Input text to normalize
        case_type: Type of casing - "title" for proper case, "upper" for uppercase, "lower" for lowercase
        
    Returns:
        Text with normalized casing
    """
    if not text:
        return text
    
    if case_type == "title":
        return text.title()
    elif case_type == "upper":
        return text.upper()
    elif case_type == "lower":
        return text.lower()
    else:
        return text


def clean_punctuation(text: str, preserve_patterns: Optional[List[str]] = None) -> str:
    """
    Remove unnecessary punctuation while preserving dates and decimals.
    
    Args:
        text: Input text to clean
        preserve_patterns: List of patterns to preserve (e.g., ["date", "decimal"])
        
    Returns:
        Text with cleaned punctuation
    """
    if not text:
        return text
    
    preserve_patterns = preserve_patterns or ["date", "decimal"]
    
    # Preserve dates (MM/DD/YYYY, DD-MM-YYYY, etc.)
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    dates = re.findall(date_pattern, text)
    date_placeholders = {date: f"__DATE_{i}__" for i, date in enumerate(dates)}
    
    for date, placeholder in date_placeholders.items():
        text = text.replace(date, placeholder)
    
    # Preserve decimals (e.g., 3.14, 98.6)
    decimal_pattern = r'\b\d+\.\d+\b'
    decimals = re.findall(decimal_pattern, text)
    decimal_placeholders = {decimal: f"__DECIMAL_{i}__" for i, decimal in enumerate(decimals)}
    
    for decimal, placeholder in decimal_placeholders.items():
        text = text.replace(decimal, placeholder)
    
    # Remove unnecessary punctuation (keep only alphanumeric, spaces, and preserved patterns)
    text = re.sub(r'[^\w\s_]', '', text)
    
    # Restore preserved patterns
    for date, placeholder in date_placeholders.items():
        text = text.replace(placeholder, date)
    
    for decimal, placeholder in decimal_placeholders.items():
        text = text.replace(placeholder, decimal)
    
    return text.strip()


def extract_numeric(text: str) -> Optional[str]:
    """
    Extract numeric values from text while preserving format.
    
    Args:
        text: Input text containing numeric values
        
    Returns:
        Extracted numeric value as string, or None if no numeric value found
    """
    if not text:
        return None
    
    # Pattern to match integers and decimals
    numeric_pattern = r'\b\d+(?:\.\d+)?\b'
    
    matches = re.findall(numeric_pattern, text)
    
    if matches:
        # Return the first numeric value found
        return matches[0]
    
    return None


def extract_all_numeric(text: str) -> List[str]:
    """
    Extract all numeric values from text.
    
    Args:
        text: Input text containing numeric values
        
    Returns:
        List of all numeric values found as strings
    """
    if not text:
        return []
    
    numeric_pattern = r'\b\d+(?:\.\d+)?\b'
    return re.findall(numeric_pattern, text)
