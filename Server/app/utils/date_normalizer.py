import re
from datetime import date
from typing import Optional, Tuple
from dateutil import parser


class DateNormalizer:
    """Utility class for parsing and normalizing dates from natural language."""

    @staticmethod
    def parse_natural_date(text: str) -> Optional[date]:
        """
        Parse natural language date string into a date object.
        
        Args:
            text: Natural language date string (e.g., "January 15th, 1980")
            
        Returns:
            Parsed date object or None if parsing fails
        """
        if not text:
            return None
        
        try:
            parsed_date = parser.parse(text, fuzzy=True)
            return parsed_date.date()
        except (ValueError, parser.ParserError):
            return None

    @staticmethod
    def normalize_to_mmddyyyy(d: date) -> str:
        """
        Convert date object to MM/DD/YYYY format string.
        
        Args:
            d: Date object to format
            
        Returns:
            Date string in MM/DD/YYYY format
        """
        return d.strftime("%m/%d/%Y")

    @staticmethod
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
        
        # Pattern for numeric dates like MM/DD/YYYY or DD/MM/YYYY
        numeric_date_pattern = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b'
        match = re.search(numeric_date_pattern, text)
        
        if match:
            first_part = int(match.group(1))
            second_part = int(match.group(2))
            
            # Ambiguous if both parts are valid for both month and day (1-12)
            if 1 <= first_part <= 12 and 1 <= second_part <= 12 and first_part != second_part:
                return True
        
        return False

    @staticmethod
    def parse_and_normalize(text: str) -> Tuple[Optional[str], bool]:
        """
        Parse natural language date and normalize to MM/DD/YYYY format.
        
        Args:
            text: Natural language date string
            
        Returns:
            Tuple of (normalized_date_string, is_ambiguous)
        """
        is_ambiguous = DateNormalizer.detect_ambiguous_date(text)
        
        parsed_date = DateNormalizer.parse_natural_date(text)
        
        if parsed_date:
            normalized = DateNormalizer.normalize_to_mmddyyyy(parsed_date)
            return (normalized, is_ambiguous)
        
        return (None, is_ambiguous)
