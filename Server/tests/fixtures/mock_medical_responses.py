"""
Mock Medical Response Fixtures

Mock Azure OpenAI responses for testing medical term extraction.
"""

from typing import Dict, Any


class MockChatCompletion:
    """Mock chat completion response"""
    
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockChoice:
    """Mock choice in completion"""
    
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message in choice"""
    
    def __init__(self, content: str):
        self.content = content


# HBsAg Mock Responses
MOCK_HBSAG_POSITIVE = {
    "status": "Positive",
    "confidence": 0.95,
    "raw_text": "Hep B positive"
}

MOCK_HBSAG_NEGATIVE = {
    "status": "Negative",
    "confidence": 0.96,
    "raw_text": "HBsAg negative"
}

MOCK_HBSAG_UNKNOWN = {
    "status": "Unknown",
    "confidence": 0.88,
    "raw_text": "status unknown"
}

# Treatment Location Mock Responses
MOCK_LOCATION_OR = {
    "location": "OR",
    "confidence": 0.94,
    "raw_text": "operating room"
}

MOCK_LOCATION_BEDSIDE = {
    "location": "Bedside",
    "confidence": 0.93,
    "raw_text": "at bedside"
}

MOCK_LOCATION_ICU_CCU = {
    "location": "ICU-CCU",
    "confidence": 0.95,
    "raw_text": "ICU-CCU"
}

MOCK_LOCATION_ER = {
    "location": "ER",
    "confidence": 0.96,
    "raw_text": "ER visit"
}

MOCK_LOCATION_MULTI_TX = {
    "location": "Multi-Tx Room",
    "confidence": 0.89,
    "raw_text": "multiple treatment rooms"
}

# Gender Mock Responses
MOCK_GENDER_MALE = {
    "gender": "Male",
    "confidence": 0.97,
    "raw_text": "patient is male"
}

MOCK_GENDER_FEMALE = {
    "gender": "Female",
    "confidence": 0.96,
    "raw_text": "Female patient"
}

# Error Mock Responses
MOCK_ERROR_TIMEOUT = {
    "error": "Request timeout",
    "status_code": 408
}

MOCK_ERROR_RATE_LIMIT = {
    "error": "Rate limit exceeded",
    "status_code": 429
}

MOCK_ERROR_SERVER = {
    "error": "Internal server error",
    "status_code": 500
}

# Low Confidence Mock Responses
MOCK_LOW_CONFIDENCE_HBSAG = {
    "status": "Positive",
    "confidence": 0.65,
    "raw_text": "maybe hep b"
}

MOCK_LOW_CONFIDENCE_LOCATION = {
    "location": "OR",
    "confidence": 0.60,
    "raw_text": "somewhere surgical"
}

# Hallucinated Mock Responses
MOCK_HALLUCINATED_HBSAG = {
    "status": "Maybe Positive",
    "confidence": 0.70,
    "raw_text": "liver issue"
}

MOCK_HALLUCINATED_LOCATION = {
    "location": "Pharmacy",
    "confidence": 0.75,
    "raw_text": "getting medication"
}

MOCK_HALLUCINATED_GENDER = {
    "gender": "Unknown",
    "confidence": 0.50,
    "raw_text": "person"
}


def get_mock_response(field_type: str, value: str) -> Dict[str, Any]:
    """
    Get mock response for field type and value
    
    Args:
        field_type: Type of field (hbsag, treatment_location, gender)
        value: Expected value
    
    Returns:
        Mock response dictionary
    """
    mock_responses = {
        "hbsag": {
            "Positive": MOCK_HBSAG_POSITIVE,
            "Negative": MOCK_HBSAG_NEGATIVE,
            "Unknown": MOCK_HBSAG_UNKNOWN
        },
        "treatment_location": {
            "OR": MOCK_LOCATION_OR,
            "Bedside": MOCK_LOCATION_BEDSIDE,
            "ICU-CCU": MOCK_LOCATION_ICU_CCU,
            "ER": MOCK_LOCATION_ER,
            "Multi-Tx Room": MOCK_LOCATION_MULTI_TX
        },
        "gender": {
            "Male": MOCK_GENDER_MALE,
            "Female": MOCK_GENDER_FEMALE
        }
    }
    
    return mock_responses.get(field_type, {}).get(value, {})
