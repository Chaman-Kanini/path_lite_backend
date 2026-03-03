# TODO: Future backend integration for LLM cleaning pipeline
# This endpoint will receive raw transcribed text from the mobile app's
# Speech-to-Text service and clean/normalize it using an LLM pipeline.
#
# Expected API Contract:
# POST /api/voice/clean
# Request Body: { "text": str, "language": str, "confidence": float }
# Response Body: { "cleaned_text": str, "confidence": float, "corrections": list }
#
# Security Requirements:
# - Requires authentication token (Bearer)
# - PII must be redacted from logs
# - Input sanitization before LLM processing
# - Rate limiting: 60 requests/minute per user

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/voice", tags=["voice"])


class CleanTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    language: str = Field(default="en-US")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class CleanTextResponse(BaseModel):
    cleaned_text: str
    confidence: float
    corrections: list[str] = []


@router.post("/clean", response_model=CleanTextResponse)
async def clean_transcription(request: CleanTextRequest) -> CleanTextResponse:
    """
    Clean and normalize transcribed text using LLM pipeline.
    TODO: Implement LLM integration when backend AI service is ready.
    """
    # Placeholder: return text as-is until LLM pipeline is implemented
    return CleanTextResponse(
        cleaned_text=request.text.strip(),
        confidence=request.confidence,
        corrections=[],
    )
