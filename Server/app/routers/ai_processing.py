from fastapi import APIRouter, Depends
from datetime import datetime
from app.models.ai import (
    ConversationCreate,
    MessageCreate,
    ConversationResponse,
    MessageResponse,
    MessageRole,
    ConversationStatus,
)
from app.dependencies import get_current_user, get_db

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
