from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from enum import Enum


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ConversationStatus(str, Enum):
    active = "active"
    completed = "completed"
    archived = "archived"


class ConversationCreate(BaseModel):
    patient_id: str
    treatment_type: Optional[str] = None


class MessageCreate(BaseModel):
    content: str
    role: MessageRole = MessageRole.user


class MessageResponse(BaseModel):
    id: str
    content: str
    role: MessageRole
    timestamp: datetime


class ConversationResponse(BaseModel):
    id: str
    patient_id: str
    messages: List[MessageResponse]
    status: ConversationStatus
    created_at: datetime
