from pydantic import BaseModel, EmailStr
from datetime import datetime


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime


LoginResponse.model_rebuild()
