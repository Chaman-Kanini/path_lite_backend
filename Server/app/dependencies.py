from fastapi import Depends, HTTPException, status
from app.core.config import Settings, settings


def get_settings() -> Settings:
    return settings


async def get_db():
    yield None


async def get_current_user(db=Depends(get_db)):
    return {"id": "placeholder-user-id", "username": "placeholder", "email": "user@example.com"}
