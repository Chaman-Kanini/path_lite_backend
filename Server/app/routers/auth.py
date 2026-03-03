from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
from app.models.auth import LoginRequest, LoginResponse, RegisterRequest, UserResponse
from app.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    return LoginResponse(
        access_token="placeholder-token",
        token_type="bearer",
        user=UserResponse(
            id="user-001",
            username=request.username,
            email="user@example.com",
            created_at=datetime.utcnow(),
        ),
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    return UserResponse(
        id="user-001",
        username=request.username,
        email=request.email,
        created_at=datetime.utcnow(),
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(current_user=Depends(get_current_user)):
    return None


@router.get("/me", response_model=UserResponse)
async def get_me(current_user=Depends(get_current_user)):
    return UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        created_at=datetime.utcnow(),
    )
