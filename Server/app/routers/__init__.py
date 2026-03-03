from app.routers.auth import router as auth_router
from app.routers.patients import router as patients_router
from app.routers.ai_processing import router as ai_router

__all__ = ["auth_router", "patients_router", "ai_router"]
