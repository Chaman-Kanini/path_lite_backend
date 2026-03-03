import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.core.config import settings
from app.core.middleware import RequestIDMiddleware, RequestLoggingMiddleware
from app.routers import auth_router, patients_router, ai_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Path-Lite backend API for patient management and AI-assisted care",
    contact={"name": "Path-Lite Team"},
    license_info={"name": "Proprietary"},
    openapi_tags=[
        {"name": "Health", "description": "Health and status endpoints"},
        {"name": "Authentication", "description": "User authentication and authorization"},
        {"name": "Patients", "description": "Patient management operations"},
        {"name": "AI Processing", "description": "AI-assisted conversation endpoints"},
    ],
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(patients_router, prefix="/api/v1")
app.include_router(ai_router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": settings.APP_VERSION}


@app.get("/api/v1/status", tags=["Health"])
async def api_status():
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}

