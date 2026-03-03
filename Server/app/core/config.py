from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List
import json


class Settings(BaseSettings):
    APP_NAME: str = "Path-Lite API"
    APP_VERSION: str = "0.1.0"
    DEBUG_MODE: bool = True

    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    LOG_LEVEL: str = "info"
    RELOAD: bool = True

    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8081"]

    SECRET_KEY: str = "dev-secret-key-do-not-use-in-production"

    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4"
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()
