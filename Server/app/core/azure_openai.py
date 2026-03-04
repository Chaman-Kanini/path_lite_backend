from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    _instance: Optional["AzureOpenAIClient"] = None
    _client: Optional[AzureOpenAI] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._initialize_client()

    def _initialize_client(self):
        try:
            self._client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                timeout=settings.AZURE_OPENAI_TIMEOUT,
                max_retries=0,
            )
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise

    @property
    def client(self) -> AzureOpenAI:
        if self._client is None:
            self._initialize_client()
        return self._client

    @retry(
        stop=stop_after_attempt(settings.AZURE_OPENAI_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def verify_connection(self) -> bool:
        try:
            response = self.client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=5,
            )
            logger.info("Azure OpenAI connection verified successfully")
            return True
        except Exception as e:
            logger.error(f"Azure OpenAI connection verification failed: {str(e)}")
            raise


_azure_openai_client: Optional[AzureOpenAIClient] = None


def get_azure_openai_client() -> AzureOpenAIClient:
    global _azure_openai_client
    if _azure_openai_client is None:
        _azure_openai_client = AzureOpenAIClient()
    return _azure_openai_client
