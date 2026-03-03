import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from main import app
from app.core.config import settings


@pytest.fixture(scope="session")
def test_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_settings():
    return settings


@pytest.fixture
def mock_user():
    return {
        "id": "test-user-id",
        "username": "testuser",
        "email": "testuser@example.com",
    }
