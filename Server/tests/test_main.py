import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.mark.unit
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.unit
def test_api_status():
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app" in data
    assert "version" in data


@pytest.mark.unit
def test_openapi_schema():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema


@pytest.mark.unit
def test_docs_accessible():
    response = client.get("/docs")
    assert response.status_code == 200


@pytest.mark.unit
def test_redoc_accessible():
    response = client.get("/redoc")
    assert response.status_code == 200


@pytest.mark.unit
def test_cors_headers():
    response = client.options(
        "/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code in (200, 204)
