import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.mark.unit
def test_login_success():
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "testuser", "password": "testpass"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "user" in data


@pytest.mark.unit
def test_login_invalid_payload():
    response = client.post("/api/v1/auth/login", json={"username": "testuser"})
    assert response.status_code == 422


@pytest.mark.unit
def test_register_success():
    response = client.post(
        "/api/v1/auth/register",
        json={"username": "newuser", "email": "newuser@example.com", "password": "securepass"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "newuser"
    assert "id" in data
    assert "created_at" in data


@pytest.mark.unit
def test_register_invalid_email():
    response = client.post(
        "/api/v1/auth/register",
        json={"username": "newuser", "email": "not-an-email", "password": "pass"},
    )
    assert response.status_code == 422


@pytest.mark.unit
def test_get_current_user():
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "username" in data
    assert "email" in data


@pytest.mark.unit
def test_logout():
    response = client.post("/api/v1/auth/logout")
    assert response.status_code == 204
