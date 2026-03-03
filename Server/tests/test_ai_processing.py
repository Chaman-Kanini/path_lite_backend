import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.mark.unit
def test_create_conversation():
    response = client.post(
        "/api/v1/ai/conversations",
        json={"patient_id": "patient-001", "treatment_type": "physiotherapy"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["patient_id"] == "patient-001"
    assert data["status"] == "active"
    assert "messages" in data


@pytest.mark.unit
def test_create_conversation_missing_patient_id():
    response = client.post("/api/v1/ai/conversations", json={})
    assert response.status_code == 422


@pytest.mark.unit
def test_send_message():
    response = client.post(
        "/api/v1/ai/conversations/conv-001/messages",
        json={"content": "How is the patient progressing?", "role": "user"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["content"] == "How is the patient progressing?"
    assert data["role"] == "user"
    assert "timestamp" in data


@pytest.mark.unit
def test_send_message_invalid_role():
    response = client.post(
        "/api/v1/ai/conversations/conv-001/messages",
        json={"content": "Hello", "role": "invalid_role"},
    )
    assert response.status_code == 422


@pytest.mark.unit
def test_get_conversation():
    response = client.get("/api/v1/ai/conversations/conv-001")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "patient_id" in data
    assert "messages" in data
    assert "status" in data
