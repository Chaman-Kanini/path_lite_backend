import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.mark.unit
def test_list_patients():
    response = client.get("/api/v1/patients")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data


@pytest.mark.unit
def test_list_patients_pagination():
    response = client.get("/api/v1/patients?page=2&page_size=5")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 2
    assert data["page_size"] == 5


@pytest.mark.unit
def test_create_patient():
    response = client.post(
        "/api/v1/patients",
        json={
            "first_name": "Jane",
            "last_name": "Smith",
            "mrn": "MRN-002",
            "dob": "1985-06-15",
            "email": "jane.smith@example.com",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["first_name"] == "Jane"
    assert data["last_name"] == "Smith"
    assert "id" in data


@pytest.mark.unit
def test_create_patient_missing_required_fields():
    response = client.post("/api/v1/patients", json={"first_name": "Jane"})
    assert response.status_code == 422


@pytest.mark.unit
def test_get_patient():
    response = client.get("/api/v1/patients/patient-001")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "first_name" in data
    assert "mrn" in data


@pytest.mark.unit
def test_update_patient():
    response = client.put(
        "/api/v1/patients/patient-001",
        json={"email": "updated@example.com"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
