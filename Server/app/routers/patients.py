from fastapi import APIRouter, Depends, Query
from datetime import datetime, date
from typing import List
from app.models.patient import (
    PatientCreate, 
    PatientUpdate, 
    PatientResponse, 
    PatientList,
    Gender,
    TreatmentLocation,
    HBsAgStatus
)
from app.dependencies import get_current_user, get_db

router = APIRouter(prefix="/patients", tags=["Patients"])

_PLACEHOLDER_PATIENT = PatientResponse(
    id="patient-001",
    mrn="MRN-001",
    firstName="John",
    middleName="M",
    lastName="Doe",
    dob="1990-01-01",
    gender=Gender.Male,
    admissionNumber="ADM-001",
    treatmentLocation=TreatmentLocation.OR,
    roomNumber="101",
    hbsAgStatus=HBsAgStatus.Negative,
    hbsAgDate=None,
    hbsAgSource=None,
    hbsAbValue=None,
    hbsAbDate=None,
    hbsAbSource=None,
    created_at=datetime.utcnow(),
    updated_at=datetime.utcnow(),
)


@router.get("", response_model=PatientList)
async def list_patients(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    return PatientList(
        items=[_PLACEHOLDER_PATIENT],
        total=1,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=PatientResponse, status_code=201)
async def create_patient(
    patient: PatientCreate,
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    return PatientResponse(
        id="patient-new",
        **patient.model_dump(),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: str,
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    return _PLACEHOLDER_PATIENT


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: str,
    patient: PatientUpdate,
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    return _PLACEHOLDER_PATIENT
