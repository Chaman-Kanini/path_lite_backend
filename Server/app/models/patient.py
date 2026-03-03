from pydantic import BaseModel
from datetime import date, datetime
from typing import List, Optional


class PatientCreate(BaseModel):
    first_name: str
    last_name: str
    mrn: str
    dob: date
    email: Optional[str] = None
    phone: Optional[str] = None


class PatientUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class PatientResponse(BaseModel):
    id: str
    first_name: str
    last_name: str
    mrn: str
    dob: date
    email: Optional[str] = None
    phone: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class PatientList(BaseModel):
    items: List[PatientResponse]
    total: int
    page: int
    page_size: int
