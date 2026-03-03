from pydantic import BaseModel
from datetime import date, datetime
from typing import List, Optional
from enum import Enum


class Gender(str, Enum):
    Male = "Male"
    Female = "Female"


class TreatmentLocation(str, Enum):
    OR = "OR"
    Bedside = "Bedside"
    ICU_CCU = "ICU-CCU"
    ER = "ER"
    Multi_Tx_Room = "Multi-Tx Room"


class HBsAgStatus(str, Enum):
    Positive = "Positive"
    Negative = "Negative"
    Unknown = "Unknown"


class SourceType(str, Enum):
    Lab = "Lab"
    CWOW = "CWOW"
    DaVita = "DaVita"
    Manual_Entry = "Manual Entry"


class PatientCreate(BaseModel):
    mrn: str
    firstName: str
    middleName: Optional[str] = None
    lastName: str
    dob: str  # Frontend sends as string in YYYY-MM-DD format
    gender: Gender
    admissionNumber: Optional[str] = None
    treatmentLocation: TreatmentLocation
    roomNumber: str
    hbsAgStatus: HBsAgStatus
    hbsAgDate: Optional[str] = None
    hbsAgSource: Optional[SourceType] = None
    hbsAbValue: Optional[int] = None
    hbsAbDate: Optional[str] = None
    hbsAbSource: Optional[SourceType] = None


class PatientUpdate(BaseModel):
    firstName: Optional[str] = None
    middleName: Optional[str] = None
    lastName: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[Gender] = None
    admissionNumber: Optional[str] = None
    treatmentLocation: Optional[TreatmentLocation] = None
    roomNumber: Optional[str] = None
    hbsAgStatus: Optional[HBsAgStatus] = None
    hbsAgDate: Optional[str] = None
    hbsAgSource: Optional[SourceType] = None
    hbsAbValue: Optional[int] = None
    hbsAbDate: Optional[str] = None
    hbsAbSource: Optional[SourceType] = None


class PatientResponse(BaseModel):
    id: str
    mrn: str
    firstName: str
    middleName: Optional[str] = None
    lastName: str
    dob: str
    gender: Gender
    admissionNumber: Optional[str] = None
    treatmentLocation: TreatmentLocation
    roomNumber: str
    hbsAgStatus: HBsAgStatus
    hbsAgDate: Optional[str] = None
    hbsAgSource: Optional[SourceType] = None
    hbsAbValue: Optional[int] = None
    hbsAbDate: Optional[str] = None
    hbsAbSource: Optional[SourceType] = None
    created_at: datetime
    updated_at: datetime


class PatientList(BaseModel):
    items: List[PatientResponse]
    total: int
    page: int
    page_size: int
