from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import re
from utils.logger import get_logger

logger = get_logger("data_models")

# PII Detection Patterns
PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "vin": r"\b[A-HJ-NPR-Z0-9]{17}\b"
}

def get_current_utc():
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)

class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., max_length=10000, description="Message content")
    timestamp: datetime = Field(default_factory=get_current_utc)
    message_id: Optional[str] = None
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()
    
    def contains_pii(self) -> Dict[str, List[str]]:
        """Detect PII in message content"""
        pii_found = {}
        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, self.content, re.IGNORECASE)
            if matches:
                pii_found[pii_type] = matches
        return pii_found
    
    def dict(self, **kwargs):
        """Override dict method to serialize datetime as ISO string for DynamoDB"""
        result = super().dict(**kwargs)
        if 'timestamp' in result:
            result['timestamp'] = result['timestamp'].isoformat()
        return result
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Conversation(BaseModel):
    conversation_id: str = Field(..., description="Unique conversation identifier")
    user_id: str = Field(..., description="User identifier")
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=get_current_utc)
    updated_at: datetime = Field(default_factory=get_current_utc)
    ticket_id: Optional[str] = None
    status: str = Field(default="active", description="Conversation status")
    
    @validator('conversation_id')
    def validate_conversation_id(cls, v):
        if not re.match(r'^[A-Za-z0-9-]+$', v):
            raise ValueError("Invalid conversation ID format")
        return v
    
    def add_message(self, message: Message):
        """Add a message to the conversation"""
        self.messages.append(message)
        self.updated_at = get_current_utc()
    
    def get_pii_summary(self) -> Dict[str, int]:
        """Get summary of PII detected in conversation"""
        pii_summary = {}
        for message in self.messages:
            pii_found = message.contains_pii()
            for pii_type, matches in pii_found.items():
                pii_summary[pii_type] = pii_summary.get(pii_type, 0) + len(matches)
        return pii_summary
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class VehicleInfo(BaseModel):
    make: str = Field(..., description="Vehicle make")
    model: str = Field(..., description="Vehicle model")
    year: int = Field(..., ge=1900, le=2024, description="Vehicle year")
    color: str = Field(..., description="Vehicle color")
    vin: str = Field(..., description="Vehicle identification number")
    license_plate: str = Field(..., description="License plate number")
    
    @validator('vin')
    def validate_vin(cls, v):
        if not re.match(r'^[A-HJ-NPR-Z0-9]{17}$', v.upper()):
            raise ValueError("Invalid VIN format")
        return v.upper()
    
    @validator('license_plate')
    def validate_license_plate(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("License plate cannot be empty")
        return v.strip().upper()

class OwnerInfo(BaseModel):
    name: str = Field(..., description="Owner full name")
    phone: str = Field(..., description="Phone number")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Current address")
    
    @validator('email')
    def validate_email(cls, v):
        if not re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$', v):
            raise ValueError("Invalid email format")
        return v.lower()
    
    @validator('phone')
    def validate_phone(cls, v):
        # Remove all non-digits
        digits_only = re.sub(r'\D', '', v)
        if len(digits_only) != 10:
            raise ValueError("Phone number must be 10 digits")
        return digits_only

class IncidentInfo(BaseModel):
    incident_date: datetime = Field(..., description="Date of incident")
    incident_time: datetime = Field(..., description="Time of incident")
    location: str = Field(..., description="Last known location")
    circumstances: str = Field(..., description="Circumstances of theft")
    
    @validator('circumstances')
    def validate_circumstances(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Circumstances must be at least 10 characters")
        return v.strip()

class InsuranceInfo(BaseModel):
    company: str = Field(..., description="Insurance company")
    policy_number: str = Field(..., description="Policy number")
    
    @validator('policy_number')
    def validate_policy_number(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Policy number cannot be empty")
        return v.strip()

class Ticket(BaseModel):
    ticket_id: str = Field(..., description="Unique ticket identifier")
    user_id: str = Field(..., description="User identifier")
    status: str = Field(default="open", description="Ticket status")
    priority: str = Field(default="medium", description="Ticket priority")
    vehicle_info: VehicleInfo
    owner_info: OwnerInfo
    incident_info: IncidentInfo
    insurance_info: InsuranceInfo
    created_at: datetime = Field(default_factory=get_current_utc)
    updated_at: datetime = Field(default_factory=get_current_utc)
    assigned_to: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    
    @validator('ticket_id')
    def validate_ticket_id(cls, v):
        if not re.match(r'^SVL-\d{8}-\d{6}-\d{3}$', v):
            raise ValueError("Invalid ticket ID format")
        return v
    
    @validator('status')
    def validate_status(cls, v):
        allowed_statuses = ["open", "in_progress", "resolved", "closed"]
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {allowed_statuses}")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        allowed_priorities = ["low", "medium", "high", "urgent"]
        if v not in allowed_priorities:
            raise ValueError(f"Priority must be one of: {allowed_priorities}")
        return v
    
    def add_note(self, note: str):
        """Add a note to the ticket"""
        self.notes.append(f"{get_current_utc().isoformat()}: {note}")
        self.updated_at = get_current_utc()
    
    def get_pii_summary(self) -> Dict[str, int]:
        """Get summary of PII in ticket"""
        pii_summary = {}
        # Check owner info
        owner_pii = self.owner_info.phone + self.owner_info.email + self.owner_info.address
        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, owner_pii, re.IGNORECASE)
            if matches:
                pii_summary[pii_type] = len(matches)
        return pii_summary
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AuditLog(BaseModel):
    log_id: str = Field(..., description="Unique log identifier")
    operation: str = Field(..., description="Operation performed")
    table_name: str = Field(..., description="DynamoDB table name")
    record_id: str = Field(..., description="Record identifier")
    user_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=get_current_utc)
    details: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 