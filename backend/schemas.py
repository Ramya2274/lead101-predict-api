from pydantic import BaseModel
from typing import Optional, List

class LeadBase(BaseModel):
    lead_id: str
    created_date: str
    source: str
    course_interest: str
    city: str
    total_calls: int
    total_whatsapp_messages: int
    total_emails: int
    email_opened: int
    whatsapp_replied: int
    form_completion_percentage: int
    response_time_hours: Optional[float] = None
    days_since_last_interaction: int
    current_stage: str
    days_in_inquiry_stage: int
    days_in_engagement_stage: int
    days_in_application_stage: int
    days_in_verification_stage: int
    counselor_id: str
    converted: int
    days_to_convert: Optional[float] = None
    conversion_probability: Optional[float] = None

class LeadCreate(LeadBase):
    pass

class LeadRead(LeadBase):
    class Config:
        from_attributes = True

class LeadListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    data: List[LeadRead]

class UploadResponse(BaseModel):
    message: str
    total_inserted: int
    skipped: int

class PredictRequest(BaseModel):
    source: str = "Unknown"
    course_interest: str = "Unknown"
    city: str = "Unknown"
    total_calls: int = 0
    total_whatsapp_messages: int = 0
    total_emails: int = 0
    email_opened: int = 0
    whatsapp_replied: int = 0
    form_completion_percentage: int = 0
    response_time_hours: Optional[float] = None
    days_since_last_interaction: int = 0
    current_stage: str = "Unknown"
    days_in_inquiry_stage: int = 0
    days_in_engagement_stage: int = 0
    days_in_application_stage: int = 0
    days_in_verification_stage: int = 0
    counselor_id: str = "Unknown"

class PredictResponse(BaseModel):
    conversion_probability: float
    will_convert: bool
    confidence: str
    risk_factors: List[str]

class BatchPredictResponse(BaseModel):
    message: str
    total_scored: int
