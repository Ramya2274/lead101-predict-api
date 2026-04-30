from sqlalchemy import Column, String, Integer, Float
from backend.database import Base

class Lead(Base):
    __tablename__ = "leads"

    lead_id = Column(String, primary_key=True, index=True)
    created_date = Column(String)
    source = Column(String)
    course_interest = Column(String)
    city = Column(String)
    total_calls = Column(Integer)
    total_whatsapp_messages = Column(Integer)
    total_emails = Column(Integer)
    email_opened = Column(Integer)
    whatsapp_replied = Column(Integer)
    form_completion_percentage = Column(Integer)
    response_time_hours = Column(Float, nullable=True)
    days_since_last_interaction = Column(Integer)
    current_stage = Column(String)
    days_in_inquiry_stage = Column(Integer)
    days_in_engagement_stage = Column(Integer)
    days_in_application_stage = Column(Integer)
    days_in_verification_stage = Column(Integer)
    counselor_id = Column(String)
    converted = Column(Integer)
    days_to_convert = Column(Float, nullable=True)
    conversion_probability = Column(Float, nullable=True)
