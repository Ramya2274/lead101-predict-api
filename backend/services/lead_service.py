import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
import numpy as np

from backend.models import Lead

async def bulk_insert_leads(df: pd.DataFrame, db: AsyncSession) -> int:
    df.columns = df.columns.str.strip()
    print("Uploaded CSV columns:", df.columns.tolist())
    print("Total rows:", len(df))

    required_columns = ['lead_id', 'created_date', 'source', 'course_interest', 'city']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Got columns: {df.columns.tolist()}")

    print("CSV Columns:", df.columns.tolist())
    print("First row:", df.iloc[0].to_dict())

    leads = []
    for _, row in df.iterrows():
        lead = Lead(
            lead_id=str(row['lead_id']),
            created_date=str(row['created_date']),
            source=str(row['source']),
            course_interest=str(row['course_interest']),
            city=str(row['city']),
            total_calls=int(row['total_calls']),
            total_whatsapp_messages=int(row['total_whatsapp_messages']),
            total_emails=int(row['total_emails']),
            email_opened=int(row['email_opened']),
            whatsapp_replied=int(row['whatsapp_replied']),
            form_completion_percentage=int(row['form_completion_percentage']),
            response_time_hours=None if pd.isna(row['response_time_hours']) 
                                else float(row['response_time_hours']),
            days_since_last_interaction=int(row['days_since_last_interaction']),
            current_stage=str(row['current_stage']),
            days_in_inquiry_stage=int(row['days_in_inquiry_stage']),
            days_in_engagement_stage=int(row['days_in_engagement_stage']),
            days_in_application_stage=int(row['days_in_application_stage']),
            days_in_verification_stage=int(row['days_in_verification_stage']),
            counselor_id=str(row['counselor_id']),
            converted=int(row['converted']),
            days_to_convert=None if pd.isna(row['days_to_convert']) 
                            else float(row['days_to_convert']),
            conversion_probability=None
        )
        leads.append(lead)

    db.add_all(leads)
    await db.commit()
    
    return len(leads)

async def get_leads(db: AsyncSession, page: int, page_size: int, filters: dict):
    query = select(Lead)
    
    if filters.get("source"):
        query = query.where(Lead.source == filters["source"])
    if filters.get("city"):
        query = query.where(Lead.city == filters["city"])
    if filters.get("course_interest"):
        query = query.where(Lead.course_interest == filters["course_interest"])
    if filters.get("converted") is not None:
        query = query.where(Lead.converted == filters["converted"])
    if filters.get("current_stage"):
        query = query.where(Lead.current_stage == filters["current_stage"])
        
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    leads = result.scalars().all()
    
    return total, leads

async def get_lead_by_id(db: AsyncSession, lead_id: str):
    query = select(Lead).where(Lead.lead_id == lead_id)
    result = await db.execute(query)
    return result.scalars().first()

async def get_all_leads_filtered(db: AsyncSession, filters: dict):
    """Fetch all leads matching filters (no pagination) for export."""
    query = select(Lead)

    if filters.get("source"):
        query = query.where(Lead.source == filters["source"])
    if filters.get("city"):
        query = query.where(Lead.city == filters["city"])
    if filters.get("course_interest"):
        query = query.where(Lead.course_interest == filters["course_interest"])
    if filters.get("converted") is not None:
        query = query.where(Lead.converted == filters["converted"])
    if filters.get("current_stage"):
        query = query.where(Lead.current_stage == filters["current_stage"])
    if filters.get("start_date"):
        query = query.where(Lead.created_date >= filters["start_date"])
    if filters.get("end_date"):
        query = query.where(Lead.created_date <= filters["end_date"])

    result = await db.execute(query)
    return result.scalars().all()

async def get_top_leads(db: AsyncSession, limit: int = 20):
    query = select(Lead).where(
        and_(
            Lead.converted == 0,
            Lead.conversion_probability.isnot(None)
        )
    ).order_by(desc(Lead.conversion_probability)).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def get_at_risk_leads(db: AsyncSession, limit: int = 20, days_inactive: int = 7):
    query = select(Lead).where(
        and_(
            Lead.converted == 0,
            Lead.conversion_probability.isnot(None),
            Lead.conversion_probability > 0.5,
            Lead.days_since_last_interaction > days_inactive
        )
    ).order_by(desc(Lead.conversion_probability)).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def get_stuck_leads_detail(db: AsyncSession, limit: int = 50):
    query = select(Lead).where(
        and_(
            Lead.converted == 0,
            or_(
                Lead.days_in_inquiry_stage > 14,
                Lead.days_in_engagement_stage > 14
            )
        )
    ).order_by(desc(Lead.conversion_probability)).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

