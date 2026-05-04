import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models import Lead

async def bulk_insert_leads(df: pd.DataFrame, db: AsyncSession) -> int:
    df.columns = df.columns.str.strip()
    required_columns = ['lead_id', 'created_date', 'source', 'course_interest', 'city']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Got columns: {df.columns.tolist()}")

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
