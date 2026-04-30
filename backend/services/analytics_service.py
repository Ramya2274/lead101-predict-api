from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from backend.models import Lead

async def get_conversion_by_source(db: AsyncSession):
    query = select(
        Lead.source,
        func.count().label('total_leads'),
        func.sum(Lead.converted).label('converted_leads'),
        (func.sum(Lead.converted) * 100.0 / func.count()).label('conversion_rate')
    ).group_by(Lead.source).order_by((func.sum(Lead.converted) * 100.0 / func.count()).desc())
    
    result = await db.execute(query)
    rows = result.all()
    return [
        {
            "source": row.source,
            "total_leads": row.total_leads,
            "converted_leads": row.converted_leads,
            "conversion_rate": float(row.conversion_rate) if row.conversion_rate else 0.0
        }
        for row in rows
    ]

async def get_conversion_by_city(db: AsyncSession):
    query = select(
        Lead.city,
        func.count().label('total_leads'),
        func.sum(Lead.converted).label('converted_leads'),
        (func.sum(Lead.converted) * 100.0 / func.count()).label('conversion_rate')
    ).group_by(Lead.city).order_by(func.count().desc())
    
    result = await db.execute(query)
    rows = result.all()
    return [
        {
            "city": row.city,
            "total_leads": row.total_leads,
            "converted_leads": row.converted_leads,
            "conversion_rate": float(row.conversion_rate) if row.conversion_rate else 0.0
        }
        for row in rows
    ]

async def get_conversion_by_course(db: AsyncSession):
    query = select(
        Lead.course_interest,
        func.count().label('total_leads'),
        func.sum(Lead.converted).label('converted_leads'),
        (func.sum(Lead.converted) * 100.0 / func.count()).label('conversion_rate')
    ).group_by(Lead.course_interest).order_by(func.count().desc())
    
    result = await db.execute(query)
    rows = result.all()
    return [
        {
            "course": row.course_interest,
            "total_leads": row.total_leads,
            "converted_leads": row.converted_leads,
            "conversion_rate": float(row.conversion_rate) if row.conversion_rate else 0.0
        }
        for row in rows
    ]

async def get_stage_funnel(db: AsyncSession):
    query = select(
        Lead.current_stage,
        func.count().label('count')
    ).group_by(Lead.current_stage)
    
    result = await db.execute(query)
    rows = result.all()
    
    stage_counts = {row.current_stage: row.count for row in rows}
    
    stages = [
        "inquiry", "engagement", "application", 
        "verification", "admission", "payment", "enrollment"
    ]
    
    return [
        {"stage": stage, "count": stage_counts.get(stage, 0)}
        for stage in stages
    ]

async def get_counselor_performance(db: AsyncSession):
    query = select(
        Lead.counselor_id,
        func.count().label('total_leads'),
        func.sum(Lead.converted).label('converted_leads'),
        (func.sum(Lead.converted) * 100.0 / func.count()).label('conversion_rate'),
        func.avg(Lead.form_completion_percentage).label('avg_form_completion'),
        func.avg(Lead.total_calls).label('avg_calls'),
        func.avg(Lead.total_whatsapp_messages).label('avg_whatsapp_messages')
    ).group_by(Lead.counselor_id).order_by((func.sum(Lead.converted) * 100.0 / func.count()).desc())
    
    result = await db.execute(query)
    rows = result.all()
    return [
        {
            "counselor_id": row.counselor_id,
            "total_leads": row.total_leads,
            "converted_leads": row.converted_leads,
            "conversion_rate": float(row.conversion_rate) if row.conversion_rate else 0.0,
            "avg_form_completion": float(row.avg_form_completion) if row.avg_form_completion else 0.0,
            "avg_calls": float(row.avg_calls) if row.avg_calls else 0.0,
            "avg_whatsapp_messages": float(row.avg_whatsapp_messages) if row.avg_whatsapp_messages else 0.0
        }
        for row in rows
    ]

async def get_monthly_leads(db: AsyncSession):
    year_col = func.substr(Lead.created_date, 1, 4)
    month_col = func.substr(Lead.created_date, 6, 2)
    
    query = select(
        year_col.label('year'),
        month_col.label('month'),
        func.count().label('total_leads'),
        func.sum(Lead.converted).label('converted_leads'),
        (func.sum(Lead.converted) * 100.0 / func.count()).label('conversion_rate')
    ).group_by(year_col, month_col).order_by(year_col, month_col)
    
    result = await db.execute(query)
    rows = result.all()
    return [
        {
            "year": row.year,
            "month": row.month,
            "total_leads": row.total_leads,
            "converted_leads": row.converted_leads,
            "conversion_rate": float(row.conversion_rate) if row.conversion_rate else 0.0
        }
        for row in rows
    ]

async def get_stuck_leads(db: AsyncSession):
    query = select(Lead).where(
        and_(
            Lead.converted == 0,
            or_(
                Lead.days_in_inquiry_stage > 14,
                Lead.days_in_engagement_stage > 14
            )
        )
    )
    
    result = await db.execute(query)
    leads = result.scalars().all()
    
    return {
        "count": len(leads),
        "leads": [
            {
                "lead_id": lead.lead_id,
                "counselor_id": lead.counselor_id,
                "current_stage": lead.current_stage
            }
            for lead in leads
        ]
    }
