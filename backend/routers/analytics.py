from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

from backend.database import get_db
from backend.services.analytics_service import (
    get_conversion_by_source,
    get_conversion_by_city,
    get_conversion_by_course,
    get_stage_funnel,
    get_counselor_performance,
    get_monthly_leads,
    get_stuck_leads
)

router = APIRouter()

@router.get("/source")
async def get_source_analytics(db: AsyncSession = Depends(get_db)):
    return await get_conversion_by_source(db)

@router.get("/city")
async def get_city_analytics(db: AsyncSession = Depends(get_db)):
    return await get_conversion_by_city(db)

@router.get("/course")
async def get_course_analytics(db: AsyncSession = Depends(get_db)):
    return await get_conversion_by_course(db)

@router.get("/funnel")
async def get_funnel_analytics(db: AsyncSession = Depends(get_db)):
    return await get_stage_funnel(db)

@router.get("/counselors")
async def get_counselors_analytics(db: AsyncSession = Depends(get_db)):
    return await get_counselor_performance(db)

@router.get("/monthly")
async def get_monthly_analytics(db: AsyncSession = Depends(get_db)):
    return await get_monthly_leads(db)

@router.get("/stuck-leads")
async def get_stuck_leads_analytics(db: AsyncSession = Depends(get_db)):
    return await get_stuck_leads(db)

@router.get("/overview")
async def get_overview_analytics(db: AsyncSession = Depends(get_db)):
    # Run all queries in parallel using asyncio.gather
    results = await asyncio.gather(
        get_conversion_by_source(db),
        get_conversion_by_city(db),
        get_conversion_by_course(db),
        get_stage_funnel(db),
        get_counselor_performance(db),
        get_monthly_leads(db),
        get_stuck_leads(db)
    )
    
    return {
        "source": results[0],
        "city": results[1],
        "course": results[2],
        "funnel": results[3],
        "counselors": results[4],
        "monthly": results[5],
        "stuck_leads": results[6]
    }
