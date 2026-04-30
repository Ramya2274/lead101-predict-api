from typing import Optional
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
from backend.auth import get_api_key

router = APIRouter()

@router.get("/source", dependencies=[Depends(get_api_key)])
async def get_source_analytics(db: AsyncSession = Depends(get_db), start_date: Optional[str] = None, end_date: Optional[str] = None):
    return await get_conversion_by_source(db, start_date, end_date)

@router.get("/city", dependencies=[Depends(get_api_key)])
async def get_city_analytics(db: AsyncSession = Depends(get_db), start_date: Optional[str] = None, end_date: Optional[str] = None):
    return await get_conversion_by_city(db, start_date, end_date)

@router.get("/course", dependencies=[Depends(get_api_key)])
async def get_course_analytics(db: AsyncSession = Depends(get_db), start_date: Optional[str] = None, end_date: Optional[str] = None):
    return await get_conversion_by_course(db, start_date, end_date)

@router.get("/funnel", dependencies=[Depends(get_api_key)])
async def get_funnel_analytics(db: AsyncSession = Depends(get_db), start_date: Optional[str] = None, end_date: Optional[str] = None):
    return await get_stage_funnel(db, start_date, end_date)

@router.get("/counselors", dependencies=[Depends(get_api_key)])
async def get_counselors_analytics(db: AsyncSession = Depends(get_db), start_date: Optional[str] = None, end_date: Optional[str] = None):
    return await get_counselor_performance(db, start_date, end_date)

@router.get("/monthly", dependencies=[Depends(get_api_key)])
async def get_monthly_analytics(db: AsyncSession = Depends(get_db), start_date: Optional[str] = None, end_date: Optional[str] = None):
    return await get_monthly_leads(db, start_date, end_date)

@router.get("/stuck-leads", dependencies=[Depends(get_api_key)])
async def get_stuck_leads_analytics(db: AsyncSession = Depends(get_db), start_date: Optional[str] = None, end_date: Optional[str] = None):
    return await get_stuck_leads(db, start_date, end_date)

@router.get("/overview", dependencies=[Depends(get_api_key)])
async def get_overview_analytics(db: AsyncSession = Depends(get_db), start_date: Optional[str] = None, end_date: Optional[str] = None):
    # Run all queries in parallel using asyncio.gather
    results = await asyncio.gather(
        get_conversion_by_source(db, start_date, end_date),
        get_conversion_by_city(db, start_date, end_date),
        get_conversion_by_course(db, start_date, end_date),
        get_stage_funnel(db, start_date, end_date),
        get_counselor_performance(db, start_date, end_date),
        get_monthly_leads(db, start_date, end_date),
        get_stuck_leads(db, start_date, end_date)
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
