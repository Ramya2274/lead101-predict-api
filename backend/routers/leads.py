from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import pandas as pd
import io
from io import StringIO, BytesIO
from sqlalchemy import delete, func, select

from backend.database import get_db
from backend.schemas import LeadListResponse, UploadResponse, LeadRead
from backend.services.lead_service import bulk_insert_leads, get_leads, get_lead_by_id, get_all_leads_filtered
from backend.models import Lead

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_leads(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Delete all existing rows
        await db.execute(delete(Lead))
        await db.commit()
        
        total_inserted = await bulk_insert_leads(df, db)
        
        return UploadResponse(
            message="Upload successful",
            total_inserted=total_inserted,
            skipped=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("", response_model=LeadListResponse)
@router.get("/", response_model=LeadListResponse, include_in_schema=False)
async def get_leads_endpoint(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1),
    source: Optional[str] = None,
    city: Optional[str] = None,
    course_interest: Optional[str] = None,
    converted: Optional[int] = None,
    current_stage: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    filters = {
        "source": source,
        "city": city,
        "course_interest": course_interest,
        "converted": converted,
        "current_stage": current_stage
    }
    
    total, leads = await get_leads(db, page, page_size, filters)
    
    return LeadListResponse(
        total=total,
        page=page,
        page_size=page_size,
        data=leads
    )

@router.get("/stats/summary")
async def get_stats_summary(db: AsyncSession = Depends(get_db)):
    total_leads_query = select(func.count(Lead.lead_id))
    total_leads = (await db.execute(total_leads_query)).scalar() or 0
    
    total_converted_query = select(func.count(Lead.lead_id)).where(Lead.converted == 1)
    total_converted = (await db.execute(total_converted_query)).scalar() or 0
    
    conversion_rate = (total_converted / total_leads * 100) if total_leads > 0 else 0
    
    distinct_sources_query = select(func.count(func.distinct(Lead.source)))
    distinct_sources = (await db.execute(distinct_sources_query)).scalar() or 0
    
    distinct_cities_query = select(func.count(func.distinct(Lead.city)))
    distinct_cities = (await db.execute(distinct_cities_query)).scalar() or 0
    
    return {
        "total_leads": total_leads,
        "total_converted": total_converted,
        "conversion_rate": conversion_rate,
        "distinct_sources": distinct_sources,
        "distinct_cities": distinct_cities
    }

@router.get("/export/csv")
async def export_leads_csv(
    source: Optional[str] = None,
    city: Optional[str] = None,
    course_interest: Optional[str] = None,
    converted: Optional[int] = None,
    current_stage: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    filters = {
        "source": source,
        "city": city,
        "course_interest": course_interest,
        "converted": converted,
        "current_stage": current_stage,
        "start_date": start_date,
        "end_date": end_date
    }
    
    leads = await get_all_leads_filtered(db, filters)
    
    # Convert leads to list of dicts for pandas
    leads_data = []
    for lead in leads:
        lead_dict = lead.__dict__.copy()
        if "_sa_instance_state" in lead_dict:
            del lead_dict["_sa_instance_state"]
        leads_data.append(lead_dict)
        
    df = pd.DataFrame(leads_data)
    
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=leads_export.csv"
        }
    )

@router.get("/export/excel")
async def export_leads_excel(
    source: Optional[str] = None,
    city: Optional[str] = None,
    course_interest: Optional[str] = None,
    converted: Optional[int] = None,
    current_stage: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    filters = {
        "source": source,
        "city": city,
        "course_interest": course_interest,
        "converted": converted,
        "current_stage": current_stage,
        "start_date": start_date,
        "end_date": end_date
    }
    
    leads = await get_all_leads_filtered(db, filters)
    
    leads_data = []
    for lead in leads:
        lead_dict = lead.__dict__.copy()
        if "_sa_instance_state" in lead_dict:
            del lead_dict["_sa_instance_state"]
        leads_data.append(lead_dict)
        
    df = pd.DataFrame(leads_data)
    
    output = BytesIO()
    df.to_excel(output, index=False, sheet_name="Leads")
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=leads_export.xlsx"
        }
    )

@router.get("/{lead_id}", response_model=LeadRead)
async def get_lead_by_id_endpoint(lead_id: str, db: AsyncSession = Depends(get_db)):
    lead = await get_lead_by_id(db, lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    return lead
