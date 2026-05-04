from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from backend.database import get_db
from backend.schemas import PredictRequest, PredictResponse, PredictResultRead, PredictResultsResponse
from backend.services.predict_service import predict_single
from backend.models import Lead

router = APIRouter()

@router.post("", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    result = predict_single(request.model_dump())
    return result

@router.get("/results", response_model=PredictResultsResponse)
async def get_predict_results(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1),
    converted: Optional[int] = None,
    source: Optional[str] = None,
    city: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    query = select(Lead)
    
    if converted is not None:
        query = query.where(Lead.converted == converted)
    if source is not None:
        query = query.where(Lead.source == source)
    if city is not None:
        query = query.where(Lead.city == city)
        
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0
    
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    leads = result.scalars().all()
    
    data = []
    for lead in leads:
        prob = lead.conversion_probability
        will_convert = prob is not None and prob >= 0.5
        confidence = "high" if prob is not None and prob > 0.75 else "medium" if prob is not None and prob > 0.5 else "low" if prob is not None else "unknown"
        
        data.append(PredictResultRead(
            lead_id=lead.lead_id,
            source=lead.source,
            course_interest=lead.course_interest,
            city=lead.city,
            current_stage=lead.current_stage,
            counselor_id=lead.counselor_id,
            conversion_probability=prob,
            will_convert=will_convert,
            confidence=confidence
        ))
        
    return PredictResultsResponse(
        total=total,
        page=page,
        page_size=page_size,
        data=data
    )
