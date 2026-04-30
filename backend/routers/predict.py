from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database import get_db
from backend.schemas import PredictRequest, PredictResponse, BatchPredictResponse
from backend.services.predict_service import predict_single, predict_batch
from backend.models import Lead
from sqlalchemy import select

from backend.auth import get_api_key

router = APIRouter()

@router.post("", response_model=PredictResponse, dependencies=[Depends(get_api_key)])
@router.post("/", response_model=PredictResponse, include_in_schema=False, dependencies=[Depends(get_api_key)])
async def predict_endpoint(request: PredictRequest):
    result = predict_single(request.model_dump())
    return result

@router.post("/batch", response_model=BatchPredictResponse, dependencies=[Depends(get_api_key)])
async def batch_predict_endpoint(db: AsyncSession = Depends(get_db)):
    total_scored = await predict_batch(db)
    return BatchPredictResponse(
        message="Batch prediction successful",
        total_scored=total_scored
    )

@router.get("/{lead_id}", response_model=PredictResponse, dependencies=[Depends(get_api_key)])
async def predict_lead_by_id(lead_id: str, db: AsyncSession = Depends(get_db)):
    query = select(Lead).where(Lead.lead_id == lead_id)
    result = await db.execute(query)
    lead = result.scalars().first()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
        
    lead_dict = {c.name: getattr(lead, c.name) for c in Lead.__table__.columns}
    
    prediction = predict_single(lead_dict)
    
    # Also update conversion_probability in DB
    lead.conversion_probability = prediction["conversion_probability"]
    await db.commit()
    
    return prediction
