from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database import get_db
from backend.schemas import PredictRequest, PredictResponse, BatchPredictResponse, ModelInfoResponse, LeadExplanationResponse
from backend.services.predict_service import predict_single, predict_batch, get_model_info, get_lead_explanation
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

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info_endpoint():
    info = get_model_info()
    return {
        "model_version": info["model_metrics"]["model_version"],
        "algorithm": info["model_metrics"]["algorithm"],
        "accuracy": info["model_metrics"]["accuracy"],
        "roc_auc": info["model_metrics"]["roc_auc"],
        "precision": info["model_metrics"]["precision"],
        "recall": info["model_metrics"]["recall"],
        "f1_score": info["model_metrics"]["f1_score"],
        "train_size": info["model_metrics"]["train_size"],
        "test_size": info["model_metrics"]["test_size"],
        "total_features": info["model_metrics"]["total_features"],
        "training_date": info["model_metrics"]["training_date"],
        "trained_on": info["model_metrics"]["trained_on"],
        "confusion_matrix": info["model_metrics"]["confusion_matrix"],
        "top_10_features": info["top_10_features"],
        "all_features": info["feature_importance"]
    }

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

@router.post("/explain", response_model=LeadExplanationResponse, dependencies=[Depends(get_api_key)])
async def explain_prediction(request: PredictRequest):
    result = get_lead_explanation(request.model_dump())
    return result

@router.get("/explain/{lead_id}", response_model=LeadExplanationResponse, dependencies=[Depends(get_api_key)])
async def explain_lead_by_id(lead_id: str, db: AsyncSession = Depends(get_db)):
    query = select(Lead).where(Lead.lead_id == lead_id)
    result = await db.execute(query)
    lead = result.scalars().first()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
        
    lead_dict = {c.name: getattr(lead, c.name) for c in Lead.__table__.columns}
    
    explanation = get_lead_explanation(lead_dict)
    
    return explanation
