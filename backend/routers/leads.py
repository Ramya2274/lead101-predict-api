from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import io
from sqlalchemy import delete

from backend.database import get_db
from backend.schemas import UploadResponse
from backend.services.lead_service import bulk_insert_leads
from backend.services.predict_service import predict_batch
from backend.models import Lead

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_leads(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        
        # Delete all existing rows
        await db.execute(delete(Lead))
        await db.commit()
        
        total_inserted = await bulk_insert_leads(df, db)
        
        # AUTO score all leads immediately
        total_scored = await predict_batch(db)
        
        return UploadResponse(
            message="Upload and scoring successful",
            total_inserted=total_inserted,
            total_scored=total_scored
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
