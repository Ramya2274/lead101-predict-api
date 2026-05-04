from fastapi import APIRouter
from prophet import Prophet
import joblib
import json
import pandas as pd
import os

router = APIRouter()

# Load model and metadata at startup
model = joblib.load("backend/ml/forecast_model.pkl")
metadata = joblib.load("backend/ml/forecast_metadata.pkl")

with open("backend/ml/next_4_weeks.json", "r") as f:
    next_4_weeks_cache = json.load(f)

@router.get("/next-week")
async def get_next_week_forecast():
    """
    Get conversion forecast for next week only
    """
    next_week = next_4_weeks_cache[0]
    return {
        "week": next_week['week'],
        "predicted_conversions": next_week['predicted'],
        "lower_bound": next_week['lower_bound'],
        "upper_bound": next_week['upper_bound'],
        "confidence_range": f"{next_week['lower_bound']} - {next_week['upper_bound']}",
        "model": "Facebook Prophet",
        "mae": metadata['mae']
    }

@router.get("/next-4-weeks")
async def get_next_4_weeks_forecast():
    """
    Get conversion forecast for next 4 weeks
    """
    return {
        "forecast": next_4_weeks_cache,
        "total_predicted": sum(w['predicted'] for w in next_4_weeks_cache),
        "model": "Facebook Prophet",
        "mae": metadata['mae'],
        "trained_on_weeks": metadata['total_weeks_trained']
    }

@router.get("/history")
async def get_forecast_history():
    """
    Get historical weekly conversion data
    """
    weekly_df = pd.read_csv("data/weekly_conversions.csv")
    weekly_df = weekly_df[weekly_df['week'] >= '2023-01-02']
    
    history = weekly_df.to_dict(orient='records')
    
    return {
        "total_weeks": len(history),
        "avg_weekly_conversions": round(
            weekly_df['converted_leads'].mean(), 1),
        "avg_weekly_leads": round(
            weekly_df['total_leads'].mean(), 1),
        "avg_conversion_rate": round(
            weekly_df['conversion_rate'].mean() * 100, 1),
        "history": history
    }

@router.get("/model-info")
async def get_model_info():
    """
    Get forecasting model information
    """
    return {
        "model_type": metadata['model_type'],
        "frequency": metadata['frequency'],
        "forecast_horizon": metadata['forecast_horizon'],
        "total_weeks_trained": metadata['total_weeks_trained'],
        "avg_weekly_conversions": round(
            metadata['avg_weekly_conversions'], 1),
        "mae": metadata['mae'],
        "interpretation": f"Model predictions are accurate within ±{metadata['mae']} leads per week"
    }
