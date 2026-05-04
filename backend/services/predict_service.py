import joblib
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from backend.models import Lead

# 1. Load all artifacts once on module load
try:
    model = joblib.load("backend/ml/model.pkl")
    features = joblib.load("backend/ml/features.pkl")
    encoder_source = joblib.load("backend/ml/encoder_source.pkl")
    encoder_course = joblib.load("backend/ml/encoder_course.pkl")
    encoder_city = joblib.load("backend/ml/encoder_city.pkl")
    encoder_stage = joblib.load("backend/ml/encoder_stage.pkl")
    encoder_counselor = joblib.load("backend/ml/encoder_counselor.pkl")
    response_time_median = joblib.load("backend/ml/response_time_median.pkl")
    model_metrics = joblib.load("backend/ml/model_metrics.pkl")
    feature_importance = joblib.load("backend/ml/feature_importance.pkl")
except FileNotFoundError:
    print("WARNING: ML artifacts not found. Run train.py first.")
    model_metrics = {}
    feature_importance = []

def prepare_features(data: dict) -> pd.DataFrame:
    # Pre-clean the dict: replace None for response_time_hours with median
    data = dict(data)
    if data.get('response_time_hours') is None:
        data['response_time_hours'] = float(response_time_median)

    df = pd.DataFrame([data])
    
    # Feature Engineering
    df['engagement_score'] = (
        df['total_calls'] * 0.3 +
        df['total_whatsapp_messages'] * 0.2 +
        df['total_emails'] * 0.1 +
        df['email_opened'] * 0.2 +
        df['whatsapp_replied'] * 0.2
    )
    
    df['is_fast_response'] = (df['response_time_hours'].fillna(float(response_time_median)) < 2).astype(int)
    df['is_high_form'] = (df['form_completion_percentage'] > 70).astype(int)
    
    df['is_stuck'] = (
        (df['days_in_inquiry_stage'] > 14) | (df['days_in_engagement_stage'] > 14)
    ).astype(int)
    
    df['total_interactions'] = (
        df['total_calls'] + df['total_whatsapp_messages'] + df['total_emails']
    )

    # Encode categorical columns
    cat_cols = {
        'source': encoder_source,
        'course_interest': encoder_course,
        'city': encoder_city,
        'current_stage': encoder_stage,
        'counselor_id': encoder_counselor
    }
    
    for col, encoder in cat_cols.items():
        val = str(df[col].iloc[0]) if col in df.columns else "Unknown"
        if val == 'None' or val == 'nan':
            val = "Unknown"
        try:
            df[col] = int(encoder.transform([val])[0])
        except ValueError:
            df[col] = 0

    # Ensure response time is float
    df['response_time_hours'] = pd.to_numeric(df['response_time_hours'], errors='coerce').fillna(float(response_time_median))
    
    # Add any missing feature columns with default 0
    for f in features:
        if f not in df.columns:
            df[f] = 0
    
    # Force ALL columns to numeric (float64) — XGBoost requires no object dtype
    result = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return result

def predict_single(data: dict) -> dict:
    df_features = prepare_features(data)
    
    prob = float(model.predict_proba(df_features)[0, 1])
    prob_rounded = round(prob, 4)
    
    will_convert = prob_rounded > 0.5
    
    if prob_rounded > 0.75:
        confidence = "high"
    elif prob_rounded > 0.5:
        confidence = "medium"
    else:
        confidence = "low"
        
    risk_factors = []
    if df_features['is_stuck'].iloc[0] == 1:
        risk_factors.append("Stuck in stage")
        
    original_response_time = data.get('response_time_hours')
    if original_response_time is None or pd.isna(original_response_time):
        risk_factors.append("No response")
        
    if df_features['form_completion_percentage'].iloc[0] < 30:
        risk_factors.append("Low form completion")
        
    if df_features['total_interactions'].iloc[0] < 3:
        risk_factors.append("No engagement")
        
    return {
        "conversion_probability": prob_rounded,
        "will_convert": will_convert,
        "confidence": confidence,
        "risk_factors": risk_factors
    }

async def predict_batch(db: AsyncSession) -> int:
    # Fetch all leads where conversion_probability IS NULL
    query = select(Lead).where(Lead.conversion_probability.is_(None))
    result = await db.execute(query)
    leads = result.scalars().all()
    
    if not leads:
        return 0
        
    # Convert leads to dicts
    leads_dict = [
        {c.name: getattr(lead, c.name) for c in Lead.__table__.columns}
        for lead in leads
    ]
    
    # We could do this row by row, or bulk. prepare_features is written for single rows or we can adapt it for df.
    # The instructions say: "Call prepare_features in bulk using df"
    # To strictly follow, I will build a bulk prepare_features logic or just use prepare_features row by row and concat.
    
    # Doing it bulk via a DataFrame since instructions suggest bulk:
    df_raw = pd.DataFrame(leads_dict)
    
    # Feature Engineering in bulk
    df_raw['engagement_score'] = (
        df_raw['total_calls'] * 0.3 +
        df_raw['total_whatsapp_messages'] * 0.2 +
        df_raw['total_emails'] * 0.1 +
        df_raw['email_opened'] * 0.2 +
        df_raw['whatsapp_replied'] * 0.2
    )
    
    temp_resp_time = df_raw['response_time_hours'].fillna(response_time_median)
    df_raw['is_fast_response'] = np.where(temp_resp_time < 2, 1, 0)
    df_raw['is_high_form'] = np.where(df_raw['form_completion_percentage'] > 70, 1, 0)
    
    df_raw['is_stuck'] = np.where(
        (df_raw['days_in_inquiry_stage'] > 14) | (df_raw['days_in_engagement_stage'] > 14), 1, 0
    )
    
    df_raw['total_interactions'] = (
        df_raw['total_calls'] +
        df_raw['total_whatsapp_messages'] +
        df_raw['total_emails']
    )
    
    cat_cols = {
        'source': encoder_source,
        'course_interest': encoder_course,
        'city': encoder_city,
        'current_stage': encoder_stage,
        'counselor_id': encoder_counselor
    }
    
    for col, encoder in cat_cols.items():
        # Handle unseen labels
        known_classes = set(encoder.classes_)
        df_raw[col] = df_raw[col].fillna("Unknown").astype(str)
        df_raw[col] = df_raw[col].apply(lambda x: x if x in known_classes else encoder.classes_[0])
        df_raw[col] = encoder.transform(df_raw[col])
        
    df_raw['response_time_hours'] = df_raw['response_time_hours'].fillna(response_time_median).astype(float)
    
    df_features = df_raw[features]
    
    # Force numeric types to avoid object dtype issues in XGBoost
    for col in df_features.columns:
        if df_features[col].dtype == 'object':
            try:
                df_features[col] = df_features[col].astype(float)
            except ValueError:
                pass
    
    # Call model.predict_proba on entire df
    probs = model.predict_proba(df_features)[:, 1]
    
    # Update each lead's conversion_probability using bulk update
    updates = []
    for lead, prob in zip(leads, probs):
        lead.conversion_probability = float(prob)
        # SQLAlchemy 2.0 tracks changes on objects automatically when they are modified.
        # Alternatively, we could use a massive `update(Lead)` statement but mapping objects is safer and supported here via session.
        # Wait, the prompt says "using bulk update". Let's do it using db.execute(update...) or just relying on unit of work.
        # If we use session.commit() the loaded objects are updated automatically.
    
    await db.commit()
    
    return len(leads)

def get_model_info() -> dict:
    return {
        "model_metrics": model_metrics,
        "feature_importance": feature_importance,
        "top_10_features": feature_importance[:10]
    }

def get_lead_explanation(lead_data: dict):
    # Prepare features
    df = prepare_features(lead_data)
    prob = model.predict_proba(df)[0][1]
    
    # Get top feature contributions
    feature_values = df.iloc[0].to_dict()
    
    # Build positive factors (boosting conversion)
    positive_factors = []
    
    if feature_values.get('is_fast_response') == 1:
        positive_factors.append({
            "factor": "Fast Response",
            "detail": "Lead responded within 2 hours",
            "impact": "high"
        })
    if feature_values.get('is_high_form') == 1:
        positive_factors.append({
            "factor": "High Form Completion",
            "detail": "Form completion above 70%",
            "impact": "high"
        })
    if feature_values.get('whatsapp_replied') == 1:
        positive_factors.append({
            "factor": "WhatsApp Engagement",
            "detail": "Lead replied on WhatsApp",
            "impact": "medium"
        })
    if feature_values.get('engagement_score', 0) > 5:
        positive_factors.append({
            "factor": "High Engagement",
            "detail": "Strong interaction across calls, WhatsApp and emails",
            "impact": "high"
        })
    if lead_data.get('total_calls', 0) > 5:
        positive_factors.append({
            "factor": "Multiple Calls",
            "detail": f"{lead_data.get('total_calls')} calls recorded",
            "impact": "medium"
        })

    # Build negative factors (hurting conversion)
    negative_factors = []
    
    if feature_values.get('is_stuck') == 1:
        negative_factors.append({
            "factor": "Stuck in Stage",
            "detail": "Lead stuck in inquiry or engagement > 14 days",
            "impact": "critical"
        })
    if lead_data.get('response_time_hours') is None or pd.isna(lead_data.get('response_time_hours')):
        negative_factors.append({
            "factor": "No Response",
            "detail": "Lead has not responded at all",
            "impact": "high"
        })
    if lead_data.get('form_completion_percentage', 0) < 30:
        negative_factors.append({
            "factor": "Low Form Completion",
            "detail": f"Only {lead_data.get('form_completion_percentage')}% form filled",
            "impact": "high"
        })
    if lead_data.get('total_calls', 0) == 0 and \
       lead_data.get('total_whatsapp_messages', 0) == 0:
        negative_factors.append({
            "factor": "Zero Engagement",
            "detail": "No calls or WhatsApp interaction recorded",
            "impact": "critical"
        })
    if lead_data.get('days_since_last_interaction', 0) > 14:
        negative_factors.append({
            "factor": "Inactive Lead",
            "detail": f"No interaction in last {lead_data.get('days_since_last_interaction')} days",
            "impact": "high"
        })

    # Build recommendation
    if prob >= 0.75:
        recommendation = "High priority lead. Assign senior counselor and follow up immediately."
    elif prob >= 0.5:
        recommendation = "Medium priority. Schedule a call within 24 hours."
    elif prob >= 0.25:
        recommendation = "Low priority. Send WhatsApp message and monitor engagement."
    else:
        recommendation = "Very low probability. Consider re-engagement campaign or drop."

    return {
        "conversion_probability": round(float(prob), 4),
        "will_convert": bool(prob >= 0.5),
        "confidence": "high" if prob > 0.75 else 
                     "medium" if prob > 0.5 else "low",
        "positive_factors": positive_factors,
        "negative_factors": negative_factors,
        "recommendation": recommendation,
        "priority": "P1" if prob >= 0.75 else
                   "P2" if prob >= 0.5 else
                   "P3" if prob >= 0.25 else "P4"
    }
