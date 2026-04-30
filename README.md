# Lead101 Predict API

A FastAPI-based REST API for Education CRM Lead Conversion Prediction using XGBoost ML model.

---

## 🚀 Live API
Base URL: https://lead101-predict-api.onrender.com
Swagger Docs: https://lead101-predict-api.onrender.com/docs

> Note: Free tier - first request may take 50 seconds to wake up after inactivity.

---

## 🔐 Authentication
All endpoints (except health check) require an API key in the request header:
`X-API-Key: lead101-secret-key-2024`

---

## 📦 Tech Stack
- FastAPI + Uvicorn
- PostgreSQL + SQLAlchemy (async)
- XGBoost ML Model
- Pandas
- Render (deployment)

---

## 📊 ML Model Performance
- Accuracy: 86.9%
- ROC-AUC: 0.92
- Training data: 5000 leads
- Algorithm: XGBoost Classifier

### Top Features by Importance
1. is_stuck (stuck in stage > 14 days)
2. current_stage
3. is_high_form (form completion > 70%)
4. source
5. engagement_score

---

## 🛠 API Endpoints

### Health
`GET /`
Returns API status

### Leads
`POST /api/leads/upload`
Upload CSV file of leads

`GET /api/leads`
Get paginated leads with filters
Params: `page`, `page_size`, `source`, `city`, `course_interest`, `converted`, `current_stage`

`GET /api/leads/{lead_id}`
Get single lead by ID

`GET /api/leads/stats/summary`
Get overall lead statistics

`GET /api/leads/top-leads`
Get top unconverted leads by ML score
Params: `limit` (default 20)

`GET /api/leads/at-risk`
Get high potential leads with no recent contact
Params: `limit` (default 20), `days_inactive` (default 7)

`GET /api/leads/stuck`
Get leads stuck in early stages
Params: `limit` (default 50)

`GET /api/leads/export/csv`
Download filtered leads as CSV
Params: `source`, `city`, `course_interest`, `converted`, `start_date`, `end_date`

`GET /api/leads/export/excel`
Download filtered leads as Excel
Params: same as CSV export

### Analytics
`GET /api/analytics/overview`
Get all analytics in single call
Params: `start_date`, `end_date`

`GET /api/analytics/source`
Conversion rates by lead source

`GET /api/analytics/city`
Lead distribution by city

`GET /api/analytics/course`
Performance by course interest

`GET /api/analytics/funnel`
Stage-wise lead funnel

`GET /api/analytics/counselors`
Counselor performance metrics

`GET /api/analytics/monthly`
Monthly lead trends

`GET /api/analytics/stuck-leads`
Count and details of stuck leads

### Predict
`POST /api/predict`
Predict conversion for a single lead
Returns: probability, confidence, risk_factors

`POST /api/predict/batch`
Score all unscored leads in DB
Returns: total_scored count

`GET /api/predict/{lead_id}`
Get prediction for existing lead by ID

---

## 📥 Quick Start

### 1. Upload Data
`POST /api/leads/upload`
Upload `lead101_realtime_5000.csv`

### 2. Score All Leads
`POST /api/predict/batch`

### 3. Get Top Leads
`GET /api/leads/top-leads?limit=20`

### 4. Get Analytics
`GET /api/analytics/overview`

---

## 🧪 Testing with Postman
1. Import `postman_collection.json` into Postman
2. Set `base_url` to your API URL
3. Set `api_key` to your API key
4. Run requests folder by folder

---

## 📁 Project Structure
```
lead101-predict-api/
  backend/
    main.py         - FastAPI app entry point
    database.py     - DB connection setup
    models.py       - SQLAlchemy models
    schemas.py      - Pydantic schemas
    auth.py         - API key authentication
    routers/
      leads.py      - Lead endpoints
      analytics.py  - Analytics endpoints
      predict.py    - Prediction endpoints
    services/
      lead_service.py      - Lead business logic
      analytics_service.py - Analytics logic
      predict_service.py   - ML prediction logic
    ml/
      train.py      - Model training script
      model.pkl     - Trained XGBoost model
  data/
    lead101_realtime_5000.csv - Training dataset
  postman_collection.json - Postman collection
  requirements.txt
  README.md
```

---

## 🔄 Local Development

1. Clone repo:
```bash
git clone https://github.com/YOUR_USERNAME/lead101-predict-api.git
```

2. Install dependencies:
```bash
pip install -r backend/requirements.txt
```

3. Set environment variables in `backend/.env`:
```
DATABASE_URL=postgresql+asyncpg://user:pass@host/db
API_KEY=lead101-secret-key-2024
```

4. Train model:
```bash
python -m backend.ml.train
```

5. Run server:
```bash
uvicorn backend.main:app --reload --port 8000
```

6. Open docs:
http://localhost:8000/docs

---

## 👥 Team
Built by Ramya - Tectra Technologies  
Lead101 CRM - Education Lead Conversion Prediction
