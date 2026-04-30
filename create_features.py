import pandas as pd

# Load original data
df = pd.read_csv("lead101_realtime_5000.csv")

# 1. engagement_score = (total_calls + total_whatsapp_messages + total_emails) / 3
df["engagement_score"] = (df["total_calls"] + df["total_whatsapp_messages"] + df["total_emails"]) / 3

# 2. is_fast_responder = 1 if response_time_hours < 5 else 0 (handle nulls as 0)
# (NaN < 5 is False, so astype(int) yields 0)
df["is_fast_responder"] = (df["response_time_hours"] < 5).astype(int)

# 3. stage_numeric = map current_stage to numbers
stage_map = {
    "inquiry": 1,
    "engagement": 2,
    "application": 3,
    "verification": 4,
    "admission": 5,
    "payment": 6,
    "enrollment": 7
}
df["stage_numeric"] = df["current_stage"].map(stage_map)

# 4. total_stage_days = sum of days in stages
df["total_stage_days"] = (
    df["days_in_inquiry_stage"] + 
    df["days_in_engagement_stage"] + 
    df["days_in_application_stage"] + 
    df["days_in_verification_stage"]
)

# 5. source_quality_score = map source
source_map = {
    "referral": 5,
    "walk_in": 4,
    "google_ads": 3,
    "website": 2,
    "facebook": 1
}
df["source_quality_score"] = df["source"].map(source_map)

# 6. has_high_form_completion = 1 if form_completion_percentage >= 70 else 0
df["has_high_form_completion"] = (df["form_completion_percentage"] >= 70).astype(int)

# Show first 10 rows with new features
new_features = [
    "engagement_score",
    "is_fast_responder",
    "stage_numeric",
    "total_stage_days",
    "source_quality_score",
    "has_high_form_completion"
]

print("--- First 10 Rows of New Features ---")
print(df[new_features].head(10).to_string())
