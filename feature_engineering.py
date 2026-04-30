import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
df = pd.read_csv("lead101_realtime_5000.csv")
print("=" * 55)
print("TASK 3: FEATURE ENGINEERING")
print("=" * 55)
print(f"Original shape: {df.shape}")

# ─────────────────────────────────────────────
# 1. Handle Missing Values
# ─────────────────────────────────────────────
print("\n--- Step 1: Handle Missing Values ---")

# response_time_hours: fill with median (robust to outliers)
median_rt = df["response_time_hours"].median()
df["response_time_hours"] = df["response_time_hours"].fillna(median_rt)
print(f"  response_time_hours  -> filled {1318} nulls with median ({median_rt:.2f})")

# days_to_convert: only meaningful for converted leads — fill with 0 for non-converted
df["days_to_convert"] = df["days_to_convert"].fillna(0)
print(f"  days_to_convert      -> filled {3098} nulls with 0 (non-converted leads)")

print(f"  Remaining nulls: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 2. Parse Dates & Extract Date Features
# ─────────────────────────────────────────────
print("\n--- Step 2: Date Features ---")
df["created_date"] = pd.to_datetime(df["created_date"])
df["lead_month"]   = df["created_date"].dt.month
df["lead_quarter"] = df["created_date"].dt.quarter
df["lead_year"]    = df["created_date"].dt.year
df["lead_dayofweek"] = df["created_date"].dt.dayofweek   # 0=Mon, 6=Sun
print("  Extracted: lead_month, lead_quarter, lead_year, lead_dayofweek")

# ─────────────────────────────────────────────
# 3. Encode Categorical Variables
# ─────────────────────────────────────────────
print("\n--- Step 3: Encode Categoricals ---")

# One-hot encode: source, course_interest, city, current_stage
cat_cols = ["source", "course_interest", "city", "current_stage"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
print(f"  One-hot encoded: {cat_cols}")

# Label encode counselor_id (ordinal-style)
df["counselor_id"] = df["counselor_id"].str.extract(r"(\d+)").astype(int)
print(f"  Label encoded : counselor_id (extracted numeric ID)")

# ─────────────────────────────────────────────
# 4. Create New Engineered Features
# ─────────────────────────────────────────────
print("\n--- Step 4: Create New Features ---")

# Total engagement touchpoints
df["total_touchpoints"] = (
    df["total_calls"] +
    df["total_whatsapp_messages"] +
    df["total_emails"]
)
print("  + total_touchpoints = calls + whatsapp + emails")

# Response engagement ratio (how fast + how many replies)
df["engagement_ratio"] = (
    df["whatsapp_replied"] + df["email_opened"]
) / (df["total_touchpoints"] + 1)  # +1 to avoid div-by-zero
print("  + engagement_ratio = (whatsapp_replied + email_opened) / total_touchpoints")

# Total days spent in pipeline
df["total_pipeline_days"] = (
    df["days_in_inquiry_stage"] +
    df["days_in_engagement_stage"] +
    df["days_in_application_stage"] +
    df["days_in_verification_stage"]
)
print("  + total_pipeline_days = sum of all stage days")

# Is fast responder (response within 24 hrs)
df["is_fast_responder"] = (df["response_time_hours"] <= 24).astype(int)
print("  + is_fast_responder = 1 if response_time_hours <= 24")

# Has gone quiet (no interaction in 30+ days)
df["is_dormant"] = (df["days_since_last_interaction"] >= 30).astype(int)
print("  + is_dormant = 1 if days_since_last_interaction >= 30")

# Form completion tier
df["form_tier"] = pd.cut(
    df["form_completion_percentage"],
    bins=[0, 33, 66, 100],
    labels=[0, 1, 2],
    include_lowest=True
).astype(int)
print("  + form_tier = 0 (low), 1 (mid), 2 (high) based on form_completion_percentage")

# ─────────────────────────────────────────────
# 5. Drop Non-Feature Columns
# ─────────────────────────────────────────────
print("\n--- Step 5: Drop Non-Feature Columns ---")
drop_cols = ["lead_id", "created_date"]
df.drop(columns=drop_cols, inplace=True)
print(f"  Dropped: {drop_cols}")

# ─────────────────────────────────────────────
# 6. Final Dataset Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL ENGINEERED DATASET SUMMARY")
print("=" * 55)
print(f"  Shape          : {df.shape}")
print(f"  Numeric cols   : {df.select_dtypes(include='number').shape[1]}")
print(f"  Remaining nulls: {df.isnull().sum().sum()}")
print(f"\n  All columns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"    {i:>3}. {col}")

# ─────────────────────────────────────────────
# 7. Save ML-Ready Dataset
# ─────────────────────────────────────────────
output_path = "lead101_ml_ready.csv"
df.to_csv(output_path, index=False)
print(f"\n  Saved to: {output_path}")

# ─────────────────────────────────────────────
# 8. Feature Correlation with Target
# ─────────────────────────────────────────────
print("\n--- Top 15 Features Correlated with 'converted' ---")
numeric_df = df.select_dtypes(include="number")
correlations = numeric_df.corr()["converted"].drop("converted").abs().sort_values(ascending=False)
print(correlations.head(15).to_string())
