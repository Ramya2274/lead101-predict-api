import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Load original data + add engineered features
# ─────────────────────────────────────────────
df = pd.read_csv("lead101_realtime_5000.csv")

# Add engineered features
df["engagement_score"] = (
    df["total_calls"] + df["total_whatsapp_messages"] + df["total_emails"]
) / 3

df["is_fast_responder"] = (df["response_time_hours"] < 5).astype(int)

stage_map = {
    "inquiry": 1, "engagement": 2, "application": 3,
    "verification": 4, "admission": 5, "payment": 6, "enrollment": 7
}
df["stage_numeric"] = df["current_stage"].map(stage_map)

df["total_stage_days"] = (
    df["days_in_inquiry_stage"] +
    df["days_in_engagement_stage"] +
    df["days_in_application_stage"] +
    df["days_in_verification_stage"]
)

source_map = {
    "referral": 5, "walk_in": 4, "google_ads": 3, "website": 2, "facebook": 1
}
df["source_quality_score"] = df["source"].map(source_map)

df["has_high_form_completion"] = (df["form_completion_percentage"] >= 70).astype(int)

# ─────────────────────────────────────────────
# 1. Define Features (X) and Target (y)
# ─────────────────────────────────────────────
drop_cols = ["lead_id", "created_date", "converted", "days_to_convert"]

# Also drop raw categoricals that have been replaced by numeric mappings
drop_also = ["source", "course_interest", "city", "current_stage", "counselor_id"]

X = df.drop(columns=drop_cols + drop_also)
y = df["converted"]

print("=" * 55)
print("ML DATA SPLIT")
print("=" * 55)
print(f"\nFeature matrix (X) shape : {X.shape}")
print(f"Target vector (y) shape  : {y.shape}")
print(f"\nClass distribution in full dataset:")
print(y.value_counts().to_string())
print(f"  Conversion rate: {y.mean()*100:.2f}%")

# ─────────────────────────────────────────────
# 2. Train/Test Split — 80/20, stratified
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ─────────────────────────────────────────────
# 3. Print Shapes & Verify Stratification
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("SPLIT RESULTS")
print("=" * 55)
print(f"\n  Train set  — X: {X_train.shape}, y: {y_train.shape}")
print(f"  Test set   — X: {X_test.shape}, y: {y_test.shape}")

print(f"\n  Train conversion rate : {y_train.mean()*100:.2f}%")
print(f"  Test  conversion rate : {y_test.mean()*100:.2f}%")
print(f"  (stratify check — should both be ~38.04%)")

# ─────────────────────────────────────────────
# 4. Save train_data.csv and test_data.csv
#    (include all original + engineered columns + target)
# ─────────────────────────────────────────────
train_df = X_train.copy()
train_df["converted"] = y_train.values
train_df.to_csv("train_data.csv", index=False)

test_df = X_test.copy()
test_df["converted"] = y_test.values
test_df.to_csv("test_data.csv", index=False)

print("\n" + "=" * 55)
print("SAVED FILES")
print("=" * 55)
print(f"  train_data.csv — {train_df.shape[0]} rows x {train_df.shape[1]} cols")
print(f"  test_data.csv  — {test_df.shape[0]} rows x {test_df.shape[1]} cols")

print("\n  Feature columns used for training:")
for i, col in enumerate(X_train.columns, 1):
    print(f"    {i:>2}. {col}")
