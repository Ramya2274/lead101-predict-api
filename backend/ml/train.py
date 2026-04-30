import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import os

def main():
    # 1. Load data
    data_path = "lead101_realtime_5000.csv"
    if not os.path.exists(data_path):
        data_path = "data/lead101_realtime_5000.csv"
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    conversion_rate = df['converted'].mean() * 100
    print(f"Conversion rate: {conversion_rate:.2f}%")

    # 2. Feature Engineering
    df['engagement_score'] = (
        df['total_calls'] * 0.3 +
        df['total_whatsapp_messages'] * 0.2 +
        df['total_emails'] * 0.1 +
        df['email_opened'] * 0.2 +
        df['whatsapp_replied'] * 0.2
    )
    
    temp_response_time = df['response_time_hours'].fillna(0)
    df['is_fast_response'] = np.where(temp_response_time < 2, 1, 0)
    
    df['is_high_form'] = np.where(df['form_completion_percentage'] > 70, 1, 0)
    
    df['is_stuck'] = np.where(
        (df['days_in_inquiry_stage'] > 14) | (df['days_in_engagement_stage'] > 14), 1, 0
    )
    
    df['total_interactions'] = (
        df['total_calls'] +
        df['total_whatsapp_messages'] +
        df['total_emails']
    )

    # 3. Encode categorical columns
    categorical_cols = ['source', 'course_interest', 'city', 'current_stage', 'counselor_id']
    encoders = {}
    
    os.makedirs("backend/ml", exist_ok=True)
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna("Unknown")
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
        # Save encoder
        if col == 'source':
            joblib.dump(le, "backend/ml/encoder_source.pkl")
        elif col == 'course_interest':
            joblib.dump(le, "backend/ml/encoder_course.pkl")
        elif col == 'city':
            joblib.dump(le, "backend/ml/encoder_city.pkl")
        elif col == 'current_stage':
            joblib.dump(le, "backend/ml/encoder_stage.pkl")
        elif col == 'counselor_id':
            joblib.dump(le, "backend/ml/encoder_counselor.pkl")

    # 4. Define feature columns (X) and Target (y)
    feature_columns = [
        'source', 'course_interest', 'city', 'total_calls',
        'total_whatsapp_messages', 'total_emails',
        'email_opened', 'whatsapp_replied',
        'form_completion_percentage', 'response_time_hours',
        'days_since_last_interaction', 'current_stage',
        'days_in_inquiry_stage', 'days_in_engagement_stage',
        'days_in_application_stage', 'days_in_verification_stage',
        'counselor_id', 'engagement_score', 'is_fast_response',
        'is_high_form', 'is_stuck', 'total_interactions'
    ]
    
    X = df[feature_columns].copy()
    y = df['converted']

    # 5. Fill nulls for response_time_hours with median
    response_time_median = X['response_time_hours'].median()
    X['response_time_hours'] = X['response_time_hours'].fillna(response_time_median)
    joblib.dump(response_time_median, "backend/ml/response_time_median.pkl")

    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Train XGBoost model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nTop 10 Feature Importances:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(10, len(indices))):
        print(f"{i+1}. {feature_columns[indices[i]]}: {importances[indices[i]]:.4f}")

    # 8. Save artifacts
    joblib.dump(model, "backend/ml/model.pkl")
    joblib.dump(feature_columns, "backend/ml/features.pkl")
    print("\nModel and artifacts saved successfully in backend/ml/")

if __name__ == "__main__":
    main()
