import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# ─────────────────────────────────────────────
# Load Train & Test Data
# ─────────────────────────────────────────────
train_df = pd.read_csv("train_data.csv")
test_df  = pd.read_csv("test_data.csv")

X_train = train_df.drop(columns=["converted"])
y_train = train_df["converted"]
X_test  = test_df.drop(columns=["converted"])
y_test  = test_df["converted"]

# Impute any remaining NaNs with column medians from train set
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test  = X_test.fillna(train_medians)

print("=" * 60)
print("TASK 5: MODEL TRAINING & EVALUATION")
print("=" * 60)
print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
print(f"  NaNs in train: {X_train.isnull().sum().sum()} | NaNs in test: {X_test.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# Scale features (needed for Logistic Regression)
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# Helper: Print Evaluation Metrics
# ─────────────────────────────────────────────
def evaluate_model(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted 0   Predicted 1")
    print(f"  Actual 0       {cm[0][0]:>8}      {cm[0][1]:>8}")
    print(f"  Actual 1       {cm[1][0]:>8}      {cm[1][1]:>8}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Not Converted", "Converted"]))
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": auc}

results = []

# ─────────────────────────────────────────────
# Model 1: Logistic Regression
# ─────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
results.append(evaluate_model("LOGISTIC REGRESSION", y_test, lr_pred, lr_prob))

# ─────────────────────────────────────────────
# Model 2: Random Forest
# ─────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
results.append(evaluate_model("RANDOM FOREST", y_test, rf_pred, rf_prob))

# ─────────────────────────────────────────────
# Model Comparison Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
summary_df = pd.DataFrame(results).set_index("Model")
print(summary_df.round(4).to_string())

# ─────────────────────────────────────────────
# Random Forest: Top Feature Importances
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TOP 10 FEATURE IMPORTANCES (Random Forest)")
print("=" * 60)
feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_imp = feat_imp.sort_values(ascending=False).head(10)
for feat, score in feat_imp.items():
    bar = "#" * int(score * 200)
    print(f"  {feat:<35} {score:.4f}  {bar}")
