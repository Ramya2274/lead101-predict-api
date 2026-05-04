import pandas as pd
import numpy as np
import joblib
import json
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

import os

data_path = "data/lead101_realtime_5000.csv"
if not os.path.exists(data_path):
    data_path = "lead101_realtime_5000.csv"
df = pd.read_csv(data_path)
df['created_date'] = pd.to_datetime(df['created_date'])
df['week'] = df['created_date'].dt.to_period('W').apply(
    lambda r: r.start_time)

weekly_df = df.groupby('week').agg(
    total_leads=('lead_id', 'count'),
    converted_leads=('converted', 'sum')
).reset_index()
weekly_df['conversion_rate'] = (
    weekly_df['converted_leads'] / weekly_df['total_leads'])
weekly_df['week'] = weekly_df['week'].astype(str)

# Remove partial first week
weekly_df = weekly_df[
    weekly_df['week'] >= '2023-01-02'].reset_index(drop=True)

print(f"Training on {len(weekly_df)} weeks")

# Train ARIMA model
y = weekly_df['converted_leads'].values

model = ARIMA(y, order=(2, 1, 2))
fitted_model = model.fit()

# Forecast next 4 weeks
forecast_result = fitted_model.forecast(steps=4)
conf_int = fitted_model.get_forecast(steps=4).conf_int()

# Generate future week dates
last_week = pd.to_datetime(weekly_df['week'].iloc[-1])
future_weeks = [
    str((last_week + pd.Timedelta(weeks=i+1)).date())
    for i in range(4)
]

conf_int_array = np.array(conf_int)

# Build forecast output
next_4_weeks = []
for i in range(4):
    predicted = max(0, round(float(forecast_result[i])))
    lower = max(0, round(float(conf_int_array[i, 0])))
    upper = max(0, round(float(conf_int_array[i, 1])))
    next_4_weeks.append({
        "week": future_weeks[i],
        "predicted": predicted,
        "lower_bound": lower,
        "upper_bound": upper
    })

print("\nNext 4 weeks forecast:")
for w in next_4_weeks:
    print(w)

# Calculate MAE on historical data
fitted_values = fitted_model.fittedvalues
mae = mean_absolute_error(
    y[1:], fitted_values[1:])
print(f"\nModel MAE: {mae:.2f} leads per week")

# Save model
joblib.dump(fitted_model, "backend/ml/forecast_model.pkl")

# Save metadata
metadata = {
    "model_type": "ARIMA(2,1,2)",
    "frequency": "Weekly",
    "forecast_horizon": "4 weeks",
    "total_weeks_trained": len(weekly_df),
    "avg_weekly_conversions": round(float(y.mean()), 1),
    "mae": round(mae, 2)
}
joblib.dump(metadata, "backend/ml/forecast_metadata.pkl")

# Save next 4 weeks
with open("backend/ml/next_4_weeks.json", "w") as f:
    json.dump(next_4_weeks, f, indent=2)

# Save weekly data
weekly_df.to_csv("data/weekly_conversions.csv", index=False)

print("\nAll artifacts saved!")
