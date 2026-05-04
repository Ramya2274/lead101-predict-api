import pandas as pd
import os
import json
import joblib
from prophet import Prophet

def main():
    data_path = "data/lead101_realtime_5000.csv"
    if not os.path.exists(data_path):
        data_path = "lead101_realtime_5000.csv"
        
    df = pd.read_csv(data_path)
    
    df['created_date'] = pd.to_datetime(df['created_date'])
    df['week'] = df['created_date'].dt.to_period('W').apply(lambda r: r.start_time)
    
    weekly_df = df.groupby('week').agg(
        total_leads=('lead_id', 'count'),
        converted_leads=('converted', 'sum')
    ).reset_index()
    
    weekly_df['conversion_rate'] = weekly_df['converted_leads'] / weekly_df['total_leads']
    
    print(weekly_df.head(20))
    print(f"Total weeks: {len(weekly_df)}")
    print(f"Avg weekly conversions: {weekly_df['converted_leads'].mean():.1f}")
    print(f"Min: {weekly_df['converted_leads'].min()}")
    print(f"Max: {weekly_df['converted_leads'].max()}")
    
    os.makedirs("data", exist_ok=True)
    weekly_df.to_csv("data/weekly_conversions.csv", index=False)
    print("Saved to data/weekly_conversions.csv")

    # Prophet requires columns named 'ds' and 'y'
    prophet_df = weekly_df[['week', 'converted_leads']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Remove first row (partial week 2022-12-26 with only 3 leads)
    prophet_df = prophet_df[prophet_df['ds'] >= '2023-01-02'].reset_index(drop=True)
    
    print(f"Training on {len(prophet_df)} weeks of data")
    print(prophet_df.tail(5))
    
    # Train Prophet model
    model = Prophet(
        weekly_seasonality=False,
        daily_seasonality=False,
        yearly_seasonality=True,
        seasonality_mode='additive',
        changepoint_prior_scale=0.1
    )
    model.fit(prophet_df)
    
    # Make future dataframe for next 4 weeks
    future = model.make_future_dataframe(periods=4, freq='W')
    forecast = model.predict(future)
    
    # Get next 4 weeks predictions
    next_4_weeks = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(4)
    next_4_weeks.columns = ['week', 'predicted', 'lower_bound', 'upper_bound']
    
    # Round predictions (can't have decimal leads)
    next_4_weeks['predicted'] = next_4_weeks['predicted'].round().astype(int)
    next_4_weeks['lower_bound'] = next_4_weeks['lower_bound'].round().astype(int)
    next_4_weeks['upper_bound'] = next_4_weeks['upper_bound'].round().astype(int)
    
    print("\nNext 4 weeks forecast:")
    print(next_4_weeks)
    
    # Calculate model accuracy on historical data
    historical_forecast = forecast[['ds', 'yhat']].head(len(prophet_df))
    historical_forecast.loc[:, 'actual'] = prophet_df['y'].values
    historical_forecast.loc[:, 'error'] = abs(
        historical_forecast['actual'] - historical_forecast['yhat']
    )
    mae = historical_forecast['error'].mean()
    print(f"\nModel MAE: {mae:.2f} leads per week")
    
    # Save model
    os.makedirs("backend/ml", exist_ok=True)
    joblib.dump(model, "backend/ml/forecast_model.pkl")
    print("Forecast model saved!")
    
    # Save forecast metadata
    metadata = {
        "total_weeks_trained": len(prophet_df),
        "avg_weekly_conversions": float(prophet_df['y'].mean()),
        "mae": round(mae, 2),
        "model_type": "Facebook Prophet",
        "frequency": "Weekly",
        "forecast_horizon": "4 weeks"
    }
    joblib.dump(metadata, "backend/ml/forecast_metadata.pkl")
    print("Metadata saved!")
    
    # Save next 4 weeks to json for quick access
    next_4_weeks['week'] = next_4_weeks['week'].astype(str)
    forecast_output = next_4_weeks.to_dict(orient='records')
    with open("backend/ml/next_4_weeks.json", "w") as f:
        json.dump(forecast_output, f, indent=2)
    print("Next 4 weeks forecast saved to JSON!")

if __name__ == "__main__":
    main()
