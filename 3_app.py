import streamlit as st
import hopsworks
import joblib
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 1. Page Config
st.set_page_config(page_title="Rawalpindi AQI Forecaster", page_icon="🌫️")
st.title("🌫️ Rawalpindi AQI Forecast (Next 3 Days)")
st.caption("Powered by the Best Performing Model from Hopsworks Registry")

load_dotenv()

# 2. Connect to Hopsworks & Get BEST Model
@st.cache_resource
def get_best_model():
    print("🚀 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    mr = project.get_model_registry()
    
    # Get the model object
    models = mr.get_models("aqi_model_rawalpindi")
    
    # Find the latest version (highest version number)
    best_model_meta = max(models, key=lambda m: m.version)
    print(f"✅ Loaded Model Version: {best_model_meta.version}")
    
    # Download
    model_dir = best_model_meta.download()
    model = joblib.load(model_dir + "/aqi_model.pkl")
    
    return model, best_model_meta.version

# 3. Fetch Future Forecast (Open-Meteo)
def get_forecast_data():
    today = datetime.now()
    end_date = today + timedelta(days=3)
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 33.60,  
        "longitude": 73.04, 
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone", 
        "start_date": today.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "timezone": "Asia/Karachi"
    }

    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame({
        "timestamp": data['hourly']['time'],
        "pm25": data['hourly']['pm2_5'],
        "pm10": data['hourly']['pm10'],
        "no2": data['hourly']['nitrogen_dioxide'],
        "ozone": data['hourly']['ozone'],
    })
    
    # Feature Engineering (Must match training!)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Daily Average for cleaner UI
    df['date_only'] = df['timestamp'].dt.date
    df_daily = df.groupby('date_only').mean().reset_index()
    
    return df_daily.head(3)

# --- MAIN APP ---
try:
    with st.spinner("Downloading Best Model & Forecast..."):
        model, version = get_best_model()
        forecast_df = get_forecast_data()
        
    st.success(f"Using Model Version {version} (Latest Champion)")

    if not forecast_df.empty:
        # Prepare features (Order matters!)
        feature_cols = ["pm25", "pm10", "no2", "ozone", "hour", "day_of_week", "month"]
        features = forecast_df[feature_cols]
        
        # Predict
        predictions = model.predict(features)
        
        # Display 3 Columns
        cols = st.columns(len(predictions))
        
        for idx, col in enumerate(cols):
            date_obj = forecast_df.iloc[idx]['date_only']
            date_str = date_obj.strftime("%A, %d %b")
            aqi_val = predictions[idx]
            
            with col:
                st.subheader(date_str)
                st.metric("Predicted AQI", f"{aqi_val:.0f}")
                
                if aqi_val < 50:
                    st.success("Good 🟢")
                elif aqi_val < 100:
                    st.warning("Moderate 🟡")
                elif aqi_val < 150:
                    st.warning("Unhealthy (Sens.) 🟠")
                elif aqi_val < 200:
                    st.error("Unhealthy 🔴")
                else:
                    st.error("Very Unhealthy 🟣")

        with st.expander("Show Technical Data"):
            st.write("Raw Forecast Inputs:")
            st.dataframe(forecast_df)
            
    else:
        st.error("Could not fetch forecast data.")

except Exception as e:
    st.error(f"Error: {e}")