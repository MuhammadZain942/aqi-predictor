import os
import hopsworks
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 1. Load your secret API key from the .env file
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

# 2. Configuration for RAWALPINDI, Pakistan
LATITUDE = 33.60
LONGITUDE = 73.04
CITY_NAME = "Rawalpindi"

def fetch_data(backfill=False):
    """
    Fetches weather & AQI data from Open-Meteo.
    If backfill=True, it gets the last 365 days (for training).
    If backfill=False, it gets the last 3 days (for daily updates).
    """
    print(f"📡 Connecting to Open-Meteo API for {CITY_NAME}...")
    
    today = datetime.now()
    if backfill:
        # Go back 1 year for training data
        start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
    else:
        # Go back 3 days just to ensure we have recent context
        start_date = (today - timedelta(days=3)).strftime('%Y-%m-%d')
        
    end_date = today.strftime('%Y-%m-%d')

    # Open-Meteo Free API Endpoint
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,european_aqi",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Karachi"
    }

    response = requests.get(url, params=params)
    data = response.json()
    
    # Check if API call was successful
    if "hourly" not in data:
        raise Exception("API Error: Could not fetch data. Check your internet connection.")

    # Convert to DataFrame
    df = pd.DataFrame({
        "timestamp": data['hourly']['time'],
        "pm25": data['hourly']['pm2_5'],
        "pm10": data['hourly']['pm10'],
        "no2": data['hourly']['nitrogen_dioxide'],
        "ozone": data['hourly']['ozone'],
        "aqi": data['hourly']['european_aqi']
    })
    
    # --- FEATURE ENGINEERING (The "Secret Sauce") ---
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Create Time Features (Helping the model learn cycles)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['date_str'] = df['timestamp'].dt.strftime('%Y-%m-%d') # For primary key
    
    # 2. Convert to Hopsworks Event Time (Milliseconds)
    df['date'] = df['timestamp'].apply(lambda x: int(x.timestamp() * 1000))
    
    # 3. Add City Name (Primary Key)
    df['city'] = CITY_NAME
    
    # 4. Clean up (Drop NaN values if any)
    df = df.dropna()
    
    print(f"✅ Downloaded {len(df)} rows of data.")
    return df

def main():
    # A. Login to Hopsworks
    print("🔑 Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    # B. Fetch Historical Data (Backfill)
    df = fetch_data(backfill=True)

    # C. Create or Get the Feature Group
    # A Feature Group is like a database table in the cloud
    print("💾 Uploading to Feature Store...")
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_data_rawalpindi",
        version=1,
        primary_key=["city", "date"], 
        event_time="date",
        description="Hourly Air Quality Data for Rawalpindi"
    )

    # D. Save the data
    aqi_fg.insert(df)
    print("🎉 Success! Data is now in Hopsworks.")

if __name__ == "__main__":
    main()