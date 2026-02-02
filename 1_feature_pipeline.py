import os
import hopsworks
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

LATITUDE = 33.60
LONGITUDE = 73.04
CITY_NAME = "Rawalpindi"

def fetch_data(backfill=False):
    print(f"📡 Connecting to Open-Meteo API for {CITY_NAME}...")
    today = datetime.now()
    # If backfill=True, get 365 days; else, just get 3 days for the hourly update
    start_date = (today - timedelta(days=365 if backfill else 3)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

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
    
    if "hourly" not in data:
        raise Exception("API Error: Could not fetch data.")

    df = pd.DataFrame({
        "timestamp": data['hourly']['time'],
        "pm25": data['hourly']['pm2_5'],
        "pm10": data['hourly']['pm10'],
        "no2": data['hourly']['nitrogen_dioxide'],
        "ozone": data['hourly']['ozone'],
        "aqi": data['hourly']['european_aqi']
    })
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['date_str'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df['date'] = df['timestamp'].apply(lambda x: int(x.timestamp() * 1000))
    df['city'] = CITY_NAME
    df = df.dropna()
    
    return df

def main():
    print("🔑 Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    # Logic to check if we need to backfill or just update
    try:
        # Check if the feature group already exists
        aqi_fg = fs.get_feature_group(name="aqi_data_rawalpindi", version=1)
        print("💾 Feature group found. Fetching daily update...")
        df = fetch_data(backfill=False)
    except:
        print("💾 Feature group not found. Performing initial backfill (1 year)...")
        df = fetch_data(backfill=True)
        aqi_fg = fs.get_or_create_feature_group(
            name="aqi_data_rawalpindi",
            version=1,
            primary_key=["city", "date"], 
            event_time="date",
            description="Hourly Air Quality Data for Rawalpindi"
        )

    aqi_fg.insert(df)
    print("🎉 Success! Data is now in Hopsworks.")

if __name__ == "__main__":
    main()