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
    
    # --- FEATURE ENGINEERING ---
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Create Time Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['date_str'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    
    # 2. Convert to Hopsworks Event Time (Milliseconds)
    df['date'] = df['timestamp'].apply(lambda x: int(x.timestamp() * 1000))
    
    # 3. Add City Name (Primary Key)
    df['city'] = CITY_NAME
    
    # 4. Clean up
    df = df.dropna()
    
    print(f"✅ Downloaded {len(df)} rows of data.")
    return df

def main():
    # A. Login to Hopsworks
    print("🔑 Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    # B. Check for existing Feature Group to decide Backfill vs Update
    try:
        aqi_fg = fs.get_feature_group(name="aqi_data_rawalpindi", version=1)
        print("💾 Feature group found. Fetching daily update...")
        df = fetch_data(backfill=False)
        
        # Small daily updates can be inserted normally
        aqi_fg.insert(df)
        
    except:
        print("💾 Feature group not found. Performing initial chunked backfill...")
        df = fetch_data(backfill=True)
        
        # Create the Feature Group
        aqi_fg = fs.get_or_create_feature_group(
            name="aqi_data_rawalpindi",
            version=1,
            primary_key=["city", "date"], 
            event_time="date",
            description="Hourly Air Quality Data for Rawalpindi"
        )

        # CHUNKING LOGIC: Upload 500 rows at a time to prevent server-side timeouts
        chunk_size = 500
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            print(f"📦 Uploading rows {i} to {i+len(chunk)}...")
            aqi_fg.insert(chunk, write_options={"wait_for_job": False})

    print("🎉 Success! Data is now in Hopsworks.")

# 3. EXECUTION BLOCK (Crucial for the script to run)
if __name__ == "__main__":
    main()