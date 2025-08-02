import pandas as pd
import requests
import numpy as np
import hopsworks
from datetime import datetime
import os

API_KEY = os.environ.get('OWM_API_KEY')
HOPSWORKS_API_KEY = os.environ.get('HOPSWORKS_API_KEY', 'vOGm8AlIgFUieggQ.GwCjZj1Utg3D5WP3qZS2vWOhVchbVHO8IbLHqDcMhMmmuadvn3dE2TDCkbgpnGUS')
LAT = 24.8607
LON = 67.0011
BASE_URL = 'http://api.openweathermap.org/data/2.5/air_pollution'
STANDARD_COLS = ['time', 'aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'no', 'nh3']

def fetch_owm_realtime():
    params = {
        'lat': LAT,
        'lon': LON,
        'appid': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        return None
    data = response.json()
    if 'list' not in data or not data['list']:
        print("No data found in response.")
        return None
    return data['list'][0]

def standardize_row(data):
    dt = datetime.utcfromtimestamp(data['dt'])
    main = data['main']
    components = data['components']
    row = {
        'time': dt,
        'aqi': main.get('aqi'),
        'pm25': components.get('pm2_5'),
        'pm10': components.get('pm10'),
        'o3': components.get('o3'),
        'no2': components.get('no2'),
        'so2': components.get('so2'),
        'co': components.get('co'),
        'no': components.get('no'),
        'nh3': components.get('nh3'),
    }
    return row

def get_latest_features_from_hopsworks():
    """Get the latest features from Hopsworks Feature Store"""
    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        feature_group = fs.get_feature_group("aqi_features", version=2)
        
        # Get latest data (last 100 records for lag feature calculation)
        df = feature_group.read()
        # Sort by time and get last 100 records
        df = df.sort_values('time').tail(100)
        return df
    except Exception as e:
        print(f"Error reading from Hopsworks: {e}")
        return pd.DataFrame(columns=STANDARD_COLS)

def add_to_hopsworks(new_df):
    """Add new features to Hopsworks Feature Store"""
    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        feature_group = fs.get_feature_group("aqi_features", version=2)
        
        feature_group.insert(new_df, write_options={"wait_for_job": False})
        print("✅ New features added to Hopsworks Feature Store")
        return True
    except Exception as e:
        print(f"❌ Error adding to Hopsworks: {e}")
        return False

def main():
    print("🔄 Hourly AQI Feature Update - Using Hopsworks")
    print("=" * 45)
    
    # Get existing features from Hopsworks
    print("📊 Getting latest features from Hopsworks...")
    df = get_latest_features_from_hopsworks()
    
    # Fetch new real-time data
    print("🌍 Fetching new AQI data from OpenWeatherMap...")
    data = fetch_owm_realtime()
    if not data:
        print("❌ No new data fetched.")
        return
    
    # Standardize new row
    new_row = standardize_row(data)
    print(f"📥 New data: AQI={new_row['aqi']}, PM2.5={new_row['pm25']}, Time={new_row['time']}")
    
    # Append new row to existing data
    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Feature engineering
    new_df['time'] = pd.to_datetime(new_df['time'], utc=True)  # Handle timezone consistently
    new_df['hour'] = new_df['time'].dt.hour.astype('int64')
    new_df['day'] = new_df['time'].dt.day.astype('int64')
    new_df['month'] = new_df['time'].dt.month.astype('int64')
    new_df['aqi_lag1'] = new_df['aqi'].shift(1)
    new_df['aqi_change_rate'] = new_df['aqi'] - new_df['aqi'].shift(1)
    
    # Add future AQI targets (assuming hourly data)
    new_df['aqi_t_24h'] = new_df['aqi'].shift(-24)
    new_df['aqi_t_48h'] = new_df['aqi'].shift(-48)
    new_df['aqi_t_72h'] = new_df['aqi'].shift(-72)
    
    # Get only the latest row with engineered features
    latest_row = new_df.tail(1)
    
    # Add to Hopsworks Feature Store
    print("📤 Adding new features to Hopsworks...")
    success = add_to_hopsworks(latest_row)
    
    if success:
        print("🎉 Hourly update completed successfully!")
    else:
        print("❌ Update failed!")

if __name__ == '__main__':
    main()