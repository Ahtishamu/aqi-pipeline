import pandas as pd
import requests
import csv
import numpy as np
from datetime import datetime
import os

API_KEY = '04f944e99e3ef35da278e90e3d8550fc'
LAT = 24.8607
LON = 67.0011
BASE_URL = 'http://api.openweathermap.org/data/2.5/air_pollution'
STANDARD_COLS = ['time', 'aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'no', 'nh3']

FEATURES_FILE = 'std_features_latest.csv'  # Use a fixed file for ongoing updates

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

def main():
    # Load existing features or create new DataFrame
    if os.path.exists(FEATURES_FILE):
        df = pd.read_csv(FEATURES_FILE)
    else:
        df = pd.DataFrame(columns=STANDARD_COLS)
    # Fetch new real-time data
    data = fetch_owm_realtime()
    if not data:
        print("No new data fetched.")
        return
    # Standardize and append new row
    new_row = standardize_row(data)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    # Feature engineering
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['aqi_lag1'] = df['aqi'].shift(1)
    df['aqi_change_rate'] = df['aqi'] - df['aqi'].shift(1)
    # Save updated features
    df.to_csv(FEATURES_FILE, index=False)
    print(f"Updated features saved to {FEATURES_FILE}")

if __name__ == '__main__':
    main() 