import requests
import pandas as pd
from datetime import datetime, timedelta
import time

API_KEY = '04f944e99e3ef35da278e90e3d8550fc'
LAT = 24.8607
LON = 67.0011
N_DAYS = 60
BASE_URL = 'http://api.openweathermap.org/data/2.5/air_pollution/history'

# Calculate UNIX timestamps for the date range
end_time = int(datetime.now().timestamp())
start_time = int((datetime.now() - timedelta(days=N_DAYS)).timestamp())

params = {
    'lat': LAT,
    'lon': LON,
    'start': start_time,
    'end': end_time,
    'appid': API_KEY
}

def fetch_historical_data():
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    data = response.json()
    if 'list' not in data:
        print("No data found in response.")
        return None
    return data['list']

def main():
    print("Fetching historical AQI data from OpenWeatherMap...")
    records = fetch_historical_data()
    if not records:
        print("No records fetched.")
        return
    rows = []
    for rec in records:
        dt = datetime.utcfromtimestamp(rec['dt'])
        main = rec['main']
        components = rec['components']
        row = {
            'time': dt,
            'aqi': main.get('aqi'),
            'co': components.get('co'),
            'no': components.get('no'),
            'no2': components.get('no2'),
            'o3': components.get('o3'),
            'so2': components.get('so2'),
            'pm2_5': components.get('pm2_5'),
            'pm10': components.get('pm10'),
            'nh3': components.get('nh3'),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    out_file = f'owm_backfill_aqi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(out_file, index=False)
    print(f"Saved backfilled data to {out_file}")

if __name__ == '__main__':
    main() 