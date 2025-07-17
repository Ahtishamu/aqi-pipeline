import requests
import csv
import numpy as np
from datetime import datetime

API_KEY = '04f944e99e3ef35da278e90e3d8550fc'
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

def save_standardized_csv(data, filename):
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
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=STANDARD_COLS)
        writer.writeheader()
        writer.writerow(row)
    print(f"Saved standardized data to {filename}")

def main():
    data = fetch_owm_realtime()
    if data:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_standardized_csv(data, f'owm_realtime_{timestamp}_std.csv')

if __name__ == '__main__':
    main() 