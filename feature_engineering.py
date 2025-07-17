import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime

# Unified columns for ML pipeline
STANDARD_COLS = ['time', 'aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'no', 'nh3']

# Find the latest CSV file (either AQICN or OWM)
csv_files = sorted(glob.glob('*.csv'), key=os.path.getctime, reverse=True)
latest_file = csv_files[0]

def load_and_standardize(filename):
    df = pd.read_csv(filename)
    # Rename columns for OWM compatibility
    if 'pm2_5' in df.columns:
        df.rename(columns={'pm2_5': 'pm25'}, inplace=True)
    # Add missing columns as NaN
    for col in STANDARD_COLS:
        if col not in df.columns:
            df[col] = np.nan
    # Reorder columns
    df = df[STANDARD_COLS]
    return df

def add_time_features(df):
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    return df

def add_lag_features(df):
    df['aqi_lag1'] = df['aqi'].shift(1)
    return df

def add_derived_features(df):
    df['aqi_change_rate'] = df['aqi'] - df['aqi'].shift(1)
    return df

def main():
    df = load_and_standardize(latest_file)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_derived_features(df)
    out_file = f'std_features_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(out_file, index=False)
    print(f"Saved standardized features to {out_file}")

if __name__ == '__main__':
    main() 