import pandas as pd
import hopsworks

FEATURES_FILE = 'std_features_latest.csv'  # Your merged features file
UPDATED_FILE = 'std_features_latest_with_targets.csv'

# 1. Load the merged features file
df = pd.read_csv(FEATURES_FILE)

# 2. Add future AQI targets (assuming hourly data)
df['aqi_t+24h'] = df['aqi'].shift(-24)
df['aqi_t+48h'] = df['aqi'].shift(-48)
df['aqi_t+72h'] = df['aqi'].shift(-72)

# 3. Save the updated file
df.to_csv(UPDATED_FILE, index=False)
print(f"Updated features with targets saved to {UPDATED_FILE}")

# 4. Upload to Hopsworks feature store
project = hopsworks.login(api_key_value="z2OpqYQG6gbehlcD")
fs = project.get_feature_store()
feature_group = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    description="AQI and pollutant features for Karachi (with 3-day targets)",
    primary_key=["time"],
    online_enabled=True
)
feature_group.insert(df)
print("Features uploaded to Hopsworks feature store!") 