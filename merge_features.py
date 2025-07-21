import pandas as pd

# Replace with your actual backfill features file name
BACKFILL_FEATURES = 'std_features_backfill.csv'  # Your processed backfill features file
LIVE_FEATURES = 'std_features_latest.csv'        # Your live features file (to be updated in-place)

# Load both datasets
backfill_df = pd.read_csv(BACKFILL_FEATURES)
live_df = pd.read_csv(LIVE_FEATURES)

# Ensure 'time' is datetime for both
df_back = backfill_df.copy()
df_live = live_df.copy()
df_back['time'] = pd.to_datetime(df_back['time'])
df_live['time'] = pd.to_datetime(df_live['time'])

# Remove any overlap: keep only backfill rows strictly before the first live row
earliest_live_time = df_live['time'].min()
df_back = df_back[df_back['time'] < earliest_live_time]

# Concatenate and sort by time
merged_df = pd.concat([df_back, df_live], ignore_index=True)
merged_df = merged_df.sort_values('time').reset_index(drop=True)

# Overwrite the live features file
merged_df.to_csv(LIVE_FEATURES, index=False)
print(f"Backfill data merged into {LIVE_FEATURES}") 