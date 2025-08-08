#!/usr/bin/env python3
"""
Export current AQI features from Hopsworks Feature Store to CSV
"""
import os
import pandas as pd
import hopsworks
from datetime import datetime

def main():
    print("📊 Exporting AQI Features from Hopsworks")
    print("=" * 50)
    
    try:
        # Connect to Hopsworks
        print("🔗 Connecting to Hopsworks...")
        
        # Try to get API key from environment, if not available prompt user
        api_key = os.environ.get('HOPSWORKS_API_KEY')
        if not api_key:
            print("❌ HOPSWORKS_API_KEY environment variable not set")
            print("💡 Please set it first: set HOPSWORKS_API_KEY=your_key_here")
            return False
            
        project = hopsworks.login(
            api_key_value=api_key,
            project='ahtisham'
        )
        
        # Get feature store
        fs = project.get_feature_store()
        
        # Get the feature group
        fg = fs.get_feature_group("aqi_features", version=2)
        
        print("📊 Reading all data from feature store...")
        # Read all data
        df = fg.read()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aqi_features_export_{timestamp}.csv"
        
        # Export to CSV
        df.to_csv(filename, index=False)
        
        print(f"✅ Data exported to: {filename}")
        print(f"📈 Total records: {len(df)}")
        print(f"📅 Date range: {df['time'].min()} to {df['time'].max()}")
        
        # Show target completeness
        total_records = len(df)
        aqi_24h_filled = df['aqi_t_24h'].notna().sum()
        aqi_48h_filled = df['aqi_t_48h'].notna().sum()
        aqi_72h_filled = df['aqi_t_72h'].notna().sum()
        
        print(f"\n📊 Target completeness:")
        print(f"   aqi_t_24h: {aqi_24h_filled}/{total_records} ({aqi_24h_filled/total_records*100:.1f}%)")
        print(f"   aqi_t_48h: {aqi_48h_filled}/{total_records} ({aqi_48h_filled/total_records*100:.1f}%)")
        print(f"   aqi_t_72h: {aqi_72h_filled}/{total_records} ({aqi_72h_filled/total_records*100:.1f}%)")
        
        # Show recent records (last 10)
        print(f"\n📋 Most recent 10 records:")
        recent_df = df.sort_values('time').tail(10)[['time', 'aqi', 'aqi_t_24h', 'aqi_t_48h', 'aqi_t_72h']]
        for _, row in recent_df.iterrows():
            print(f"   {row['time']}: AQI={row['aqi']}, t+24h={row['aqi_t_24h']}, t+48h={row['aqi_t_48h']}, t+72h={row['aqi_t_72h']}")
        
        # Show records where targets were recently updated
        print(f"\n🔍 Records with all targets filled (showing last 10):")
        complete_targets = df[(df['aqi_t_24h'].notna()) & (df['aqi_t_48h'].notna()) & (df['aqi_t_72h'].notna())]
        if len(complete_targets) > 0:
            recent_complete = complete_targets.sort_values('time').tail(10)[['time', 'aqi', 'aqi_t_24h', 'aqi_t_48h', 'aqi_t_72h']]
            for _, row in recent_complete.iterrows():
                print(f"   {row['time']}: AQI={row['aqi']}, t+24h={row['aqi_t_24h']}, t+48h={row['aqi_t_48h']}, t+72h={row['aqi_t_72h']}")
        else:
            print("   No records with all targets filled yet")
        
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main()