import pandas as pd
import hopsworks
from datetime import datetime

def upload_existing_features():
    """Upload all existing features from CSV to Hopsworks Feature Store"""
    
    print("🚀 Uploading existing features to Hopsworks Feature Store")
    print("=" * 55)
    
    # Load existing features
    print("📊 Loading existing features...")
    df = pd.read_csv('std_features_latest.csv')
    print(f"✅ Loaded {len(df)} records")
    
    # Convert time to proper datetime format
    df['time'] = pd.to_datetime(df['time'])
    print(f"📅 Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Add future AQI targets (assuming hourly data)
    print("\n🎯 Adding prediction targets...")
    df['aqi_t_24h'] = df['aqi'].shift(-24)
    df['aqi_t_48h'] = df['aqi'].shift(-48)
    df['aqi_t_72h'] = df['aqi'].shift(-72)
    print(f"✅ Added target features: aqi_t_24h, aqi_t_48h, aqi_t_72h")
    print(f"📋 Updated columns: {list(df.columns)}")
    
    # Connect to Hopsworks
    print("\n🔗 Connecting to Hopsworks...")
    try:
        project = hopsworks.login(api_key_value="vOGm8AlIgFUieggQ.GwCjZj1Utg3D5WP3qZS2vWOhVchbVHO8IbLHqDcMhMmmuadvn3dE2TDCkbgpnGUS")
        print(f"✅ Connected to project: {project.name}")
        
        fs = project.get_feature_store()
        print("✅ Feature store ready")
        
        # Create feature group
        print("\n📋 Creating feature group...")
        feature_group = fs.get_or_create_feature_group(
            name="aqi_features",
            version=2,  # New version with target features
            description="AQI and pollutant features for Karachi with time-series data and prediction targets",
            primary_key=["time"],
            event_time="time",
            online_enabled=False  # Disable online mode to avoid timestamp primary key issue
        )
        print("✅ Feature group 'aqi_features' version 2 created/ready")
        
        # Upload data
        print(f"\n📤 Uploading {len(df)} records to feature store...")
        feature_group.insert(df, write_options={"wait_for_job": True, "use_spark": False})
        
        print("🎉 SUCCESS! All existing features uploaded to Hopsworks!")
        print(f"\n📊 Summary:")
        print(f"   • Records uploaded: {len(df)}")
        print(f"   • Features: {len(df.columns)} columns")
        print(f"   • Date range: {df['time'].min()} to {df['time'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = upload_existing_features()
    if success:
        print("\n✅ Upload completed successfully")
    else:
        print("\n❌ Upload failed")
