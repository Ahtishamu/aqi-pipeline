import pandas as pd
import hopsworks
from datetime import datetime, timedelta
import os

HOPSWORKS_API_KEY = os.environ.get('HOPSWORKS_API_KEY')

def update_daily_targets():
    """Daily automation to update target features for records from 3 days ago"""
    
    print("📅 Daily Target Feature Update - Backfill Mode")
    print("=" * 50)
    
    try:
        # Connect to Hopsworks
        print("🔗 Connecting to Hopsworks...")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        print(f"✅ Connected to project: {project.name}")
        
        fs = project.get_feature_store()
        feature_group = fs.get_feature_group("aqi_features", version=2)
        
        # Read all data
        print("📊 Reading data from feature store...")
        df = feature_group.read()
        df = df.sort_values('time').reset_index(drop=True)
        df['time'] = pd.to_datetime(df['time'])
        
        print(f"📈 Total records: {len(df)}")
        
        # Find records that need target updates (3+ days old with missing targets)
        cutoff_date = datetime.now().replace(tzinfo=None) - timedelta(days=3)
        print(f"🎯 Looking for records before: {cutoff_date}")
        
        # Find records that can have their targets filled
        records_to_update = []
        updates_made = 0
        
        for i, row in df.iterrows():
            # Convert time to timezone-naive for comparison
            record_time = pd.to_datetime(row['time'])
            if hasattr(record_time, 'tz_localize'):
                record_time = record_time.tz_localize(None)
            
            # Only process records older than 3 days
            if record_time < cutoff_date:
                updated = False
                
                # Check if aqi_t_24h can be filled
                if pd.isna(row['aqi_t_24h']):
                    target_24h_idx = i + 24
                    if target_24h_idx < len(df):
                        df.at[i, 'aqi_t_24h'] = df.iloc[target_24h_idx]['aqi']
                        updated = True
                
                # Check if aqi_t_48h can be filled
                if pd.isna(row['aqi_t_48h']):
                    target_48h_idx = i + 48
                    if target_48h_idx < len(df):
                        df.at[i, 'aqi_t_48h'] = df.iloc[target_48h_idx]['aqi']
                        updated = True
                
                # Check if aqi_t_72h can be filled
                if pd.isna(row['aqi_t_72h']):
                    target_72h_idx = i + 72
                    if target_72h_idx < len(df):
                        df.at[i, 'aqi_t_72h'] = df.iloc[target_72h_idx]['aqi']
                        updated = True
                
                if updated:
                    records_to_update.append(i)
                    updates_made += 1
        
        print(f"🔄 Found {updates_made} records that can be updated")
        
        if updates_made > 0:
            # Show examples of what was updated
            print(f"\n📋 Examples of updates made:")
            for i in records_to_update[:5]:  # Show first 5 examples
                record_time = df.iloc[i]['time']
                aqi_24h = df.iloc[i]['aqi_t_24h']
                print(f"   {record_time}: aqi_t_24h = {aqi_24h}")
            
            if len(records_to_update) > 5:
                print(f"   ... and {len(records_to_update) - 5} more records")
            
            # Create updated records DataFrame for insertion
            updated_records = df.iloc[records_to_update].copy()
            
            print(f"\n📤 Updating {len(updated_records)} records in feature store...")
            
            # Insert updated records (Hopsworks will handle upserts based on primary key)
            feature_group.insert(updated_records, write_options={"wait_for_job": False})
            
            print(f"✅ Successfully updated {updates_made} target features!")
            
            # Export updated records for verification
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"daily_target_updates_{timestamp}.csv"
            updated_records.to_csv(filename, index=False)
            print(f"💾 Updated records exported to: {filename}")
            
        else:
            print(f"ℹ️ No records found that need target updates today")
            print(f"💡 This is normal if all eligible records already have their targets filled")
        
        # Show current completeness stats
        total_records = len(df)
        complete_24h = df['aqi_t_24h'].notna().sum()
        complete_48h = df['aqi_t_48h'].notna().sum()
        complete_72h = df['aqi_t_72h'].notna().sum()
        
        print(f"\n📊 Current target completeness:")
        print(f"   aqi_t_24h: {complete_24h}/{total_records} ({complete_24h/total_records*100:.1f}%)")
        print(f"   aqi_t_48h: {complete_48h}/{total_records} ({complete_48h/total_records*100:.1f}%)")
        print(f"   aqi_t_72h: {complete_72h}/{total_records} ({complete_72h/total_records*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = update_daily_targets()
    if success:
        print(f"\n🎉 Daily target update completed!")
        print(f"⏰ This automation should run every day to keep targets current")
    else:
        print(f"\n❌ Daily update failed!")
