#!/usr/bin/env python3
"""
Check data freshness to determine if model retraining is needed
"""
import os
import sys
import pandas as pd
import hopsworks
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_freshness(min_new_records: int = 23):  # ~1 day of data (24 hours - 1 for buffer)
    """
    Check if we have enough fresh data that warrants retraining.
    
    Args:
        min_new_records: Minimum new records needed to trigger retraining
    
    Returns:
        bool: True if should train, False otherwise
    """
    try:
        # Connect to Hopsworks
        api_key = os.environ.get('HOPSWORKS_API_KEY')
        if not api_key:
            logger.error("HOPSWORKS_API_KEY not found")
            return False
        
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        
        # Get feature group
        fg = fs.get_feature_group("aqi_features", version=2)
        df = fg.read()
        
        if df.empty:
            logger.warning("No data found in feature store")
            return False
        
        # Check for existing models in registry
        try:
            models = mr.get_models("aqi_prediction_24h")
            if not models:
                logger.info("No existing models found - initial training needed")
                return True
            
            latest_model = models[0]  # Most recent model
            model_creation = latest_model.created
            logger.info(f"Latest model created: {model_creation}")
        except:
            logger.info("No existing models - initial training needed")
            return True
        
        # Check latest data timestamp
        df['time'] = pd.to_datetime(df['time'])
        latest_data = df['time'].max()
        
        # Count new records since last model training
        new_data = df[df['time'] > model_creation]
        new_record_count = len(new_data)
        
        logger.info(f"Latest data: {latest_data}")
        logger.info(f"New records since last training: {new_record_count}")
        logger.info(f"Minimum records needed: {min_new_records}")
        
        # Check if we have enough new data to warrant retraining
        should_train = new_record_count >= min_new_records
        
        if should_train:
            logger.info("✅ Sufficient new data for retraining")
        else:
            logger.info(f"⏳ Not enough new data ({new_record_count}/{min_new_records})")
        
        return should_train
        if len(recent_data) < 100:  # Need at least 100 recent data points
            logger.warning(f"Only {len(recent_data)} recent data points, need at least 100")
            return False
        
        # Check target completeness for the model horizons
        horizons = ['aqi_t_24h', 'aqi_t_48h', 'aqi_t_72h']
        for horizon in horizons:
            if horizon in df.columns:
                completeness = df[horizon].notna().mean()
                logger.info(f"{horizon} completeness: {completeness:.2%}")
                if completeness < 0.1:  # Need at least 10% complete targets
                    logger.warning(f"{horizon} has low completeness: {completeness:.2%}")
        
        logger.info("✅ Data freshness check passed - proceeding with training")
        return True
        
    except Exception as e:
        logger.error(f"Error checking data freshness: {e}")
        return False

def main():
    should_train = check_data_freshness()
    
    # Set GitHub Actions output
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"should_train={str(should_train).lower()}\n")
    
    # Also print for debugging
    print(f"should_train={str(should_train).lower()}")
    
    # Exit with appropriate code
    sys.exit(0 if should_train else 1)

if __name__ == '__main__':
    main()
