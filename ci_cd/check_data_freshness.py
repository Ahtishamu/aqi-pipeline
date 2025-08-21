#!/usr/bin/env python3
"""
Check data freshness to determine if model retraining is needed
"""
import os
import sys
import pandas as pd
import hopsworks
from datetime import datetime, timezone
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _normalize_model_created(created_raw) -> Optional[pd.Timestamp]:
    """Convert various possible 'created' representations from a Hopsworks model object
    into a timezone-aware pandas Timestamp (UTC). Returns None if cannot parse.
    Handles:
      - datetime objects
      - pandas Timestamp
      - ISO 8601 strings
      - Integer / float epoch (s / ms / us / ns)
    """
    if created_raw is None:
        return None
    try:
        # Already a pandas Timestamp
        if isinstance(created_raw, pd.Timestamp):
            if created_raw.tz is None:
                return created_raw.tz_localize('UTC')
            return created_raw.tz_convert('UTC')
        # Python datetime
        if isinstance(created_raw, datetime):
            if created_raw.tzinfo is None:
                created_raw = created_raw.replace(tzinfo=timezone.utc)
            return pd.Timestamp(created_raw).tz_convert('UTC')
        # Numeric epoch
        if isinstance(created_raw, (int, float)):
            val = float(created_raw)
            # Heuristic for unit
            # seconds ~1e9, ms ~1e12, us ~1e15, ns ~1e18
            if val > 1e17:  # ns
                ts = pd.to_datetime(val, unit='ns', utc=True)
            elif val > 1e14:  # us
                ts = pd.to_datetime(val, unit='us', utc=True)
            elif val > 1e11:  # ms
                ts = pd.to_datetime(val, unit='ms', utc=True)
            else:  # seconds
                ts = pd.to_datetime(val, unit='s', utc=True)
            return ts
        # String (try ISO first)
        if isinstance(created_raw, str):
            try:
                ts = pd.to_datetime(created_raw, utc=True, errors='raise')
                if ts.tz is None:
                    ts = ts.tz_localize('UTC')
                else:
                    ts = ts.tz_convert('UTC')
                return ts
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Could not normalize model creation time: {e}")
    return None

def check_data_freshness(min_new_records: int = 23, horizon: str = None):  # ~1 day of data (24 hours - 1 for buffer)
    """
    Check if we have enough fresh data that warrants retraining.
    
    Args:
        min_new_records: Minimum new records needed to trigger retraining
        horizon: Specific horizon to check (24h, 48h, 72h), or None to check overall freshness
    
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
        
        # Ensure time column is datetime (timezone-aware UTC)
        if 'time' not in df.columns:
            logger.error("'time' column missing in feature group data")
            return False
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
        df = df.dropna(subset=['time'])
        if df.empty:
            logger.warning("All 'time' values are NaT after conversion")
            return False
        
        latest_data_ts = df['time'].max()
        logger.info(f"Latest data timestamp: {latest_data_ts}")
        
        # Fetch existing models; if none, trigger initial training
        try:
            # Check which model to use based on horizon
            if horizon == "24h":
                model_name = "aqi_prediction_24h"
            elif horizon == "48h":
                model_name = "aqi_prediction_48h"
            elif horizon == "72h":
                model_name = "aqi_prediction_72h"
            else:
                # For overall freshness check, use the most recently trained model across all horizons
                model_names = ["aqi_prediction_24h", "aqi_prediction_48h", "aqi_prediction_72h"]
                all_model_times = []
                
                for model_name in model_names:
                    try:
                        models = mr.get_models(model_name) or []
                        for m in models:
                            created_attr = getattr(m, 'created', None)
                            norm = _normalize_model_created(created_attr)
                            if norm is not None:
                                all_model_times.append((norm, m, model_name))
                    except Exception as e:
                        logger.debug(f"Could not retrieve {model_name} models: {e}")
                        continue
                
                if not all_model_times:
                    logger.info("No existing models found for any horizon - initial training needed")
                    return True
                
                # Sort by creation time, get the most recent across all horizons
                all_model_times.sort(key=lambda x: x[0], reverse=True)
                latest_model_time, latest_model, latest_model_name = all_model_times[0]
                logger.info(f"Latest model across all horizons: {latest_model_name} created at {latest_model_time}")
                
            if horizon:
                # Check specific horizon
                models = mr.get_models(model_name) or []
                if not models:
                    logger.info(f"No existing {model_name} models found - training needed")
                    return True
                # Sort by created descending using normalized timestamps
                model_times = []
                for m in models:
                    created_attr = getattr(m, 'created', None)
                    norm = _normalize_model_created(created_attr)
                    if norm is None:
                        logger.debug(f"Skipping model without parsable 'created' value: {created_attr}")
                        continue
                    model_times.append((norm, m))
                if not model_times:
                    logger.info(f"No {model_name} models with valid creation timestamps - training needed")
                    return True
                model_times.sort(key=lambda x: x[0], reverse=True)
                latest_model_time, latest_model = model_times[0]
                logger.info(f"Latest {model_name} model creation timestamp (UTC): {latest_model_time}")
                
        except Exception as e:
            logger.info(f"Could not retrieve existing models ({e}) - initial training needed")
            return True
        
        # Compare timestamps
        # Count new records strictly after latest model creation
        new_records_mask = df['time'] > latest_model_time
        new_record_count = int(new_records_mask.sum())
        logger.info(f"New records since last training: {new_record_count}")
        logger.info(f"Minimum records needed: {min_new_records}")
        
        should_train = new_record_count >= min_new_records
        if should_train:
            logger.info("Sufficient new data for retraining")
        else:
            logger.info(f"⏳ Not enough new data ({new_record_count}/{min_new_records})")
        
        return should_train
        
    except Exception as e:
        logger.error(f"Error checking data freshness: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check data freshness for model retraining")
    parser.add_argument("--horizon", choices=["24h", "48h", "72h"], 
                       help="Specific horizon to check (default: overall freshness)")
    args = parser.parse_args()
    
    should_train = check_data_freshness(horizon=args.horizon)
    
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
