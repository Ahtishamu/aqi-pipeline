#!/usr/bin/env python3
"""
Update Hopsworks Model Registry with trained models
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import joblib
import hopsworks
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_model_registry(horizon: str, model_dir: str, best_model_name: str):
    """
    Update Hopsworks Model Registry with the trained model.
    
    Args:
        horizon (str): Model horizon ('24h', '48h', '72h')
        model_dir (str): Directory containing model files
        best_model_name (str): Name of the best performing model
    """
    try:
        logger.info(f"📦 Updating model registry for {horizon}")
        
        # Connect to Hopsworks
        api_key = os.environ.get('HOPSWORKS_API_KEY')
        if not api_key:
            logger.error("HOPSWORKS_API_KEY not found")
            return False
        
        project = hopsworks.login(api_key_value=api_key)
        mr = project.get_model_registry()
        
        # Find the latest model files in directory
        model_dir_path = Path(model_dir)
        model_files = list(model_dir_path.glob("best_model_*.pkl"))
        results_files = list(model_dir_path.glob("model_results_*.csv"))
        
        if not model_files or not results_files:
            logger.error(f"Model files not found in {model_dir}")
            return False
        
        # Get latest files (most recent timestamp)
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        latest_results_file = max(results_files, key=lambda x: x.stat().st_mtime)
        
        # Load results to get metrics
        results_df = pd.read_csv(latest_results_file)
        
        # Find best model metrics more safely
        model_search_name = best_model_name.replace('_', ' ').title()  # Convert to title case
        best_model_row = results_df[results_df['Model'].str.contains(model_search_name, case=False, na=False)]
        
        if best_model_row.empty:
            # Fallback: use the first row (should be the best based on how results are saved)
            best_metrics = results_df.iloc[0]
            logger.warning(f"Could not find exact match for {model_search_name}, using first row")
        else:
            best_metrics = best_model_row.iloc[0]
        
        # Create model in registry
        model_name = f"aqi_prediction_{horizon}"
        model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        
        # Prepare model schema (simplified for your features)
        input_schema = [
            {"name": "aqi", "type": "double"},
            {"name": "pm25", "type": "double"}, 
            {"name": "pm10", "type": "double"},
            {"name": "o3", "type": "double"},
            {"name": "no2", "type": "double"},
            {"name": "so2", "type": "double"},
            {"name": "co", "type": "double"},
            {"name": "no", "type": "double"},
            {"name": "nh3", "type": "double"},
            {"name": "hour", "type": "int"},
            {"name": "day", "type": "int"},
            {"name": "month", "type": "int"},
            {"name": "aqi_lag1", "type": "double"},
            {"name": "aqi_change_rate", "type": "double"},
        ]
        
        output_schema = [
            {"name": f"aqi_t_{horizon}", "type": "double"}
        ]
        
        # Model metadata
        model_metadata = {
            "model_type": best_model_name,
            "horizon": horizon,
            "training_date": str(pd.Timestamp.now()),
            "rmse": float(best_metrics['rmse']),
            "mae": float(best_metrics['mae']),
            "r2": float(best_metrics['r2']),
            "feature_count": len(input_schema),
            "training_framework": "scikit-learn"
        }
        
        # Register model
        model_registry = mr.python.create_model(
            name=model_name,
            version=model_version,
            description=f"AQI prediction model for {horizon} horizon using {best_model_name}",
            input_example=None,
            model_schema={
                "input_schema": input_schema,
                "output_schema": output_schema
            },
            metrics={
                "rmse": float(best_metrics['rmse']),
                "mae": float(best_metrics['mae']), 
                "r2_score": float(best_metrics['r2'])
            }
        )
        
        # Upload model file
        model_registry.save(str(latest_model_file))
        
        logger.info(f"✅ Model {model_name} v{model_version} registered successfully")
        logger.info(f"📊 Metrics - RMSE: {best_metrics['rmse']:.4f}, R²: {best_metrics['r2']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update model registry: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Update model registry')
    parser.add_argument('--horizon', choices=['24h', '48h', '72h'], 
                       required=True, help='Model horizon')
    
    args = parser.parse_args()
    
    success = update_model_registry(args.horizon)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
