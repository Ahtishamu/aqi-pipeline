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
        
        # Create model in registry using the model registry directly
        model_name = f"aqi_prediction_{horizon}"
        
        # Load the saved model
        model_dir_path = Path(model_dir)
        model_path_pattern = f"best_model_aqi_t_{horizon}_*.pkl"
        model_files = list(model_dir_path.glob(model_path_pattern))
        
        if not model_files:
            logger.error(f"No model file found matching pattern: {model_path_pattern} in {model_dir}")
            return False
            
        actual_model_path = model_files[0]
        
        # Register model using the model registry create_model method
        # Use the correct method name for Hopsworks 4.x
        try:
            # Try the sklearn model approach
            import joblib
            model_object = joblib.load(actual_model_path)
            
            # Create the model using the Hopsworks model registry
            registered_model = mr.sklearn.create_model(
                name=model_name,
                model=model_object,
                description=f"AQI prediction model for {horizon} horizon using {best_model_name}",
                metrics={
                    "rmse": float(best_metrics.get('RMSE', best_metrics.get('rmse', 0))),
                    "mae": float(best_metrics.get('MAE', best_metrics.get('mae', 0))), 
                    "r2_score": float(best_metrics.get('R2', best_metrics.get('r2', 0)))
                }
            )
        except Exception as sklearn_error:
            logger.warning(f"sklearn approach failed: {sklearn_error}")
            # Fallback to python model approach
            registered_model = mr.python.create_model(
                name=model_name,
                model_path=str(actual_model_path),
                description=f"AQI prediction model for {horizon} horizon using {best_model_name}",
                metrics={
                    "rmse": float(best_metrics.get('RMSE', best_metrics.get('rmse', 0))),
                    "mae": float(best_metrics.get('MAE', best_metrics.get('mae', 0))), 
                    "r2_score": float(best_metrics.get('R2', best_metrics.get('r2', 0)))
                }
            )
        
        logger.info(f"✅ Model {model_name} registered successfully")
        logger.info(f"📊 Metrics - RMSE: {best_metrics.get('RMSE', best_metrics.get('rmse', 0)):.4f}, R²: {best_metrics.get('R2', best_metrics.get('r2', 0)):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update model registry: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Update model registry')
    parser.add_argument('--horizon', choices=['24h', '48h', '72h'], 
                       required=True, help='Model horizon')
    parser.add_argument('--model-dir', type=str,
                       required=True, help='Directory containing model files')
    parser.add_argument('--best-model', type=str,
                       required=True, help='Name of the best model')
    
    args = parser.parse_args()
    
    success = update_model_registry(args.horizon, args.model_dir, args.best_model)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
