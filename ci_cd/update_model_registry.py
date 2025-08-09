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
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

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
        
        # Debug: Log all available model names
        logger.info(f"Available models in results: {results_df['Model'].tolist()}")
        
        # Find best model metrics with multiple search strategies
        best_model_row = pd.DataFrame()
        
        # Strategy 1: Try exact match with the provided name
        best_model_row = results_df[results_df['Model'].str.lower() == best_model_name.lower()]
        
        if best_model_row.empty:
            # Strategy 2: Try with underscore to space conversion
            model_search_name = best_model_name.replace('_', ' ')
            best_model_row = results_df[results_df['Model'].str.contains(model_search_name, case=False, na=False)]
            logger.info(f"Trying search with: {model_search_name}")
        
        if best_model_row.empty:
            # Strategy 3: Try with title case conversion
            model_search_name = best_model_name.replace('_', ' ').title()
            best_model_row = results_df[results_df['Model'].str.contains(model_search_name, case=False, na=False)]
            logger.info(f"Trying search with: {model_search_name}")
        
        if best_model_row.empty:
            # Strategy 4: Try searching for key parts (e.g., "random" and "forest")
            if 'random' in best_model_name.lower() and 'forest' in best_model_name.lower():
                best_model_row = results_df[
                    results_df['Model'].str.contains('random', case=False, na=False) & 
                    results_df['Model'].str.contains('forest', case=False, na=False)
                ]
                logger.info(f"Trying search for Random Forest variants")
        
        if best_model_row.empty:
            # Fallback: use the first row (should be the best based on how results are saved)
            best_metrics = results_df.iloc[0]
            logger.warning(f"Could not find exact match for {best_model_name}, using first row with model: {best_metrics.get('Model', 'Unknown')}")
        else:
            best_metrics = best_model_row.iloc[0]
            logger.info(f"Found matching model: {best_metrics.get('Model', 'Unknown')}")
        
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
        
        # Register model using the correct Hopsworks API
        # Use the model registry to create and upload the model
        try:
            # Load the trained model
            import joblib
            model_object = joblib.load(actual_model_path)
            
            # Create model in Hopsworks Model Registry using the correct API
            # For Hopsworks 4.x, we need to use the model registry directly
            registered_model = mr.create_model(
                name=model_name,
                description=f"AQI prediction model for {horizon} horizon using {best_model_name}"
            )
            
            # Save the model object to the created model
            registered_model.save(model_object, model_schema=None)
            
            # Update model metrics separately if the API supports it
            try:
                registered_model.set_metrics({
                    "rmse": float(best_metrics.get('RMSE', best_metrics.get('rmse', 0))),
                    "mae": float(best_metrics.get('MAE', best_metrics.get('mae', 0))), 
                    "r2_score": float(best_metrics.get('R2', best_metrics.get('r2', 0)))
                })
            except Exception as metrics_error:
                logger.warning(f"Could not set metrics: {metrics_error}")
                
        except Exception as main_error:
            logger.warning(f"Standard approach failed: {main_error}")
            # Try alternative approach - direct file upload
            try:
                # Create model without the model object first
                registered_model = mr.create_model(
                    name=model_name,
                    description=f"AQI prediction model for {horizon} horizon using {best_model_name}"
                )
                
                # Upload the model file directly
                registered_model.save(str(actual_model_path))
                
            except Exception as fallback_error:
                logger.error(f"All model registration approaches failed: {fallback_error}")
                return False
        
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
