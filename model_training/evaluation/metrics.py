#!/usr/bin/env python3
"""
Evaluation metrics for AQI prediction models
"""
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics for model evaluation.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        
    Returns:
        dict: Dictionary with metrics (rmse, mae, r2)
    """
    try:
        # Convert to numpy arrays and flatten
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Remove NaN values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            logger.error("No valid data points after removing NaN/inf values")
            return {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)}")
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Handle edge cases
        if np.isnan(rmse) or np.isinf(rmse):
            rmse = float('inf')
        if np.isnan(mae) or np.isinf(mae):
            mae = float('inf')
        if np.isnan(r2) or np.isinf(r2):
            r2 = -float('inf')
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}

def calculate_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate additional regression metrics.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        
    Returns:
        dict: Dictionary with additional metrics
    """
    try:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Remove NaN values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {}
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
        # Mean Bias Error (MBE)
        mbe = np.mean(y_pred - y_true)
        
        # Normalized RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        nrmse = rmse / (np.max(y_true) - np.min(y_true)) if np.max(y_true) != np.min(y_true) else 0
        
        return {
            'mape': float(mape),
            'mbe': float(mbe),
            'nrmse': float(nrmse)
        }
        
    except Exception as e:
        logger.error(f"Error calculating additional metrics: {str(e)}")
        return {}

def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Pretty print model metrics.
    
    Args:
        metrics (dict): Dictionary of metrics
        model_name (str): Name of the model
    """
    print(f"\n📊 {model_name} Performance Metrics:")
    print("=" * 40)
    
    if 'rmse' in metrics:
        print(f"🎯 RMSE: {metrics['rmse']:.4f}")
    if 'mae' in metrics:
        print(f"📏 MAE:  {metrics['mae']:.4f}")
    if 'r2' in metrics:
        print(f"📈 R²:   {metrics['r2']:.4f}")
    if 'mape' in metrics:
        print(f"📊 MAPE: {metrics['mape']:.2f}%")
    if 'mbe' in metrics:
        print(f"⚖️  MBE:  {metrics['mbe']:.4f}")
    if 'nrmse' in metrics:
        print(f"📐 NRMSE: {metrics['nrmse']:.4f}")

def compare_models(model_results: Dict[str, Dict[str, float]], 
                  primary_metric: str = 'rmse') -> str:
    """
    Compare multiple models and select the best one.
    
    Args:
        model_results (dict): Dictionary of model names and their metrics
        primary_metric (str): Primary metric for comparison
        
    Returns:
        str: Name of the best model
    """
    if not model_results:
        return None
    
    valid_models = {name: metrics for name, metrics in model_results.items() 
                   if primary_metric in metrics and not np.isinf(metrics[primary_metric])}
    
    if not valid_models:
        logger.warning("No valid models found for comparison")
        return None
    
    if primary_metric in ['rmse', 'mae', 'mape', 'nrmse']:
        # Lower is better
        best_model = min(valid_models, key=lambda k: valid_models[k][primary_metric])
    elif primary_metric == 'r2':
        # Higher is better
        best_model = max(valid_models, key=lambda k: valid_models[k][primary_metric])
    else:
        logger.warning(f"Unknown metric: {primary_metric}. Using RMSE.")
        best_model = min(valid_models, key=lambda k: valid_models[k].get('rmse', float('inf')))
    
    return best_model

def create_metrics_dataframe(model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a DataFrame of model comparison results.
    
    Args:
        model_results (dict): Dictionary of model names and their metrics
        
    Returns:
        pd.DataFrame: DataFrame with model comparison
    """
    if not model_results:
        return pd.DataFrame()
    
    df = pd.DataFrame(model_results).T
    df.index.name = 'Model'
    
    # Sort by RMSE (lower is better)
    if 'rmse' in df.columns:
        df = df.sort_values('rmse')
    
    return df.reset_index()
