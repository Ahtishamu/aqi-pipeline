#!/usr/bin/env python3
"""
Abstract base class for all AQI prediction models
"""
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import joblib
import os

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all machine learning models."""
    
    def __init__(self, name: str, target_col: str = None):
        """
        Initialize the base model.
        
        Args:
            name (str): Name of the model
            target_col (str, optional): Target column name. Defaults to None.
        """
        self.name = name
        self.target_col = target_col
        self.model = None
        self.feature_importances_ = None
        self.feature_names_ = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Train the model on the given data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target values
            
        Returns:
            bool: True if training successful, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predictions or None if failed
        """
        pass
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances if available.
        
        Returns:
            np.array or None: Feature importance scores
        """
        return self.feature_importances_
    
    def get_feature_names(self) -> Optional[list]:
        """
        Get feature names used during training.
        
        Returns:
            list or None: Feature names
        """
        return self.feature_names_
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_trained:
                logger.error(f"Cannot save untrained model: {self.name}")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model and metadata
            model_data = {
                'model': self.model,
                'name': self.name,
                'target_col': self.target_col,
                'feature_names': self.feature_names_,
                'feature_importances': self.feature_importances_,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model {self.name} saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {self.name}: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.name = model_data['name']
            self.target_col = model_data['target_col']
            self.feature_names_ = model_data['feature_names']
            self.feature_importances_ = model_data['feature_importances']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model {self.name} loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {str(e)}")
            return False
    
    def validate_input(self, X: pd.DataFrame) -> bool:
        """
        Validate input data format and features.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(X, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame")
            return False
        
        if X.empty:
            logger.error("Input DataFrame is empty")
            return False
        
        if self.feature_names_ is not None:
            missing_features = set(self.feature_names_) - set(X.columns)
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
                return False
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            dict: Model information
        """
        return {
            'name': self.name,
            'target_col': self.target_col,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names_) if self.feature_names_ else 0,
            'has_feature_importances': self.feature_importances_ is not None
        }
