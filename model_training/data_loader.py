#!/usr/bin/env python3
"""
Data loader for fetching AQI features from Hopsworks Feature Store
"""
import os
import pandas as pd
import hopsworks
import logging
from typing import Tuple, Optional, List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AQIDataLoader:
    """Data loader for AQI features from Hopsworks Feature Store."""
    
    def __init__(self, project_name: str = 'ahtisham'):
        """
        Initialize the data loader.
        
        Args:
            project_name (str): Hopsworks project name
        """
        self.project_name = project_name
        self.project = None
        self.fs = None
        self.connected = False
    
    def connect(self) -> bool:
        """
        Connect to Hopsworks.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            api_key = os.environ.get('HOPSWORKS_API_KEY')
            if not api_key:
                logger.error("HOPSWORKS_API_KEY environment variable not set")
                return False
            
            logger.info("Connecting to Hopsworks...")
            self.project = hopsworks.login(
                api_key_value=api_key,
                project=self.project_name
            )
            
            self.fs = self.project.get_feature_store()
            self.connected = True
            logger.info(f"✅ Connected to project: {self.project.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {str(e)}")
            return False
    
    def load_training_data(self, target_col: str = 'aqi_t_24h', 
                          feature_group_name: str = 'aqi_features',
                          version: int = 2) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Load training data from Hopsworks Feature Store.
        
        Args:
            target_col (str): Target column name
            feature_group_name (str): Feature group name
            version (int): Feature group version
            
        Returns:
            tuple: (features DataFrame, target Series) or (None, None) if failed
        """
        if not self.connected:
            if not self.connect():
                return None, None
        
        try:
            logger.info(f"Loading data for target: {target_col}")
            
            # Get feature group
            fg = self.fs.get_feature_group(feature_group_name, version=version)
            
            # Read all data
            df = fg.read()
            logger.info(f"Loaded {len(df)} records from feature store")
            
            # Check if target column exists
            if target_col not in df.columns:
                logger.error(f"Target column '{target_col}' not found in data")
                return None, None
            
            # Remove records with missing target values
            initial_count = len(df)
            df = df.dropna(subset=[target_col])
            final_count = len(df)
            
            if final_count == 0:
                logger.error(f"No records with valid {target_col} values found")
                return None, None
            
            logger.info(f"Filtered to {final_count} records with valid targets "
                       f"(removed {initial_count - final_count} records)")
            
            # Separate features and target
            feature_columns = [col for col in df.columns if col not in ['time', target_col]]
            X = df[feature_columns].copy()
            y = df[target_col].copy()
            
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Target shape: {y.shape}")
            logger.info(f"Feature columns: {list(X.columns)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            return None, None
    
    def preprocess_features(self, X: pd.DataFrame, y: pd.Series, 
                           handle_missing: str = 'mean',
                           scale_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess features for training.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            handle_missing (str): Strategy for handling missing values ('mean', 'median', 'drop')
            scale_features (bool): Whether to scale features
            
        Returns:
            tuple: (preprocessed features, target)
        """
        try:
            logger.info("Preprocessing features...")
            
            # Convert all columns to numeric, coercing errors to NaN
            X = X.apply(pd.to_numeric, errors='coerce')
            y = pd.to_numeric(y, errors='coerce')
            
            # Remove completely empty columns
            empty_cols = X.columns[X.isna().all()]
            if len(empty_cols) > 0:
                logger.warning(f"Dropping empty columns: {list(empty_cols)}")
                X = X.drop(columns=empty_cols)
            
            # Handle missing values
            if handle_missing == 'drop':
                # Drop rows with any missing values
                initial_count = len(X)
                mask = X.notna().all(axis=1) & y.notna()
                X = X[mask]
                y = y[mask]
                logger.info(f"Dropped {initial_count - len(X)} rows with missing values")
            else:
                # Impute missing values
                imputer = SimpleImputer(strategy=handle_missing)
                X_imputed = imputer.fit_transform(X)
                X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
                
                # Handle missing target values
                y = y.fillna(y.mean())
                logger.info(f"Imputed missing values using {handle_missing} strategy")
            
            # Scale features if requested
            if scale_features:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                logger.info("Features scaled using StandardScaler")
            
            # Check for constant columns
            constant_cols = X.columns[X.nunique() <= 1]
            if len(constant_cols) > 0:
                logger.warning(f"Removing constant columns: {list(constant_cols)}")
                X = X.drop(columns=constant_cols)
            
            logger.info(f"Final preprocessed data shape: {X.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return X, y
    
    def get_feature_info(self, feature_group_name: str = 'aqi_features', 
                        version: int = 2) -> Optional[dict]:
        """
        Get information about the feature group.
        
        Args:
            feature_group_name (str): Feature group name
            version (int): Feature group version
            
        Returns:
            dict: Feature group information
        """
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            fg = self.fs.get_feature_group(feature_group_name, version=version)
            
            # Get basic info
            info = {
                'name': fg.name,
                'version': fg.version,
                'description': fg.description,
                'features': [f.name for f in fg.features],
                'primary_key': fg.primary_key,
                'event_time': fg.event_time
            }
            
            # Get data statistics
            df = fg.read()
            info['record_count'] = len(df)
            info['date_range'] = f"{df['time'].min()} to {df['time'].max()}"
            
            # Check target completeness
            for target in ['aqi_t_24h', 'aqi_t_48h', 'aqi_t_72h']:
                if target in df.columns:
                    filled = df[target].notna().sum()
                    total = len(df)
                    info[f'{target}_completeness'] = f"{filled}/{total} ({filled/total*100:.1f}%)"
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get feature info: {str(e)}")
            return None
    
    def get_time_series_split_indices(self, df: pd.DataFrame, 
                                     train_ratio: float = 0.8) -> Tuple[List[int], List[int]]:
        """
        Create time-based train/test split indices.
        
        Args:
            df (pd.DataFrame): DataFrame with time column
            train_ratio (float): Ratio of data for training
            
        Returns:
            tuple: (train_indices, test_indices)
        """
        try:
            # Sort by time to ensure chronological order
            df_sorted = df.sort_values('time') if 'time' in df.columns else df
            
            split_point = int(len(df_sorted) * train_ratio)
            train_indices = df_sorted.index[:split_point].tolist()
            test_indices = df_sorted.index[split_point:].tolist()
            
            logger.info(f"Time-based split: {len(train_indices)} train, {len(test_indices)} test")
            return train_indices, test_indices
            
        except Exception as e:
            logger.error(f"Error creating time series split: {str(e)}")
            # Fallback to simple split
            split_point = int(len(df) * train_ratio)
            return list(range(split_point)), list(range(split_point, len(df)))
