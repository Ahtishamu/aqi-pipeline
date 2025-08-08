#!/usr/bin/env python3
"""
Random Forest model for AQI prediction
"""
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from model_training.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class RandomForestModel(BaseModel):
    """Random Forest regression model implementation."""
    
    def __init__(self, name: str = "RandomForest", target_col: str = None,
                 n_estimators: int = 100, max_depth: int = None, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', random_state: int = 42):
        """
        Initialize the Random Forest model.
        
        Args:
            name (str): Model name
            target_col (str): Target column name
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth
            min_samples_split (int): Minimum samples required to split
            min_samples_leaf (int): Minimum samples required in leaf
            max_features (str): Number of features to consider for split
            random_state (int): Random seed
        """
        super().__init__(name=name, target_col=target_col)
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
            verbose=0
        )
        
        # Store hyperparameters
        self.hyperparams = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Train the Random Forest model.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target values
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # Validate inputs
            if not self.validate_input(X):
                return False
            
            if len(X) != len(y):
                logger.error(f"Feature and target length mismatch: {len(X)} vs {len(y)}")
                return False
            
            # Store feature names
            self.feature_names_ = X.columns.tolist()
            
            logger.info(f"Training {self.name} model with {len(X)} samples and {len(X.columns)} features")
            logger.info(f"Hyperparameters: {self.hyperparams}")
            
            # Handle target as DataFrame if needed
            if isinstance(y, pd.DataFrame):
                if self.target_col and self.target_col in y.columns:
                    y = y[self.target_col]
                else:
                    y = y.iloc[:, 0]  # Take first column
            
            # Train the model
            self.model.fit(X, y)
            
            # Extract feature importances
            self.feature_importances_ = self.model.feature_importances_
            
            self.is_trained = True
            logger.info(f"✅ {self.name} model trained successfully")
            
            # Log training information
            train_score = self.model.score(X, y)
            logger.info(f"Training R² score: {train_score:.4f}")
            logger.info(f"Number of estimators: {self.model.n_estimators}")
            
            # Log feature importance statistics
            if self.feature_importances_ is not None:
                top_features = np.argsort(self.feature_importances_)[-5:][::-1]
                logger.info("Top 5 important features:")
                for i, feat_idx in enumerate(top_features):
                    feat_name = self.feature_names_[feat_idx]
                    importance = self.feature_importances_[feat_idx]
                    logger.info(f"  {i+1}. {feat_name}: {importance:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train {self.name} model: {str(e)}")
            self.is_trained = False
            return False
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the Random Forest model.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predictions or None if failed
        """
        if not self.is_trained:
            logger.error("Model not trained. Call train() first.")
            return None
        
        try:
            # Validate input
            if not self.validate_input(X):
                return None
            
            # Ensure feature order matches training
            if self.feature_names_:
                missing_features = set(self.feature_names_) - set(X.columns)
                if missing_features:
                    logger.error(f"Missing features for prediction: {missing_features}")
                    return None
                X = X[self.feature_names_]
            
            # Make predictions
            predictions = self.model.predict(X)
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions with {self.name}: {str(e)}")
            return None
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple:
        """
        Make predictions with uncertainty estimates using individual trees.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            tuple: (predictions, standard_deviations) or (None, None) if failed
        """
        if not self.is_trained:
            logger.error("Model not trained. Call train() first.")
            return None, None
        
        try:
            # Validate input
            if not self.validate_input(X):
                return None, None
            
            # Ensure feature order matches training
            if self.feature_names_:
                X = X[self.feature_names_]
            
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            
            # Calculate mean and standard deviation
            predictions = np.mean(tree_predictions, axis=0)
            uncertainties = np.std(tree_predictions, axis=0)
            
            logger.info(f"Generated {len(predictions)} predictions with uncertainty estimates")
            return predictions, uncertainties
            
        except Exception as e:
            logger.error(f"Failed to make predictions with uncertainty: {str(e)}")
            return None, None
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importances as a sorted DataFrame.
        
        Returns:
            pd.DataFrame: Feature importances sorted by importance
        """
        if not self.is_trained or self.feature_importances_ is None:
            return pd.DataFrame()
        
        try:
            importance_df = pd.DataFrame({
                'feature': self.feature_names_,
                'importance': self.feature_importances_
            })
            
            # Sort by importance (descending)
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Add cumulative importance
            importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
            
            return importance_df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Failed to create feature importance DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def get_oob_score(self) -> float:
        """
        Get out-of-bag score if available.
        
        Returns:
            float: OOB score or None if not available
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'oob_score_'):
            return self.model.oob_score_
        else:
            logger.warning("OOB score not available. Set oob_score=True in model initialization.")
            return None
    
    def get_tree_depths(self) -> list:
        """
        Get the depth of each tree in the forest.
        
        Returns:
            list: Tree depths
        """
        if not self.is_trained:
            return []
        
        try:
            depths = [tree.tree_.max_depth for tree in self.model.estimators_]
            return depths
        except Exception as e:
            logger.error(f"Failed to get tree depths: {str(e)}")
            return []
    
    def partial_dependence(self, X: pd.DataFrame, feature_names: list) -> dict:
        """
        Calculate partial dependence for specified features.
        
        Args:
            X (pd.DataFrame): Features
            feature_names (list): Features to calculate partial dependence for
            
        Returns:
            dict: Partial dependence results
        """
        if not self.is_trained:
            return {}
        
        try:
            from sklearn.inspection import partial_dependence
            
            # Get feature indices
            feature_indices = [self.feature_names_.index(name) for name in feature_names 
                             if name in self.feature_names_]
            
            if not feature_indices:
                logger.warning("No valid features found for partial dependence")
                return {}
            
            # Calculate partial dependence
            pd_results = partial_dependence(
                self.model, X[self.feature_names_], feature_indices
            )
            
            results = {}
            for i, feat_idx in enumerate(feature_indices):
                feat_name = self.feature_names_[feat_idx]
                results[feat_name] = {
                    'values': pd_results['values'][i],
                    'grid_values': pd_results['grid_values'][i]
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to calculate partial dependence: {str(e)}")
            return {}
    
    def get_model_info(self) -> dict:
        """
        Get detailed model information.
        
        Returns:
            dict: Model information
        """
        info = super().get_model_info()
        info.update(self.hyperparams)
        
        if self.is_trained:
            info.update({
                'oob_score': self.get_oob_score(),
                'n_trees': len(self.model.estimators_),
                'avg_tree_depth': np.mean(self.get_tree_depths()) if self.get_tree_depths() else None,
                'max_tree_depth': np.max(self.get_tree_depths()) if self.get_tree_depths() else None
            })
        
        return info
