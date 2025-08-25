#!/usr/bin/env python3
"""
Gradient Boosting model for AQI prediction
"""
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import GradientBoostingRegressor
from model_training.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class GradientBoostingModel(BaseModel):    
    def __init__(self, name: str = "GradientBoosting", target_col: str = None,
                 n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, subsample: float = 1.0,
                 random_state: int = 42):
        super().__init__(name=name, target_col=target_col)
        
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
            verbose=0
        )
        
        # Store hyperparameters
        self.hyperparams = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'subsample': subsample,
            'random_state': random_state
        }
        
        self.training_scores_ = None
        self.validation_scores_ = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              monitor_validation: bool = True, validation_fraction: float = 0.1) -> bool:
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
            
            # Set up validation monitoring
            if monitor_validation and len(X) > 100:  # Only if we have enough data
                self.model.set_params(
                    validation_fraction=validation_fraction,
                    n_iter_no_change=10,  # Early stopping
                    tol=1e-4
                )
            
            # Train the model
            self.model.fit(X, y)
            
            # Extract feature importances
            self.feature_importances_ = self.model.feature_importances_
            
            # Store training scores
            self.training_scores_ = self.model.train_score_
            if hasattr(self.model, 'validation_scores_'):
                self.validation_scores_ = self.model.validation_scores_
            
            self.is_trained = True
            logger.info(f"✅ {self.name} model trained successfully")
            
            # Log training information
            train_score = self.model.score(X, y)
            logger.info(f"Training R² score: {train_score:.4f}")
            logger.info(f"Number of estimators used: {self.model.n_estimators_}")
            
            if self.training_scores_ is not None:
                final_train_score = self.training_scores_[-1]
                logger.info(f"Final training loss: {final_train_score:.4f}")
            
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
    
    def staged_predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            logger.error("Model not trained. Call train() first.")
            return None
        
        try:
            # Validate input
            if not self.validate_input(X):
                return None
            
            # Ensure feature order matches training
            if self.feature_names_:
                X = X[self.feature_names_]
            
            # Get staged predictions
            staged_preds = list(self.model.staged_predict(X))
            
            return np.array(staged_preds)
            
        except Exception as e:
            logger.error(f"Failed to get staged predictions: {str(e)}")
            return None
    
    def get_feature_importance_df(self) -> pd.DataFrame:
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
    
    def get_learning_curve(self) -> dict:
        if not self.is_trained:
            return {}
        
        result = {}
        
        if self.training_scores_ is not None:
            result['training_scores'] = self.training_scores_
            result['n_estimators'] = range(1, len(self.training_scores_) + 1)
        
        if self.validation_scores_ is not None:
            result['validation_scores'] = self.validation_scores_
        
        return result
    
    def find_optimal_n_estimators(self, X: pd.DataFrame, y: pd.Series,
                                 test_size: float = 0.2) -> int:
        if not self.is_trained:
            logger.error("Model must be trained first")
            return self.model.n_estimators
        
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Get staged predictions on test set
            if self.feature_names_:
                X_test = X_test[self.feature_names_]
            
            staged_preds = list(self.model.staged_predict(X_test))
            
            # Calculate test errors
            test_errors = []
            for pred in staged_preds:
                error = mean_squared_error(y_test, pred)
                test_errors.append(error)
            
            # Find optimal number of estimators (minimum test error)
            optimal_n = np.argmin(test_errors) + 1
            
            logger.info(f"Optimal number of estimators: {optimal_n}")
            return optimal_n
            
        except Exception as e:
            logger.error(f"Failed to find optimal n_estimators: {str(e)}")
            return self.model.n_estimators
    
    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update(self.hyperparams)
        
        if self.is_trained:
            info.update({
                'n_estimators_used': self.model.n_estimators_,
                'has_validation_scores': self.validation_scores_ is not None,
                'final_training_loss': self.training_scores_[-1] if self.training_scores_ is not None else None,
                'final_validation_loss': self.validation_scores_[-1] if self.validation_scores_ is not None else None
            })
        
        return info
