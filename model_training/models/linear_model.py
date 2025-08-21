#!/usr/bin/env python3
"""
Linear regression models for AQI prediction
"""
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from model_training.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class LinearModel(BaseModel):
 """Linear regression model implementation with variants."""
 
 def __init__(self, name: str = "LinearRegression", target_col: str = None, 
 model_type: str = 'linear', alpha: float = 1.0):
 """
 Initialize the linear model.
 
 Args:
 name (str): Model name
 target_col (str): Target column name
 model_type (str): Type of linear model ('linear', 'ridge', 'lasso')
 alpha (float): Regularization strength for Ridge/Lasso
 """
 super().__init__(name=name, target_col=target_col)
 self.model_type = model_type
 self.alpha = alpha
 self.scaler = StandardScaler()
 self.use_scaling = True
 
 # Initialize the appropriate model
 if model_type == 'ridge':
 self.model = Ridge(alpha=alpha, random_state=42)
 elif model_type == 'lasso':
 self.model = Lasso(alpha=alpha, random_state=42, max_iter=2000)
 else:
 self.model = LinearRegression()
 
 def train(self, X: pd.DataFrame, y: pd.Series) -> bool:
 """
 Train the linear model.
 
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
 
 # Handle target as DataFrame if needed
 if isinstance(y, pd.DataFrame):
 if self.target_col and self.target_col in y.columns:
 y = y[self.target_col]
 else:
 y = y.iloc[:, 0] # Take first column
 
 # Scale features if enabled
 if self.use_scaling:
 X_scaled = self.scaler.fit_transform(X)
 X_train = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
 else:
 X_train = X
 
 # Train the model
 self.model.fit(X_train, y)
 
 # Extract feature importances (coefficients)
 if hasattr(self.model, 'coef_'):
 self.feature_importances_ = np.abs(self.model.coef_)
 
 self.is_trained = True
 logger.info(f" {self.name} model trained successfully")
 
 # Log model information
 if hasattr(self.model, 'score'):
 train_score = self.model.score(X_train, y)
 logger.info(f"Training R² score: {train_score:.4f}")
 
 if hasattr(self.model, 'coef_'):
 logger.info(f"Number of coefficients: {len(self.model.coef_)}")
 if hasattr(self.model, 'intercept_'):
 logger.info(f"Intercept: {self.model.intercept_:.4f}")
 
 return True
 
 except Exception as e:
 logger.error(f"Failed to train {self.name} model: {str(e)}")
 self.is_trained = False
 return False
 
 def predict(self, X: pd.DataFrame) -> np.ndarray:
 """
 Make predictions with the linear model.
 
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
 
 # Scale features if scaling was used during training
 if self.use_scaling:
 X_scaled = self.scaler.transform(X)
 X_pred = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
 else:
 X_pred = X
 
 # Make predictions
 predictions = self.model.predict(X_pred)
 
 logger.info(f"Generated {len(predictions)} predictions")
 return predictions
 
 except Exception as e:
 logger.error(f"Failed to make predictions with {self.name}: {str(e)}")
 return None
 
 def get_coefficients(self) -> pd.DataFrame:
 """
 Get model coefficients with feature names.
 
 Returns:
 pd.DataFrame: Coefficients with feature names
 """
 if not self.is_trained or not hasattr(self.model, 'coef_'):
 return pd.DataFrame()
 
 try:
 coef_df = pd.DataFrame({
 'feature': self.feature_names_,
 'coefficient': self.model.coef_,
 'abs_coefficient': np.abs(self.model.coef_)
 })
 
 # Sort by absolute coefficient value
 coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
 
 return coef_df.reset_index(drop=True)
 
 except Exception as e:
 logger.error(f"Failed to get coefficients: {str(e)}")
 return pd.DataFrame()
 
 def get_regularization_path(self, X: pd.DataFrame, y: pd.Series, 
 alphas: np.ndarray = None) -> dict:
 """
 Get regularization path for Ridge/Lasso models.
 
 Args:
 X (pd.DataFrame): Features
 y (pd.Series): Target
 alphas (np.ndarray): Alpha values to test
 
 Returns:
 dict: Regularization path results
 """
 if self.model_type not in ['ridge', 'lasso']:
 logger.warning("Regularization path only available for Ridge/Lasso models")
 return {}
 
 try:
 if alphas is None:
 alphas = np.logspace(-3, 2, 50)
 
 if self.use_scaling:
 X_scaled = self.scaler.fit_transform(X)
 X_train = pd.DataFrame(X_scaled, columns=X.columns)
 else:
 X_train = X
 
 coef_path = []
 scores = []
 
 for alpha in alphas:
 if self.model_type == 'ridge':
 temp_model = Ridge(alpha=alpha, random_state=42)
 else:
 temp_model = Lasso(alpha=alpha, random_state=42, max_iter=2000)
 
 temp_model.fit(X_train, y)
 coef_path.append(temp_model.coef_)
 scores.append(temp_model.score(X_train, y))
 
 return {
 'alphas': alphas,
 'coefficients': np.array(coef_path),
 'scores': np.array(scores),
 'feature_names': self.feature_names_
 }
 
 except Exception as e:
 logger.error(f"Failed to compute regularization path: {str(e)}")
 return {}
 
 def get_model_info(self) -> dict:
 """
 Get detailed model information.
 
 Returns:
 dict: Model information
 """
 info = super().get_model_info()
 info.update({
 'model_type': self.model_type,
 'alpha': self.alpha,
 'use_scaling': self.use_scaling
 })
 
 if self.is_trained and hasattr(self.model, 'coef_'):
 info.update({
 'n_coefficients': len(self.model.coef_),
 'intercept': float(self.model.intercept_) if hasattr(self.model, 'intercept_') else None,
 'max_coef': float(np.max(np.abs(self.model.coef_))),
 'min_coef': float(np.min(np.abs(self.model.coef_)))
 })
 
 return info
