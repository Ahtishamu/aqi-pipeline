#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import joblib
import os
from datetime import datetime

# Import model classes
from model_training.models.linear_model import LinearModel
from model_training.models.random_forest_model import RandomForestModel
from model_training.models.gradient_boosting_model import GradientBoostingModel
from model_training.models.neural_network_model import NeuralNetworkModel

# Import evaluation
from model_training.evaluation.metrics import calculate_metrics, compare_models, print_metrics, create_metrics_dataframe

logger = logging.getLogger(__name__)

class AQIModelTrainer:
    
    def __init__(self, target_col: str = 'aqi_t_24h', random_state: int = 42):
        self.target_col = target_col
        self.random_state = random_state
        self.models = {}
        self.model_results = {}
        self.best_model_name = None
        self.best_model = None
        self.training_data_info = {}
        
        logger.info(f"Initialized AQI Model Trainer for target: {target_col}")
    
    def initialize_models(self) -> Dict[str, object]:
        models = {
            'linear': LinearModel(
                name="Linear",
                target_col=self.target_col,
                model_type='linear'
            ),
            'ridge': LinearModel(
                name="Ridge",
                target_col=self.target_col,
                model_type='ridge',
                alpha=1.0
            ),
            'lasso': LinearModel(
                name="Lasso",
                target_col=self.target_col,
                model_type='lasso',
                alpha=0.1
            ),
            'random_forest': RandomForestModel(
                name="RandomForest",
                target_col=self.target_col,
                n_estimators=100,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state
            ),
            'gradient_boosting': GradientBoostingModel(
                name="GradientBoosting",
                target_col=self.target_col,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=self.random_state
            ),
            'neural_network': NeuralNetworkModel(
                name="NeuralNetwork",
                target_col=self.target_col,
                hidden_layers=[64, 32, 16],
                activation='relu',
                dropout_rate=0.2,
                learning_rate=0.001,
                epochs=200,
                batch_size=32,
                validation_split=0.2,
                random_state=self.random_state
            )
        }
        
        logger.info(f"Initialized {len(models)} model types")
        return models
    
    def train_single_model(self, model: object, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_test: pd.DataFrame, 
                          y_test: pd.Series) -> Dict[str, float]:
        try:
            logger.info(f"Training {model.name} model...")
            
            # Train the model
            success = model.train(X_train, y_train)
            if not success:
                logger.error(f"Failed to train {model.name}")
                return {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
            
            # Make predictions
            y_pred = model.predict(X_test)
            if y_pred is None:
                logger.error(f"Failed to get predictions from {model.name}")
                return {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred)
            
            logger.info(f"{model.name} - RMSE: {metrics['rmse']:.4f}, "
                       f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training {model.name}: {str(e)}")
            return {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2, 
                        time_based_split: bool = True) -> Dict[str, Dict[str, float]]:
        logger.info(f"Training all models with {len(X)} samples and {len(X.columns)} features")
        
        # Store training data info
        self.training_data_info = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'target': self.target_col,
            'test_size': test_size,
            'time_based_split': time_based_split,
            'feature_names': X.columns.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Split data
        if time_based_split and 'time' in X.index.names:
            # Sort by time and split chronologically
            sorted_indices = X.index.argsort()
            split_point = int(len(X) * (1 - test_size))
            
            train_indices = sorted_indices[:split_point]
            test_indices = sorted_indices[split_point:]
            
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            
            logger.info("Using time-based train/test split")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            logger.info("Using random train/test split")
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Initialize models
        self.models = self.initialize_models()
        self.model_results = {}
        
        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name.upper()} MODEL")
            logger.info(f"{'='*50}")
            
            try:
                metrics = self.train_single_model(model, X_train, y_train, X_test, y_test)
                self.model_results[model_name] = metrics
                
                # Print metrics
                print_metrics(metrics, model.name)
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                self.model_results[model_name] = {
                    'rmse': float('inf'), 
                    'mae': float('inf'), 
                    'r2': -float('inf')
                }
        
        # Select best model
        self.best_model_name = compare_models(self.model_results, 'rmse')
        if self.best_model_name:
            self.best_model = self.models[self.best_model_name]
            logger.info(f"\nBest model: {self.best_model_name}")
            print_metrics(self.model_results[self.best_model_name], "BEST MODEL")
        
        return self.model_results
    
    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series, 
                             cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        logger.info(f"Starting {cv_folds}-fold time series cross-validation")
        
        # Use TimeSeriesSplit for proper time series CV
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_results = {}
        
        # Initialize models
        models = self.initialize_models()
        
        for model_name, model_class in models.items():
            logger.info(f"\nCross-validating {model_name}...")
            
            fold_scores = {'rmse': [], 'mae': [], 'r2': []}
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                logger.info(f"  Fold {fold + 1}/{cv_folds}")
                
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Create fresh model instance for this fold
                if model_name == 'linear':
                    model = LinearModel(name="Linear", target_col=self.target_col)
                elif model_name == 'ridge':
                    model = LinearModel(name="Ridge", target_col=self.target_col, 
                                      model_type='ridge', alpha=1.0)
                elif model_name == 'lasso':
                    model = LinearModel(name="Lasso", target_col=self.target_col, 
                                      model_type='lasso', alpha=0.1)
                elif model_name == 'random_forest':
                    model = RandomForestModel(name="RandomForest", target_col=self.target_col,
                                            random_state=self.random_state)
                elif model_name == 'gradient_boosting':
                    model = GradientBoostingModel(name="GradientBoosting", 
                                                target_col=self.target_col,
                                                random_state=self.random_state)
                elif model_name == 'neural_network':
                    model = NeuralNetworkModel(name="NeuralNetwork", target_col=self.target_col,
                                             epochs=100, random_state=self.random_state)
                
                # Train and evaluate
                metrics = self.train_single_model(model, X_train, y_train, X_test, y_test)
                
                # Store fold results
                for metric in fold_scores:
                    if not np.isinf(metrics[metric]):
                        fold_scores[metric].append(metrics[metric])
            
            # Calculate mean and std for each metric
            cv_results[model_name] = {}
            for metric in fold_scores:
                if fold_scores[metric]:  # Only if we have valid scores
                    cv_results[model_name][f'{metric}_mean'] = np.mean(fold_scores[metric])
                    cv_results[model_name][f'{metric}_std'] = np.std(fold_scores[metric])
                else:
                    cv_results[model_name][f'{metric}_mean'] = float('inf') if metric != 'r2' else -float('inf')
                    cv_results[model_name][f'{metric}_std'] = 0
            
            logger.info(f"  {model_name} CV RMSE: {cv_results[model_name]['rmse_mean']:.4f} "
                       f"(±{cv_results[model_name]['rmse_std']:.4f})")
        
        return cv_results
    
    def get_feature_importance_analysis(self) -> pd.DataFrame:
        importance_results = []
        
        for model_name, model in self.models.items():
            if hasattr(model, 'get_feature_importance_df'):
                try:
                    importance_df = model.get_feature_importance_df()
                    if not importance_df.empty:
                        importance_df['model'] = model_name
                        importance_results.append(importance_df)
                except Exception as e:
                    logger.error(f"Failed to get feature importance for {model_name}: {str(e)}")
            elif hasattr(model, 'get_coefficients'):
                try:
                    coef_df = model.get_coefficients()
                    if not coef_df.empty:
                        # Convert coefficients to importance-like format
                        importance_df = pd.DataFrame({
                            'feature': coef_df['feature'],
                            'importance': coef_df['abs_coefficient'],
                            'model': model_name
                        })
                        importance_results.append(importance_df)
                except Exception as e:
                    logger.error(f"Failed to get coefficients for {model_name}: {str(e)}")
        
        if importance_results:
            combined_df = pd.concat(importance_results, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def save_results(self, output_dir: str = "model_training_results") -> bool:
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model results
            results_df = create_metrics_dataframe(self.model_results)
            results_file = os.path.join(output_dir, f"model_results_{timestamp}.csv")
            results_df.to_csv(results_file, index=False)
            logger.info(f"Model results saved to: {results_file}")
            
            # Save training info
            import json
            info_file = os.path.join(output_dir, f"training_info_{timestamp}.json")
            with open(info_file, 'w') as f:
                json.dump(self.training_data_info, f, indent=2)
            logger.info(f"Training info saved to: {info_file}")
            
            # Save best model
            if self.best_model:
                model_file = os.path.join(output_dir, f"best_model_{self.target_col}_{timestamp}.pkl")
                success = self.best_model.save_model(model_file)
                if success:
                    logger.info(f"Best model saved to: {model_file}")
            
            # Save feature importance analysis
            importance_df = self.get_feature_importance_analysis()
            if not importance_df.empty:
                importance_file = os.path.join(output_dir, f"feature_importance_{timestamp}.csv")
                importance_df.to_csv(importance_file, index=False)
                logger.info(f"Feature importance saved to: {importance_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            return False
    
    def get_model_summary(self) -> dict:
        summary = {
            'target_column': self.target_col,
            'training_data': self.training_data_info,
            'models_trained': list(self.models.keys()) if self.models else [],
            'best_model': self.best_model_name,
            'model_results': self.model_results
        }
        
        if self.best_model:
            summary['best_model_info'] = self.best_model.get_model_info()
        
        return summary
