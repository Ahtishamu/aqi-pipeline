#!/usr/bin/env python3
"""
Neural Network model for AQI prediction using TensorFlow/Keras
"""
import pandas as pd
import numpy as np
import logging
from model_training.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# Try to import TensorFlow, provide fallback if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import StandardScaler
    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. Neural Network model will not work.")
    TF_AVAILABLE = False

class NeuralNetworkModel(BaseModel):
    """Neural Network regression model implementation using TensorFlow/Keras."""
    
    def __init__(self, name: str = "NeuralNetwork", target_col: str = None,
                 hidden_layers: list = [64, 32], activation: str = 'relu',
                 dropout_rate: float = 0.2, learning_rate: float = 0.001,
                 epochs: int = 100, batch_size: int = 32, 
                 validation_split: float = 0.2, random_state: int = 42):
        """
        Initialize the Neural Network model.
        
        Args:
            name (str): Model name
            target_col (str): Target column name
            hidden_layers (list): List of hidden layer sizes
            activation (str): Activation function
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data for validation
            random_state (int): Random seed
        """
        super().__init__(name=name, target_col=target_col)
        
        if not TF_AVAILABLE:
            logger.error("TensorFlow is not available. Cannot use Neural Network model.")
            self.model = None
            return
        
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.random_state = random_state
        
        # Initialize scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Training history
        self.history = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Store hyperparameters
        self.hyperparams = {
            'hidden_layers': hidden_layers,
            'activation': activation,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'random_state': random_state
        }
    
    def _build_model(self, input_dim: int) -> keras.Model:
        """
        Build the neural network architecture.
        
        Args:
            input_dim (int): Number of input features
            
        Returns:
            keras.Model: Compiled model
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not available")
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(units, activation=self.activation, 
                                 name=f'hidden_{i+1}'))
            if self.dropout_rate > 0:
                model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X: pd.DataFrame, y: pd.Series, verbose: int = 0) -> bool:
        """
        Train the Neural Network model.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target values
            verbose (int): Verbosity level for training
            
        Returns:
            bool: True if training successful, False otherwise
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow is not available")
            return False
        
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
            logger.info(f"Architecture: {len(X.columns)} -> {' -> '.join(map(str, self.hidden_layers))} -> 1")
            
            # Handle target as DataFrame if needed
            if isinstance(y, pd.DataFrame):
                if self.target_col and self.target_col in y.columns:
                    y = y[self.target_col]
                else:
                    y = y.iloc[:, 0]  # Take first column
            
            # Scale features and target
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
            
            # Build model
            self.model = self._build_model(X_scaled.shape[1])
            
            # Set up callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=verbose
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6,
                    verbose=verbose
                )
            ]
            
            # Train the model
            self.history = self.model.fit(
                X_scaled, y_scaled,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            
            self.is_trained = True
            logger.info(f"✅ {self.name} model trained successfully")
            
            # Log training information
            final_loss = self.history.history['loss'][-1]
            final_val_loss = self.history.history['val_loss'][-1]
            epochs_trained = len(self.history.history['loss'])
            
            logger.info(f"Training completed in {epochs_trained} epochs")
            logger.info(f"Final training loss: {final_loss:.4f}")
            logger.info(f"Final validation loss: {final_val_loss:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train {self.name} model: {str(e)}")
            self.is_trained = False
            return False
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the Neural Network model.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predictions or None if failed
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow is not available")
            return None
        
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
            
            # Scale features
            X_scaled = self.scaler_X.transform(X)
            
            # Make predictions
            predictions_scaled = self.model.predict(X_scaled, verbose=0)
            
            # Inverse transform predictions
            predictions = self.scaler_y.inverse_transform(predictions_scaled).flatten()
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions with {self.name}: {str(e)}")
            return None
    
    def get_training_history(self) -> dict:
        """
        Get training history data.
        
        Returns:
            dict: Training history
        """
        if self.history is None:
            return {}
        
        return {
            'epochs': range(1, len(self.history.history['loss']) + 1),
            'loss': self.history.history['loss'],
            'val_loss': self.history.history['val_loss'],
            'mae': self.history.history.get('mae', []),
            'val_mae': self.history.history.get('val_mae', [])
        }
    
    def plot_training_history(self):
        """
        Plot training history (loss and metrics).
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            history = self.history.history
            epochs = range(1, len(history['loss']) + 1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
            ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot MAE if available
            if 'mae' in history:
                ax2.plot(epochs, history['mae'], 'b-', label='Training MAE')
                ax2.plot(epochs, history['val_mae'], 'r-', label='Validation MAE')
                ax2.set_title('Model MAE')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MAE')
                ax2.legend()
                ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to plot training history: {str(e)}")
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            str: Model summary
        """
        if not self.is_trained or self.model is None:
            return "Model not trained"
        
        try:
            import io
            import contextlib
            
            # Capture model summary
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.model.summary()
            summary = buffer.getvalue()
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get model summary: {str(e)}")
            return f"Error getting summary: {str(e)}"
    
    def get_model_info(self) -> dict:
        """
        Get detailed model information.
        
        Returns:
            dict: Model information
        """
        info = super().get_model_info()
        info.update(self.hyperparams)
        
        if self.is_trained and self.history:
            history = self.history.history
            info.update({
                'epochs_trained': len(history['loss']),
                'final_training_loss': history['loss'][-1],
                'final_validation_loss': history['val_loss'][-1],
                'best_validation_loss': min(history['val_loss']),
                'total_parameters': self.model.count_params() if self.model else 0
            })
        
        return info
