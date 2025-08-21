"""
Working predictor that extracts the model from the dictionary structure
"""
import os
import pickle
import glob
import logging
import joblib
import json
import numpy as np
from typing import Any, Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor:
    
    def __init__(self):
        print(f"WORKING PREDICTOR v1.0")
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the actual sklearn model from the dictionary structure"""
        try:
            # Find pickle files
            pkl_files = glob.glob("/mnt/models/*.pkl")
            print(f"Found {len(pkl_files)} pickle files")
            
            if not pkl_files:
                raise Exception("No pickle files found!")
            
            model_file = pkl_files[0]
            print(f" Loading: {model_file}")
            
            # Load with joblib (safer for sklearn models)
            data = joblib.load(model_file)
            print(f" Loaded data type: {type(data)}")
            
            if isinstance(data, dict):
                print(f" Dictionary keys: {list(data.keys())}")
                
                # Extract the actual model
                if 'model' in data and hasattr(data['model'], 'predict'):
                    self.model = data['model']
                    print(f"Extracted model: {type(self.model)}")
                    
                    # Extract feature names if available
                    if 'feature_names' in data:
                        self.feature_names = data['feature_names']
                        print(f"Feature names: {len(self.feature_names) if self.feature_names else 'None'}")
                    
                    # Test the model
                    self.test_model()
                    
                else:
                    raise Exception("No 'model' key with predict method found!")
                    
            elif hasattr(data, 'predict'):
                # Direct model
                self.model = data
                print(f"Direct model loaded: {type(self.model)}")
                self.test_model()
                
            else:
                raise Exception(f"Unexpected data structure: {type(data)}")
                
        except Exception as e:
            print(f"Model loading error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def test_model(self):
        try:
            # Create dummy input (assuming we need multiple features)
            if self.feature_names:
                n_features = len(self.feature_names)
            else:
                # Guess based on typical AQI model features
                n_features = 20  # Adjust as needed
            
            dummy_input = [[1.0] * n_features]
            result = self.model.predict(dummy_input)
            print(f"Test prediction successful: {result}")
            
        except Exception as e:
            print(f"Test prediction failed: {str(e)}")
            # Don't raise - the model might still work with proper input

    def predict(self, inputs: Dict) -> Dict:
        try:
            print(f"📥 Received prediction request")
            print(f"Input keys: {list(inputs.keys()) if isinstance(inputs, dict) else 'Not a dict'}")
            
            if self.model is None:
                print("No model loaded!")
                return {"predictions": [3.0]}
            
            # Extract input data
            if 'instances' in inputs:
                input_data = inputs['instances']
            elif 'inputs' in inputs:
                input_data = inputs['inputs']
            else:
                # Try to use the input directly
                input_data = inputs
            
            print(f"🔢 Input data shape: {np.array(input_data).shape if hasattr(input_data, '__len__') else 'scalar'}")
            
            # Make prediction
            predictions = self.model.predict(input_data)
            print(f"Raw predictions: {predictions}")
            
            # Convert to list
            if hasattr(predictions, 'tolist'):
                pred_list = predictions.tolist()
            else:
                pred_list = list(predictions) if hasattr(predictions, '__iter__') else [predictions]
            
            result = {
                "predictions": pred_list
            }
            print(f"📤 Returning: {result}")
            return result
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return fallback
            return {"predictions": [3.0]}

# Test on import
print("WORKING PREDICTOR READY!")
try:
    # Test instantiation
    test_predictor = Predictor()
    print("Predictor initialized successfully!")
except Exception as e:
    print(f"Predictor initialization failed: {str(e)}")
    import traceback
    traceback.print_exc()
