#!/usr/bin/env python3
"""
SHAP and LIME explainability for AQI prediction models
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# SHAP and LIME imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from model_training.data_loader import AQIDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    """Model explainability using SHAP and LIME"""
    
    def __init__(self, model_path: str, horizon: str = '24h'):
        self.model_path = model_path
        self.horizon = horizon
        self.model = None
        self.explainer = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        
    def load_model_and_data(self):
        """Load trained model and test data"""
        try:
            # Load model
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Loaded model from {self.model_path}")
            
            # Load data
            data_loader = AQIDataLoader()
            target_mapping = {'24h': 'aqi_t_24h', '48h': 'aqi_t_48h', '72h': 'aqi_t_72h'}
            target_col = target_mapping.get(self.horizon, 'aqi_t_24h')
            
            X, y = data_loader.load_training_data(target_col=target_col)
            if X is not None and y is not None:
                X, y = data_loader.preprocess_features(X, y)
                
                # Split for explanation (use last 20% as test set)
                split_idx = int(len(X) * 0.8)
                self.X_train = X.iloc[:split_idx]
                self.X_test = X.iloc[split_idx:]
                self.y_test = y.iloc[split_idx:]
                self.feature_names = list(X.columns)
                
                logger.info(f"✅ Loaded data: {len(self.X_train)} train, {len(self.X_test)} test samples")
                return True
            else:
                logger.error("Failed to load data")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model/data: {e}")
            return False
    
    def explain_with_shap(self, save_plots: bool = True):
        """Generate SHAP explanations"""
        if not SHAP_AVAILABLE:
            logger.error("SHAP not installed. Install with: pip install shap")
            return
            
        if self.model is None:
            logger.error("Model not loaded")
            return
            
        try:
            logger.info("🔍 Generating SHAP explanations...")
            
            # Create SHAP explainer based on model type
            model_name = type(self.model).__name__.lower()
            
            if 'forest' in model_name or 'tree' in model_name:
                # Tree-based models
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(self.X_test)
            else:
                # Other models (use KernelExplainer)
                explainer = shap.KernelExplainer(
                    self.model.predict, 
                    self.X_train.sample(min(100, len(self.X_train)))  # Background data
                )
                shap_values = explainer.shap_values(self.X_test.iloc[:50])  # Limit for speed
            
            if save_plots:
                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False)
                plt.title(f'SHAP Summary Plot - AQI {self.horizon} Prediction')
                plt.tight_layout()
                plt.savefig(f'shap_summary_{self.horizon}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Feature importance plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, 
                                plot_type="bar", show=False)
                plt.title(f'SHAP Feature Importance - AQI {self.horizon} Prediction')
                plt.tight_layout()
                plt.savefig(f'shap_importance_{self.horizon}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Waterfall plot for first prediction
                if hasattr(shap, 'waterfall_plot'):
                    plt.figure(figsize=(10, 6))
                    shap.waterfall_plot(
                        explainer.expected_value, 
                        shap_values[0], 
                        self.X_test.iloc[0], 
                        feature_names=self.feature_names,
                        show=False
                    )
                    plt.title(f'SHAP Waterfall Plot - Single Prediction (AQI {self.horizon})')
                    plt.tight_layout()
                    plt.savefig(f'shap_waterfall_{self.horizon}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                
                logger.info(f"✅ SHAP plots saved for {self.horizon} model")
            
            # Generate feature importance ranking
            feature_importance = np.abs(shap_values).mean(0)
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 Top SHAP Feature Importance (AQI {self.horizon}):")
            for _, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
            
            return shap_values, importance_df
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return None, None
    
    def explain_with_lime(self, num_samples: int = 5):
        """Generate LIME explanations for individual predictions"""
        if not LIME_AVAILABLE:
            logger.error("LIME not installed. Install with: pip install lime")
            return
            
        if self.model is None:
            logger.error("Model not loaded")
            return
            
        try:
            logger.info("🔍 Generating LIME explanations...")
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train.values,
                feature_names=self.feature_names,
                mode='regression',
                discretize_continuous=True
            )
            
            explanations = []
            
            for i in range(min(num_samples, len(self.X_test))):
                # Explain prediction
                exp = explainer.explain_instance(
                    self.X_test.iloc[i].values,
                    self.model.predict,
                    num_features=len(self.feature_names)
                )
                
                # Save explanation
                exp.save_to_file(f'lime_explanation_{self.horizon}_sample_{i+1}.html')
                explanations.append(exp)
                
                # Print top features for this prediction
                actual_value = self.y_test.iloc[i]
                predicted_value = self.model.predict(self.X_test.iloc[i:i+1])[0]
                
                print(f"\n📊 LIME Explanation {i+1} (AQI {self.horizon}):")
                print(f"   Actual: {actual_value:.2f}, Predicted: {predicted_value:.2f}")
                print("   Top contributing features:")
                
                for feature, importance in exp.as_list()[:5]:
                    print(f"     {feature}: {importance:+.4f}")
            
            logger.info(f"✅ LIME explanations saved for {num_samples} samples")
            return explanations
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return None
    
    def generate_explanation_report(self):
        """Generate comprehensive explainability report"""
        try:
            print(f"\n🔍 MODEL EXPLAINABILITY ANALYSIS - AQI {self.horizon}")
            print("=" * 60)
            
            # Load model and data
            if not self.load_model_and_data():
                return False
            
            # SHAP explanations
            if SHAP_AVAILABLE:
                print("\n📊 Generating SHAP explanations...")
                shap_values, shap_importance = self.explain_with_shap()
            else:
                print("\n⚠️ SHAP not available. Install with: pip install shap")
            
            # LIME explanations
            if LIME_AVAILABLE:
                print("\n🔍 Generating LIME explanations...")
                lime_explanations = self.explain_with_lime(num_samples=3)
            else:
                print("\n⚠️ LIME not available. Install with: pip install lime")
            
            # Model performance summary
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(self.X_test)
                rmse = np.sqrt(np.mean((self.y_test - predictions) ** 2))
                mae = np.mean(np.abs(self.y_test - predictions))
                
                print(f"\n📈 Model Performance on Test Set:")
                print(f"   RMSE: {rmse:.4f}")
                print(f"   MAE: {mae:.4f}")
                print(f"   Test samples: {len(self.X_test)}")
            
            print(f"\n✅ Explainability analysis complete for AQI {self.horizon} model!")
            
            if SHAP_AVAILABLE:
                print(f"📊 SHAP plots saved: shap_summary_{self.horizon}.png, shap_importance_{self.horizon}.png")
            if LIME_AVAILABLE:
                print(f"🔍 LIME explanations saved: lime_explanation_{self.horizon}_sample_*.html")
            
            return True
            
        except Exception as e:
            logger.error(f"Explanation report generation failed: {e}")
            return False

def main():
    """Run explainability analysis for all available models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate model explainability analysis')
    parser.add_argument('--horizon', choices=['24h', '48h', '72h', 'all'], 
                       default='all', help='Model horizon to analyze')
    parser.add_argument('--model-dir', default='model_results', 
                       help='Directory containing trained models')
    
    args = parser.parse_args()
    
    horizons = ['24h', '48h', '72h'] if args.horizon == 'all' else [args.horizon]
    
    for horizon in horizons:
        model_dir = Path(args.model_dir) / f"aqi_{horizon}"
        
        # Find the latest model file
        model_files = list(model_dir.glob("best_model_*.pkl"))
        
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            print(f"\n🎯 Analyzing {horizon} model: {latest_model}")
            
            explainer = ModelExplainer(str(latest_model), horizon)
            explainer.generate_explanation_report()
        else:
            print(f"\n❌ No trained model found for {horizon} in {model_dir}")

if __name__ == "__main__":
    main()
