#!/usr/bin/env python3
"""
Model training pipeline for CI/CD
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_training.data_loader import AQIDataLoader
from model_training.model_trainer import AQIModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model_pipeline(horizon: str):
    """
    Train model for specific horizon in CI/CD environment.
    
    Args:
        horizon (str): Model horizon ('24h', '48h', '72h')
    """
    try:
        logger.info(f"🚀 Starting model training pipeline for {horizon}")
        
        # Check if training should be forced
        force_retrain = os.environ.get('FORCE_RETRAIN', 'false').lower() == 'true'
        
        if not force_retrain:
            # Check if we need to retrain based on data freshness
            from ci_cd.check_data_freshness import check_data_freshness
            should_train = check_data_freshness()
            
            if not should_train:
                logger.info(f"⏭️ Skipping training for {horizon} - insufficient new data")
                return True  # Not an error, just not needed
        else:
            logger.info(f"🔄 Force retraining enabled for {horizon}")
        
        # Determine target column
        target_mapping = {
            '24h': 'aqi_t_24h',
            '48h': 'aqi_t_48h', 
            '72h': 'aqi_t_72h'
        }
        
        if horizon not in target_mapping:
            raise ValueError(f"Invalid horizon: {horizon}")
        
        target_col = target_mapping[horizon]
        
        # Initialize data loader
        data_loader = AQIDataLoader()
        
        # Load training data
        logger.info(f"📊 Loading training data for {target_col}")
        X, y = data_loader.load_training_data(target_col=target_col)
        
        if X is None or y is None:
            logger.error("Failed to load training data")
            return False
        
        # Preprocess data
        logger.info("🔧 Preprocessing data")
        X, y = data_loader.preprocess_features(X, y)
        
        # Initialize trainer
        trainer = AQIModelTrainer(
            target_col=target_col,
            random_state=42
        )
        
        # Train all models
        logger.info("🎯 Training models")
        results = trainer.train_all_models(X, y)
        
        if not results:
            logger.error("Model training failed")
            return False
        
        # Save results and update model registry
        output_dir = project_root / "model_results" / f"aqi_{horizon}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training results using the trainer's method
        success = trainer.save_results(str(output_dir))
        
        if success:
            # Update model registry with new model
            from ci_cd.update_model_registry import update_model_registry
            registry_success = update_model_registry(horizon, str(output_dir), trainer.best_model_name)
            
            logger.info(f"✅ Training completed for {horizon}")
            logger.info(f"Best model: {trainer.best_model_name}")
            logger.info(f"Results saved to: {output_dir}")
            
            if registry_success:
                logger.info("✅ Model registry updated successfully")
            else:
                logger.warning("⚠️ Model registry update failed")
        else:
            logger.error("Failed to save results")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train AQI prediction model')
    parser.add_argument('--horizon', choices=['24h', '48h', '72h'], 
                       required=True, help='Model horizon to train')
    
    args = parser.parse_args()
    
    success = train_model_pipeline(args.horizon)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
