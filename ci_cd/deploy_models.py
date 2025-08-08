#!/usr/bin/env python3
"""
Deploy best models to production
"""
import os
import sys
import json
import logging
from pathlib import Path
import hopsworks
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_models():
    """
    Deploy the best models to production in Hopsworks.
    """
    try:
        logger.info("🚀 Deploying models to production")
        
        # Connect to Hopsworks
        api_key = os.environ.get('HOPSWORKS_API_KEY')
        if not api_key:
            logger.error("HOPSWORKS_API_KEY not found")
            return False
        
        project = hopsworks.login(api_key_value=api_key)
        mr = project.get_model_registry()
        
        # Load evaluation results
        project_root = Path(__file__).parent.parent
        reports_dir = project_root / "reports"
        summary_file = reports_dir / "evaluation_summary.json"
        
        if not summary_file.exists():
            logger.error("Evaluation summary not found")
            return False
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        best_models = summary.get('best_models', {})
        
        if not best_models:
            logger.error("No best models found in summary")
            return False
        
        # Deploy each horizon's best model
        deployed_models = {}
        
        for horizon, model_info in best_models.items():
            try:
                model_name = f"aqi_prediction_{horizon}"
                
                # Get the latest version of the model
                model_versions = mr.get_model(model_name)
                
                if not model_versions:
                    logger.warning(f"No model versions found for {model_name}")
                    continue
                
                # Get the latest version
                latest_version = max(model_versions, key=lambda x: x.version)
                
                # Create deployment
                deployment_name = f"aqi-{horizon}-predictor"
                
                # Check if deployment already exists
                try:
                    existing_deployment = ms.get_deployment(deployment_name)
                    logger.info(f"Updating existing deployment: {deployment_name}")
                    
                    # Update the deployment with new model
                    existing_deployment.update(model_version=latest_version)
                    
                except Exception:
                    # Create new deployment
                    logger.info(f"Creating new deployment: {deployment_name}")
                    
                    deployment = latest_version.deploy(
                        name=deployment_name,
                        description=f"AQI prediction service for {horizon} horizon",
                        script_file="predictor.py",  # Would need to create this
                        resources={"cpu": 1, "memory": 2},
                        instances=1
                    )
                
                deployed_models[horizon] = {
                    'model_name': model_name,
                    'version': latest_version.version,
                    'deployment_name': deployment_name,
                    'metrics': model_info
                }
                
                logger.info(f"✅ Deployed {horizon} model (RMSE: {model_info['test_rmse']:.4f})")
                
            except Exception as e:
                logger.error(f"Failed to deploy {horizon} model: {e}")
                continue
        
        # Save deployment summary
        deployment_summary = {
            'deployment_date': str(pd.Timestamp.now()),
            'deployed_models': deployed_models,
            'total_deployed': len(deployed_models)
        }
        
        deployment_file = reports_dir / "deployment_summary.json"
        with open(deployment_file, 'w') as f:
            json.dump(deployment_summary, f, indent=2)
        
        logger.info(f"✅ Successfully deployed {len(deployed_models)} models")
        
        # Print deployment summary
        print("\n" + "="*50)
        print("DEPLOYMENT SUMMARY")
        print("="*50)
        
        for horizon, deployment_info in deployed_models.items():
            print(f"\n{horizon} Model:")
            print(f"  Deployment: {deployment_info['deployment_name']}")
            print(f"  Version: {deployment_info['version']}")
            print(f"  RMSE: {deployment_info['metrics']['test_rmse']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        return False

def main():
    success = deploy_models()
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
