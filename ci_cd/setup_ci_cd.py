#!/usr/bin/env python3
"""
Setup script for CI/CD pipeline
"""
import os
from pathlib import Path

def setup_ci_cd():
    print("🔧 Setting up CI/CD pipeline for AQI model training")
    
    project_root = Path(__file__).parent.parent
    
    # Create required directories
    directories = [
        "model_results",
        "model_results/aqi_24h",
        "model_results/aqi_48h", 
        "model_results/aqi_72h",
        "reports",
        "models"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
        print(f" Created directory: {directory}")
    
    # Create CI/CD __init__.py
    ci_cd_init = project_root / "ci_cd" / "__init__.py"
    ci_cd_init.touch()
    
    print("\n Required GitHub Secrets:")
    print("- HOPSWORKS_API_KEY: Your Hopsworks API key")
    print("- OWM_API_KEY: Your OpenWeatherMap API key")
    
    print("\n CI/CD Pipeline Features:")
    print("-  Daily automated training at 2 AM UTC")
    print("-  Data freshness checking")
    print("-  Multi-horizon model training (24h, 48h, 72h)")
    print("-  Model evaluation and comparison")
    print("-  Automated model registry updates")
    print("-  HTML report generation")
    print("-  Manual training trigger")
    print("-  Artifact storage and retention")
    
    print("\n Workflow Files Created:")
    print("- .github/workflows/daily-model-training.yml")
    print("- .github/workflows/manual-training.yml")
    
    print("\n CI/CD Scripts Created:")
    print("- ci_cd/check_data_freshness.py")
    print("- ci_cd/train_model_pipeline.py") 
    print("- ci_cd/update_model_registry.py")
    print("- ci_cd/evaluate_models.py")
    print("- ci_cd/deploy_models.py")
    print("- ci_cd/generate_model_report.py")
    
    print("\n Next Steps:")
    print("1. Add the required secrets to your GitHub repository")
    print("2. Commit and push the CI/CD files to your repository")
    print("3. The daily workflow will run automatically at 2 AM UTC")
    print("4. Use the manual workflow for on-demand training")
    print("5. Check the Actions tab in GitHub to monitor pipeline runs")
    
    print("\n CI/CD setup complete!")

if __name__ == '__main__':
    setup_ci_cd()
