#!/usr/bin/env python3
"""
Evaluate and compare models across different horizons
"""
import os
import json
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_models():
    """
    Evaluate and compare models across all horizons.
    """
    try:
        logger.info("📊 Evaluating and comparing models")
        
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "model_results"
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Collect results from all horizons
        all_results = {}
        horizons = ['24h', '48h', '72h']
        
        for horizon in horizons:
            horizon_dir = results_dir / f"aqi_{horizon}"
            
            # Look for the most recent results files
            if horizon_dir.exists():
                # Find the most recent training_info file
                info_files = list(horizon_dir.glob("training_info_*.json"))
                results_files = list(horizon_dir.glob("model_results_*.csv"))
                
                if info_files and results_files:
                    # Get the most recent files
                    latest_info = max(info_files, key=lambda x: x.stat().st_mtime)
                    latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
                    
                    # Load results
                    with open(latest_info, 'r') as f:
                        info_data = json.load(f)
                    
                    results_df = pd.read_csv(latest_results)
                    
                    # Convert to the expected format
                    all_results[horizon] = {
                        'info': info_data,
                        'results': results_df.to_dict('records')
                    }
                    logger.info(f"✅ Loaded results for {horizon}")
                else:
                    logger.warning(f"⚠️ Results files not found for {horizon}")
            else:
                logger.warning(f"⚠️ Results directory not found for {horizon}")
        
        if not all_results:
            logger.error("No model results found")
            return False
        
        # Create comparison DataFrame
        comparison_data = []
        
        for horizon, data in all_results.items():
            # Get the best model info from training info
            info = data.get('info', {})
            results = data.get('results', [])
            
            # Add data for each model from results CSV
            for result_row in results:
                comparison_data.append({
                    'horizon': horizon,
                    'model': result_row.get('Model', 'Unknown'),
                    'test_rmse': result_row.get('RMSE', result_row.get('rmse', float('inf'))),
                    'test_mae': result_row.get('MAE', result_row.get('mae', float('inf'))),
                    'test_r2': result_row.get('R2', result_row.get('r2', -float('inf'))),
                    'cv_rmse_mean': result_row.get('CV_RMSE_Mean', result_row.get('RMSE', float('inf'))),
                    'cv_rmse_std': result_row.get('CV_RMSE_Std', 0),
                    'train_time': info.get('total_training_time', 0)
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_file = reports_dir / "model_comparison.csv"
        df.to_csv(comparison_file, index=False)
        
        # Find best models per horizon
        best_models = {}
        for horizon in horizons:
            horizon_data = df[df['horizon'] == horizon]
            if not horizon_data.empty:
                best_idx = horizon_data['test_rmse'].idxmin()
                best_models[horizon] = horizon_data.loc[best_idx]
        
        # Generate summary report
        summary_report = {
            'evaluation_date': str(pd.Timestamp.now()),
            'total_models_evaluated': len(df),
            'horizons_evaluated': len(all_results),
            'best_models': {}
        }
        
        for horizon, best_model in best_models.items():
            summary_report['best_models'][horizon] = {
                'model_name': best_model['model'],
                'test_rmse': float(best_model['test_rmse']),
                'test_mae': float(best_model['test_mae']),
                'test_r2': float(best_model['test_r2']),
                'cv_rmse_mean': float(best_model['cv_rmse_mean']),
                'train_time': float(best_model['train_time'])
            }
        
        # Save summary
        summary_file = reports_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        # Create visualizations
        create_model_comparison_plots(df, reports_dir)
        
        logger.info("✅ Model evaluation completed")
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL EVALUATION SUMMARY")
        print("="*50)
        
        for horizon, best_model in best_models.items():
            print(f"\n{horizon} Best Model: {best_model['model']}")
            print(f"  RMSE: {best_model['test_rmse']:.4f}")
            print(f"  MAE:  {best_model['test_mae']:.4f}")
            print(f"  R²:   {best_model['test_r2']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {e}")
        return False

def create_model_comparison_plots(df: pd.DataFrame, output_dir: Path):
    """
    Create comparison plots for model evaluation.
    """
    try:
        plt.style.use('default')
        
        # 1. RMSE comparison across horizons
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='horizon', y='test_rmse', hue='model')
        plt.title('Model Performance Comparison (RMSE)', fontsize=14, fontweight='bold')
        plt.ylabel('Test RMSE')
        plt.xlabel('Prediction Horizon')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'rmse_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. R² comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='horizon', y='test_r2', hue='model')
        plt.title('Model Performance Comparison (R²)', fontsize=14, fontweight='bold')
        plt.ylabel('Test R²')
        plt.xlabel('Prediction Horizon')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'r2_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Training time comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='horizon', y='train_time', hue='model')
        plt.title('Model Training Time Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Training Time (seconds)')
        plt.xlabel('Prediction Horizon')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Cross-validation stability
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='horizon', y='cv_rmse_std', hue='model')
        plt.title('Model Stability (CV RMSE Standard Deviation)', fontsize=14, fontweight='bold')
        plt.ylabel('CV RMSE Std Dev')
        plt.xlabel('Prediction Horizon')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'stability_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📈 Comparison plots created")
        
    except Exception as e:
        logger.warning(f"Could not create plots: {e}")

def main():
    success = evaluate_models()
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
