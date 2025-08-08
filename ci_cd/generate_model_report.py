#!/usr/bin/env python3
"""
Generate comprehensive model report
"""
import os
import json
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_model_report():
    """
    Generate a comprehensive HTML report of model training and evaluation.
    """
    try:
        logger.info("📄 Generating model report")
        
        project_root = Path(__file__).parent.parent
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Load data
        summary_file = reports_dir / "evaluation_summary.json"
        comparison_file = reports_dir / "model_comparison.csv"
        
        if not summary_file.exists():
            logger.error("Evaluation summary not found")
            return False
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        if comparison_file.exists():
            comparison_df = pd.read_csv(comparison_file)
        else:
            comparison_df = pd.DataFrame()
        
        # Generate HTML report
        html_content = generate_html_report(summary, comparison_df)
        
        # Save report
        report_file = reports_dir / f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        # Also save as latest
        latest_report = reports_dir / "latest_model_report.html"
        with open(latest_report, 'w') as f:
            f.write(html_content)
        
        logger.info(f"✅ Report generated: {report_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        return False

def generate_html_report(summary: dict, comparison_df: pd.DataFrame) -> str:
    """
    Generate HTML content for the model report.
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AQI Model Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
            .metric-card {{ background-color: #e3f2fd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .best-model {{ background-color: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4caf50; }}
            .warning {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .metric {{ font-weight: bold; color: #333; }}
            .timestamp {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌍 AQI Model Training Report</h1>
            <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p>Evaluation Date: {summary.get('evaluation_date', 'Unknown')}</p>
        </div>
        
        <h2>📊 Training Summary</h2>
        <div class="metric-card">
            <p><span class="metric">Total Models Evaluated:</span> {summary.get('total_models_evaluated', 0)}</p>
            <p><span class="metric">Horizons Evaluated:</span> {summary.get('horizons_evaluated', 0)}</p>
        </div>
        
        <h2>🏆 Best Models by Horizon</h2>
    """
    
    # Add best models section
    best_models = summary.get('best_models', {})
    for horizon, model_info in best_models.items():
        html += f"""
        <div class="best-model">
            <h3>{horizon} Prediction</h3>
            <p><span class="metric">Best Model:</span> {model_info.get('model_name', 'Unknown')}</p>
            <p><span class="metric">Test RMSE:</span> {model_info.get('test_rmse', 0):.4f}</p>
            <p><span class="metric">Test MAE:</span> {model_info.get('test_mae', 0):.4f}</p>
            <p><span class="metric">Test R²:</span> {model_info.get('test_r2', 0):.4f}</p>
            <p><span class="metric">CV RMSE Mean:</span> {model_info.get('cv_rmse_mean', 0):.4f}</p>
            <p><span class="metric">Training Time:</span> {model_info.get('train_time', 0):.2f} seconds</p>
        </div>
        """
    
    # Add detailed comparison table if available
    if not comparison_df.empty:
        html += """
        <h2>📋 Detailed Model Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Horizon</th>
                    <th>Model</th>
                    <th>Test RMSE</th>
                    <th>Test MAE</th>
                    <th>Test R²</th>
                    <th>CV RMSE Mean</th>
                    <th>CV RMSE Std</th>
                    <th>Training Time (s)</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for _, row in comparison_df.iterrows():
            html += f"""
                <tr>
                    <td>{row['horizon']}</td>
                    <td>{row['model']}</td>
                    <td>{row['test_rmse']:.4f}</td>
                    <td>{row['test_mae']:.4f}</td>
                    <td>{row['test_r2']:.4f}</td>
                    <td>{row['cv_rmse_mean']:.4f}</td>
                    <td>{row['cv_rmse_std']:.4f}</td>
                    <td>{row['train_time']:.2f}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
    
    # Add visualizations section
    html += """
        <h2>📈 Performance Visualizations</h2>
        <p>The following charts are available in the reports directory:</p>
        <ul>
            <li><strong>rmse_comparison.png</strong> - RMSE comparison across models and horizons</li>
            <li><strong>r2_comparison.png</strong> - R² comparison across models and horizons</li>
            <li><strong>training_time_comparison.png</strong> - Training time comparison</li>
            <li><strong>stability_comparison.png</strong> - Model stability (CV standard deviation)</li>
        </ul>
        
        <h2>🔄 Next Steps</h2>
        <div class="metric-card">
            <ul>
                <li>Review model performance metrics and select best models for deployment</li>
                <li>Monitor model performance in production</li>
                <li>Consider retraining if performance degrades</li>
                <li>Evaluate feature importance and consider feature engineering</li>
            </ul>
        </div>
        
        <h2>📞 Support</h2>
        <div class="metric-card">
            <p>For questions or issues with the AQI prediction models:</p>
            <ul>
                <li>Check the model training logs in the CI/CD pipeline</li>
                <li>Review the feature data quality in Hopsworks</li>
                <li>Validate the model predictions against known good data</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html

def main():
    success = generate_model_report()
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
