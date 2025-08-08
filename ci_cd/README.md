# CI/CD Pipeline for AQI Model Training

This directory contains the CI/CD pipeline scripts for automated AQI model training.

## 🚀 Features

- **Daily Automated Training**: Runs at 2 AM UTC daily
- **Data Freshness Checks**: Only trains when fresh data is available
- **Multi-Horizon Models**: Trains 24h, 48h, and 72h prediction models
- **Model Evaluation**: Compares all models and selects the best
- **Automated Deployment**: Updates Hopsworks Model Registry
- **Comprehensive Reporting**: Generates HTML reports with visualizations
- **Manual Triggers**: On-demand training via GitHub Actions

## 📁 Files

### Workflow Files (`.github/workflows/`)
- `daily-model-training.yml`: Main daily training pipeline
- `manual-training.yml`: Manual training trigger

### Pipeline Scripts (`ci_cd/`)
- `check_data_freshness.py`: Validates data recency and quality
- `train_model_pipeline.py`: Core model training pipeline
- `update_model_registry.py`: Updates Hopsworks Model Registry
- `evaluate_models.py`: Model evaluation and comparison
- `deploy_models.py`: Production model deployment
- `generate_model_report.py`: HTML report generation
- `setup_ci_cd.py`: Setup script for initialization

## ⚙️ Setup

1. **Run Setup Script**:
   ```bash
   python ci_cd/setup_ci_cd.py
   ```

2. **Add GitHub Secrets**:
   - `HOPSWORKS_API_KEY`: Your Hopsworks API key
   - `OWM_API_KEY`: Your OpenWeatherMap API key
   - `SLACK_WEBHOOK_URL`: (Optional) For notifications

3. **Commit and Push**:
   ```bash
   git add .github/ ci_cd/
   git commit -m "Add CI/CD pipeline for AQI model training"
   git push
   ```

## 🔄 Pipeline Flow

### Daily Training Pipeline
1. **Data Freshness Check**: Validates data quality and recency
2. **Model Training**: Trains models for all horizons (24h, 48h, 72h)
3. **Model Evaluation**: Compares performance across models
4. **Model Registry Update**: Registers best models in Hopsworks
5. **Report Generation**: Creates HTML report with visualizations
6. **Deployment**: Deploys best models to production
7. **Notification**: Sends status updates (if configured)

### Manual Training
- Trigger via GitHub Actions UI
- Choose specific horizon or train all
- Option to force retrain even with stale data
- Select specific model types to train

## 📊 Outputs

### Model Artifacts
- `model_results/aqi_{horizon}/`: Training results and models
- `models/`: Serialized model files
- `reports/`: HTML reports and visualizations

### Reports
- Model comparison tables (CSV)
- Performance visualizations (PNG)
- Comprehensive HTML reports
- Deployment summaries (JSON)

## 🛠️ Customization

### Training Schedule
Edit the cron expression in `daily-model-training.yml`:
```yaml
schedule:
  - cron: '0 2 * * *'  # 2 AM UTC daily
```

### Model Selection
Modify `train_model_pipeline.py` to include/exclude specific models:
```python
# Add or remove model types in ModelTrainer
trainer = ModelTrainer(...)
```

### Evaluation Metrics
Update `evaluate_models.py` to change comparison criteria:
```python
# Change sorting key for best model selection
best_idx = horizon_data['test_rmse'].idxmin()  # Use different metric
```

## 🔧 Troubleshooting

### Common Issues

1. **Data Freshness Failures**:
   - Check Hopsworks connection
   - Verify feature group exists and has recent data
   - Review hourly update scripts

2. **Training Failures**:
   - Check dependency installation
   - Verify API keys are correctly set
   - Review training data quality

3. **Deployment Issues**:
   - Ensure Hopsworks Model Registry access
   - Check model file paths and formats
   - Verify deployment configuration

### Debugging
- Check GitHub Actions logs for detailed error messages
- Enable debug logging in individual scripts
- Use manual workflow with specific parameters for testing

## 📈 Monitoring

### Key Metrics to Monitor
- Training success rate
- Model performance trends (RMSE, MAE, R²)
- Data freshness and availability
- Pipeline execution time

### Alerts
Configure Slack notifications for:
- Training failures
- Significant performance degradation
- Data freshness issues
- Deployment problems

## 🔒 Security

- API keys stored as GitHub Secrets
- No sensitive data in logs
- Artifact retention policies applied
- Limited scope for deployment permissions

## 📞 Support

For issues with the CI/CD pipeline:
1. Check GitHub Actions logs
2. Review individual script outputs
3. Validate Hopsworks connectivity
4. Verify data quality and availability
