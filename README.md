# 🌍 AQI Prediction Pipeline

Advanced Air Quality Index prediction system using multiple forecasting models, featuring automated data collection, ML training, and hazard alerting.

## 🎯 Project Overview

This pipeline predicts AQI levels for Karachi, Pakistan using:
- **Statistical Models:** Linear, Ridge, Lasso regression
- **Ensemble Methods:** Random Forest, Gradient Boosting
- **Deep Learning:** Neural Networks (TensorFlow)
- **Time Series:** Multi-horizon forecasting (24h, 48h, 72h)

## 🚀 Features

### ✅ **Guidelines Implementation**
- **📊 EDA Analysis:** Comprehensive exploratory data analysis with trend identification
- **🤖 Multiple Models:** Statistical modeling to deep learning approaches
- **🐳 Containerized:** Docker and Docker Compose deployment
- **🔍 Explainability:** SHAP and LIME feature importance analysis
- **🚨 Hazard Alerts:** Automated notifications for dangerous AQI levels

### 🔄 **Automation**
- **Hourly:** Real-time feature collection from OpenWeatherMap
- **Daily:** Target feature backfilling and model retraining
- **Weekly:** Smart model retraining with data freshness checks
- **CI/CD:** GitHub Actions for automated ML pipeline

### 📈 **MLOps**
- **Feature Store:** Hopsworks for feature management
- **Model Registry:** Automated model versioning and deployment
- **Model Monitoring:** Performance tracking and drift detection
- **Experiment Tracking:** Comprehensive model comparison

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Sources   │────│ Feature Store   │────│ Model Training  │
│                 │    │                 │    │                 │
│ • OpenWeatherMap│    │ • Hopsworks     │    │ • Multiple      │
│ • Real-time API │    │ • Time Series   │    │   Algorithms    │
│ • Historical    │    │ • Features      │    │ • Auto Selection│
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Automation    │    │ Model Registry  │    │ Alert System    │
│                 │    │                 │    │                 │
│ • Hourly Update │    │ • Versioning    │    │ • Email/Slack   │
│ • Daily Retrain │    │ • A/B Testing   │    │ • AQI Monitoring│
│ • CI/CD Pipeline│    │ • Deployment    │    │ • Health Alerts │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Installation

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (for containerized deployment)
- Hopsworks account
- OpenWeatherMap API key

### Quick Start

1. **Clone Repository**
   ```bash
   git clone https://github.com/ahtishamu/aqi-pipeline.git
   cd aqi-pipeline
   ```

2. **Environment Setup**
   ```bash
   # Install dependencies
   pip install -r model_training_requirements.txt
   
   # Set environment variables
   export HOPSWORKS_API_KEY="your_hopsworks_key"
   export OWM_API_KEY="your_openweathermap_key"
   ```

3. **Docker Deployment**
   ```bash
   # Create .env file
   echo "HOPSWORKS_API_KEY=your_key" > .env
   echo "OWM_API_KEY=your_key" >> .env
   
   # Deploy with Docker Compose
   docker-compose up -d
   ```

## 📊 Usage

### **1. Exploratory Data Analysis**
```bash
python eda_analysis.py
```
Generates comprehensive EDA including:
- Temporal trends and seasonal patterns
- Pollutant correlations
- Pollution event analysis
- Statistical summaries and visualizations

### **2. Model Training**
```bash
# Train specific horizon
python ci_cd/train_model_pipeline.py --horizon 24h

# Train all models
python ci_cd/train_model_pipeline.py --horizon all
```

### **3. Model Explainability**
```bash
# Generate SHAP and LIME explanations
python model_explainability.py --horizon 24h

# Analyze all models
python model_explainability.py --horizon all
```

### **4. Alert Monitoring**
```bash
# Check current AQI and send alerts if needed
python aqi_alert_system.py
```

## 🤖 Models & Performance

| Model | 24h RMSE | 48h RMSE | 72h RMSE | Features |
|-------|----------|----------|----------|----------|
| **Random Forest** | **0.495** | **0.447** | TBD | Best overall performance |
| Gradient Boosting | 0.513 | 0.495 | TBD | Strong ensemble method |
| Neural Network | 0.596 | 0.594 | TBD | Deep learning approach |
| Ridge Regression | 0.680 | 0.666 | TBD | Regularized linear model |
| Linear Regression | 0.680 | 0.663 | TBD | Baseline model |
| Lasso Regression | 0.696 | 0.682 | TBD | Sparse feature selection |

## 🔍 Model Explainability

### **SHAP Analysis**
- Global feature importance ranking
- Feature interaction effects
- Individual prediction explanations
- Waterfall plots for decision transparency

### **LIME Analysis**
- Local interpretable explanations
- Instance-level feature contributions
- Model-agnostic explanations
- HTML reports for detailed analysis

## 🚨 Alert System

### **Alert Levels**
- **Good (AQI 1-2):** No alerts
- **Moderate (AQI 3):** Sensitive group warnings
- **Poor (AQI 4):** Public health warnings
- **Very Poor (AQI 5):** Emergency alerts

### **Notification Channels**
- **Email:** Detailed alerts with recommendations
- **Slack:** Real-time team notifications
- **Logs:** Persistent alert history

### **Alert Triggers**
- AQI level thresholds (4+ triggers warnings)
- Rapid deterioration detection
- Extended pollution episodes
- Custom threshold configurations

## 🐳 Docker Deployment

### **Services**
- **aqi-hourly:** Real-time feature collection
- **aqi-daily:** Daily target updates
- **aqi-training:** Weekly model retraining
- **aqi-alerts:** Continuous hazard monitoring

### **Configuration**
```bash
# Environment variables in .env file
HOPSWORKS_API_KEY=your_hopsworks_key
OWM_API_KEY=your_openweathermap_key
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
ALERT_RECIPIENTS=recipient1@email.com,recipient2@email.com
SLACK_WEBHOOK_URL=your_slack_webhook
```

## 📁 Project Structure

```
aqi-pipeline/
├── 📊 eda_analysis.py              # Exploratory data analysis
├── 🔍 model_explainability.py     # SHAP/LIME explanations
├── 🚨 aqi_alert_system.py         # Hazard alert system
├── 🐳 Dockerfile                   # Container configuration
├── 🐳 docker-compose.yml          # Multi-service deployment
│
├── model_training/                 # ML training package
│   ├── data_loader.py             # Hopsworks integration
│   ├── model_trainer.py           # Training orchestration
│   ├── models/                    # Model implementations
│   └── evaluation/                # Metrics and evaluation
│
├── ci_cd/                         # CI/CD automation
│   ├── train_model_pipeline.py    # Training pipeline
│   ├── check_data_freshness.py    # Smart retraining logic
│   ├── evaluate_models.py         # Model evaluation
│   └── generate_model_report.py   # Reporting system
│
├── .github/workflows/             # GitHub Actions
│   ├── daily-model-training.yml   # Automated training
│   ├── manual-training.yml        # Manual triggers
│   ├── aqi_hourly.yml            # Hourly automation
│   └── aqi_daily_targets.yml     # Daily automation
│
├── 🔄 update_features_hourly.py   # Real-time data collection
├── 📅 update_targets_daily.py     # Target feature updates
└── 📊 model_results/              # Training outputs
```

## 🚀 CI/CD Pipeline

### **Automated Workflows**
- **Hourly:** Feature collection and storage
- **Daily:** Target backfilling and data quality checks
- **Weekly:** Smart model retraining based on data freshness
- **Manual:** On-demand training with custom parameters

### **Quality Gates**
- Data freshness validation
- Model performance thresholds
- Feature completeness checks
- Automated testing and validation

## 📈 Monitoring & Observability

### **Model Performance**
- Cross-validation metrics (RMSE, MAE, R²)
- Feature importance tracking
- Prediction confidence intervals
- Model drift detection

### **Data Quality**
- Missing data monitoring
- Feature distribution tracking
- Outlier detection
- Data freshness validation

### **System Health**
- API response monitoring
- Pipeline execution tracking
- Error rate monitoring
- Resource usage metrics

## 🔧 Configuration

### **Environment Variables**
```bash
# Required
HOPSWORKS_API_KEY=your_hopsworks_api_key
OWM_API_KEY=your_openweathermap_key

# Optional - Alerts
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
ALERT_RECIPIENTS=email1@domain.com,email2@domain.com
SLACK_WEBHOOK_URL=your_slack_webhook_url

# Optional - SMTP
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenWeatherMap** for air quality data
- **Hopsworks** for feature store platform
- **Scikit-learn** for machine learning algorithms
- **SHAP/LIME** for model explainability
- **TensorFlow** for deep learning capabilities

---

### 📞 Support

For questions or issues:
- 📧 Email: ahtishamu@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/ahtishamu/aqi-pipeline/issues)
- 📖 Docs: [Project Wiki](https://github.com/ahtishamu/aqi-pipeline/wiki)

**Built with ❤️ for better air quality monitoring**