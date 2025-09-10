# AQI Prediction Pipeline

Real-time Air Quality Index prediction system for Karachi, Pakistan. Features automated data collection, ML Models, and a dashboard for AQI viewing.

## 🌐 Live Dashboard

**[View Live Dashboard →](https://myaqipredictionapp.streamlit.app/)**

Access the real-time AQI dashboard showing current air quality and 72-hour forecasts for Karachi.

## Overview

This pipeline provides 72-hour AQI forecasts using machine learning models trained on weather pollutant data. The system automatically collects data, trains models, and serves predictions through a webpage.

**Key Features:**
- Real-time AQI monitoring and 3-day forecasts
- Automated data collection from OpenWeatherMap API
- Multi-horizon forecasting (24h, 48h, 72h predictions)
- Streamlit dashboard
- Model serving with Hopsworks deployment
- Automated CI/CD pipeline for model retraining

## Architecture

The system consists of several components:

**Data Pipeline:**
- Hourly feature collection from OpenWeatherMap
- Feature engineering with lag and future variables 
- Automated target backfilling for training data

**ML Pipeline:**
- Multiple model training (Random Forest, Decision Tree, etc.)
- Best model selection based on RMSE
- Three separate models for 24h, 48h, and 72h

**Serving & Dashboard:**
- Models deployed on Hopsworks
- Streamlit web app for visualization
- Current AQI display with 3-day hourly forecasts

## Getting Started

### Quick Access

**🚀 [Try the Live Dashboard](https://myaqipredictionapp.streamlit.app/)** - No installation required!

### Local Development

#### Prerequisites

- Python 3.9+
- Hopsworks account with API key
- OpenWeatherMap API key

#### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ahtishamu/aqi-pipeline.git
   cd aqi-pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install -r model_training_requirements.txt
   pip install -r dashboard/requirements.txt
   ```

3. **Configure environment**
   
   Create `dashboard/config.py`:
   ```python
   HOPSWORKS_API_KEY = "your_hopsworks_api_key"
   HOPSWORKS_PROJECT_NAME = "your_project_name"
   OWM_API_KEY = "your_openweathermap_key"
   ```

4. **Run the dashboard**
   ```bash
   cd dashboard
   streamlit run simple_app.py
   ```

### Docker Deployment

```bash
# Build and run with Docker
docker build -t aqi-dashboard .
docker run -p 8501:8501 aqi-dashboard
```

## Usage


### Automated Pipeline

The system includes automated workflows:

**Daily Model Training** (GitHub Actions):
- Checks data freshness for each model horizon
- Retrains models when sufficient new data is available  
- Updates model registry with best performing models

**Manual Training Trigger**:
- Can force retrain specific horizons via GitHub Actions
- Useful for testing or emergency model updates

## Technical Details

### Model Training

**Algorithms tested:**
- Random Forest Regressor
- Decision Tree Regressor  
- Gradient Boosting Regressor
- Linear Regression

**Model selection:**
- Best model selected by lowest RMSE
- Separate models for each prediction horizon

### Model Serving

Models are deployed on Hopsworks and accessed via:
- Model registry API for inference
- Predictor scripts for containerized serving
- Connection pooling for dashboard performance