from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import hopsworks
import joblib
import pandas as pd
from functools import lru_cache

app = FastAPI(title="AQI Prediction API", version="1.0")

HORIZONS = ["24h", "48h", "72h"]

class PredictionResponse(BaseModel):
    horizon: str
    prediction: float

class MultiPredictionResponse(BaseModel):
    predictions: dict

@lru_cache(maxsize=1)
def _get_project():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY not set")
    return hopsworks.login(api_key_value=api_key)

@lru_cache(maxsize=4)
def _get_model(horizon: str):
    project = _get_project()
    mr = project.get_model_registry()
    name = f"aqi_prediction_{horizon}"
    models = mr.get_models(name)
    if not models:
        raise RuntimeError(f"No model found for {horizon}")
    model = models[0]
    local_dir = model.download()
    # Find first pkl
    import pathlib
    pkls = list(pathlib.Path(local_dir).rglob('*.pkl'))
    if not pkls:
        raise RuntimeError("Model artifact pkl not found")
    return joblib.load(pkls[0])

def _latest_feature_row():
    project = _get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=2)
    df = fg.read()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    if df.empty:
        raise RuntimeError("Feature group empty")
    # Build derived features like training (basic subset)
    if 'aqi_lag1' not in df.columns:
        df['aqi_lag1'] = df['aqi'].shift(1)
    if 'aqi_change_rate' not in df.columns:
        df['aqi_change_rate'] = df['aqi'] - df['aqi_lag1']
    row = df.iloc[-1:].copy()
    t = row['time'].iloc[0]
    row['hour'] = t.hour
    row['day'] = t.day
    row['month'] = t.month
    # Select feature columns (exclude targets and raw time)
    drop_cols = [c for c in ['time','aqi_t_24h','aqi_t_48h','aqi_t_72h'] if c in row.columns]
    features = row.drop(columns=drop_cols)
    return features

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict/{horizon}", response_model=PredictionResponse)
def predict_horizon(horizon: str):
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail="Invalid horizon")
    model = _get_model(horizon)
    x = _latest_feature_row()
    # Align columns if model has attribute feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        cols = model.feature_names_in_
        x = x.reindex(columns=cols, fill_value=0)
    pred = float(model.predict(x)[0])
    return PredictionResponse(horizon=horizon, prediction=pred)

@app.get("/predict", response_model=MultiPredictionResponse)
def predict_all():
    x = _latest_feature_row()
    results = {}
    for h in HORIZONS:
        model = _get_model(h)
        x_use = x
        if hasattr(model, 'feature_names_in_'):
            x_use = x.reindex(columns=model.feature_names_in_, fill_value=0)
        results[h] = float(model.predict(x_use)[0])
    return MultiPredictionResponse(predictions=results)
