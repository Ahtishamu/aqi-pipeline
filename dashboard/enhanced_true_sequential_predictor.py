
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from connection_manager import get_connection_manager
from aqi_validation import calculate_aqi_from_pollutants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTrueSequentialPredictor:
    
    def __init__(self):
        self.connection_manager = get_connection_manager()
        
    def create_features_from_data(self, aqi, pm25, pm10, o3, no2, so2, co, hour, day, month):
        return [
            float(aqi), float(pm25), float(pm10), float(o3), float(no2), float(so2), float(co),
            0.01, 0.0, float(hour), float(day), float(month),
            float(aqi), 0.0, float(aqi), float(aqi), float(aqi)
        ]
    
    def predict_with_model(self, features, model_name):
        try:
            deployments = self.connection_manager.get_deployments()
            if model_name not in deployments:
                logger.warning(f"⚠️ Model {model_name} not found, using fallback")
                return None
                
            deployment = deployments[model_name]
            payload = {'instances': [features]}
            
            # Use the deployment object's predict method (like batch predictor)
            result = deployment.predict(payload)
            
            if result and 'predictions' in result and len(result['predictions']) > 0:
                return float(result['predictions'][0])
            else:
                logger.warning(f"⚠️ No predictions in response from {model_name}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error calling model {model_name}: {e}")
            return None
    
    def evolve_pollutants_based_on_aqi(self, current_data, predicted_aqi, hour):
        # Get current values
        current_aqi = current_data['aqi']
        aqi_change = predicted_aqi - current_aqi
        
        # Base evolution (small natural variations)
        np.random.seed(hour)  # For reproducible results
        
        # Evolve pollutants based on AQI trend and time patterns
        hour_of_day = (current_data['timestamp'] + timedelta(hours=hour)).hour
        
        # Time-based patterns (rush hours, etc.)
        rush_hour_factor = 1.2 if hour_of_day in [7, 8, 17, 18, 19] else 1.0
        night_factor = 0.9 if 22 <= hour_of_day or hour_of_day <= 6 else 1.0
        
        # Calculate evolution factors
        base_change = aqi_change * 0.3  # Moderate influence from AQI prediction
        time_factor = rush_hour_factor * night_factor
        
        # Evolve each pollutant with realistic bounds
        evolved = {
            'pm25': max(5.0, min(50.0, 
                current_data['pm25'] + base_change * 2.0 + np.random.normal(0, 0.5))),
            'pm10': max(10.0, min(100.0,
                current_data['pm10'] + base_change * 3.0 + np.random.normal(0, 1.0))),
            'o3': max(15.0, min(150.0,
                current_data['o3'] + base_change * 4.0 * time_factor + np.random.normal(0, 2.0))),
            'no2': max(0.0, min(50.0,
                current_data['no2'] + base_change * 0.5 * rush_hour_factor + np.random.normal(0, 0.2))),
            'so2': max(0.0, min(20.0,
                current_data['so2'] + np.random.normal(0, 0.1))),
            'co': max(50.0, min(5000.0,
                current_data['co'] + base_change * 10.0 * rush_hour_factor + np.random.normal(0, 10.0)))
        }
        
        return evolved
    
    def predict_sequential_aqi(self, hours=72):
        logger.info(f" Starting enhanced sequential AQI prediction for {hours} hours...")
        
        try:
            # Get current data
            df = self.connection_manager.get_cached_data()
            if df.empty:
                logger.error("No feature data available")
                return []
            
            # Get latest record
            latest = df.iloc[-1]
            current_time = pd.to_datetime(latest['time']).replace(tzinfo=pytz.UTC)
            
            # Initialize current state
            current_data = {
                'timestamp': current_time,
                'aqi': float(latest['aqi']),
                'pm25': float(latest['pm25']),
                'pm10': float(latest['pm10']),
                'o3': float(latest['o3']),
                'no2': float(latest['no2']),
                'so2': float(latest['so2']),
                'co': float(latest['co'])
            }
            
            logger.info(f" Starting from: {current_time}")
            logger.info(f" Initial: AQI={current_data['aqi']}, PM2.5={current_data['pm25']:.1f}")
            
            forecasts = []
            
            # Sequential prediction for each hour
            for hour in range(1, hours + 1):
                target_time = current_time + timedelta(hours=hour)
                
                # Determine which model to use
                if hour <= 24:
                    model_name = "aqiprediction24h"
                elif hour <= 48:
                    model_name = "aqiprediction48h"
                else:
                    model_name = "aqiprediction72h"
                
                # Create features from current data
                features = self.create_features_from_data(
                    current_data['aqi'],
                    current_data['pm25'],
                    current_data['pm10'],
                    current_data['o3'],
                    current_data['no2'],
                    current_data['so2'],
                    current_data['co'],
                    target_time.hour,
                    target_time.day,
                    target_time.month
                )
                
                predicted_aqi_ml = self.predict_with_model(features, model_name)
                
                # Evolve pollutants based on ML prediction
                evolved_pollutants = self.evolve_pollutants_based_on_aqi(
                    current_data, predicted_aqi_ml, hour
                )
                
                # Calculate actual AQI from evolved pollutants
                calculated_aqi, main_pollutant = calculate_aqi_from_pollutants(
                    evolved_pollutants['pm25'],
                    evolved_pollutants['pm10'],
                    evolved_pollutants['o3'],
                    evolved_pollutants['no2'],
                    evolved_pollutants['so2'],
                    evolved_pollutants['co']
                )
                
                # Create forecast point
                forecast_point = {
                    'time': target_time,
                    'hour': hour,
                    'aqi': calculated_aqi,
                    'pm25': round(evolved_pollutants['pm25'], 1),
                    'pm10': round(evolved_pollutants['pm10'], 1),
                    'o3': round(evolved_pollutants['o3'], 1),
                    'no2': round(evolved_pollutants['no2'], 2),
                    'so2': round(evolved_pollutants['so2'], 2),
                    'co': round(evolved_pollutants['co'], 1),
                    'main_pollutant': main_pollutant,
                    'method': 'enhanced_sequential',
                    'ml_guidance': predicted_aqi_ml
                }
                
                forecasts.append(forecast_point)
                
                # Update current_data for next iteration
                current_data.update({
                    'timestamp': target_time,
                    'aqi': calculated_aqi,
                    'pm25': evolved_pollutants['pm25'],
                    'pm10': evolved_pollutants['pm10'],
                    'o3': evolved_pollutants['o3'],
                    'no2': evolved_pollutants['no2'],
                    'so2': evolved_pollutants['so2'],
                    'co': evolved_pollutants['co']
                })
                
                # Log progress
                if hour <= 5 or hour in [24, 25, 48, 49, 72] or hour % 24 == 0:
                    logger.info(f"🔮 Hour {hour}: AQI {calculated_aqi}, Model: {model_name}")
            
            logger.info(f"Generated {len(forecasts)} enhanced sequential predictions")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error in enhanced sequential prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_forecast(self, hours=72):
        forecasts = self.predict_sequential_aqi(hours)
        
        # Convert to dashboard format
        dashboard_data = []
        for f in forecasts:
            dashboard_data.append({
                'datetime': f['time'],  # Use 'time' key from forecast data
                'aqi': f['aqi'],
                'pm2_5': f['pm25'],  # Dashboard expects pm2_5, not pm25
                'pm10': f['pm10'],
                'o3': f['o3'],
                'no2': f['no2'],
                'so2': f['so2'],
                'co': f['co']
            })
        
        return dashboard_data

def test_enhanced_predictor():
    predictor = EnhancedTrueSequentialPredictor()
    
    # Test with first 12 hours
    forecasts = predictor.predict_sequential_aqi(hours=12)
    
    if forecasts:
        logger.info(f"Generated {len(forecasts)} forecasts")
    else:
        logger.warning("No forecasts generated")

if __name__ == "__main__":
    test_enhanced_predictor()
