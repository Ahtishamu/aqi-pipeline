
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import os
from typing import Optional, Dict, List, Tuple
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from config import DashboardConfig

# Configure page
st.set_page_config(
    page_title=DashboardConfig.DASHBOARD_TITLE,
    layout=DashboardConfig.LAYOUT,
    page_icon=DashboardConfig.PAGE_ICON
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .aqi-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    /* Hide sidebar completely */
    .css-1d391kg {display: none;}
    .css-1lcbmhc {display: none;}
    .css-1g0d7rf {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    .sidebar {display: none;}
</style>
""", unsafe_allow_html=True)

def initialize_hopsworks():
    try:
        import hopsworks
        
        api_key = os.getenv('HOPSWORKS_API_KEY')
        if not api_key:
            st.warning("🔑 **HOPSWORKS_API_KEY not found.** Using demo mode.")
            return None, None, None
        
        try:
            project = hopsworks.login(api_key_value=api_key)
            fs = project.get_feature_store()
            mr = project.get_model_registry()
            logger.info("Connected to Hopsworks successfully!")
            return project, fs, mr
        except Exception as login_error:
            st.error(f"Failed to connect to Hopsworks: {login_error}")
            return None, None, None
            
    except ImportError:
        st.error("Hopsworks library not installed. Run: `pip install hopsworks`")
        return None, None, None

def get_current_aqi(fs) -> Optional[Dict]:
    if not fs:
        return None
        
    try:
        # Use the same connection manager as the forecast
        from connection_manager import get_connection_manager
        cm = get_connection_manager()
        
        # Get cached data (same as forecast uses)
        data = cm.get_cached_data()
        if data.empty:
            logger.error("No cached data available")
            return None
        
        # Get the most recent record (same as forecast uses)
        latest_data = data.iloc[-1]
        latest_time = pd.to_datetime(latest_data['time'])
        
        current_aqi_info = {
            'aqi': latest_data['aqi'],
            'datetime': latest_time,
            'pm2_5': latest_data.get('pm25', 0),  # Use pm25 from cached data
            'pm10': latest_data.get('pm10', 0),
            'o3': latest_data.get('o3', 0),
            'no2': latest_data.get('no2', 0),
            'so2': latest_data.get('so2', 0),
            'co': latest_data.get('co', 0)
        }
        
        logger.info(f"Current AQI: {current_aqi_info['aqi']} at {current_aqi_info['datetime']}")
        return current_aqi_info
        
    except Exception as e:
        st.error(f"Error fetching current AQI: {e}")
        return None

def get_aqi_color_and_description(aqi_value):
    if aqi_value <= 1:
        return "#00E400", "Good", "Air quality is satisfactory"
    elif aqi_value <= 2:
        return "#FFFF00", "Fair", "Air quality is acceptable"
    elif aqi_value <= 3:
        return "#FF7E00", "Moderate", "Air quality is of moderate concern"
    elif aqi_value <= 4:
        return "#FF0000", "Poor", "Air quality is unhealthy"
    else:
        return "#8F3F97", "Very Poor", "Air quality is very unhealthy"

def format_pollutant_value(value, pollutant):
    if pollutant in ['pm2_5', 'pm10', 'o3', 'no2', 'so2']:
        return f"{value:.1f} μg/m³"
    elif pollutant == 'co':
        return f"{value:.1f} mg/m³"
    else:
        return f"{value:.1f}"

def display_current_aqi(current_aqi_info):
    st.header("Current Air Quality")
    
    if not current_aqi_info:
        st.warning("⚠️ Unable to fetch current AQI data")
        return
    
    aqi_value = current_aqi_info['aqi']
    color, level, description = get_aqi_color_and_description(aqi_value)
    
    # Main AQI display
    st.markdown(f"""
    <div class="aqi-card" style="background-color: {color}; color: {'white' if aqi_value > 2 else 'black'};">
        <h1 style="margin: 0; font-size: 3em;">{int(aqi_value)}</h1>
        <h2 style="margin: 10px 0;">{level}</h2>
        <p style="margin: 0; font-size: 1.1em;">{description}</p>
        <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.8;">
            Last updated: {current_aqi_info['datetime']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pollutant breakdown
    st.subheader("🔬 Pollutant Levels")
    
    pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
    pollutant_names = {
        'pm2_5': 'PM2.5',
        'pm10': 'PM10', 
        'o3': 'Ozone',
        'no2': 'NO₂',
        'so2': 'SO₂',
        'co': 'CO'
    }
    
    cols = st.columns(3)
    for i, pollutant in enumerate(pollutants):
        col = cols[i % 3]
        value = current_aqi_info.get(pollutant, 0)
        formatted_value = format_pollutant_value(value, pollutant)
        
        with col:
            st.metric(
                label=pollutant_names[pollutant],
                value=formatted_value
            )

def display_forecast():
    """Display AQI forecast using enhanced true sequential predictor."""
    st.header("🔮 AQI Forecast")
    
    # Use enhanced true sequential AQI predictor (realistic pollutant evolution!)
    try:
        from enhanced_true_sequential_predictor import EnhancedTrueSequentialPredictor
        
        with st.spinner("🔮 Getting AQI predictions from your trained models..."):
            predictor = EnhancedTrueSequentialPredictor()
            forecast_data = predictor.get_forecast(DashboardConfig.FORECAST_HOURS)
    except Exception as e:
        st.error(f"Error with enhanced predictor: {e}")
        return
    
    if not forecast_data:
        st.warning("⚠️ Unable to generate forecast")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(forecast_data)
    
    # Detailed hourly chart
    st.subheader("72-Hour AQI Forecast")
    
    # Create interactive chart
    fig = go.Figure()
    
    # Add AQI line
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['aqi'],
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6),
        hovertemplate=(
            '<b>%{x}</b><br>' +
            'AQI: %{y:.1f}<br>' +
            'PM2.5: %{customdata[0]:.1f} μg/m³<br>' +
            'PM10: %{customdata[1]:.1f} μg/m³<br>' +
            'O₃: %{customdata[2]:.1f} μg/m³<br>' +
            'NO₂: %{customdata[3]:.1f} μg/m³<br>' +
            'SO₂: %{customdata[4]:.1f} μg/m³<br>' +
            'CO: %{customdata[5]:.1f} mg/m³' +
            '<extra></extra>'
        ),
        customdata=df[['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']].values
    ))
    
    # Add AQI level zones
    fig.add_hline(y=1, line_dash="dash", line_color="green", 
                  annotation_text="Good", annotation_position="left")
    fig.add_hline(y=2, line_dash="dash", line_color="yellow", 
                  annotation_text="Fair", annotation_position="left")
    fig.add_hline(y=3, line_dash="dash", line_color="orange", 
                  annotation_text="Moderate", annotation_position="left")
    fig.add_hline(y=4, line_dash="dash", line_color="red", 
                  annotation_text="Poor", annotation_position="left")
    fig.add_hline(y=5, line_dash="dash", line_color="purple", 
                  annotation_text="Very Poor", annotation_position="left")
    
    fig.update_layout(
        title="Hourly AQI Predictions with Pollutant Details",
        xaxis_title="Time",
        yaxis_title="AQI (OpenWeather Scale 1-5)",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxes(range=[0, 6])  # Fixed: update_yaxes not update_yaxis
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed data table (expandable)
    with st.expander("View Detailed Forecast Data"):
        # Format the dataframe for display
        display_df = df.copy()
        display_df['AQI'] = display_df['aqi'].round(1)
        display_df['PM2.5'] = display_df['pm2_5'].round(1)
        display_df['PM10'] = display_df['pm10'].round(1)
        display_df['O₃'] = display_df['o3'].round(1)
        display_df['NO₂'] = display_df['no2'].round(1)
        display_df['SO₂'] = display_df['so2'].round(1)
        display_df['CO'] = display_df['co'].round(1)
        
        # Select columns for display
        display_cols = ['datetime', 'AQI', 'PM2.5', 'PM10', 'O₃', 'NO₂', 'SO₂', 'CO']
        st.dataframe(display_df[display_cols], use_container_width=True)

def main():
    """Main dashboard application."""
    st.title(f"{DashboardConfig.PAGE_ICON} {DashboardConfig.DASHBOARD_TITLE}")
    st.markdown("Real-time air quality monitoring with ML-powered forecasts")
    
    # Initialize Hopsworks
    project, fs, mr = initialize_hopsworks()
    
    # Get current AQI
    current_aqi_info = get_current_aqi(fs)
    
    # Display current AQI
    display_current_aqi(current_aqi_info)
    
    # Add some spacing
    st.markdown("---")
    
    # Display forecast
    display_forecast()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; padding: 20px;">
        <p>🤖 Powered by machine learning models trained on real air quality data</p>
        <p>Data from OpenWeather API | Models deployed on Hopsworks</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
