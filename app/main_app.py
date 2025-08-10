import streamlit as st
import requests
import os
import pandas as pd
import hopsworks

API_BASE = os.getenv('AQI_API_BASE', 'http://localhost:8000')

st.set_page_config(page_title='AQI Forecast Dashboard', layout='wide')
st.title('AQI Forecast Dashboard')

if 'predictions' not in st.session_state or st.button('Refresh'):
    try:
        resp = requests.get(f"{API_BASE}/predict")
        st.session_state['predictions'] = resp.json().get('predictions', {})
    except Exception as e:
        st.error(f"API error: {e}")

preds = st.session_state.get('predictions', {})
col1, col2, col3, col4 = st.columns(4)

# Fetch current AQI from feature store
try:
    api_key = os.getenv('HOPSWORKS_API_KEY')
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_feature_group('aqi_features', version=2)
    df = fg.read()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    current_aqi = df['aqi'].iloc[-1]
    history = df.tail(168)
except Exception as e:
    current_aqi = None
    history = pd.DataFrame()
    st.warning(f"Could not load feature data: {e}")

col1.metric('Current AQI', f"{current_aqi:.0f}" if current_aqi is not None else 'N/A')
for (h, c) in zip(['24h','48h','72h'], [col2, col3, col4]):
    if h in preds:
        c.metric(f'Forecast {h}', f"{preds[h]:.0f}")
    else:
        c.metric(f'Forecast {h}', 'N/A')

st.subheader('Recent AQI History (last 168 records)')
if not history.empty:
    st.line_chart(history.set_index('time')['aqi'])
else:
    st.info('No history available')

st.subheader('Raw Predictions JSON')
st.json(preds)
