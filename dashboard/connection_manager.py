
import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import streamlit as st
from config import DashboardConfig

logger = logging.getLogger(__name__)

class HopsworksConnectionManager:
    
    _instance = None
    _project = None
    _fs = None
    _mr = None
    _ms = None
    _deployments = None
    _feature_data = None
    _deployments_dict = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_connection(self):
        if self._project is None:
            try:
                logger.info("Establishing Hopsworks connection...")
                import hopsworks
                
                self._project = hopsworks.login(
                    api_key_value=DashboardConfig.HOPSWORKS_API_KEY,
                    project=DashboardConfig.HOPSWORKS_PROJECT
                )
                
                self._fs = self._project.get_feature_store()
                self._mr = self._project.get_model_registry()
                self._ms = self._project.get_model_serving()
                
                logger.info("Hopsworks connection established")
                
            except Exception as e:
                logger.error(f"Failed to connect to Hopsworks: {e}")
                raise
        
        return self._project, self._fs, self._mr, self._ms
    
    def get_deployments(self):
        if self._deployments_dict is None:
            try:
                _, _, _, ms = self.get_connection()
                deployments = ms.get_deployments()
                
                # Create deployment lookup dictionary
                self._deployments_dict = {}
                for deployment in deployments:
                    self._deployments_dict[deployment.name] = deployment
                
                logger.info(f"Cached {len(self._deployments_dict)} model deployments")
                
            except Exception as e:
                logger.error(f"Failed to get deployments: {e}")
                self._deployments_dict = {}
        
        return self._deployments_dict
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_cached_data(_self):
        if _self._feature_data is None:
            try:
                logger.info("Reading feature store data...")
                _, fs, _, _ = _self.get_connection()
                
                fg = fs.get_feature_group(
                    DashboardConfig.FEATURE_GROUP_NAME, 
                    version=DashboardConfig.FEATURE_GROUP_VERSION
                )
                
                _self._feature_data = fg.read()
                _self._feature_data['time'] = pd.to_datetime(_self._feature_data['time'])
                _self._feature_data = _self._feature_data.sort_values('time')
                
                logger.info(f"Cached {len(_self._feature_data)} records from feature store")
                
            except Exception as e:
                logger.error(f"Failed to read feature data: {e}")
                raise
        
        return _self._feature_data
    
    def get_latest_record(self):
        data = self.get_cached_data()
        if data is not None and not data.empty:
            return data.iloc[-1]
        return None
    
    def cleanup(self):
        self._project = None
        self._fs = None
        self._mr = None
        self._ms = None
        self._deployments_dict = None
        self._feature_data = None
        logger.info("Connection manager reset")

# Singleton instance getter
@st.cache_resource
def get_connection_manager():
    return HopsworksConnectionManager()
