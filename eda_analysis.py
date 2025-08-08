#!/usr/bin/env python3
"""
Exploratory Data Analysis for AQI Time Series Data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hopsworks
import os
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQIExplorer:
    """EDA toolkit for AQI time series analysis"""
    
    def __init__(self):
        self.data = None
        self.api_key = os.environ.get('HOPSWORKS_API_KEY')
        
    def load_data(self):
        """Load data from Hopsworks Feature Store"""
        try:
            project = hopsworks.login(api_key_value=self.api_key)
            fs = project.get_feature_store()
            fg = fs.get_feature_group("aqi_features", version=2)
            
            self.data = fg.read()
            self.data['time'] = pd.to_datetime(self.data['time'])
            self.data = self.data.sort_values('time').reset_index(drop=True)
            
            logger.info(f"✅ Loaded {len(self.data)} records")
            logger.info(f"📅 Date range: {self.data['time'].min()} to {self.data['time'].max()}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            
    def basic_stats(self):
        """Generate basic statistical summary"""
        if self.data is None:
            logger.error("No data loaded")
            return
            
        print("📊 BASIC STATISTICS")
        print("=" * 50)
        
        # Numerical columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        stats_df = self.data[numeric_cols].describe()
        
        print(f"\n📈 Summary Statistics:")
        print(stats_df.round(2))
        
        # Missing data analysis
        print(f"\n🔍 Missing Data Analysis:")
        missing_data = self.data.isnull().sum()
        missing_pct = (missing_data / len(self.data)) * 100
        
        for col in missing_data.index:
            if missing_data[col] > 0:
                print(f"   {col}: {missing_data[col]} ({missing_pct[col]:.1f}%)")
        
        # AQI distribution
        print(f"\n🎯 AQI Distribution:")
        if 'aqi' in self.data.columns:
            aqi_counts = self.data['aqi'].value_counts().sort_index()
            print(f"   Good (1-2): {aqi_counts.get(1, 0) + aqi_counts.get(2, 0)}")
            print(f"   Moderate (3): {aqi_counts.get(3, 0)}")
            print(f"   Poor (4): {aqi_counts.get(4, 0)}")
            print(f"   Very Poor (5): {aqi_counts.get(5, 0)}")
        
    def trend_analysis(self):
        """Analyze temporal trends and patterns"""
        if self.data is None:
            logger.error("No data loaded")
            return
            
        print("\n📈 TREND ANALYSIS")
        print("=" * 50)
        
        # Add time features
        self.data['hour'] = self.data['time'].dt.hour
        self.data['day_of_week'] = self.data['time'].dt.dayofweek
        self.data['month'] = self.data['time'].dt.month
        self.data['year'] = self.data['time'].dt.year
        
        # Hourly patterns
        if 'aqi' in self.data.columns:
            hourly_avg = self.data.groupby('hour')['aqi'].mean()
            print(f"\n⏰ Hourly AQI Patterns:")
            peak_hour = hourly_avg.idxmax()
            lowest_hour = hourly_avg.idxmin()
            print(f"   Peak pollution: {peak_hour}:00 (AQI: {hourly_avg[peak_hour]:.2f})")
            print(f"   Cleanest air: {lowest_hour}:00 (AQI: {hourly_avg[lowest_hour]:.2f})")
            
            # Weekly patterns
            weekly_avg = self.data.groupby('day_of_week')['aqi'].mean()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            print(f"\n📅 Weekly AQI Patterns:")
            for i, day in enumerate(days):
                print(f"   {day}: {weekly_avg.get(i, 0):.2f}")
            
            # Seasonal patterns
            seasonal_avg = self.data.groupby('month')['aqi'].mean()
            print(f"\n🌍 Seasonal AQI Patterns:")
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for i, month in enumerate(months, 1):
                print(f"   {month}: {seasonal_avg.get(i, 0):.2f}")
        
    def correlation_analysis(self):
        """Analyze correlations between pollutants"""
        if self.data is None:
            logger.error("No data loaded")
            return
            
        print("\n🔗 CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Select numeric pollutant columns
        pollutant_cols = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'no', 'nh3']
        available_cols = [col for col in pollutant_cols if col in self.data.columns]
        
        if len(available_cols) > 1:
            corr_matrix = self.data[available_cols].corr()
            
            print(f"\n📊 Pollutant Correlations (with AQI):")
            if 'aqi' in available_cols:
                aqi_corr = corr_matrix['aqi'].sort_values(ascending=False)
                for pollutant, correlation in aqi_corr.items():
                    if pollutant != 'aqi':
                        print(f"   {pollutant.upper()}: {correlation:.3f}")
            
            # Find strongest correlations (excluding self-correlations)
            corr_pairs = []
            for i in range(len(available_cols)):
                for j in range(i+1, len(available_cols)):
                    col1, col2 = available_cols[i], available_cols[j]
                    corr_val = corr_matrix.loc[col1, col2]
                    corr_pairs.append((abs(corr_val), col1, col2, corr_val))
            
            corr_pairs.sort(reverse=True)
            print(f"\n🔝 Top Pollutant Correlations:")
            for abs_corr, col1, col2, corr_val in corr_pairs[:5]:
                print(f"   {col1.upper()} - {col2.upper()}: {corr_val:.3f}")
    
    def pollution_events(self):
        """Identify significant pollution events"""
        if self.data is None or 'aqi' not in self.data.columns:
            logger.error("No AQI data available")
            return
            
        print("\n🚨 POLLUTION EVENTS")
        print("=" * 50)
        
        # Define thresholds
        very_poor_threshold = 4  # AQI level 4-5
        hazardous_threshold = 5  # AQI level 5
        
        # Find pollution events
        very_poor_events = self.data[self.data['aqi'] >= very_poor_threshold]
        hazardous_events = self.data[self.data['aqi'] == hazardous_threshold]
        
        print(f"\n⚠️ Pollution Event Statistics:")
        print(f"   Very Poor Air Quality (AQI 4-5): {len(very_poor_events)} records ({len(very_poor_events)/len(self.data)*100:.1f}%)")
        print(f"   Hazardous Air Quality (AQI 5): {len(hazardous_events)} records ({len(hazardous_events)/len(self.data)*100:.1f}%)")
        
        if len(very_poor_events) > 0:
            # Find longest pollution episodes
            very_poor_events = very_poor_events.sort_values('time').reset_index(drop=True)
            episodes = []
            current_episode = [very_poor_events.iloc[0]]
            
            for i in range(1, len(very_poor_events)):
                prev_time = very_poor_events.iloc[i-1]['time']
                curr_time = very_poor_events.iloc[i]['time']
                
                # If gap is more than 2 hours, start new episode
                if (curr_time - prev_time).total_seconds() > 7200:
                    episodes.append(current_episode)
                    current_episode = [very_poor_events.iloc[i]]
                else:
                    current_episode.append(very_poor_events.iloc[i])
            
            episodes.append(current_episode)
            
            # Find longest episode
            longest_episode = max(episodes, key=len)
            print(f"\n📅 Longest Pollution Episode:")
            print(f"   Duration: {len(longest_episode)} hours")
            print(f"   Start: {longest_episode[0]['time']}")
            print(f"   End: {longest_episode[-1]['time']}")
            print(f"   Peak AQI: {max(record['aqi'] for record in longest_episode)}")
    
    def generate_plots(self):
        """Generate comprehensive EDA plots"""
        if self.data is None:
            logger.error("No data loaded")
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AQI Time Series - Exploratory Data Analysis', fontsize=16)
        
        # 1. Time series plot
        if 'aqi' in self.data.columns:
            axes[0,0].plot(self.data['time'], self.data['aqi'], alpha=0.7, linewidth=0.5)
            axes[0,0].set_title('AQI Time Series')
            axes[0,0].set_ylabel('AQI Level')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. AQI distribution
        if 'aqi' in self.data.columns:
            self.data['aqi'].hist(bins=20, ax=axes[0,1], alpha=0.7)
            axes[0,1].set_title('AQI Distribution')
            axes[0,1].set_xlabel('AQI Level')
            axes[0,1].set_ylabel('Frequency')
        
        # 3. Hourly patterns
        if 'hour' in self.data.columns or 'aqi' in self.data.columns:
            if 'hour' not in self.data.columns:
                self.data['hour'] = self.data['time'].dt.hour
            hourly_avg = self.data.groupby('hour')['aqi'].mean()
            hourly_avg.plot(kind='bar', ax=axes[0,2])
            axes[0,2].set_title('Average AQI by Hour')
            axes[0,2].set_xlabel('Hour of Day')
            axes[0,2].set_ylabel('Average AQI')
        
        # 4. Pollutant correlations heatmap
        pollutant_cols = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        available_cols = [col for col in pollutant_cols if col in self.data.columns]
        if len(available_cols) > 2:
            corr_matrix = self.data[available_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=axes[1,0], square=True, fmt='.2f')
            axes[1,0].set_title('Pollutant Correlations')
        
        # 5. Monthly trends
        if 'month' in self.data.columns or 'aqi' in self.data.columns:
            if 'month' not in self.data.columns:
                self.data['month'] = self.data['time'].dt.month
            monthly_avg = self.data.groupby('month')['aqi'].mean()
            monthly_avg.plot(kind='line', marker='o', ax=axes[1,1])
            axes[1,1].set_title('Average AQI by Month')
            axes[1,1].set_xlabel('Month')
            axes[1,1].set_ylabel('Average AQI')
            axes[1,1].set_xticks(range(1, 13))
        
        # 6. Pollution level distribution
        if 'aqi' in self.data.columns:
            aqi_labels = {1: 'Good', 2: 'Fair', 3: 'Moderate', 4: 'Poor', 5: 'Very Poor'}
            aqi_counts = self.data['aqi'].value_counts().sort_index()
            
            # Create pie chart
            labels = [aqi_labels.get(level, f'Level {level}') for level in aqi_counts.index]
            axes[1,2].pie(aqi_counts.values, labels=labels, autopct='%1.1f%%')
            axes[1,2].set_title('AQI Level Distribution')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aqi_eda_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 EDA plots saved to: {filename}")
        plt.show()
        
    def run_full_analysis(self):
        """Run complete EDA analysis"""
        print("🔍 AQI EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        self.load_data()
        if self.data is not None:
            self.basic_stats()
            self.trend_analysis()
            self.correlation_analysis()
            self.pollution_events()
            self.generate_plots()
            
            print(f"\n✅ EDA Analysis Complete!")
            print(f"📋 Dataset: {len(self.data)} records")
            print(f"📅 Period: {self.data['time'].min()} to {self.data['time'].max()}")

def main():
    """Main EDA execution"""
    explorer = AQIExplorer()
    explorer.run_full_analysis()

if __name__ == "__main__":
    main()
