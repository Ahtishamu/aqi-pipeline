"""
OpenWeather AQI Validation Tool
Check if predicted pollutant values match the correct AQI categories
"""

def get_openweather_aqi_ranges():
    """
    OpenWeather AQI scale and pollutant ranges
    Based on: https://openweathermap.org/api/air-pollution
    """
    return {
        1: {  # Good
            "description": "Good",
            "pm25": (0, 10),      # μg/m³
            "pm10": (0, 20),      # μg/m³
            "o3": (0, 60),        # μg/m³
            "no2": (0, 40),       # μg/m³
            "so2": (0, 20),       # μg/m³
            "co": (0, 4400),      # μg/m³
        },
        2: {  # Fair
            "description": "Fair", 
            "pm25": (10, 25),
            "pm10": (20, 50),
            "o3": (60, 100),
            "no2": (40, 70),
            "so2": (20, 80),
            "co": (4400, 9400),
        },
        3: {  # Moderate
            "description": "Moderate",
            "pm25": (25, 50),
            "pm10": (50, 100),
            "o3": (100, 140),
            "no2": (70, 150),
            "so2": (80, 250),
            "co": (9400, 12400),
        },
        4: {  # Poor
            "description": "Poor",
            "pm25": (50, 75),
            "pm10": (100, 200),
            "o3": (140, 180),
            "no2": (150, 200),
            "so2": (250, 350),
            "co": (12400, 15400),
        },
        5: {  # Very Poor
            "description": "Very Poor",
            "pm25": (75, float('inf')),
            "pm10": (200, float('inf')),
            "o3": (180, float('inf')),
            "no2": (200, float('inf')),
            "so2": (350, float('inf')),
            "co": (15400, float('inf')),
        }
    }

def calculate_aqi_from_pollutants(pm25, pm10, o3, no2, so2, co):
    """
    Calculate the correct AQI based on pollutant concentrations
    Returns the highest AQI category from all pollutants
    """
    ranges = get_openweather_aqi_ranges()
    pollutants = {
        'pm25': pm25,
        'pm10': pm10, 
        'o3': o3,
        'no2': no2,
        'so2': so2,
        'co': co
    }
    
    max_aqi = 1
    contributing_pollutant = None
    
    for pollutant, value in pollutants.items():
        for aqi in range(1, 6):
            min_val, max_val = ranges[aqi][pollutant]
            if min_val <= value < max_val:
                if aqi > max_aqi:
                    max_aqi = aqi
                    contributing_pollutant = pollutant
                break
        else:
            # If no range found, assume highest category
            if 5 > max_aqi:
                max_aqi = 5
                contributing_pollutant = pollutant
    
    return max_aqi, contributing_pollutant

def get_main_pollutant(pm25, pm10, o3, no2, so2, co):
    """
    Get the main pollutant that contributes to the highest AQI
    """
    _, main_pollutant = calculate_aqi_from_pollutants(pm25, pm10, o3, no2, so2, co)
    return main_pollutant

def validate_prediction(predicted_aqi, pm25, pm10, o3, no2, so2, co):
    """
    Validate if the predicted AQI matches the pollutant concentrations
    """
    calculated_aqi, main_pollutant = calculate_aqi_from_pollutants(pm25, pm10, o3, no2, so2, co)
    
    is_correct = abs(predicted_aqi - calculated_aqi) <= 1  # Allow 1 level tolerance
    
    return {
        'predicted_aqi': predicted_aqi,
        'calculated_aqi': calculated_aqi,
        'is_correct': is_correct,
        'main_pollutant': main_pollutant,
        'pollutants': {
            'pm25': pm25,
            'pm10': pm10,
            'o3': o3,
            'no2': no2,
            'so2': so2,
            'co': co
        }
    }

def print_aqi_ranges():
    """Print the OpenWeather AQI ranges for reference"""
    ranges = get_openweather_aqi_ranges()
    
    print("📊 OpenWeather AQI Scale and Pollutant Ranges:")
    print("=" * 80)
    
    for aqi, data in ranges.items():
        print(f"\n🔹 AQI {aqi} - {data['description']}:")
        print(f"   PM2.5: {data['pm25'][0]}-{data['pm25'][1] if data['pm25'][1] != float('inf') else '∞'} μg/m³")
        print(f"   PM10:  {data['pm10'][0]}-{data['pm10'][1] if data['pm10'][1] != float('inf') else '∞'} μg/m³")
        print(f"   O3:    {data['o3'][0]}-{data['o3'][1] if data['o3'][1] != float('inf') else '∞'} μg/m³")
        print(f"   NO2:   {data['no2'][0]}-{data['no2'][1] if data['no2'][1] != float('inf') else '∞'} μg/m³")
        print(f"   SO2:   {data['so2'][0]}-{data['so2'][1] if data['so2'][1] != float('inf') else '∞'} μg/m³")
        print(f"   CO:    {data['co'][0]}-{data['co'][1] if data['co'][1] != float('inf') else '∞'} μg/m³")

if __name__ == "__main__":
    print_aqi_ranges()
    
    # Test with current example
    print("\n🧪 Testing current prediction:")
    result = validate_prediction(
        predicted_aqi=3.0,
        pm25=12.1,   # Should be AQI 2 (Fair: 10-25)
        pm10=27.8,   # Should be AQI 2 (Fair: 20-50) 
        o3=33.1,     # Should be AQI 1 (Good: 0-60)
        no2=0.1,     # Should be AQI 1 (Good: 0-40)
        so2=0.1,     # Should be AQI 1 (Good: 0-20)
        co=87.55     # Should be AQI 1 (Good: 0-4400)
    )
    
    print(f"Predicted AQI: {result['predicted_aqi']}")
    print(f"Calculated AQI: {result['calculated_aqi']}")
    print(f"Main pollutant: {result['main_pollutant']}")
    print(f"Is correct: {'✅' if result['is_correct'] else '❌'}")
