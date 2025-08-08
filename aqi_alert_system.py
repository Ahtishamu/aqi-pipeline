#!/usr/bin/env python3
"""
AQI Hazard Alert System
Monitor AQI levels and send alerts for dangerous conditions
"""
import pandas as pd
import numpy as np
import hopsworks
import os
import smtplib
import requests
import logging
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQIAlertSystem:
    """Alert system for hazardous AQI levels"""
    
    def __init__(self):
        self.api_key = os.environ.get('HOPSWORKS_API_KEY')
        self.email_config = {
            'smtp_server': os.environ.get('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.environ.get('SMTP_PORT', '587')),
            'email_user': os.environ.get('EMAIL_USER'),
            'email_password': os.environ.get('EMAIL_PASSWORD'),
            'alert_recipients': os.environ.get('ALERT_RECIPIENTS', '').split(',')
        }
        self.slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
        
        # AQI thresholds and descriptions
        self.aqi_levels = {
            1: {'name': 'Good', 'description': 'Air quality is satisfactory', 'color': 'green', 'action': 'none'},
            2: {'name': 'Fair', 'description': 'Air quality is acceptable', 'color': 'yellow', 'action': 'none'},
            3: {'name': 'Moderate', 'description': 'Air quality is moderately polluted', 'color': 'orange', 'action': 'sensitive_caution'},
            4: {'name': 'Poor', 'description': 'Air quality is poor - health warnings', 'color': 'red', 'action': 'alert'},
            5: {'name': 'Very Poor', 'description': 'Air quality is very poor - health alerts', 'color': 'purple', 'action': 'emergency'}
        }
        
    def get_current_aqi(self):
        """Get current AQI from Hopsworks Feature Store"""
        try:
            project = hopsworks.login(api_key_value=self.api_key)
            fs = project.get_feature_store()
            fg = fs.get_feature_group("aqi_features", version=2)
            
            # Get latest data
            df = fg.read()
            df = df.sort_values('time').tail(1)
            
            if len(df) > 0:
                latest_record = df.iloc[0]
                return {
                    'aqi': latest_record.get('aqi'),
                    'pm25': latest_record.get('pm25'),
                    'pm10': latest_record.get('pm10'),
                    'time': latest_record.get('time'),
                    'location': 'Karachi, Pakistan'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current AQI: {e}")
            return None
    
    def check_alert_conditions(self, current_data):
        """Check if alert conditions are met"""
        if not current_data or current_data['aqi'] is None:
            return None
            
        aqi = current_data['aqi']
        aqi_info = self.aqi_levels.get(aqi, self.aqi_levels[5])  # Default to worst case
        
        alerts = []
        
        # Check for poor/very poor air quality
        if aqi >= 4:
            alert_type = 'EMERGENCY' if aqi == 5 else 'WARNING'
            alerts.append({
                'type': alert_type,
                'level': aqi,
                'name': aqi_info['name'],
                'description': aqi_info['description'],
                'action': aqi_info['action'],
                'time': current_data['time'],
                'location': current_data['location'],
                'pm25': current_data.get('pm25'),
                'pm10': current_data.get('pm10')
            })
        
        # Check for rapid deterioration (compare with recent history)
        try:
            project = hopsworks.login(api_key_value=self.api_key)
            fs = project.get_feature_store()
            fg = fs.get_feature_group("aqi_features", version=2)
            
            # Get last 6 hours of data
            df = fg.read()
            df = df.sort_values('time').tail(6)
            
            if len(df) >= 3:
                recent_aqi = df['aqi'].tail(3).mean()
                if aqi > recent_aqi + 1:  # AQI increased by more than 1 level
                    alerts.append({
                        'type': 'DETERIORATION',
                        'level': aqi,
                        'name': 'Rapid Air Quality Deterioration',
                        'description': f'AQI increased from {recent_aqi:.1f} to {aqi}',
                        'action': 'monitor',
                        'time': current_data['time'],
                        'location': current_data['location']
                    })
        except:
            pass  # Skip deterioration check if data unavailable
        
        return alerts
    
    def send_email_alert(self, alert):
        """Send email alert"""
        if not self.email_config['email_user'] or not self.email_config['email_password']:
            logger.warning("Email credentials not configured")
            return False
            
        try:
            # Create email
            msg = MimeMultipart()
            msg['From'] = self.email_config['email_user']
            msg['To'] = ', '.join(self.email_config['alert_recipients'])
            msg['Subject'] = f"🚨 AQI ALERT: {alert['name']} - {alert['location']}"
            
            # Email body
            body = f"""
AQI ALERT NOTIFICATION

Alert Type: {alert['type']}
Location: {alert['location']}
Time: {alert['time']}
AQI Level: {alert['level']} ({alert['name']})

Description: {alert['description']}

Current Conditions:
• PM2.5: {alert.get('pm25', 'N/A')} μg/m³
• PM10: {alert.get('pm10', 'N/A')} μg/m³

Recommended Actions:
"""
            
            if alert['action'] == 'emergency':
                body += """
• Stay indoors and keep windows closed
• Avoid outdoor activities
• Use air purifiers if available
• Wear N95 masks when going outside
• Seek medical attention if experiencing breathing difficulties
"""
            elif alert['action'] == 'alert':
                body += """
• Limit outdoor activities, especially strenuous exercise
• Consider wearing masks when outside
• Keep windows closed
• Children and elderly should stay indoors
"""
            elif alert['action'] == 'sensitive_caution':
                body += """
• Sensitive individuals should limit outdoor activities
• Consider reducing time spent outdoors
"""
            
            body += f"""

Monitor live updates at: https://app.hopsworks.ai

This is an automated alert from the AQI Monitoring System.
Time generated: {datetime.now()}
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['email_user'], self.email_config['email_password'])
            text = msg.as_string()
            server.sendmail(self.email_config['email_user'], self.email_config['alert_recipients'], text)
            server.quit()
            
            logger.info(f"✅ Email alert sent for {alert['type']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_slack_alert(self, alert):
        """Send Slack alert"""
        if not self.slack_webhook:
            logger.warning("Slack webhook not configured")
            return False
            
        try:
            # Color coding for alerts
            color_map = {
                'EMERGENCY': '#8B0000',  # Dark red
                'WARNING': '#FF4500',    # Orange red
                'DETERIORATION': '#FFA500'  # Orange
            }
            
            color = color_map.get(alert['type'], '#FF0000')
            
            # Create Slack message
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"🚨 AQI ALERT: {alert['name']}",
                        "text": alert['description'],
                        "fields": [
                            {
                                "title": "Location",
                                "value": alert['location'],
                                "short": True
                            },
                            {
                                "title": "AQI Level",
                                "value": f"{alert['level']} ({alert['name']})",
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": str(alert['time']),
                                "short": False
                            }
                        ],
                        "footer": "AQI Monitoring System",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            # Add pollutant info if available
            if alert.get('pm25') or alert.get('pm10'):
                pollutant_text = ""
                if alert.get('pm25'):
                    pollutant_text += f"PM2.5: {alert['pm25']} μg/m³  "
                if alert.get('pm10'):
                    pollutant_text += f"PM10: {alert['pm10']} μg/m³"
                
                payload["attachments"][0]["fields"].append({
                    "title": "Pollutant Levels",
                    "value": pollutant_text,
                    "short": False
                })
            
            # Send to Slack
            response = requests.post(self.slack_webhook, json=payload)
            
            if response.status_code == 200:
                logger.info(f"✅ Slack alert sent for {alert['type']}")
                return True
            else:
                logger.error(f"Slack alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def log_alert(self, alert):
        """Log alert to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alert_log = {
                'timestamp': timestamp,
                'alert': alert
            }
            
            # Append to alerts log file
            with open('aqi_alerts.log', 'a') as f:
                f.write(f"{json.dumps(alert_log)}\n")
            
            logger.info(f"✅ Alert logged: {alert['type']} - AQI {alert['level']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
            return False
    
    def run_alert_check(self):
        """Main alert checking routine"""
        try:
            logger.info("🔍 Checking AQI alert conditions...")
            
            # Get current AQI data
            current_data = self.get_current_aqi()
            
            if not current_data:
                logger.warning("No current AQI data available")
                return False
            
            logger.info(f"📊 Current AQI: {current_data['aqi']} at {current_data['time']}")
            
            # Check for alerts
            alerts = self.check_alert_conditions(current_data)
            
            if not alerts:
                logger.info("✅ No alerts triggered - air quality within acceptable range")
                return True
            
            # Process each alert
            for alert in alerts:
                logger.warning(f"🚨 ALERT TRIGGERED: {alert['type']} - {alert['name']}")
                
                # Log alert
                self.log_alert(alert)
                
                # Send notifications
                email_sent = self.send_email_alert(alert)
                slack_sent = self.send_slack_alert(alert)
                
                print(f"\n🚨 AQI ALERT: {alert['type']}")
                print(f"   Level: {alert['level']} ({alert['name']})")
                print(f"   Location: {alert['location']}")
                print(f"   Time: {alert['time']}")
                print(f"   Description: {alert['description']}")
                print(f"   Notifications sent: Email={email_sent}, Slack={slack_sent}")
            
            return True
            
        except Exception as e:
            logger.error(f"Alert check failed: {e}")
            return False

def main():
    """Main alert system execution"""
    print("🚨 AQI HAZARD ALERT SYSTEM")
    print("=" * 40)
    
    alert_system = AQIAlertSystem()
    
    # Configuration check
    print("\n⚙️ Configuration Status:")
    print(f"   Hopsworks API: {'✅' if alert_system.api_key else '❌'}")
    print(f"   Email alerts: {'✅' if alert_system.email_config['email_user'] else '❌'}")
    print(f"   Slack alerts: {'✅' if alert_system.slack_webhook else '❌'}")
    
    # Run alert check
    success = alert_system.run_alert_check()
    
    if success:
        print(f"\n✅ Alert check completed successfully")
    else:
        print(f"\n❌ Alert check failed")

if __name__ == "__main__":
    main()
