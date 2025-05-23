import torch
import requests
import openai
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque
import pickle
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
import folium
from flask import Flask, render_template, request, jsonify
import threading
import time
import schedule
import json
import logging
import openrouteservice
from openrouteservice.exceptions import ApiError
from dotenv import load_dotenv
import sys
import io
from fpdf import FPDF
import base64

# Load environment variables
load_dotenv()

# Fix console encoding for Windows
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'UTF-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

class Config:
    @property
    def mapbox_token(self):
        return os.getenv('MAPBOX_TOKEN')

    @property
    def EMAIL_DISPLAY_NAME(self):
        return os.getenv('EMAIL_DISPLAY_NAME', 'Traffic Monitor')

    @property
    def GOOGLE_MAPS_KEY(self):
        return os.getenv('GOOGLE_MAPS_KEY')
    
    @property
    def OPENAI_KEY(self):
        return os.getenv('OPENAI_KEY')
    
    @property
    def WHAPI_TOKEN(self):
        return os.getenv('WHAPI_TOKEN')
    
    @property
    def ORS_API_KEY(self):
        return os.getenv('ORS_API_KEY')
    
    @property
    def SMTP_SERVER(self):
        return os.getenv('SMTP_SERVER')
    
    @property
    def SMTP_PORT(self):
        return int(os.getenv('SMTP_PORT', 587))
    
    @property
    def EMAIL_USER(self):
        return os.getenv('EMAIL_USER')
    
    @property
    def EMAIL_PASSWORD(self):
        return os.getenv('EMAIL_PASSWORD')
    
    @property
    def WHATSAPP_NUMBERS(self):
        numbers = os.getenv('WHATSAPP_NUMBERS', '')
        return [num.strip() for num in numbers.split(',') if num.strip()]
    
    @property
    def EMAIL_ADDRESSES(self):
        emails = os.getenv('EMAIL_ADDRESSES', '')
        return [email.strip() for email in emails.split(',') if email.strip()]
    
    MODEL_PATH = "lagos_traffic_model.pth"
    CACHE_PATH = "traffic_cache.json"
    LOG_PATH = "traffic_monitor.log"
    ROUTES_FILE = "routes.json"
        
    @property
    def MONITORED_ROUTES(self):
        if not hasattr(self, '_routes'):
            self._load_routes()
        return self._routes
    
    @MONITORED_ROUTES.setter
    def MONITORED_ROUTES(self, value):
        self._routes = value
        self._save_routes()
        
    def _load_routes(self):
        try:
            if os.path.exists(self.ROUTES_FILE):
                with open(self.ROUTES_FILE) as f:
                    self._routes = json.load(f)
            else:
                self._routes = [
                    {
                        "origin": "Eko Hotel, VI", 
                        "destination": "Falomo Bridge, Ikoyi", 
                        "coords": {
                            "origin": [6.4318, 3.4216], 
                            "destination": [6.4389, 3.4278],
                            "waypoints": [[6.4335, 3.4247], [6.4362, 3.4263]]
                        },
                        "notification_email": os.getenv('DEFAULT_NOTIFICATION_EMAIL', '')
                    }
                ]
                self._save_routes()
        except Exception as e:
            logging.error(f"Error loading routes: {str(e)}")
            self._routes = []
        
    def _save_routes(self):
        try:
            with open(self.ROUTES_FILE, 'w') as f:
                json.dump(self._routes, f)
        except Exception as e:
            logging.error(f"Error saving routes: {str(e)}")

Config = Config()

class WhatsAppClient:
    def __init__(self, token):
        self.base_url = "https://gate.whapi.cloud"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        })
        self.logger = logging.getLogger('WhatsAppClient')

    def send_message(self, phone, message, media_url=None):
        payload = {
            "to": self._format_phone(phone),
            "body": message
        }
        if media_url:
            payload["media"] = {"url": media_url}
            endpoint = f"{self.base_url}/messages/media"
        else:
            endpoint = f"{self.base_url}/messages/text"
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            self.logger.info(f"Message sent to {phone[:6]}...")
            return response.json()
        except Exception as e:
            self.logger.error(f"WHAPI Error: {str(e)}")
            return None

    @staticmethod
    def _format_phone(phone):
        phone = phone.lstrip("+").replace(" ", "")
        if not phone.startswith("234") and len(phone) == 10:
            return f"234{phone}"
        return phone

class TrafficPredictor(nn.Module):
    def __init__(self, input_size=7):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 24)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LagosTrafficSystem:
    def __init__(self):
        self.history = self._load_cache()
        self.model = self._load_model()
        self.whapi = WhatsAppClient(Config.WHAPI_TOKEN)
        self.logger = logging.getLogger('LagosTrafficSystem')

    def _load_model(self):
        model = TrafficPredictor()
        if os.path.exists(Config.MODEL_PATH):
            try:
                model.load_state_dict(torch.load(Config.MODEL_PATH))
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
                model = TrafficPredictor()
        return model

    def _load_cache(self):
        if os.path.exists(Config.CACHE_PATH):
            try:
                with open(Config.CACHE_PATH) as f:
                    return deque(json.load(f), maxlen=5000)
            except Exception as e:
                self.logger.error(f"Error loading cache: {str(e)}")
        return deque(maxlen=5000)

    def _save_cache(self):
        try:
            with open(Config.CACHE_PATH, 'w') as f:
                json.dump(list(self.history)[-1000:], f)
        except Exception as e:
            self.logger.error(f"Error saving cache: {str(e)}")

    def get_realtime_traffic(self, route):
        try:
            if "Third Mainland Bridge" in route['origin']:
                return self._get_bridge_traffic(route)

            client = openrouteservice.Client(key=Config.ORS_API_KEY)
            coordinates = [
                [route['coords']['origin'][1], route['coords']['origin'][0]]
            ]

            if route['coords'].get('waypoints'):
                for wp in route['coords']['waypoints']:
                    coordinates.append([wp[1], wp[0]])

            coordinates.append([route['coords']['destination'][1], route['coords']['destination'][0]])

            result = client.directions(
                coordinates=coordinates,
                profile='driving-car',
                format='json'
            )

            summary = result['routes'][0]['summary']
            duration_min = summary['duration'] / 60
            distance_km = summary['distance'] / 1000

            return {
                "route": f"{route['origin']} → {route['destination']}",
                "duration": duration_min,
                "distance": distance_km,
                "timestamp": datetime.now().isoformat(),
                "health_index": self._calculate_health_index(route, duration_min),
                "recommendation": self._generate_recommendation(route, duration_min)
            }

        except Exception as e:
            self.logger.error(f"Traffic check error: {str(e)}")
            return None

    def _get_bridge_traffic(self, route):
        try:
            client = openrouteservice.Client(key=Config.ORS_API_KEY)
            coordinates = [
                [route['coords']['origin'][1], route['coords']['origin'][0]],
                [route['coords']['destination'][1], route['coords']['destination'][0]]
            ]

            result = client.directions(
                coordinates=coordinates,
                profile='driving-car',
                format='json',
                options={'avoid_features': ['ferries']}
            )

            summary = result['routes'][0]['summary']
            base_duration = summary['duration'] / 60
            
            current_hour = datetime.now().hour
            if current_hour in [7,8,16,17,18]:
                duration_min = base_duration * 1.8
            else:
                duration_min = base_duration * 1.2
                
            return {
                "route": f"{route['origin']} → {route['destination']}",
                "duration": duration_min,
                "distance": summary['distance'] / 1000,
                "timestamp": datetime.now().isoformat(),
                "health_index": self._calculate_bridge_health(duration_min),
                "recommendation": self._generate_bridge_recommendation(duration_min)
            }
            
        except Exception as e:
            self.logger.error(f"Bridge traffic error: {str(e)}")
            return {
                "route": f"{route['origin']} → {route['destination']}",
                "duration": 45.0,
                "distance": 11.8,
                "timestamp": datetime.now().isoformat(),
                "health_index": 6,
                "recommendation": "⚠️ Using estimated bridge traffic data"
            }

    def _calculate_health_index(self, route, duration):
        base_score = min(10, int(50 / duration)) if duration > 0 else 5
        
        if "VI" in route["origin"] or "Ikoyi" in route["origin"]:
            if datetime.now().hour in [7,8,16,17,18]:
                base_score = max(0, base_score - 2)
        
        if datetime.now().weekday() in [4,5]:
            base_score = max(0, base_score - 1)
            
        return min(10, max(0, base_score))

    def _calculate_bridge_health(self, duration):
        if duration > 60: return 2
        if duration > 45: return 4
        if duration > 30: return 6
        return 8

    def _generate_recommendation(self, route, duration):
        recommendations = []
        current_hour = datetime.now().hour
        
        if current_hour in [7,8,16,17,18]:
            recommendations.append("⏰ Rush hour traffic - Consider alternatives")
        
        if "VI" in route["origin"] and duration > 30:
            recommendations.append("📍 Victoria Island congestion - Use Lekki alternatives")
        if "Ikoyi" in route["destination"] and current_hour in [7,8]:
            recommendations.append("🏫 School traffic alert - Allow extra time")
        
        if duration > 45:
            recommendations.append("🚨 Severe congestion - Delay travel if possible")
        elif duration > 30:
            recommendations.append("⚠️ Moderate traffic - Check alternate routes")
        
        if datetime.now().weekday() == 4:
            recommendations.append("🗓️ Friday traffic - Leave before 4PM or after 7PM")
        
        return "\n".join(recommendations) if recommendations else "✅ Normal traffic conditions"

    def _generate_bridge_recommendation(self, duration):
        recs = []
        current_hour = datetime.now().hour
        
        if duration > 60:
            recs.append("🚨 AVOID BRIDGE - Extreme congestion")
        elif duration > 45:
            recs.append("⚠️ Heavy bridge traffic - Consider alternatives")
            
        if current_hour in [7,8]:
            recs.append("🌉 Morning rush - Expect delays")
        elif current_hour in [16,17,18]:
            recs.append("🌉 Evening rush - Add 30+ mins")
            
        if datetime.now().weekday() == 4:
            recs.append("🗓️ Friday - Leave early or late")
            
        return "\n".join(recs) if recs else "✅ Normal bridge traffic"

    def predict_congestion(self, route):
        current = self.get_realtime_traffic(route)
        if not current:
            return None
            
        features = torch.FloatTensor([
            current['duration'],
            current['distance'],
            datetime.now().hour,
            datetime.now().weekday(),
            1 if datetime.now().weekday() >= 5 else 0,
            1 if "VI" in route["origin"] else 0,
            1 if "Ikoyi" in route["origin"] else 0
        ])
        
        with torch.no_grad():
            predictions = self.model(features).numpy()
        
        predictions[7:10] *= 1.4
        predictions[16:20] *= 1.6
        if datetime.now().weekday() == 4:
            predictions[15:22] *= 1.3
        if "VI" in route["origin"]:
            predictions[12:14] *= 0.7
            
        return predictions

    def generate_visualization(self, route, predictions):
        try:
            m = folium.Map(location=route['coords']['origin'], zoom_start=14)
            route_coords = [route['coords']['origin']]
            if 'waypoints' in route['coords']:
                route_coords.extend(route['coords']['waypoints'])
            route_coords.append(route['coords']['destination'])
            
            folium.PolyLine(route_coords, color='blue', weight=5).add_to(m)
            folium.Marker(route['coords']['origin'], popup="Origin").add_to(m)
            folium.Marker(route['coords']['destination'], popup="Destination").add_to(m)
            
            os.makedirs('static', exist_ok=True)
            img_path = f"static/map_{route['origin'][:3]}.png"
            m.save(img_path)
            return img_path
        except Exception as e:
            self.logger.error(f"Visualization error: {str(e)}")
            return None

    def generate_pdf_report(self, route, current, predictions):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.add_font('Arial', '', 'arial.ttf') 
            pdf.set_font('Arial', '', 12)
            
            route_text = f"Route: {route['origin']} to {route['destination']}"
            
            pdf.cell(200, 10, txt="LAGOS TRAFFIC REPORT", ln=1, align='C')
            pdf.ln(10)
            pdf.cell(200, 10, txt=route_text, ln=1)
            pdf.cell(200, 10, txt=f"Current Duration: {current['duration']:.1f} minutes", ln=1)
            pdf.cell(200, 10, txt=f"Traffic Health Index: {current['health_index']}/10", ln=1)
            pdf.ln(5)
            
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(200, 10, txt="Recommendations:", ln=1)
            pdf.set_font('DejaVu', '', 12)
            for line in current['recommendation'].replace('→', 'to').split('\n'):
                pdf.cell(200, 10, txt=line, ln=1)
            
            os.makedirs('reports', exist_ok=True)
            filename = f"reports/traffic_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            pdf.output(filename)
            
            with open(filename, "rb") as f:
                pdf_data = base64.b64encode(f.read()).decode('utf-8')
            
            return pdf_data, filename
            
        except Exception as e:
            error_msg = str(e).encode('ascii', errors='ignore').decode('ascii')
            self.logger.error(f"PDF generation error: {error_msg}")
            return None, None

    def send_alerts(self, route, current, predictions):
        if current['health_index'] < 5:
            message = f"""
            🚦 LAGOS TRAFFIC ALERT 🚦
            Route: {route['origin']} → {route['destination']}
            Current: {current['duration']:.1f} mins (Health: {current['health_index']}/10)
            Peak Today: {np.max(predictions):.1f} mins at {np.argmax(predictions)}:00
            
            Recommendations:
            {current['recommendation']}
            """
            
            map_img = None
            pdf_data = None
            pdf_filename = None
            
            try:
                map_img = self.generate_visualization(route, predictions)
                pdf_data, pdf_filename = self.generate_pdf_report(route, current, predictions)
                
                if pdf_data:
                    self._send_email(route, message, map_img, pdf_filename)
                
                for number in Config.WHATSAPP_NUMBERS:
                    self.whapi.send_message(number, message, map_img)
                    
            except Exception as e:
                self.logger.error(f"Error in send_alerts: {e}")
            finally:
                if pdf_filename and os.path.exists(pdf_filename):
                    try:
                        os.remove(pdf_filename)
                    except Exception as e:
                        self.logger.error(f"Error cleaning up PDF: {e}")

    def _send_email(self, route, message, image_path=None, pdf_path=None):
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"Traffic Alert: {route['origin']}"
            msg['From'] = f"{Config.EMAIL_DISPLAY_NAME} <{Config.EMAIL_USER}>"
            msg.attach(MIMEText(message.replace('*', '')))
            
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    msg.attach(img)
            
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as f:
                    pdf = MIMEApplication(f.read(), _subtype='pdf')
                    pdf.add_header('Content-Disposition', 'attachment',
                                filename=os.path.basename(pdf_path))
                    msg.attach(pdf)
            
            with smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT) as server:
                server.starttls()
                server.login(Config.EMAIL_USER, Config.EMAIL_PASSWORD)
                for email in Config.EMAIL_ADDRESSES:
                    msg['To'] = email
                    server.send_message(msg)
        except Exception as e:
            self.logger.error(f"Email error: {str(e)}")

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/health')
def health_check():
    checks = {
        "model_loaded": os.path.exists(Config.MODEL_PATH),
        "email_configured": all([Config.EMAIL_USER, Config.EMAIL_PASSWORD]),
        "ors_api_key": bool(Config.ORS_API_KEY),
        "openai_key": bool(Config.OPENAI_KEY),
        "log_file_access": os.access(Config.LOG_PATH, os.W_OK)
    }

    if all(checks.values()):
        return jsonify(status="OK", checks=checks), 200
    else:
        return jsonify(status="FAIL", checks=checks), 500

@app.route('/api/routes', methods=['GET', 'POST'])
def manage_routes():
    if request.method == 'POST':
        new_route = request.json
        if 'notification_email' not in new_route:
            return jsonify({"status": "error", "message": "Notification email is required"}), 400
            
        Config.MONITORED_ROUTES.append(new_route)
        return jsonify({"status": "success", "message": "Route added"})
    
    return jsonify({"routes": Config.MONITORED_ROUTES})

@app.route('/routes')
def routes_management():
    return render_template('routes.html', mapbox_token=Config.mapbox_token)

@app.route('/api/routes/<int:route_id>', methods=['DELETE'])
def delete_route(route_id):
    try:
        del Config.MONITORED_ROUTES[route_id]
        return jsonify({"status": "success", "message": "Route deleted"})
    except IndexError:
        return jsonify({"status": "error", "message": "Invalid route ID"}), 404
        
@app.route('/api/monitor')
def monitor_all():
    system = LagosTrafficSystem()
    results = []
    for route in Config.MONITORED_ROUTES:
        current = system.get_realtime_traffic(route)
        if current:
            predictions = system.predict_congestion(route)
            results.append({
                "route": {
                    "origin": route['origin'],
                    "destination": route['destination'],
                    "coords": {
                        "origin": route['coords']['origin'],
                        "destination": route['coords']['destination'],
                        "waypoints": route['coords'].get('waypoints', [])
                    },
                    "notification_email": route.get('notification_email', '')
                },
                "current": current,
                "predictions": predictions.tolist() if predictions is not None else []
            })
    return jsonify({
        "status": "success",
        "data": results,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/geocode', methods=['GET'])
def geocode_location():
    query = request.args.get('query', '')
    
    try:
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json"
        params = {
            'access_token': Config.mapbox_token,
            'country': 'ng',
            'proximity': '3.3792,6.5244',
            'limit': 5
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        suggestions = []
        for feature in data.get('features', []):
            suggestions.append({
                'name': feature['place_name'],
                'coords': [feature['center'][1], feature['center'][0]]
            })
        
        return jsonify({
            'status': 'success',
            'suggestions': suggestions
        })
        
    except Exception as e:
        logging.error(f"Geocoding error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve location suggestions'
        }), 500

def run_monitoring(use_app_context=True):
    try:
        if use_app_context:
            with app.app_context():
                _run_monitoring_logic()
        else:
            _run_monitoring_logic()
    except Exception as e:
        logging.error(f"Monitoring failed: {str(e)}")
        raise

def _run_monitoring_logic():
    system = LagosTrafficSystem()
    logging.info("Starting traffic monitoring cycle")
    
    for route in Config.MONITORED_ROUTES:
        try:
            logging.info(f"Checking route: {route['origin']} → {route['destination']}")
            
            current = system.get_realtime_traffic(route)
            if not current:
                logging.warning(f"No data received for route: {route['origin']}")
                continue
                
            predictions = system.predict_congestion(route)
            
            logging.info(
                f"Route: {route['origin']} → {route['destination']}\n"
                f"Duration: {current['duration']:.1f} mins\n"
                f"Health Index: {current['health_index']}/10"
            )
            
            if current['health_index'] < 5:
                logging.warning("Traffic alert triggered - sending notifications")
                system.send_alerts(route, current, predictions)
                
        except Exception as e:
            logging.error(f"Error processing route {route['origin']}: {str(e)}")
            continue
            
    logging.info("Monitoring cycle completed successfully")

def scheduler_thread():
    schedule.every(15).minutes.do(run_monitoring)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    try:
        scheduler = threading.Thread(target=scheduler_thread, daemon=True)
        scheduler.start()
        logging.info("Background scheduler started")
    except Exception as e:
        logging.error(f"Failed to start scheduler: {str(e)}")
    
    try:
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        logging.error(f"Flask app failed: {str(e)}")
    finally:
        logging.info("Shutting down...")