# prediction/views.py - COMPLETE FILE WITH OSRM FALLBACK AND CORS FIX
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
import joblib
import os
import numpy as np
import json
import traceback
import pandas as pd
import requests
import math
from .models import PredictionLog, CustomUser, RiskZone
from django.contrib.auth.hashers import make_password, check_password
from django.core.files.storage import FileSystemStorage
import logging

logger = logging.getLogger(__name__)

ACCIDENT_WEIGHT = 1.0
FATALITY_WEIGHT = 6.0
SEVERITY_WEIGHT = 0.12
WEIGHT_CAP_RAW_PER_KM = 200.0

MODEL_PATH = os.path.join(settings.BASE_DIR, 'prediction', 'ml_model.pkl')
_model = None

# --- Model Loading ---
def load_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            try:
                _model = joblib.load(MODEL_PATH)
                print("✔ Model loaded from:", MODEL_PATH)
            except Exception as e:
                print("✖ Failed loading model:", e)
                traceback.print_exc()
                _model = None
        else:
            print("✖ Model file not found at:", MODEL_PATH)
            _model = None
    return _model

# --- Page Views ---
def home(request):
    return render(request, 'home.html')

def alerts(request):
    return render(request, 'alerts.html')

def analytics(request):
    return render(request, 'analytics.html')

# --- Prediction View ---
@csrf_exempt
def predict_page(request):
    if request.method == 'GET':
        return render(request, 'predict.html')
    
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            latitude = data.get('latitude')
            longitude = data.get('longitude')
            temperature = data.get('temperature')
            weather_condition = data.get('weather_condition')
            humidity = data.get('humidity')
            wind_speed = data.get('wind_speed')
            visibility = data.get('visibility')
            speed = data.get('speed')
            road_condition = data.get('road_condition')
            vehicle_type = data.get('vehicle_type')
            traffic_density = data.get('traffic_density')
            time_of_day = data.get('time_of_day')
            
            if not all([latitude, longitude, speed, road_condition, vehicle_type, traffic_density, time_of_day]):
                return JsonResponse({'error': 'Missing required fields'}, status=400)
            
            risk_score = calculate_risk_score(
                float(speed), road_condition, vehicle_type, traffic_density,
                time_of_day, temperature, weather_condition, humidity,
                wind_speed, visibility
            )
            
            if risk_score < 30:
                risk_level = "Low"
                advice = "Conditions are favorable. Continue driving safely and stay alert."
            elif risk_score < 60:
                risk_level = "Medium"
                advice = "Exercise caution. Maintain safe following distance and reduce speed in adverse conditions."
            else:
                risk_level = "High"
                advice = "High risk detected! Consider delaying your trip if possible. Drive with extreme caution."
            
            return JsonResponse({
                'risk_score': risk_score,
                'risk_level': risk_level,
                'advice': advice,
                'success': True
            })
            
        except Exception as e:
            logger.exception("Prediction error")
            return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)
    
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

# --- Authentication Views ---
def register_view(request):
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')
        
        errors = []
        
        if not name:
            errors.append('Name is required')
        if not email:
            errors.append('Email is required')
        elif '@' not in email:
            errors.append('Invalid email format')
        if not password:
            errors.append('Password is required')
        elif len(password) < 8:
            errors.append('Password must be at least 8 characters')
        if password != confirm_password:
            errors.append('Passwords do not match')
        
        if email and CustomUser.objects.filter(email=email).exists():
            errors.append('Email already registered')
        
        if errors:
            return render(request, 'register.html', {'errors': errors})
        
        try:
            user = CustomUser(name=name, email=email)
            user.set_password(password)
            user.save()
            messages.success(request, 'Registration successful! Please login.')
            return redirect('login')
        except Exception as e:
            return render(request, 'register.html', {'errors': [f'Registration failed: {str(e)}']})
    
    return render(request, 'register.html')

def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        
        errors = []
        
        if not email:
            errors.append('Email is required')
        if not password:
            errors.append('Password is required')
        
        if errors:
            return render(request, 'login.html', {'errors': errors})
        
        try:
            user = CustomUser.objects.get(email=email)
            if user.check_password(password):
                request.session['user_id'] = user.id
                request.session['user_name'] = user.name
                request.session['user_email'] = user.email
                messages.success(request, f'Welcome back, {user.name}!')
                return redirect('dashboard')
            else:
                errors.append('Invalid email or password')
        except CustomUser.DoesNotExist:
            errors.append('Invalid email or password')
        except Exception as e:
            errors.append(f'Login failed: {str(e)}')
        
        return render(request, 'login.html', {'errors': errors})
    
    return render(request, 'login.html')

def logout_view(request):
    request.session.flush()
    messages.info(request, 'You have been logged out')
    return redirect('home')

# --- Original ML Model Prediction ---
@csrf_exempt
def predict_json(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        payload = json.loads(request.body.decode('utf-8'))
    except Exception as e:
        return JsonResponse({'error': 'invalid_json', 'details': str(e)}, status=400)

    try:
        features = [
            float(payload.get('Did_Police_Officer_Attend_Scene_of_Accident', 1)),
            float(payload.get('Age_of_Driver', 35)),
            float(payload.get('Vehicle_Type', 3)),
            float(payload.get('Age_of_Vehicle', 5)),
            float(payload.get('Engine_Capacity_(CC)', 1500)),
            float(payload.get('Day_of_Week', 3)),
            float(payload.get('Weather_Conditions', 1)),
            float(payload.get('Road_Surface_Conditions', 1)),
            float(payload.get('Light_Conditions', 1)),
            float(payload.get('Sex_of_Driver', 1)),
            float(payload.get('Speed_limit', 30))
        ]
    except Exception as e:
        return JsonResponse({'error': 'bad_feature_types', 'details': str(e)}, status=400)

    model = load_model()
    if model is None:
        return JsonResponse({'error': 'model_not_loaded'}, status=500)

    arr = np.array(features).reshape(1, -1)
    try:
        pred = model.predict(arr)[0]
    except Exception as e:
        return JsonResponse({'error': 'prediction_failed', 'details': str(e)}, status=500)

    severity_map = {1: 'FATAL', 2: 'SERIOUS', 3: 'SLIGHT'}
    label = severity_map.get(int(pred), 'UNKNOWN')

    try:
        log = PredictionLog.objects.create(input_json=payload, predicted_code=int(pred), predicted_label=label)
        log_id = log.id
    except Exception as e:
        log_id = None

    return JsonResponse({
        'predicted_code': int(pred),
        'predicted_label': label,
        'log_id': log_id,
        'prediction': int(pred)
    })

# --- Risk Score Calculation ---
def calculate_risk_score(speed, road_condition, vehicle_type, traffic_density,
                         time_of_day, temperature, weather_condition, humidity,
                         wind_speed, visibility):
    score = 0
    
    if float(speed) > 100:
        score += 30
    elif float(speed) > 80:
        score += 20
    elif float(speed) > 60:
        score += 10
    else:
        score += 5
    
    road_scores = {'dry': 0, 'wet': 10, 'muddy': 15, 'snow': 20}
    score += road_scores.get(str(road_condition).lower(), 0)
    
    if weather_condition:
        weather_scores = {
            'Clear': 0, 'Clouds': 5, 'Rain': 15, 'Drizzle': 10,
            'Thunderstorm': 20, 'Snow': 20, 'Mist': 12, 'Fog': 15
        }
        score += weather_scores.get(weather_condition, 5)
    
    traffic_scores = {'light': 0, 'moderate': 5, 'heavy': 10, 'congested': 15}
    score += traffic_scores.get(str(traffic_density).lower(), 0)
    
    time_scores = {'morning': 5, 'afternoon': 0, 'evening': 5, 'night': 10}
    score += time_scores.get(str(time_of_day).lower(), 0)
    
    if visibility:
        try:
            if float(visibility) < 2:
                score += 10
            elif float(visibility) < 5:
                score += 5
        except (ValueError, TypeError):
            pass
    
    vehicle_modifiers = {
        'motorcycle': 1.2, 'car': 1.0, 'suv': 0.9,
        'bus': 1.1, 'commercial': 1.15
    }
    score = score * vehicle_modifiers.get(str(vehicle_type).lower(), 1.0)
    
    return min(100, max(0, int(score)))

# --- Risk Zones View ---
def risk_zones(request):
    from sklearn.cluster import KMeans

    try:
        df = pd.read_csv('prediction/ml/accidents.csv')
    except FileNotFoundError:
        return render(request, 'risk_zones.html', {
            'error': 'Accidents data file not found.',
            'zones': [], 'risk_counts': {}
        })
    except Exception as e:
        return render(request, 'risk_zones.html', {
            'error': f'Error loading data: {str(e)}',
            'zones': [], 'risk_counts': {}
        })

    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        return render(request, 'risk_zones.html', {
            'error': 'Latitude or Longitude columns not found in data.',
            'zones': [], 'risk_counts': {}
        })

    df = df[['Latitude', 'Longitude']].dropna()
    
    if df.empty:
        return render(request, 'risk_zones.html', {
            'error': 'No valid location data to cluster.',
            'zones': [], 'risk_counts': {}
        })

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])

    mapping = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    df['Risk_Level'] = df['Cluster'].map(mapping)

    zones = df.to_dict(orient='records')
    risk_counts = df['Risk_Level'].value_counts().to_dict()

    return render(request, 'risk_zones.html', {
        'zones': zones,
        'risk_counts': risk_counts
    })

# --- Data Loading ---
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ml_dir = os.path.join(base_dir, "ml")

    candidates = [
        os.path.join(ml_dir, "telangana_hotspots.csv"),
        os.path.join(ml_dir, "hyderabad_accidents_cleaned.csv"),
        os.path.join(ml_dir, "accidents.csv"),
        os.path.join(ml_dir, "Road.csv"),
    ]

    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, encoding="utf-8", low_memory=False)
                df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
                if set(["Latitude", "Longitude"]).issubset(df.columns):
                    return df
            except Exception:
                continue

    if os.path.isdir(ml_dir):
        for f in os.listdir(ml_dir):
            if f.lower().endswith(".csv"):
                p = os.path.join(ml_dir, f)
                try:
                    df = pd.read_csv(p, encoding="utf-8", low_memory=False)
                    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
                    if set(["Latitude", "Longitude"]).issubset(df.columns):
                        return df
                except Exception:
                    continue

    return pd.DataFrame()

# --- Haversine Distance ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.asin(math.sqrt(a))

def haversine_m(lat1, lon1, lat2, lon2):
    return haversine_distance(lat1, lon1, lat2, lon2)

# --- Point to Segment Distance ---
def point_segment_distance_meters(px, py, x1, y1, x2, y2):
    lat_ref = math.radians((px + x1 + x2) / 3.0)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2*lat_ref) + 1.175 * math.cos(4*lat_ref)
    m_per_deg_lon = 111412.84 * math.cos(lat_ref) - 93.5 * math.cos(3*lat_ref)
    
    Px = px * m_per_deg_lat
    Py = py * m_per_deg_lon
    Ax = x1 * m_per_deg_lat
    Ay = y1 * m_per_deg_lon
    Bx = x2 * m_per_deg_lat
    By = y2 * m_per_deg_lon
    
    ABx = Bx - Ax
    ABy = By - Ay
    APx = Px - Ax
    APy = Py - Ay
    
    denom = ABx*ABx + ABy*ABy
    if denom == 0:
        return math.hypot(APx, APy)
    
    t = max(0, min(1, (APx*ABx + APy*ABy) / denom))
    projx = Ax + ABx * t
    projy = Ay + ABy * t
    return math.hypot(Px - projx, Py - projy)

# --- Hotspot Loading ---
BASE_APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOTSPOT_CSV_PATH = os.path.join(BASE_PROJECT_DIR, "ml", "telangana_hotspots.csv")

def load_hotspots_from_csv():
    hotspots = []
    if not os.path.exists(HOTSPOT_CSV_PATH):
        logger.warning(f"Hotspot CSV not found at: {HOTSPOT_CSV_PATH}")
        return hotspots
    
    try:
        df = pd.read_csv(HOTSPOT_CSV_PATH)
        logger.info(f"✓ Loaded {len(df)} hotspots from CSV")
        for _, r in df.iterrows():
            try:
                lat = float(r['Latitude'])
                lng = float(r['Longitude'])
                radius = float(r.get('Radius_Meters', 100))
                risk = str(r.get('Risk_Level', 'Low'))
                hotspots.append({
                    "name": r.get('Location_Name', ''),
                    "lat": lat,
                    "lng": lng,
                    "risk": risk,
                    "hazard": r.get('Primary_Hazard', ''),
                    "radius": radius
                })
            except Exception as e:
                logger.error(f"Error parsing hotspot row: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Error reading hotspots CSV: {str(e)}")
    
    logger.info(f"✓ Successfully loaded {len(hotspots)} valid hotspots")
    return hotspots

HOTSPOTS = load_hotspots_from_csv()

# --- Get ORS API Key ---
def get_ors_api_key():
    key = getattr(settings, "OPENROUTESERVICE_API_KEY", None) or \
          getattr(settings, "ORS_API_KEY", None)
    
    if not key:
        logger.error("ORS_API_KEY not found in settings")
        return None
    
    return key.strip()

# ========================================
# OSRM ROUTING (FREE - NO API KEY NEEDED)
# ========================================
def get_routes_from_osrm(start_lat, start_lng, end_lat, end_lng, alternatives=3):
    """
    Use OSRM (Open Source Routing Machine) as routing provider
    FREE - No API key required!
    """
    base_url = "http://router.project-osrm.org/route/v1/driving"
    
    coords = f"{start_lng},{start_lat};{end_lng},{end_lat}"
    url = f"{base_url}/{coords}"
    
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "false",
        "alternatives": "true",
        "continue_straight": "false"
    }
    
    logger.info(f"Requesting routes from OSRM: {start_lat},{start_lng} to {end_lat},{end_lng}")
    
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        if data.get("code") != "Ok":
            logger.error(f"OSRM error: {data.get('message', 'Unknown error')}")
            raise RuntimeError(f"OSRM routing failed: {data.get('message', 'Unknown error')}")
        
        routes = []
        for route_data in data.get("routes", []):
            geom = route_data.get("geometry", {}).get("coordinates", [])
            route_coords = [(c[1], c[0]) for c in geom]
            
            routes.append({
                "geometry": route_coords,
                "distance_m": route_data.get("distance"),
                "duration_s": route_data.get("duration")
            })
        
        logger.info(f"✓ OSRM returned {len(routes)} route(s)")
        
        if len(routes) < 3:
            logger.warning(f"⚠️ OSRM only found {len(routes)} route(s)")
        
        return routes
        
    except requests.exceptions.RequestException as e:
        logger.error(f"OSRM API request failed: {str(e)}")
        raise RuntimeError(f"OSRM routing service error: {str(e)}")

# ========================================
# ORS ROUTING (BACKUP - NEEDS API KEY)
# ========================================
def get_routes_from_ors(start_lat, start_lng, end_lat, end_lng, alternatives=3):
    """Request routes from OpenRouteService API"""
    key = get_ors_api_key()
    if not key:
        raise RuntimeError("ORS_API_KEY not configured")
    
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    
    body = {
        "coordinates": [[start_lng, start_lat], [end_lng, end_lat]],
        "instructions": False,
        "preference": "recommended",
        "units": "m",
        "geometry": True,
        "alternative_routes": {
            "share_factor": 0.6,
            "target_count": int(alternatives)
        }
    }
    
    headers = {
        "Authorization": key,
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json, application/geo+json"
    }
    
    logger.info(f"Requesting routes from ORS: {start_lat},{start_lng} to {end_lat},{end_lng}")
    
    try:
        r = requests.post(url, json=body, headers=headers, timeout=20)
        
        logger.info(f"ORS Response Status: {r.status_code}")
        
        if r.status_code != 200:
            logger.error(f"ORS API Error: {r.text[:500]}")
        
        if r.status_code == 401:
            raise RuntimeError("Invalid ORS API key")
        elif r.status_code == 403:
            raise RuntimeError("ORS API access denied")
        elif r.status_code >= 500:
            raise RuntimeError("ORS service temporarily unavailable")
        
        r.raise_for_status()
        data = r.json()
        
        routes = []
        
        if "features" in data:
            for feat in data.get("features", []):
                geom = feat.get("geometry", {}).get("coordinates", [])
                route_coords = [(c[1], c[0]) for c in geom]
                props = feat.get("properties", {}) or {}
                summary = props.get("summary", {}) or {}
                routes.append({
                    "geometry": route_coords,
                    "distance_m": summary.get("distance"),
                    "duration_s": summary.get("duration")
                })
        elif "routes" in data:
            for route in data.get("routes", []):
                geom = route.get("geometry", {}).get("coordinates", [])
                route_coords = [(c[1], c[0]) for c in geom]
                summary = route.get("summary", {}) or {}
                routes.append({
                    "geometry": route_coords,
                    "distance_m": summary.get("distance"),
                    "duration_s": summary.get("duration")
                })
        
        if not routes:
            raise RuntimeError("No routes in ORS response")
        
        logger.info(f"✓ Successfully retrieved {len(routes)} routes from ORS")
        return routes
        
    except requests.exceptions.RequestException as e:
        logger.error(f"ORS API request failed: {str(e)}")
        raise RuntimeError(f"ORS error: {str(e)}")

# Add this function after get_routes_from_ors and before score_route_improved


def generate_alternative_routes(base_route, num_alternatives=2):
    """
    Generate synthetic alternative routes when API only returns 1 route
    Creates variations by slightly modifying waypoints
    """
    import random
    
    alternatives = []
    geometry = base_route['geometry']
    distance_m = base_route.get('distance_m', 0)
    duration_s = base_route.get('duration_s', 0)
    
    if len(geometry) < 10:
        logger.warning("Route too short to generate alternatives")
        return alternatives
    
    for alt_num in range(num_alternatives):
        # Create a slightly different route by adding small detours
        new_geometry = geometry.copy()
        
        # Add small random offsets to middle waypoints (±0.001 degrees ≈ ±100m)
        offset_factor = 0.0008 * (alt_num + 1)  # Different offset for each alternative
        
        for i in range(len(new_geometry)):
            if i % 5 == (alt_num + 1) % 5:  # Offset every 5th point differently
                lat, lng = new_geometry[i]
                # Add random offset
                lat_offset = random.uniform(-offset_factor, offset_factor)
                lng_offset = random.uniform(-offset_factor, offset_factor)
                new_geometry[i] = (lat + lat_offset, lng + lng_offset)
        
        # Calculate slightly different distance/duration
        distance_variation = random.uniform(1.02, 1.15)  # 2-15% longer
        duration_variation = random.uniform(1.03, 1.18)  # 3-18% longer
        
        alternatives.append({
            "geometry": new_geometry,
            "distance_m": distance_m * distance_variation,
            "duration_s": duration_s * duration_variation,
            "is_synthetic": True
        })
    
    logger.info(f"✓ Generated {len(alternatives)} synthetic alternative routes")
    return alternatives

# ========================================
# IMPROVED ROUTE SCORING
# ========================================
def score_route_improved(route_coords, hotspots):
    """
    Enhanced route scoring with distance-weighted risk and route length normalization
    Returns a dictionary with detailed risk metrics
    """
    n = len(route_coords)
    
    if n == 0:
        return {
            "raw_score": 0,
            "risk_score": 0,
            "high_hits": 0,
            "med_hits": 0,
            "low_hits": 0,
            "total_hits": 0,
            "route_length_km": 0,
            "risk_density": 0
        }
    
    # Calculate actual route length in kilometers
    route_length_m = 0
    for i in range(len(route_coords) - 1):
        lat1, lng1 = route_coords[i]
        lat2, lng2 = route_coords[i + 1]
        route_length_m += haversine_m(lat1, lng1, lat2, lng2)
    
    route_length_km = route_length_m / 1000.0
    
    if route_length_km == 0:
        route_length_km = 0.001  # Avoid division by zero
    
    # Adaptive sampling: ensure we check enough points
    min_points = 50
    max_points = 300
    
    if n <= min_points:
        sample_indices = list(range(n))
    else:
        points_per_km = 20
        target_samples = min(max_points, max(min_points, int(route_length_km * points_per_km)))
        step = max(1, n // target_samples)
        sample_indices = list(range(0, n, step))
        if (n - 1) not in sample_indices:
            sample_indices.append(n - 1)
    
    logger.info(f"Scoring route: {n} coords, {route_length_km:.2f}km, sampling {len(sample_indices)} points")
    
    # Track hotspots with distance-weighted scoring
    hotspot_impacts = {}
    
    for idx in sample_indices:
        lat, lng = route_coords[idx]
        
        for h_idx, hotspot in enumerate(hotspots):
            try:
                distance_m = haversine_m(lat, lng, hotspot['lat'], hotspot['lng'])
                radius = hotspot.get('radius', 100)
                
                if distance_m <= radius:
                    proximity_factor = 1.0 - (distance_m / radius) ** 2
                    
                    risk_level = str(hotspot.get('risk', 'low')).lower()
                    if 'high' in risk_level:
                        risk_multiplier = 10.0
                    elif 'med' in risk_level or 'medium' in risk_level:
                        risk_multiplier = 5.0
                    else:
                        risk_multiplier = 2.0
                    
                    impact = proximity_factor * risk_multiplier
                    
                    if h_idx not in hotspot_impacts:
                        hotspot_impacts[h_idx] = {
                            'max_impact': impact,
                            'total_impact': impact,
                            'risk': risk_level,
                            'hits': 1
                        }
                    else:
                        hotspot_impacts[h_idx]['max_impact'] = max(
                            hotspot_impacts[h_idx]['max_impact'], 
                            impact
                        )
                        hotspot_impacts[h_idx]['total_impact'] += impact
                        hotspot_impacts[h_idx]['hits'] += 1
                        
            except Exception as e:
                logger.error(f"Error processing hotspot {h_idx}: {str(e)}")
                continue
    
    # Aggregate results
    high_hits = sum(1 for h in hotspot_impacts.values() if 'high' in h['risk'])
    med_hits = sum(1 for h in hotspot_impacts.values() if 'med' in h['risk'] or 'medium' in h['risk'])
    low_hits = sum(1 for h in hotspot_impacts.values() if 'high' not in h['risk'] and 'med' not in h['risk'] and 'medium' not in h['risk'])
    total_hits = len(hotspot_impacts)
    
    raw_score = sum(h['max_impact'] for h in hotspot_impacts.values())
    risk_density = raw_score / route_length_km if route_length_km > 0 else 0
    
    logger.info(f"Route analysis: {total_hits} hotspots (H:{high_hits}, M:{med_hits}, L:{low_hits}), "
                f"raw={raw_score:.2f}, density={risk_density:.2f}/km")
    
    return {
        "raw_score": raw_score,
        "risk_density": risk_density,
        "high_hits": high_hits,
        "med_hits": med_hits,
        "low_hits": low_hits,
        "total_hits": total_hits,
        "route_length_km": route_length_km,
        "hotspot_details": hotspot_impacts
    }


def normalize_risk_scores(scored_routes):
    """Normalize risk scores to 0-100 scale with proper distribution"""
    if not scored_routes:
        return scored_routes
    
    densities = [r['risk_density'] for r in scored_routes]
    
    if not densities or max(densities) == 0:
        for route in scored_routes:
            route['risk_score'] = 5.0
        return scored_routes
    
    max_density = max(densities)
    min_density = min(densities)
    density_range = max_density - min_density
    
    for route in scored_routes:
        density = route['risk_density']
        
        if density_range > 0:
            normalized = ((density - min_density) / density_range) * 100
        else:
            normalized = 50.0
        
        if route['high_hits'] > 0:
            base_risk = 50.0 + (route['high_hits'] * 8)
            normalized = max(normalized, base_risk)
        elif route['med_hits'] > 0:
            base_risk = 30.0 + (route['med_hits'] * 5)
            normalized = max(normalized, base_risk)
        elif route['low_hits'] > 0:
            base_risk = 15.0 + (route['low_hits'] * 3)
            normalized = max(normalized, base_risk)
        else:
            normalized = min(normalized, 10.0)
        
        route['risk_score'] = min(100.0, round(normalized, 1))
        
        logger.info(f"Route risk: {route['risk_score']}% "
                   f"(density={density:.2f}, length={route['route_length_km']:.2f}km, "
                   f"hits: H={route['high_hits']}, M={route['med_hits']}, L={route['low_hits']})")
    
    return scored_routes


# ========================================
# MAIN COMPUTE ROUTES FUNCTION
# ========================================
# Update the compute_routes function to use synthetic routes
@csrf_exempt
def compute_routes(request):
    """
    Enhanced route computation with improved risk scoring
    Tries OSRM first (free), falls back to ORS if needed
    Generates synthetic alternatives if only 1 route returned
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    
    try:
        payload = json.loads(request.body.decode('utf-8'))
    except Exception as e:
        logger.error(f"Invalid JSON: {str(e)}")
        return JsonResponse({"error":"invalid_json","details":str(e)}, status=400)

    start = payload.get("start")
    end = payload.get("end")
    
    if not start or not end:
        return JsonResponse({"error":"start and end coordinates required"}, status=400)
    
    try:
        s_lat = float(start.get("lat"))
        s_lng = float(start.get("lng"))
        e_lat = float(end.get("lat"))
        e_lng = float(end.get("lng"))
    except (ValueError, TypeError, AttributeError) as e:
        return JsonResponse({"error":"invalid coordinates format"}, status=400)
    
    logger.info(f"Computing routes: ({s_lat},{s_lng}) -> ({e_lat},{e_lng})")
    
    routes = None
    provider_used = None
    used_synthetic = False
    
    # Try OSRM first (free, no API key needed!)
    try:
        routes = get_routes_from_osrm(s_lat, s_lng, e_lat, e_lng, alternatives=3)
        provider_used = "OSRM (Free)"
        
        # If only 1 route, generate synthetic alternatives
        if len(routes) == 1:
            logger.info("OSRM returned only 1 route, generating synthetic alternatives...")
            synthetic_routes = generate_alternative_routes(routes[0], num_alternatives=2)
            routes.extend(synthetic_routes)
            used_synthetic = True
            logger.info(f"✓ Now have {len(routes)} routes (1 real + {len(synthetic_routes)} synthetic)")
        
        # If less than 2 routes and synthetic generation failed, try ORS
        elif len(routes) < 2:
            logger.info(f"OSRM returned {len(routes)} route(s), trying ORS for more alternatives...")
            try:
                ors_routes = get_routes_from_ors(s_lat, s_lng, e_lat, e_lng, alternatives=3)
                if len(ors_routes) > len(routes):
                    routes = ors_routes
                    provider_used = "OpenRouteService"
                    logger.info(f"✓ ORS provided {len(routes)} routes")
                elif len(routes) == 1:
                    # ORS also returned 1, generate synthetic
                    synthetic_routes = generate_alternative_routes(routes[0], num_alternatives=2)
                    routes.extend(synthetic_routes)
                    used_synthetic = True
            except Exception as ors_err:
                logger.warning(f"ORS fallback failed: {str(ors_err)}")
                # Generate synthetic alternatives as last resort
                if len(routes) == 1:
                    synthetic_routes = generate_alternative_routes(routes[0], num_alternatives=2)
                    routes.extend(synthetic_routes)
                    used_synthetic = True
        
    except Exception as osrm_error:
        logger.warning(f"OSRM failed: {str(osrm_error)}, trying ORS fallback...")
        
        # Fallback to ORS (requires API key)
        try:
            routes = get_routes_from_ors(s_lat, s_lng, e_lat, e_lng, alternatives=3)
            provider_used = "OpenRouteService"
            
            # If ORS also returns only 1 route, generate synthetic
            if len(routes) == 1:
                logger.info("ORS returned only 1 route, generating synthetic alternatives...")
                synthetic_routes = generate_alternative_routes(routes[0], num_alternatives=2)
                routes.extend(synthetic_routes)
                used_synthetic = True
                
        except Exception as ors_error:
            logger.error(f"All routing services failed. OSRM: {osrm_error}, ORS: {ors_error}")
            return JsonResponse({
                "error": "all_routing_services_failed",
                "details": f"Could not fetch routes. Please try again later."
            }, status=502)
    
    if not routes:
        return JsonResponse({
            "error": "no_routes_found",
            "details": "No routes available for this journey"
        }, status=404)
    
    # Score routes with improved algorithm
    hotspots = HOTSPOTS
    logger.info(f"Scoring {len(routes)} routes against {len(hotspots)} hotspots")
    
    scored = []
    for r in routes:
        score_data = score_route_improved(r['geometry'], hotspots)
        
        scored.append({
            "geometry": [[lng, lat] for lat, lng in r['geometry']],
            "distance_m": r.get("distance_m"),
            "duration_s": r.get("duration_s"),
            "raw_score": score_data["raw_score"],
            "risk_density": score_data["risk_density"],
            "high_hits": score_data["high_hits"],
            "med_hits": score_data["med_hits"],
            "low_hits": score_data["low_hits"],
            "total_hits": score_data["total_hits"],
            "route_length_km": score_data["route_length_km"],
            "hits": score_data["total_hits"],
            "is_synthetic": r.get("is_synthetic", False)
        })
    
    # Normalize risk scores across all routes
    scored = normalize_risk_scores(scored)
    
    # Sort by risk score (safest first)
    scored_sorted = sorted(scored, key=lambda x: x["risk_score"])
    
    # Add route type indicators
    for idx, route in enumerate(scored_sorted):
        if idx == 0:
            route['route_type'] = 'safest'
        elif idx == len(scored_sorted) - 1:
            route['route_type'] = 'fastest'
        else:
            route['route_type'] = 'balanced'
    
    provider_info = provider_used
    if used_synthetic:
        provider_info += " + Synthetic Alternatives"
    
    logger.info(f"✓ Successfully computed {len(scored_sorted)} routes using {provider_info}")
    logger.info(f"Route details: {[(r['distance_m'], r['duration_s'], r['risk_score']) for r in scored_sorted]}")
    
    return JsonResponse({
        "routes": scored_sorted,
        "start": {"lat": s_lat, "lng": s_lng},
        "end": {"lat": e_lat, "lng": e_lng},
        "provider": provider_info,
        "has_synthetic_routes": used_synthetic,
        "analysis": {
            "total_routes": len(scored_sorted),
            "safest_route_risk": scored_sorted[0]["risk_score"] if scored_sorted else 0,
            "riskiest_route_risk": scored_sorted[-1]["risk_score"] if scored_sorted else 0,
            "total_hotspots_loaded": len(hotspots),
            "note": "Alternative routes generated when only one main route available" if used_synthetic else None
        }
    })


# --- Safe Route View ---
def safe_route_view(request):
    return render(request, "safe_route.html")


# --- Debug Data ---
def debug_data(request):
    df = load_data()
    return JsonResponse({
        "columns": list(df.columns) if not df.empty else [],
        "sample": df.head(20).to_dict(orient="records") if not df.empty else [],
        "empty": df.empty
    })


# --- Get Risk Zones ---
def get_risk_zones(request):
    return JsonResponse({"zones": HOTSPOTS})


# --- Geocode Server Side ---
def geocode_text_server(q):
    key = get_ors_api_key()
    if not key:
        logger.error("Cannot geocode: ORS API key not configured")
        return None
    
    url = "https://api.openrouteservice.org/geocode/search"
    params = {"api_key": key, "text": q, "size": 1}
    
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        
        if data.get("features"):
            feat = data["features"][0]
            coords = feat.get("geometry", {}).get("coordinates", [None, None])
            label = feat.get("properties", {}).get("label", q)
            return coords[1], coords[0], label
        return None
    except Exception as e:
        logger.error(f"Geocoding failed: {str(e)}")
        return None


# --- Geocode Proxy ---
def geocode_proxy(request):
    q = request.GET.get('q') or request.GET.get('text') or ''
    if not q:
        return JsonResponse({"features": []})
    
    key = get_ors_api_key()
    if not key:
        return JsonResponse({"error": "ORS_API_KEY not configured"}, status=500)
    
    url = "https://api.openrouteservice.org/geocode/autocomplete"
    params = {"api_key": key, "text": q, "size": 8}
    
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        
        items = []
        for f in data.get("features", []):
            props = f.get("properties", {})
            geom = f.get("geometry", {}).get("coordinates", [None, None])
            items.append({
                "label": props.get("label") or props.get("name") or props.get("locality") or "",
                "lat": geom[1],
                "lng": geom[0],
                "osm_type": props.get("osm_type"),
                "country": props.get("country"),
            })
        return JsonResponse({"features": items})
    except requests.RequestException as e:
        logger.error(f"Geocode proxy failed: {str(e)}")
        return JsonResponse({"error": "geocode_failed", "details": str(e)}, status=502)


# --- Nominatim Proxy (CORS-SAFE) ---
@csrf_exempt
def nominatim_proxy(request):
    """Proxy for Nominatim geocoding to avoid CORS issues"""
    query = request.GET.get("q", "")
    
    if not query or len(query) < 3:
        return JsonResponse([], safe=False)
    
    url = "https://nominatim.openstreetmap.org/search"
    
    params = {
        "format": "json",
        "q": query,
        "limit": 5,
        "addressdetails": 1
    }
    
    headers = {
        "User-Agent": "RoadSafetyApp/1.0 (Django Backend)",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Transform the response to our format
        results = []
        for item in data:
            results.append({
                "display_name": item.get("display_name", ""),
                "lat": item.get("lat", ""),
                "lon": item.get("lon", ""),
                "type": item.get("type", ""),
                "importance": item.get("importance", 0)
            })
        
        return JsonResponse(results, safe=False)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Nominatim proxy error: {str(e)}")
        return JsonResponse({"error": "Geocoding service unavailable"}, status=503)


# --- Dashboard Data ---
def dashboard_data(request):
    df = load_data()

    if df.empty:
        return JsonResponse({"error": "Dataset not found"}, status=404)

    def safe_counts(df, col):
        if col in df.columns:
            return df[col].astype(str).value_counts().to_dict()
        return {}

    weather = safe_counts(df, "Weather")
    time = safe_counts(df, "Time")
    severity = safe_counts(df, "Risk_Level")
    road = safe_counts(df, "Road_Type")
    total_accidents = int(df["Accidents"].sum()) if "Accidents" in df.columns else len(df)

    avg_severity_score = float(df["Severity_Score"].mean()) if "Severity_Score" in df.columns else None
    total_fatalities = int(df["Fatalities"].sum()) if "Fatalities" in df.columns else None

    return JsonResponse({
        "weather": weather,
        "time": time,
        "severity": severity,
        "road": road,
        "total_accidents": total_accidents,
        "avg_severity_score": avg_severity_score,
        "total_fatalities": total_fatalities
    })