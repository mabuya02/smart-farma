import os
import requests
import time
import logging
from typing import Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace with your actual WeatherAPI key
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "a8f656b81fb548bf82c125713251705")

# Fallback coordinates for known locations
FALLBACK_COORDINATES = {
    "nakuru": {"lat": -0.2802724, "lon": 36.0712048, "display_name": "Nakuru, Kenya"},
    "nairobi": {"lat": -1.286389, "lon": 36.817223, "display_name": "Nairobi, Kenya"},
    "kiambu": {"lat": -1.036395, "lon": 36.8431312, "display_name": "Kiambu, Central Kenya, Kenya"}
}

# Fallback soil data for Kiambu
FALLBACK_SOIL_DATA = {
    (-1.036395, 36.8431312): {"N": 3.0, "P": 50.0, "K": 50.0, "ph": 6.0}
}

def get_lat_lon(address: str, retries: int = 3, delay: int = 2, timeout: int = 15) -> Optional[Dict[str, any]]:
    """
    Fetch latitude and longitude for a given address using Nominatim with retries.
    Uses fallback coordinates for known locations if API fails.
    """
    # Normalize address for fallback lookup
    address_lower = address.lower().strip()

    # Try Nominatim API
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json',
        'limit': 1
    }
    headers = {
        'User-Agent': 'SmartFarmApp/1.0 (mainamanasseh02@gmail.com)'
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            results = response.json()
            if results:
                result = {
                    'lat': float(results[0]['lat']),
                    'lon': float(results[0]['lon']),
                    'display_name': results[0].get('display_name', address)
                }
                logger.info(f"Fetched coordinates for {address}: {result}")
                return result
            logger.warning(f"No geocoding result for address: {address} (attempt {attempt}/{retries})")
        except requests.exceptions.RequestException as e:
            logger.error(f"Geocoding error for {address} (attempt {attempt}/{retries}): {str(e)}")
            if attempt < retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    # Use fallback coordinates if available
    if address_lower in FALLBACK_COORDINATES:
        logger.warning(f"Using fallback coordinates for {address}: {FALLBACK_COORDINATES[address_lower]}")
        return FALLBACK_COORDINATES[address_lower]

    logger.error(f"Failed to fetch coordinates for {address} after {retries} attempts")
    return None

def fetch_soil_data(lat: float, lon: float, retries: int = 3, delay: int = 2, timeout: int = 15) -> Dict[str, float]:
    """
    Fetch soil data from SoilGrids API for the given latitude and longitude.
    Returns a dictionary with N, P, K, ph for the 0-5cm depth layer.
    """
    # Check for fallback soil data
    coord_key = (lat, lon)
    if coord_key in FALLBACK_SOIL_DATA:
        logger.warning(f"Using fallback soil data for lat={lat}, lon={lon}: {FALLBACK_SOIL_DATA[coord_key]}")
        return FALLBACK_SOIL_DATA[coord_key]

    SOILGRID_API_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lat": lat,
        "lon": lon,
        "depths": "0-5cm",
        "properties": "nitrogen,phh2o"
    }
    headers = {'Accept': 'application/json'}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(SOILGRID_API_URL, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            properties = data.get("properties", {})
            if not properties:
                logger.warning(f"SoilGrids API returned empty properties for lat: {lat}, lon: {lon}")
                continue

            soil = {'N': None, 'P': None, 'K': None, 'ph': None}
            for layer in properties.get("layers", []):
                if layer["name"] not in ["nitrogen", "phh2o"]:
                    continue
                depth_data = next((d for d in layer.get("depths", []) if d.get("label") == "0-5cm"), None)
                if not depth_data:
                    continue
                mean_value = depth_data.get("values", {}).get("mean")
                if mean_value is None:
                    continue

                if layer["name"] == "nitrogen":
                    soil["N"] = mean_value / 100.0  # Convert cg/kg to g/kg
                elif layer["name"] == "phh2o":
                    soil["ph"] = mean_value / 10.0  # Convert pH*10 to pH

            # Set fallback values for P and K (not available in SoilGrids)
            soil["P"] = 50.0  # g/kg
            soil["K"] = 50.0  # g/kg

            # Use fallback for N or ph if None
            if soil["N"] is None:
                soil["N"] = 50.0  # g/kg
                logger.warning(f"Using fallback for N: {soil['N']}")
            if soil["ph"] is None:
                soil["ph"] = 6.5  # Neutral pH
                logger.warning(f"Using fallback for ph: {soil['ph']}")

            logger.info(f"Fetched soil data for lat={lat}, lon={lon}: {soil}")
            return soil

        except requests.exceptions.RequestException as e:
            logger.error(f"SoilGrids API error (attempt {attempt}/{retries}): {str(e)}")
            if attempt < retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    logger.error(f"Failed to fetch soil data after {retries} attempts for lat: {lat}, lon: {lon}")
    return {
        "N": 50.0,  # g/kg
        "P": 50.0,  # g/kg
        "K": 50.0,  # g/kg
        "ph": 6.5   # Neutral pH
    }

def fetch_weather_data(city_name: str) -> Dict[str, Optional[float]]:
    """
    Fetch weather data (temperature, humidity, rainfall) from WeatherAPI.
    """
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": WEATHERAPI_KEY,
        "q": city_name,
        "days": 1,
        "aqi": "no",
        "alerts": "no"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data['current']
        forecast = data['forecast']['forecastday'][0]['day']

        weather = {
            "temperature": float(current['temp_c']),  # °C
            "humidity": float(current['humidity']),   # %
            "rainfall": float(forecast.get('totalprecip_mm', 0.0))  # mm
        }
        logger.info(f"Fetched weather data for {city_name}: {weather}")
        return weather

    except Exception as e:
        logger.error(f"Error fetching weather data for {city_name}: {str(e)}")
        return {
            "temperature": 25.0,  # °C
            "humidity": 60.0,     # %
            "rainfall": 100.0     # mm
        }

def get_model_input_features(location_name: str) -> Optional[Dict[str, float]]:
    """
    Fetch and combine soil and weather data for model input.
    """
    location = get_lat_lon(location_name)
    if not location or 'lat' not in location or 'lon' not in location:
        logger.error(f"Could not retrieve coordinates for {location_name}")
        return None

    lat = location['lat']
    lon = location['lon']

    soil_data = fetch_soil_data(lat, lon)
    weather_data = fetch_weather_data(location_name)

    model_input = {
        "N": soil_data["N"],
        "P": soil_data["P"],
        "K": soil_data["K"],
        "ph": soil_data["ph"],
        "temperature": weather_data["temperature"],
        "humidity": weather_data["humidity"],
        "rainfall": weather_data["rainfall"]
    }

    # Check for None values
    if any(v is None for v in model_input.values()):
        logger.error(f"Incomplete model input for {location_name}: {model_input}")
        return None

    logger.info("\n✅ Model Input Features")
    logger.info("========================")
    for k, v in model_input.items():
        logger.info(f"{k}: {v}")

    return model_input

def standardize_model_inputs(features: Dict[str, float]) -> Dict[str, float]:
    """
    Standardize model input features to match training data ranges.
    """
    standardized = features.copy()
    ranges = {
        "N": {"min": 0.0, "max": 500.0},  # g/kg
        "P": {"min": 0.0, "max": 500.0},  # g/kg
        "K": {"min": 0.0, "max": 500.0},  # g/kg
        "ph": {"min": 3.5, "max": 10.0},  # pH units
        "temperature": {"min": 8.0, "max": 44.0},  # °C
        "humidity": {"min": 14.0, "max": 100.0},  # %
        "rainfall": {"min": 20.0, "max": 300.0}   # mm
    }
    fallback_values = {
        "N": 50.0,  # g/kg
        "P": 50.0,  # g/kg
        "K": 50.0,  # g/kg
        "ph": 6.5,
        "temperature": 25.0,  # °C
        "humidity": 60.0,     # %
        "rainfall": 100.0     # mm
    }

    for key in features:
        if features.get(key) is None or not isinstance(features[key], (int, float)):
            logger.warning(f"{key} is missing or invalid. Using fallback value: {fallback_values[key]}")
            standardized[key] = fallback_values[key]
        else:
            value = features[key]
            min_val = ranges[key]["min"]
            max_val = ranges[key]["max"]
            standardized[key] = max(min_val, min(max_val, value))
            if value != standardized[key]:
                logger.info(f"Clipped {key} from {value} to {standardized[key]}")

    return standardized

def print_standardization_summary(original: Dict[str, float], standardized: Dict[str, float]) -> None:
    """
    Log the original and standardized feature values for debugging.
    """
    logger.info("\n===== Standardization Summary =====")
    logger.info("{:<15} {:<15} {:<15} {:<15}".format("Feature", "Original", "Standardized", "Status"))
    logger.info("-" * 60)

    for key in original:
        original_val = original[key]
        standardized_val = standardized.get(key, "N/A")
        status = "Modified" if original_val != standardized_val else "Unchanged"
        logger.info("{:<15} {:<15} {:<15} {:<15}".format(
            key, str(original_val), str(standardized_val), status))

    logger.info("=" * 60)