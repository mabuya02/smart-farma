from flask import Blueprint, render_template, request, jsonify
from apps.data.util import get_lat_lon, fetch_soil_data, get_model_input_features, fetch_weather_data, standardize_model_inputs, print_standardization_summary
import joblib
import os
import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

blueprint = Blueprint('data_blueprint', __name__, url_prefix='/data')

# Calculate ROOT_DIR as the project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR = os.path.join(ROOT_DIR, 'apps', 'model', 'version', 'v1')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_crop_rec_tuned.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')

# Verify paths for debugging
logger.info(f"ROOT_DIR: {ROOT_DIR}")
logger.info(f"MODEL_PATH: {MODEL_PATH}")
logger.info(f"LABEL_ENCODER_PATH: {LABEL_ENCODER_PATH}")

# Load model and label encoder
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    logger.info(f"Label encoder loaded successfully from {LABEL_ENCODER_PATH}")
    logger.info(f"Label encoder classes: {list(label_encoder.classes_)}")
except Exception as e:
    logger.error(f"Error loading model or label encoder: {str(e)}")
    model = None
    label_encoder = None

@blueprint.route('/chat')
def chat():
    return render_template('home/virtual-reality.html')

@blueprint.route('/predictions')
def prediction():
    return render_template('predictions/ml_predictions.html')

@blueprint.route('/location')
def locations():
    return render_template('locations/view_locations.html')

@blueprint.route('/soil')
def soil():
    return render_template('soil/view_soil.html')

@blueprint.route('/weather')
def weather():
    return render_template('weather/view_weather.html')

@blueprint.route('/geocode')
def geocode():
    address = request.args.get('address')
    if not address:
        return jsonify({'error': 'Address parameter is required'}), 400
    
    result = get_lat_lon(address)
    if not result:
        return jsonify({'error': 'No geocoding result found'}), 404
    
    return jsonify(result)

@blueprint.route('/soil-info', methods=['GET'])
def get_soil_info():
    address = request.args.get('address')
    if not address:
        return jsonify({'error': 'Address is required as a query parameter'}), 400

    geo_data = get_lat_lon(address)
    if not geo_data:
        return jsonify({'error': 'Failed to get geolocation data'}), 500

    lat = geo_data.get('lat')
    lon = geo_data.get('lon')
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude could not be found'}), 500

    soil_data = fetch_soil_data(lat, lon)
    if soil_data is None:
        return jsonify({'error': 'Failed to fetch soil data from SoilGrids'}), 500

    response_data = {
        'location': {
            'address': geo_data.get('display_name'),
            'lat': lat,
            'lon': lon
        },
        'soil': soil_data
    }
    
    return jsonify(response_data)

@blueprint.route('/weather-info', methods=['GET'])
def get_weather_info():
    city = request.args.get('city')
    if not city:
        return jsonify({"error": "City parameter is required"}), 400
    
    weather = fetch_weather_data(city)
    if all(value is None for value in weather.values()):
        return jsonify({"error": "Failed to fetch weather data"}), 500
    
    return jsonify({
        "city": city,
        "weather": weather
    })

@blueprint.route('/model-input', methods=['GET'])
def model_input():
    location = request.args.get('location')
    if not location:
        return jsonify({'error': 'Location parameter is required'}), 400

    features = get_model_input_features(location)
    if not features:
        return jsonify({'error': 'Failed to fetch complete model input features'}), 500

    # Standardize the features for model input
    standardized_features = standardize_model_inputs(features)
    
    # Log the standardization for debugging
    print_standardization_summary(features, standardized_features)

    return jsonify({
        'location': location,
        'features': standardized_features,
        'original_features': features
    }), 200


@blueprint.route('/predict', methods=['POST'])
def predict():
    if not model or not label_encoder:
        return jsonify({'error': 'Model or label encoder not loaded'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON data is required in the request body'}), 400
    
    # Handle features in either root or 'features' key
    features = data.get('features', data)
    desired_crop = data.get('desired_crop', '').lower().strip()

    # The exact feature order used during model training
    required_features = ['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall']
    
    # Normalize feature keys to lowercase for case-insensitive matching
    features = {key.lower(): value for key, value in features.items()}
    required_features_lower = [key.lower() for key in required_features]
    
    # Log expected vs received feature names
    received_features = list(features.keys())
    logger.info(f"Required features (order matters): {required_features}")
    logger.info(f"Received features: {received_features}")
    
    # Check that all required features are present
    if not all(key in features for key in required_features_lower):
        missing = [key for key in required_features_lower if key not in features]
        return jsonify({'error': f'Missing required features: {missing}'}), 400

    # Validate feature value ranges
    feature_ranges = {
        'n': (0, 200),  # Nitrogen in ppm
        'p': (0, 150),  # Phosphorus in ppm
        'k': (0, 200),  # Potassium in ppm
        'ph': (0, 14),  # pH range
        'temperature': (-10, 50),  # Celsius
        'humidity': (0, 100),  # Percentage
        'rainfall': (0, 1000)  # mm
    }
    
    try:
        # Create a list with features in the EXACT order expected by the model
        feature_values = [float(features[key]) for key in required_features_lower]
        
        # Validate ranges
        for key, value in zip(required_features_lower, feature_values):
            min_val, max_val = feature_ranges[key]
            if not (min_val <= value <= max_val):
                return jsonify({'error': f'Value for {key} ({value}) is out of valid range [{min_val}, {max_val}]'}), 400
        
        # Convert to DataFrame with correct column names and order
        feature_df = pd.DataFrame([feature_values], columns=required_features)
        
        # Apply standardization to match training
        standardized_features = standardize_model_inputs(feature_df.to_dict(orient='records')[0])
        # Ensure standardized_features is a dict with the correct keys
        if not all(key in standardized_features for key in required_features):
            logger.error(f"Standardized features missing required keys: {standardized_features}")
            return jsonify({'error': 'Standardized features do not match required features'}), 500
        
        # Reconstruct DataFrame with standardized values in correct order
        standardized_values = [float(standardized_features[key]) for key in required_features]
        feature_df = pd.DataFrame([standardized_values], columns=required_features)
        
        logger.info(f"Processing features with shape {feature_df.shape}:")
        logger.info(f"Column order: {list(feature_df.columns)}")
        logger.info(f"Standardized values: {feature_df.iloc[0].tolist()}")
        
        # Make prediction using numpy array to bypass column name issues
        feature_array = feature_df[required_features].to_numpy()
        probabilities = model.predict_proba(feature_array)[0]
        predicted_class_idx = model.predict(feature_array)[0]
        
        # Validate predicted class index
        if predicted_class_idx >= len(label_encoder.classes_):
            logger.error(f"Predicted class index {predicted_class_idx} is out of bounds for classes: {label_encoder.classes_}")
            return jsonify({'error': 'Invalid prediction index from model'}), 500

        predicted_crop = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = float(probabilities[predicted_class_idx])

        # Get top 4 predictions (primary + top 3 alternatives)
        top_indices = np.argsort(probabilities)[-4:][::-1]  # Descending order
        predictions = []
        for idx in top_indices:
            if idx < len(label_encoder.classes_):
                predictions.append({
                    'crop': label_encoder.inverse_transform([idx])[0],
                    'probability': float(probabilities[idx])
                })
            else:
                logger.warning(f"Skipping invalid class index {idx}")

        # Ensure primary prediction is the first in the list
        if predictions and predictions[0]['crop'] != predicted_crop:
            predictions = sorted(predictions, key=lambda x: x['crop'] == predicted_crop, reverse=True)

        # Check suitability of desired crop (if provided)
        suitability = None
        if desired_crop:
            try:
                # Check if the desired crop is in the label encoder classes
                if desired_crop not in label_encoder.classes_:
                    suitability = {
                        'crop': desired_crop,
                        'status': 'unknown',
                        'message': 'The specified crop is not recognized by the model.'
                    }
                else:
                    desired_class_idx = label_encoder.transform([desired_crop])[0]
                    desired_confidence = probabilities[desired_class_idx]
                    if desired_confidence >= 0.7:
                        status = 'highly suitable'
                        message = f"{desired_crop.capitalize()} is highly suitable for your location (confidence: {(desired_confidence * 100):.1f}%)."
                    elif desired_confidence >= 0.3:
                        status = 'moderately suitable'
                        message = f"{desired_crop.capitalize()} is moderately suitable for your location (confidence: {(desired_confidence * 100):.1f}%). Consider the recommended crops for better results."
                    else:
                        status = 'not suitable'
                        message = f"{desired_crop.capitalize()} is not suitable for your location (confidence: {(desired_confidence * 100):.1f}%). The recommended crops are better suited."
                    suitability = {
                        'crop': desired_crop,
                        'status': status,
                        'message': message,
                        'confidence': float(desired_confidence)
                    }
            except Exception as e:
                logger.error(f"Error checking crop suitability: {str(e)}")
                suitability = {
                    'crop': desired_crop,
                    'status': 'error',
                    'message': 'Could not evaluate suitability for this crop.'
                }

        response = {
            'predictions': predictions,
            'suitability': suitability
        }
        logger.info(f"Prediction successful: {response}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}. Input features: {features}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500