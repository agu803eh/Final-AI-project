import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Flask application
app = Flask(__name__)

# Global variables for model and statistics
model_iso = None
training_stats = None

def load_or_create_model():
    """Load existing model or create and train a new one"""
    global model_iso, training_stats
    
    model_path = 'health_model.pkl'
    stats_path = 'training_stats.pkl'
    
    if os.path.exists(model_path) and os.path.exists(stats_path):
        # Load existing model and stats
        model_iso = joblib.load(model_path)
        training_stats = joblib.load(stats_path)
        logger.info("Loaded existing model and statistics")
    else:
        # Create and train new model
        logger.info("Creating new model...")
        
        # Create training data (in production, load from database/file)
        np.random.seed(42)
        n_samples = 2000
        merged_df = pd.DataFrame({
            'heart_rate': np.random.normal(70, 10, n_samples),
            'bvp': np.random.normal(0, 20, n_samples),
            'heart_rate_mean_60s': np.random.normal(70, 5, n_samples),
            'heart_rate_std_60s': np.random.normal(5, 2, n_samples).clip(0),
            'bvp_mean_60s': np.random.normal(0, 10, n_samples),
            'bvp_std_60s': np.random.normal(15, 5, n_samples).clip(0),
        }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=n_samples, freq='s')))

        # Introduce anomalies
        merged_df.iloc[100:110, merged_df.columns.get_loc('heart_rate')] = np.random.normal(150, 20, 10)
        merged_df.iloc[200:210, merged_df.columns.get_loc('bvp')] = np.random.normal(200, 50, 10)

        # Feature engineering
        merged_df['heart_rate_std_10s'] = merged_df['heart_rate'].rolling(window='10s').std()
        merged_df = merged_df.fillna(method='bfill').fillna(method='ffill')

        # Train model
        features = ['heart_rate', 'bvp', 'heart_rate_mean_60s', 'heart_rate_std_60s', 
                   'bvp_mean_60s', 'bvp_std_60s', 'heart_rate_std_10s']
        X_train_iso = merged_df[features]

        model_iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        model_iso.fit(X_train_iso)

        # Store training statistics
        training_stats = {
            'heart_rate_mean': merged_df['heart_rate'].mean(),
            'heart_rate_std': merged_df['heart_rate'].std(),
            'bvp_mean': merged_df['bvp'].mean(),
            'bvp_std': merged_df['bvp'].std(),
            'features': features
        }

        # Save model and stats
        joblib.dump(model_iso, model_path)
        joblib.dump(training_stats, stats_path)
        logger.info("Model trained and saved")

def validate_input_data(data):
    """Validate incoming data"""
    if not isinstance(data, dict):
        return False, "Data must be a JSON object"
    
    required_fields = ['heart_rate', 'bvp']
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        # Check if value is numeric
        try:
            float(data[field])
        except (ValueError, TypeError):
            return False, f"Field '{field}' must be a numeric value"
    
    # Check for reasonable ranges
    if not (30 <= data['heart_rate'] <= 250):
        return False, "Heart rate must be between 30 and 250 bpm"
    
    if not (-1000 <= data['bvp'] <= 1000):
        return False, "BVP value out of reasonable range"
    
    return True, "Valid"

def detect_and_recommend(heart_rate_data, bvp_data):
    """
    Processes incoming health data, detects anomalies using the trained model,
    and generates recommendations.
    """
    try:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'heart_rate': [heart_rate_data],
            'bvp': [bvp_data],
            'heart_rate_mean_60s': [heart_rate_data],  # Simplified
            'heart_rate_std_60s': [0],  # Simplified
            'bvp_mean_60s': [bvp_data],  # Simplified
            'bvp_std_60s': [0],  # Simplified
            'heart_rate_std_10s': [0]  # Simplified
        })

        # Select features for prediction
        input_features = input_data[training_stats['features']]

        # Predict anomaly
        anomaly_prediction = model_iso.predict(input_features)
        anomaly_score = model_iso.score_samples(input_features)[0]

        # Convert prediction to label
        anomaly_label = 'abnormal' if anomaly_prediction[0] == -1 else 'normal'

        # Generate detailed recommendations
        recommendation = generate_recommendation(heart_rate_data, bvp_data, anomaly_label)

        results = {
            "timestamp": datetime.now().isoformat(),
            "status": anomaly_label,
            "anomaly_score": float(anomaly_score),
            "recommendation": recommendation,
            "confidence": "high" if abs(anomaly_score) > 0.1 else "medium"
        }

        return results

    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "recommendation": "Unable to process data. Please try again.",
            "error": str(e)
        }

def generate_recommendation(heart_rate, bvp, anomaly_status):
    """Generate detailed health recommendations"""
    if anomaly_status == 'normal':
        return "Your health metrics appear normal. Keep up the good work!"
    
    recommendations = []
    
    # Heart rate analysis
    hr_mean = training_stats['heart_rate_mean']
    hr_std = training_stats['heart_rate_std']
    
    if heart_rate > hr_mean + 2 * hr_std:
        recommendations.append("High heart rate detected. Consider resting and staying hydrated.")
    elif heart_rate < hr_mean - 2 * hr_std:
        recommendations.append("Low heart rate detected. If persistent, consult a healthcare provider.")
    
    # BVP analysis
    bvp_std = training_stats['bvp_std']
    if abs(bvp) > 2 * bvp_std:
        recommendations.append("Irregular blood volume pulse detected. Monitor your stress levels.")
    
    if not recommendations:
        recommendations.append("Anomaly detected in your health metrics. Monitor your condition closely.")
    
    return " ".join(recommendations)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_iso is not None
    })

# Main prediction endpoint
@app.route('/predict_anomaly', methods=['POST'])
def predict_anomaly():
    """
    Receives health data via POST request, detects anomalies, and returns recommendations.
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate input
        is_valid, message = validate_input_data(data)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        # Extract values
        heart_rate_value = float(data['heart_rate'])
        bvp_value = float(data['bvp'])
        
        # Process and get results
        results = detect_and_recommend(heart_rate_value, bvp_value)
        
        # Log the prediction
        logger.info(f"Prediction made: HR={heart_rate_value}, BVP={bvp_value}, Status={results.get('status', 'unknown')}")
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in predict_anomaly: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }), 500

# Batch prediction endpoint
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Process multiple health data points at once"""
    try:
        data = request.get_json()
        
        if not isinstance(data, dict) or 'measurements' not in data:
            return jsonify({"error": "Expected JSON with 'measurements' array"}), 400
        
        measurements = data['measurements']
        if not isinstance(measurements, list):
            return jsonify({"error": "'measurements' must be an array"}), 400
        
        results = []
        for i, measurement in enumerate(measurements):
            is_valid, message = validate_input_data(measurement)
            if not is_valid:
                results.append({
                    "index": i,
                    "error": message,
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            result = detect_and_recommend(
                float(measurement['heart_rate']),
                float(measurement['bvp'])
            )
            result['index'] = i
            results.append(result)
        
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Initialize model when app starts
load_or_create_model()

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
