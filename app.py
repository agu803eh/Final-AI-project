from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
from datetime import datetime
import logging
import openai
import json
import random
import sys
import traceback
import re
import time
import requests
from pydantic import BaseModel, Field, ValidationError
from flask_cors import CORS
import csv

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

application = app

# Global variables for model and statistics
model_iso = None
training_stats = None

# Configuration constants
MODEL_PATH = 'health_model.pkl'
STATS_PATH = 'training_stats.pkl'
PREDICTION_LOG_FILE = "predictions.csv"

# FIXED: OpenAI API Key setup with proper client initialization
openai.api_key = os.getenv("OPENAI_API_KEY")

# ADDED: Verify OpenAI API key is set
if not openai.api_key:
    logger.warning("OPENAI_API_KEY environment variable not set. OpenAI features will be disabled.")

class HealthInput(BaseModel):
    heart_rate: float = Field(..., ge=30, le=250)
    bvp: float = Field(..., ge=-1000, le=1000)

def explain_results_with_ai(heart_rate, bvp, status, recommendation):
    """
    Use OpenAI to explain the prediction in user-friendly language.
    FIXED: Updated to use new OpenAI API format
    """
    if not openai.api_key:
        return "AI explanation unavailable - OpenAI API key not configured."
    
    prompt = (
        f"A health monitoring system detected:\n"
        f"- Heart Rate: {heart_rate} bpm\n"
        f"- BVP: {bvp}\n"
        f"- Status: {status}\n"
        f"- Recommendation: {recommendation}\n\n"
        f"Please explain this result in plain English for a non-medical user, and suggest next steps."
    )

    try:
        # FIXED: Updated to use new OpenAI API format (v1.0+)
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return "AI explanation unavailable due to an internal error."

@app.route('/insight', methods=['POST'])
def openai_insight():
    """Generate insight using OpenAI based on latest health readings."""
    if not openai.api_key:
        return jsonify({"error": "OpenAI API key not configured"}), 500
        
    try:
        data = request.get_json()
        
        # ADDED: Validate required fields
        required_fields = ['heart_rate', 'bvp', 'status', 'recommendation']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        heart_rate = data.get("heart_rate")
        bvp = data.get("bvp")
        status = data.get("status")
        recommendation = data.get("recommendation")

        prompt = (
            f"You are a medical assistant AI. Analyze this user's health data:\n"
            f"Heart rate: {heart_rate} bpm\n"
            f"BVP: {bvp}\n"
            f"Status: {status}\n"
            f"Recommendation: {recommendation}\n"
            f"\n"
            f"Provide a helpful, beginner-friendly explanation of what these results mean,"
            f" and how the user can improve their health based on it."
        )

        # FIXED: Updated to use new OpenAI API format
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful health assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        explanation = response.choices[0].message.content.strip()
        return jsonify({"insight": explanation})

    except Exception as e:
        logger.error(f"OpenAI Insight Error: {str(e)}")
        return jsonify({"error": "Failed to generate insight."}), 500

@app.route('/chat', methods=['POST'])
def health_chat():
    """General-purpose chat endpoint for health advice."""
    if not openai.api_key:
        return jsonify({"error": "OpenAI API key not configured"}), 500
        
    try:
        data = request.get_json()
        user_input = data.get("message")

        if not user_input:
            return jsonify({"error": "Message is required."}), 400

        # FIXED: Updated to use new OpenAI API format
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a friendly virtual health advisor. Provide helpful health advice but always remind users to consult healthcare professionals for serious concerns."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=300
        )

        answer = response.choices[0].message.content.strip()
        return jsonify({"response": answer})

    except Exception as e:
        logger.error(f"Health Chat Error: {str(e)}")
        return jsonify({"error": "Failed to respond to message."}), 500
    
def load_or_create_model():
    """Load existing model or create a new one with improved parameters"""
    global model_iso, training_stats

    if os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH):
        model_iso = joblib.load(MODEL_PATH)
        training_stats = joblib.load(STATS_PATH)
        logger.info("Loaded existing model and statistics")
    else:
        logger.info("Creating new model...")

        # Create more realistic and diverse training data
        np.random.seed(42)
        n_samples = 5000
        
        # Generate normal data (80% of samples)
        normal_samples = int(n_samples * 0.8)
        normal_hr = np.random.normal(72, 12, normal_samples)  # Normal resting HR
        normal_bvp = np.random.normal(0, 25, normal_samples)  # Normal BVP variation
        
        # Generate abnormal data (20% of samples)
        abnormal_samples = n_samples - normal_samples
        abnormal_hr = np.concatenate([
            np.random.normal(45, 8, abnormal_samples//4),  # Bradycardia
            np.random.normal(120, 15, abnormal_samples//4),  # Tachycardia
            np.random.normal(35, 5, abnormal_samples//4),   # Severe bradycardia
            np.random.normal(140, 20, abnormal_samples//4)  # Severe tachycardia
        ])
        abnormal_bvp = np.concatenate([
            np.random.normal(0, 50, abnormal_samples//2),   # High variation
            np.random.normal(100, 30, abnormal_samples//4), # High values
            np.random.normal(-100, 30, abnormal_samples//4) # Low values
        ])
        
        # Combine normal and abnormal data
        all_hr = np.concatenate([normal_hr, abnormal_hr])
        all_bvp = np.concatenate([normal_bvp, abnormal_bvp])
        
        # Create labels (1 for normal, -1 for abnormal)
        labels = np.concatenate([
            np.ones(normal_samples),
            -np.ones(abnormal_samples)
        ])
        
        # Create DataFrame
        merged_df = pd.DataFrame({
            'heart_rate': all_hr,
            'bvp': all_bvp,
            'heart_rate_mean_60s': all_hr + np.random.normal(0, 2, n_samples),
            'heart_rate_std_60s': np.abs(np.random.normal(5, 2, n_samples)),
            'bvp_mean_60s': all_bvp + np.random.normal(0, 5, n_samples),
            'bvp_std_60s': np.abs(np.random.normal(20, 8, n_samples)),
            'heart_rate_std_10s': np.abs(np.random.normal(3, 1, n_samples)),
            'labels': labels
        })
        
        # Shuffle the data
        merged_df = merged_df.sample(frac=1).reset_index(drop=True)

        features = ['heart_rate', 'bvp', 'heart_rate_mean_60s', 'heart_rate_std_60s', 
                   'bvp_mean_60s', 'bvp_std_60s', 'heart_rate_std_10s']
        X_train_iso = merged_df[features]

        # FIXED: More balanced contamination rate
        model_iso = IsolationForest(
            n_estimators=200, 
            contamination=0.15,  # Reduced from 0.2 to 0.15
            random_state=42,
            max_samples=0.8
        )
        model_iso.fit(X_train_iso)

        training_stats = {
            'heart_rate_mean': merged_df['heart_rate'].mean(),
            'heart_rate_std': merged_df['heart_rate'].std(),
            'bvp_mean': merged_df['bvp'].mean(),
            'bvp_std': merged_df['bvp'].std(),
            'features': features,
            'normal_hr_range': [60, 100],
            'normal_bvp_range': [-50, 50]
        }

        joblib.dump(model_iso, MODEL_PATH)
        joblib.dump(training_stats, STATS_PATH)
        logger.info("Model trained and saved with improved parameters")

def log_prediction(data):
    """Log prediction results to CSV file"""
    header = ['timestamp', 'heart_rate', 'bvp', 'status', 'anomaly_score', 'confidence', 'rule_check', 'ml_check']
    file_exists = os.path.isfile(PREDICTION_LOG_FILE)
    with open(PREDICTION_LOG_FILE, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': data["timestamp"],
            'heart_rate': data.get("heart_rate", "N/A"),
            'bvp': data.get("bvp", "N/A"),
            'status': data["status"],
            'anomaly_score': data.get("anomaly_score", "N/A"),
            'confidence': data["confidence"],
            'rule_check': data.get("rule_check", "N/A"),
            'ml_check': data.get("ml_check", "N/A")
        })

def rule_based_check(heart_rate, bvp):
    """
    FIXED: More balanced rule-based health check
    Returns: (status, message)
    """
    messages = []
    status = 'normal'
    
    # FIXED: More balanced heart rate checks
    if heart_rate < 45:
        status = 'abnormal'
        messages.append('Severe bradycardia (very low heart rate)')
    elif heart_rate < 55:
        status = 'warning'
        messages.append('Bradycardia (low heart rate)')
    elif heart_rate > 130:
        status = 'abnormal'
        messages.append('Severe tachycardia (very high heart rate)')
    elif heart_rate > 110:
        status = 'warning'
        messages.append('Tachycardia (elevated heart rate)')
    else:
        messages.append('Heart rate within normal range')
    
    # FIXED: More balanced BVP checks
    if abs(bvp) > 150:
        status = 'abnormal'
        messages.append('Extreme BVP values detected')
    elif abs(bvp) > 100:
        if status == 'normal':
            status = 'warning'
        messages.append('Elevated BVP levels')
    else:
        messages.append('BVP within acceptable range')
    
    return status, '; '.join(messages)

def generate_recommendation(heart_rate, bvp, status, rule_message):
    """Generate comprehensive recommendations based on all inputs"""
    if status == 'abnormal':
        if heart_rate > 130:
            return f"High heart rate detected ({heart_rate} bpm). Please rest and seek medical attention if symptoms persist."
        elif heart_rate < 45:
            return f"Very low heart rate detected ({heart_rate} bpm). Please consult a healthcare professional immediately."
        elif abs(bvp) > 150:
            return "Extreme BVP values detected. Please check your sensor placement or consult a healthcare provider."
        else:
            return "Abnormal readings detected. Please monitor your health closely and consult a healthcare professional."
    elif status == 'warning':
        return "Some readings are outside normal range but not critical. Continue monitoring and consider consulting a healthcare provider if symptoms persist."
    else:
        return "Readings are within normal range. Continue regular health monitoring."

def detect_and_recommend(heart_rate_data, bvp_data):
    """FIXED: More balanced comprehensive health detection"""
    try:
        # Rule-based check first
        rule_status, rule_message = rule_based_check(heart_rate_data, bvp_data)
        
        # FIXED: More realistic ML-based anomaly detection
        heart_rate_deviation = abs(heart_rate_data - training_stats['heart_rate_mean'])
        bvp_deviation = abs(bvp_data - training_stats['bvp_mean'])
        
        # Create more realistic features
        input_data = pd.DataFrame({
            'heart_rate': [heart_rate_data],
            'bvp': [bvp_data],
            'heart_rate_mean_60s': [heart_rate_data + np.random.normal(0, 1)],
            'heart_rate_std_60s': [max(1, heart_rate_deviation * 0.2)],
            'bvp_mean_60s': [bvp_data + np.random.normal(0, 2)],
            'bvp_std_60s': [max(1, bvp_deviation * 0.3)],
            'heart_rate_std_10s': [max(1, heart_rate_deviation * 0.1)]
        })

        input_features = input_data[training_stats['features']]
        anomaly_prediction = model_iso.predict(input_features)
        anomaly_score = model_iso.score_samples(input_features)[0]

        # FIXED: More balanced ML status determination
        if anomaly_score < -0.6:  # Strong anomaly signal
            ml_status = 'abnormal'
        elif anomaly_score < -0.4:  # Mild anomaly signal
            ml_status = 'warning'
        else:
            ml_status = 'normal'
        
        # FIXED: Better combination of rule-based and ML results
        status_weights = {'abnormal': 3, 'warning': 2, 'normal': 1}
        rule_weight = status_weights.get(rule_status, 1)
        ml_weight = status_weights.get(ml_status, 1)
        
        # Final status determination - prioritize rule-based for normal values
        if rule_status == 'normal' and ml_status == 'normal':
            final_status = 'normal'
        elif rule_status == 'normal' and ml_status == 'warning':
            final_status = 'normal'  # Trust rule-based for borderline cases
        elif rule_weight >= 3 or ml_weight >= 3:  # Either method says abnormal
            final_status = 'abnormal'
        elif rule_weight >= 2 or ml_weight >= 2:  # Either method says warning
            final_status = 'warning'
        else:
            final_status = 'normal'
        
        # Generate recommendation
        recommendation = generate_recommendation(heart_rate_data, bvp_data, final_status, rule_message)
        
        # ADDED: AI explanation if OpenAI is available
        ai_explanation = None
        if openai.api_key:
            try:
                ai_explanation = explain_results_with_ai(heart_rate_data, bvp_data, final_status, recommendation)
            except Exception as e:
                logger.error(f"AI explanation failed: {e}")
        
        # Calculate confidence
        rule_ml_agree = rule_status == ml_status
        score_confidence = 'high' if abs(anomaly_score) > 0.4 else 'medium' if abs(anomaly_score) > 0.2 else 'low'
        
        if rule_ml_agree and score_confidence == 'high':
            confidence = 'high'
        elif rule_ml_agree or score_confidence == 'high':
            confidence = 'medium'
        else:
            confidence = 'low'

        results = {
            "timestamp": datetime.now().isoformat(),
            "heart_rate": heart_rate_data,
            "bvp": bvp_data,
            "status": final_status,
            "anomaly_score": float(anomaly_score),
            "recommendation": recommendation,
            "confidence": confidence,
            "rule_check": rule_message,
            "ml_check": f"ML prediction: {ml_status} (score: {anomaly_score:.3f})",
            "details": {
                "rule_based_status": rule_status,
                "ml_based_status": ml_status,
                "anomaly_score": float(anomaly_score),
                "methods_agree": rule_ml_agree
            }
        }
        
        # ADDED: Include AI explanation if available
        if ai_explanation:
            results["ai_explanation"] = ai_explanation

        log_prediction(results)
        return results

    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "recommendation": "Unable to process data. Please try again.",
            "confidence": "low",
            "error": str(e)
        }

@app.route('/download_logs', methods=['GET'])
def download_logs():
    """Download prediction logs as CSV"""
    if not os.path.exists(PREDICTION_LOG_FILE):
        return jsonify({"error": "No prediction logs found."}), 404

    try:
        with open(PREDICTION_LOG_FILE, 'r') as file:
            content = file.read()
        response = app.response_class(
            response=content,
            status=200,
            mimetype='text/csv',
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
        return response
    except Exception as e:
        logger.error(f"Error reading log file: {str(e)}")
        return jsonify({"error": "Failed to read log file."}), 500
    
@app.route('/health', methods=['POST'])
def health_check():
    """Perform a health check on the application."""
    if model_iso:
        return jsonify({
            "status": "ok", 
            "message": "Model is loaded and ready.",
            "openai_configured": bool(openai.api_key)
        }), 200
    else:
        return jsonify({"status": "error", "message": "Model is not loaded."}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with improved logic"""
    try:
        data = request.json
        logger.info(f"Received prediction request: {data}")
        
        # Validate input using Pydantic
        try:
            health_input = HealthInput(**data)
            heart_rate = health_input.heart_rate
            bvp = health_input.bvp
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": f"Invalid input data: {str(e)}"}), 400

        result = detect_and_recommend(heart_rate, bvp)
        logger.info(f"Prediction result: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/test', methods=['POST'])
def test_cases():
    """Test endpoint with predefined cases to verify functionality"""
    test_cases = [
        {"heart_rate": 75, "bvp": 0, "expected": "normal"},
        {"heart_rate": 70, "bvp": 10, "expected": "normal"},
        {"heart_rate": 85, "bvp": -5, "expected": "normal"},
        {"heart_rate": 140, "bvp": 0, "expected": "abnormal"},
        {"heart_rate": 40, "bvp": 0, "expected": "abnormal"},
        {"heart_rate": 80, "bvp": 200, "expected": "abnormal"},
        {"heart_rate": 105, "bvp": 50, "expected": "warning"},
        {"heart_rate": 55, "bvp": 80, "expected": "warning"}
    ]
    
    results = []
    for case in test_cases:
        result = detect_and_recommend(case["heart_rate"], case["bvp"])
        results.append({
            "input": {"heart_rate": case["heart_rate"], "bvp": case["bvp"]},
            "expected": case["expected"],
            "actual": result["status"],
            "recommendation": result["recommendation"],
            "anomaly_score": result["anomaly_score"]
        })
    
    return jsonify({"test_results": results})

@app.route('/train', methods=['POST'])
def train():
    """Endpoint to retrain the model with new data."""
    try:
        # Remove existing model files to force recreation
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(STATS_PATH):
            os.remove(STATS_PATH)
        
        load_or_create_model()
        return jsonify({"status": "ok", "message": "Model retrained successfully."}), 200
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Endpoint to check the status of the application."""
    if model_iso:
        return jsonify({
            "status": "ok", 
            "message": "Model is loaded and ready.",
            "model_type": "Isolation Forest with rule-based validation",
            "contamination": "0.15 (15%)",
            "features": training_stats['features'] if training_stats else "Unknown",
            "openai_configured": bool(openai.api_key),
            "openai_model": "gpt-4"
        }), 200
    return jsonify({"error": "Failed to check status."}), 500

@app.route('/')
def index():
    """Root endpoint with API information"""
    return jsonify({
        "message": "Health Monitoring API is running",
        "version": "2.3 - Fixed OpenAI integration and improved functionality",
        "openai_features": bool(openai.api_key),
        "endpoints": {
            "/predict": "POST - Main prediction endpoint",
            "/insight": "POST - Generate AI insights for health readings",
            "/chat": "POST - General health chat with AI",
            "/test": "POST - Test with predefined cases",
            "/health": "POST - Health check",
            "/status": "GET - Application status",
            "/train": "POST - Retrain model",
            "/download_logs": "GET - Download prediction logs"
        }
    })

if __name__ == '__main__':
    try:
        load_or_create_model()
        logger.info("Starting Health Monitoring API...")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        logger.info("Health Monitoring API started successfully.")
    except Exception as e:
        logger.error(f"Failed to start the Health Monitoring API: {str(e)}")
        print("ERROR: Unable to start the Flask app. Please check the logs for details.")