from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from datetime import datetime
import logging
from openai import OpenAI
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
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
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

# FIXED: Better OpenAI client initialization with multiple methods
openai_client = None
    
# REPLACE lines 40-84 in your app.py with this SINGLE function:

def initialize_openai_client():
    """Initialize OpenAI client with multiple fallback methods"""
    global openai_client
    
    # Method 1: Try environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Method 2: Try different common environment variable names
    if not api_key:
        api_key = os.getenv("OPENAI_KEY")
    
    # Method 3: Try reading from config file
    if not api_key:
        try:
            config_file = "config.json"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    api_key = config.get("openai_api_key")
        except Exception as e:
            logger.warning(f"Could not read config file: {e}")
    
    if api_key and api_key.strip() and api_key != "your_openai_api_key_here":
        try:
            # Initialize OpenAI client
            openai_client = OpenAI(api_key=api_key.strip())
            
            # Test the connection with a simple request
            test_response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                timeout=10
            )
            logger.info("✅ OpenAI client initialized and tested successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            openai_client = None
            return False
    else:
        logger.warning("⚠️  OPENAI_API_KEY not found or invalid. OpenAI features will be disabled.")
        logger.info("💡 To enable OpenAI features:")
        logger.info("   1. Set OPENAI_API_KEY environment variable")
        logger.info("   2. Create .env file with OPENAI_API_KEY=your_key_here")
        logger.info("   3. Create config.json with {\"openai_api_key\": \"your_key_here\"}")
        openai_client = None
        return False

class HealthInput(BaseModel):
    heart_rate: float = Field(..., ge=30, le=250)
    bvp: float = Field(..., ge=-1000, le=1000)

def explain_results_with_ai(heart_rate, bvp, status, recommendation):
    """Use OpenAI to explain the prediction in user-friendly language."""
    if not openai_client:
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
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using gpt-3.5-turbo for cost efficiency
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
            timeout=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return f"AI explanation unavailable due to an error: {str(e)}"

@app.route('/insight', methods=['POST'])
def openai_insight():
    """Generate insight using OpenAI based on latest health readings."""
    if not openai_client:
        return jsonify({
            "error": "OpenAI API key not configured",
            "details": "Please set OPENAI_API_KEY environment variable or create .env file"
        }), 500
        
    try:
        data = request.get_json()
        
        # Validate required fields
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
            f"Provide a helpful, beginner-friendly explanation of what these results mean, "
            f"and how the user can improve their health based on it."
        )

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful health assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
            timeout=30
        )

        explanation = response.choices[0].message.content.strip()
        return jsonify({"insight": explanation})

    except Exception as e:
        logger.error(f"OpenAI Insight Error: {str(e)}")
        return jsonify({"error": f"Failed to generate insight: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def health_chat():
    """General-purpose chat endpoint for health advice."""
    logger.info("Chat endpoint called")
    
    if not openai_client:
        logger.error("OpenAI client not initialized")
        return jsonify({
            "error": "OpenAI API key not configured",
            "response": "I'm sorry, but the AI chat feature is not available right now. Please configure your OpenAI API key.",
            "details": "OpenAI API key not found. Please set OPENAI_API_KEY environment variable or create .env file with your API key"
        }), 200  # Changed to 200 so frontend gets the response
        
    try:
        data = request.get_json()
        logger.info(f"Received chat data: {data}")
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "response": "Please provide a message to chat with me."
            }), 400
            
        user_input = data.get("message", "").strip()
        logger.info(f"User message: {user_input}")

        if not user_input:
            return jsonify({
                "error": "Message is required",
                "response": "Please provide a message to chat with me."
            }), 400

        try:
            logger.info("Making OpenAI API call...")
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a friendly virtual health advisor. Provide helpful health advice but always remind users to consult healthcare professionals for serious concerns. Keep responses concise and practical."},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=300,
                timeout=30
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"OpenAI response received: {answer[:100]}...")
            
            return jsonify({
                "response": answer,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as api_error:
            logger.error(f"OpenAI API Error: {api_error}")
            error_msg = f"Sorry, I'm having trouble connecting to the AI service right now. Error: {str(api_error)}"
            return jsonify({
                "error": f"OpenAI API Error: {str(api_error)}",
                "response": error_msg,
                "status": "error"
            }), 200  # Return 200 so frontend gets the error message

    except Exception as e:
        logger.error(f"Health Chat Error: {str(e)}", exc_info=True)
        error_msg = f"I encountered an error while processing your message. Please try again. Error: {str(e)}"
        return jsonify({
            "error": f"Failed to respond to message: {str(e)}",
            "response": error_msg,
            "status": "error"
        }), 200  # Return 200 so frontend gets the error message

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

        model_iso = IsolationForest(
            n_estimators=200, 
            contamination=0.15,
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
    """More balanced rule-based health check"""
    messages = []
    status = 'normal'
    
    # Heart rate checks
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
    
    # BVP checks
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
    """Comprehensive health detection with improved logic"""
    try:
        # Rule-based check first
        rule_status, rule_message = rule_based_check(heart_rate_data, bvp_data)
        
        # ML-based anomaly detection
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

        # ML status determination
        if anomaly_score < -0.6:
            ml_status = 'abnormal'
        elif anomaly_score < -0.4:
            ml_status = 'warning'
        else:
            ml_status = 'normal'
        
        # Combine rule-based and ML results
        status_weights = {'abnormal': 3, 'warning': 2, 'normal': 1}
        rule_weight = status_weights.get(rule_status, 1)
        ml_weight = status_weights.get(ml_status, 1)
        
        # Final status determination
        if rule_status == 'normal' and ml_status == 'normal':
            final_status = 'normal'
        elif rule_status == 'normal' and ml_status == 'warning':
            final_status = 'normal'
        elif rule_weight >= 3 or ml_weight >= 3:
            final_status = 'abnormal'
        elif rule_weight >= 2 or ml_weight >= 2:
            final_status = 'warning'
        else:
            final_status = 'normal'
        
        # Generate recommendation
        recommendation = generate_recommendation(heart_rate_data, bvp_data, final_status, rule_message)
        
        # AI explanation if OpenAI is available
        ai_explanation = None
        if openai_client:
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
        
        # Include AI explanation if available
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
            "openai_configured": bool(openai_client),
            "openai_status": "✅ Connected" if openai_client else "❌ Not configured"
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
    openai_status = "✅ Connected and tested" if openai_client else "❌ Not configured"
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    
    if model_iso:
        return jsonify({
            "status": "ok", 
            "message": "Model is loaded and ready.",
            "model_type": "Isolation Forest with rule-based validation",
            "contamination": "0.15 (15%)",
            "features": training_stats['features'] if training_stats else "Unknown",
            "openai_configured": bool(openai_client),
            "openai_status": openai_status,
            "openai_model": "gpt-3.5-turbo" if openai_client else "Not configured",
            "api_key_in_env": api_key_present,
            "debug_info": {
                "openai_client_exists": openai_client is not None,
                "env_api_key_exists": api_key_present,
                "env_api_key_length": len(os.getenv("OPENAI_API_KEY", "")) if api_key_present else 0
            }
        }), 200
    return jsonify({"error": "Model not loaded, failed to check status."}), 500

# NEW: API key setup endpoint
@app.route('/setup-openai', methods=['POST'])
def setup_openai():
    """Endpoint to set up OpenAI API key"""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({
                "status": "error",
                "message": "API key is required"
            }), 400
        
        # Validate API key format
        if not api_key.startswith('sk-'):
            return jsonify({
                "status": "error",
                "message": "Invalid API key format. OpenAI keys should start with 'sk-'"
            }), 400
        
        try:
            # Test the API key first
            test_client = OpenAI(api_key=api_key)
            test_response = test_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                timeout=10
            )
            
            # If test succeeds, save to .env file
            env_content = f"OPENAI_API_KEY={api_key}\n"
            with open('.env', 'w') as f:
                f.write(env_content)
            
            # Update environment and reinitialize client
            os.environ['OPENAI_API_KEY'] = api_key
            load_dotenv(override=True)  # Reload .env file
            success = initialize_openai_client()
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": "OpenAI API key configured and tested successfully"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "API key saved but failed to initialize client"
                }), 500
                
        except Exception as test_error:
            return jsonify({
                "status": "error",
                "message": f"Invalid API key or connection failed: {str(test_error)}"
            }), 400
            
    except Exception as e:
        logger.error(f"Error setting up OpenAI: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Failed to set up OpenAI: {str(e)}"
        }), 500

@app.route('/')
def index():
    """Root endpoint with API information"""
    return jsonify({
        "message": "Health Monitoring API is running",
        "version": "2.5 - Enhanced OpenAI integration with multiple setup methods",
        "openai_features": bool(openai_client),
        "openai_status": "✅ Connected" if openai_client else "❌ Not configured",
        "endpoints": {
            "/predict": "POST - Main prediction endpoint",
            "/insight": "POST - Generate AI insights for health readings",
            "/chat": "POST - General health chat with AI",
            "/setup-openai": "POST - Set up OpenAI API key",
            "/test": "POST - Test with predefined cases",
            "/health": "POST - Health check",
            "/status": "GET - Application status",
            "/train": "POST - Retrain model",
            "/download_logs": "GET - Download prediction logs"
        },
        "setup_instructions": {
            "method_1": "Set OPENAI_API_KEY environment variable",
            "method_2": "Create .env file with OPENAI_API_KEY=your_key",
            "method_3": "Create config.json with openai_api_key field",
            "method_4": "Use /setup-openai endpoint to configure via API"
        }
    })

if __name__ == '__main__':
    try:
        # Initialize OpenAI client
        initialize_openai_client()
        
        # Load or create model
        load_or_create_model()
        
        logger.info("🚀 Starting Health Monitoring API...")
        logger.info(f"📊 Model status: {'✅ Ready' if model_iso else '❌ Not loaded'}")
        logger.info(f"🤖 OpenAI status: {'✅ Connected' if openai_client else '❌ Not configured'}")
        
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        
    except Exception as e:
        logger.error(f"❌ Failed to start the Health Monitoring API: {str(e)}")
        print("ERROR: Unable to start the Flask app. Please check the logs for details.")