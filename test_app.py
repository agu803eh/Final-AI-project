import requests
import json

# Make sure your Flask app is running first!
base_url = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print("✅ Health check passed!\n")
    except Exception as e:
        print(f"❌ Health check failed: {e}\n")

def test_prediction():
    """Test the prediction endpoint"""
    print("Testing prediction...")
    
    # Test data
    test_data = {
        "heart_rate": 75,
        "bvp": 10
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict_anomaly",
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Prediction test passed!\n")
    except Exception as e:
        print(f"❌ Prediction test failed: {e}\n")

def test_abnormal_data():
    """Test with abnormal data"""
    print("Testing with abnormal data...")
    
    # Abnormal test data
    abnormal_data = {
        "heart_rate": 180,  # Very high heart rate
        "bvp": 150          # Very high BVP
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict_anomaly",
            headers={"Content-Type": "application/json"},
            json=abnormal_data
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Abnormal data test passed!\n")
    except Exception as e:
        print(f"❌ Abnormal data test failed: {e}\n")

def test_batch_prediction():
    """Test batch prediction"""
    print("Testing batch prediction...")
    
    batch_data = {
        "measurements": [
            {"heart_rate": 70, "bvp": 5},
            {"heart_rate": 180, "bvp": 150},  # Abnormal
            {"heart_rate": 65, "bvp": 8}
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict_batch",
            headers={"Content-Type": "application/json"},
            json=batch_data
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Batch prediction test passed!\n")
    except Exception as e:
        print(f"❌ Batch prediction test failed: {e}\n")

def test_invalid_data():
    """Test with invalid data"""
    print("Testing with invalid data...")
    
    invalid_data = {
        "heart_rate": "invalid",  # Should be a number
        "bvp": 10
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict_anomaly",
            headers={"Content-Type": "application/json"},
            json=invalid_data
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Invalid data test passed (error handled correctly)!\n")
    except Exception as e:
        print(f"❌ Invalid data test failed: {e}\n")

if __name__ == "__main__":
    print("Running Health Monitor App Tests")
    print("=" * 50)
    print("Make sure your Flask app is running on localhost:5000")
    print("=" * 50)
    
    # Install requests if you don't have it
    try:
        import requests
    except ImportError:
        print("❌ Please install requests: pip install requests")
        exit(1)
    
    # Run all tests
    test_health_check()
    test_prediction()
    test_abnormal_data()
    test_batch_prediction()
    test_invalid_data()
    
    print("All tests completed!")

    # If you see connection errors like "raise err" or "sock.connect(sa)",
    # it usually means your Flask app is not running or not accessible at http://localhost:5000.
    # Make sure to start your Flask server before running these tests.
    # Example (in your Flask app directory):
    #   python app.py
