# AI-Powered Health Monitoring System

## Project Overview

This project develops an AI-powered system for real-time health monitoring using data collected from wearable devices. The system analyzes key health metrics such as heart rate, blood oxygen levels, and activity levels to detect anomalies and provide personalized health recommendations.

**Goals:**

*   Analyze real-time health metrics from wearable devices.
*   Detect abnormal health conditions (e.g., irregular heartbeats, low blood oxygen) using AI models.
*   Provide personalized and actionable health recommendations based on user data.

**Key Features:**

*   Real-Time Health Monitoring: Collect and analyze data from wearable devices.
*   Anomaly Detection: Use AI to detect abnormal health conditions.
*   Personalized Recommendations: Provide actionable health advice based on user data.
*   User-Friendly Interface: Develop a mobile or web app for users to view their health data and recommendations (Planned).


## Setup

To set up the project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd AI-Powered-Health-Monitoring-System
    ```

    Replace `<repository_url>` with the actual URL of the project repository.

2.  **Install dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    # Create a virtual environment
    python -m venv venv
    # Activate the virtual environment
    # On macOS and Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate

    # Install the required packages
    pip install -r requirements.txt
    ```

    Make sure you have a `requirements.txt` file in the project root containing all necessary libraries (e.g., `flask`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `requests`, `flask-cors`). You can generate this file using `pip freeze > requirements.txt` after installing the libraries.

3.  **Environment Variables:**

    Currently, there are no mandatory environment variables required for the basic functionality of the anomaly detection API. However, for future extensions (e.g., database connections, external API keys), you would typically set them up using a `.env` file and a library like `python-dotenv`.

    ```bash
    touch .env
    ```

    Add any necessary environment variables to the `.env` file in the format `KEY=VALUE`.


## Dataset

The primary dataset used in this project is the **PhysioNet Wearable Device Dataset (1.0.1)**, available at [https://physionet.org/files/wearable-device-dataset/1.0.1/](https://physionet.org/files/wearable-device-dataset/1.0.1/).

This project specifically utilizes the **Heart Rate (HR)** and **Blood Volume Pulse (BVP)** data files from this dataset.

The dataset was downloaded programmatically using `wget`.

### Preprocessing

The initial preprocessing of the HR and BVP data involved the following steps:

1.  **Loading Data:** The raw CSV files for HR and BVP were loaded into pandas DataFrames, skipping the initial metadata rows.
2.  **Extracting Metadata:** The start time and sampling rate for each sensor were extracted from the first two rows of their respective CSV files.
3.  **Creating Time Index:** A `DatetimeIndex` was created for each DataFrame using the extracted start time and the index of the data points, scaled by the sampling rate.
4.  **Resampling and Merging:** The higher-frequency BVP data was resampled to match the 1 Hz frequency of the HR data using the mean value within each second. The HR and resampled BVP DataFrames were then merged based on their time indices.
5.  **Handling Missing Values:** Missing values introduced during merging or present in the original data were handled using a combination of backward fill (`bfill`) and forward fill (`ffill`) to ensure a complete time series for feature engineering.


## Anomaly Detection Model

The core of the health monitoring system is the anomaly detection model, which identifies unusual patterns in the physiological data.

### Model Choice

We utilize the **Isolation Forest** algorithm for anomaly detection. Isolation Forest is an unsupervised learning algorithm that works by isolating anomalies rather than profiling normal data. It is particularly effective for high-dimensional datasets and does not require prior labeling of anomalies, making it suitable for detecting unexpected health events.

### Engineered Features

To provide the Isolation Forest model with relevant information, we engineered several features from the raw Heart Rate and BVP data. These features capture both the instantaneous values and the temporal characteristics of the signals:

*   **Heart Rate (Instantaneous):** The raw heart rate reading at each time point.
*   **BVP (Instantaneous):** The raw Blood Volume Pulse reading at each time point.
*   **Rolling Mean of Heart Rate (60s window):** The average heart rate over a 60-second sliding window, providing a smoothed trend.
*   **Rolling Standard Deviation of Heart Rate (60s window):** The variability of heart rate over a 60-second sliding window, indicating fluctuations.
*   **Rolling Mean of BVP (60s window):** The average BVP over a 60-second sliding window.
*   **Rolling Standard Deviation of BVP (60s window):** The variability of BVP over a 60-second sliding window.
*   **Rolling Standard Deviation of Heart Rate (10s window) - Proxy for HRV:** The variability of heart rate over a shorter, 10-second window, serving as a simplified proxy for Heart Rate Variability (HRV), which is an important indicator of autonomic nervous system activity and can be sensitive to stress or health issues.

These engineered features are calculated after the initial data loading and preprocessing steps, including resampling, merging, and handling missing values.

### Model Training

The Isolation Forest model is trained on the preprocessed data with the engineered features. As an unsupervised algorithm, it learns to identify outliers based on the structure of the data itself. The training process involves fitting the `IsolationForest` model to the feature set. The `contamination` parameter is set to 'auto', which the algorithm uses to estimate the proportion of outliers in the data. In a production environment, the trained model is saved and loaded for making predictions on new, incoming data without retraining.


## API Endpoints

The health monitoring system exposes the following API endpoints:

### `POST /predict_anomaly`

*   **Method:** `POST`
*   **Path:** `/predict_anomaly`
*   **Description:** Receives a single set of health metric readings (Heart Rate and BVP), performs real-time anomaly detection using the trained model, and returns the anomaly status and a personalized health recommendation.
*   **Request Format:**
    ```json
    {
      "heart_rate": 85.5,
      "bvp": -10.2
    }
    ```
    The request body should be a JSON object containing:
    *   `heart_rate` (number): The heart rate reading.
    *   `bvp` (number): The Blood Volume Pulse reading.
*   **Response:**
    *   **Success (200 OK):** Returns a JSON object with the anomaly detection results and recommendation.
        ```json
        {
          "timestamp": "2023-10-27T10:30:00.123456",
          "status": "normal" | "abnormal" | "error",
          "anomaly_score": -0.05,
          "recommendation": "Data looks normal." | "High heart rate detected. Consider resting." | "Unable to process data. Please try again.",
          "confidence": "high" | "medium"
        }
        ```
        *   `timestamp` (string): ISO 8601 formatted timestamp of the prediction.
        *   `status` (string): The anomaly status ('normal', 'abnormal', or 'error').
        *   `anomaly_score` (number): The anomaly score from the Isolation Forest model (lower values indicate higher anomaly likelihood).
        *   `recommendation` (string): A personalized health recommendation based on the anomaly status and detected patterns.
        *   `confidence` (string): An indicator of the confidence in the confidence in the anomaly detection ('high' or 'medium').
    *   **Error (400 Bad Request):** Returns a JSON object with an error message if the input data is invalid.
        ```json
        {
          "error": "Invalid input data. Please provide 'heart_rate' and 'bvp'."
        }
        ```
    *   **Error (500 Internal Server Error):** Returns a JSON object with an error message if an internal server error occurs.
        ```json
        {
          "error": "Internal server error",
          "timestamp": "..."
        }
        ```

### `GET /health`

*   **Method:** `GET`
*   **Path:** `/health`
*   **Description:** A simple health check endpoint to verify if the API is running and the model is loaded.
*   **Request Format:** No request body required.
*   **Response:**
    *   **Success (200 OK):** Returns a JSON object indicating the health status.
        ```json
        {
          "status": "healthy",
          "timestamp": "2023-10-27T10:30:00.123456",
          "model_loaded": true | false
        }
        ```
        *   `status` (string): The health status ('healthy').
        *   `timestamp` (string): ISO 8601 formatted timestamp of the health check.
        *   `model_loaded` (boolean): Indicates whether the anomaly detection model has been successfully loaded.

### `POST /predict_batch`

*   **Method:** `POST`
*   **Path:** `/predict_batch`
*   **Description:** Receives a batch of health metric readings, performs anomaly detection for each data point, and returns a list of results.
*   **Request Format:**
    ```json
    {
      "measurements": [
        {
          "heart_rate": 85.5,
          "bvp": -10.2
        },
        {
          "heart_rate": 120.0,
          "bvp": 50.5
        },
        ...
      ]
    }
    ```
    The request body should be a JSON object containing:
    *   `measurements` (array of objects): A list where each object has:
        *   `heart_rate` (number): The heart rate reading.
        *   `bvp` (number): The Blood Volume Pulse reading.
*   **Response:**
    *   **Success (200 OK):** Returns a JSON object containing a list of results for each measurement in the batch.
        ```json
        {
          "results": [
            {
              "index": 0,
              "timestamp": "...",
              "status": "normal" | "abnormal" | "error",
              "anomaly_score": -0.05,
              "recommendation": "...",
              "confidence": "..."
            },
            {
              "index": 1,
              "timestamp": "...",
              "status": "...",
              "anomaly_score": ..., 
              "recommendation": "...",
              "confidence": "..."
            },
            ...
          ]
        }
        ```
        Each object in the `results` array has the same structure as the response for `/predict_anomaly`, with an additional `index` field indicating the position in the input `measurements` array. Individual results can have an "error" field if validation failed for that specific measurement.
    *   **Error (400 Bad Request):** Returns a JSON object with an error message if the overall request format is invalid (e.g., missing `measurements` key or `measurements` is not an array).
        ```json
        {
          "error": "Expected JSON with 'measurements' array"
        }
        ```
    *   **Error (500 Internal Server Error):** Returns a JSON object with an error message if an internal server error occurs during batch processing.
        ```json
        {
          "error": "Internal server error"
        }
        ```


## Usage

### Running the API

To run the Flask API, execute the main script (e.g., `app.py` if you've saved the code there):

```bash
python app.py
```

The API will typically run on `http://127.0.0.1:5000` in debug mode.

### Sending Data to the API

You can interact with the API endpoints using tools like `curl`, Postman, or by writing a simple script using the `requests` library in Python.

#### Example using `curl`

To send data to the `/predict_anomaly` endpoint:

```bash
curl -X POST \
  http://127.0.0.1:5000/predict_anomaly \
  -H 'Content-Type: application/json' \
  -d '{"heart_rate": 85.5, "bvp": -10.2}'
```

To send data to the `/predict_batch` endpoint:

```bash
curl -X POST \
  http://127.00.1:5000/predict_batch \
  -H 'Content-Type: application/json' \
  -d '{"measurements": [{"heart_rate": 85.5, "bvp": -10.2}, {"heart_rate": 120.0, "bvp": 50.5}]}'
```

To check the health of the API:

```bash
curl http://127.0.0.1:5000/health
```

#### Example using Python `requests`

```python
import requests
import json

# Replace with your API endpoint URL
url = 'http://127.0.0.1:5000/predict_anomaly'
headers = {'Content-Type': 'application/json'}

# Sample data for /predict_anomaly
health_data_single = {
    'heart_rate': 85.5,
    'bvp': -10.2
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(health_data_single))
    response.raise_for_status() # Raise an exception for bad status codes
    print("Predict Anomaly Response:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error sending request to /predict_anomaly: {e}")

# Sample data for /predict_batch
url_batch = 'http://127.0.0.1:5000/predict_batch'
health_data_batch = {
    "measurements": [
        {"heart_rate": 85.5, "bvp": -10.2},
        {"heart_rate": 120.0, "bvp": 50.5},
        {"heart_rate": 72.0, "bvp": 5.0}
    ]
}

try:
    response_batch = requests.post(url_batch, headers=headers, data=json.dumps(health_data_batch))
    response_batch.raise_for_status() # Raise an exception for bad status codes
    print("
Predict Batch Response:")
    print(response_batch.json())
except requests.exceptions.RequestException as e:
    print(f"Error sending request to /predict_batch: {e}")

# Example for /health
url_health = 'http://127.0.0.1:5000/health'
try:
    response_health = requests.get(url_health)
    response_health.raise_for_status() # Raise an exception for bad status codes
    print("
Health Check Response:")
    print(response_health.json())
except requests.exceptions.RequestException as e:
    print(f"Error sending request to /health: {e}")

```

Remember to replace `http://127.0.0.1:5000` with the actual address and port where your Flask app is running if it's different.


## Future Improvements

This project provides a foundation for an AI-powered health monitoring system. Several areas can be explored for future development and enhancement:

*   **Advanced Anomaly Detection Techniques:** Investigate and implement more sophisticated anomaly detection algorithms, such as Isolation Forest variations, One-Class SVM, or deep learning-based methods (e.g., LSTMs for time series anomaly detection) to potentially improve accuracy and robustness.
*   **Incorporating Additional Health Metrics:** Integrate and analyze data from other sensors available in the PhysioNet dataset or from other wearable devices, such as:
    *   Blood Oxygen levels
    *   Activity levels (from Accelerometer data)
    *   Skin Temperature
    *   Interbeat Interval (IBI) for more precise HRV analysis
    This will provide a more comprehensive view of the user's health state.
*   **Refining Feature Engineering:** Explore more advanced feature engineering techniques tailored to time series data and physiological signals. This could include frequency domain analysis of BVP, time-frequency features, or features derived from IBI data for detailed HRV analysis (e.g., RMSSD, SDNN, pNN50).
*   **Sophisticated Recommendation Engine:** Develop a more intelligent recommendation system that considers the type, severity, duration, and context of the detected anomaly. This could involve rule-based systems, knowledge graphs, or even machine learning models trained on labeled health event data to provide more personalized and actionable advice.
*   **Building a User Interface:** Create a user-friendly mobile or web application where users can visualize their health data, receive real-time anomaly alerts, and view personalized health recommendations.
*   **Real-time Data Streaming and Processing:** Implement a pipeline for real-time data ingestion from wearable devices (simulated or actual) and processing to enable truly real-time monitoring and anomaly detection.
*   **Deployment Strategies:** Explore different deployment options for the Flask API, such as containerization (Docker), cloud platforms (AWS, Google Cloud, Azure), or edge devices for localized processing.
*   **Model Retraining and Updates:** Implement a strategy for periodically retraining the anomaly detection model with new data to adapt to changes in user behavior or health patterns.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact Information

For any questions or inquiries, please contact:

*   **Agu Miracle:** 
*   **Agu803eh@gmail.com:** 
*   **https://github.com/agu803eh/Final-AI-project.git:**

Summary:
Data Analysis Key Findings
The project utilizes the PhysioNet Wearable Device Dataset (1.0.1), specifically focusing on Heart Rate (HR) and Blood Volume Pulse (BVP) data.
Data preprocessing involves loading HR and BVP data, extracting metadata (start time, sampling rate), creating a time index, resampling BVP to match HR's 1 Hz frequency, merging the data, and handling missing values using backward and forward fill.
The anomaly detection model is based on the Isolation Forest algorithm, chosen for its effectiveness in unsupervised anomaly detection without requiring labeled data.
Engineered features used for the model include instantaneous HR and BVP, and rolling mean and standard deviation for both HR and BVP over a 60-second window, plus a 10-second rolling standard deviation of HR as a proxy for Heart Rate Variability (HRV).
The project includes a Flask API with three endpoints: /predict_anomaly (for single readings), /health (for status checks), and /predict_batch (for multiple readings).
The /predict_anomaly and /predict_batch endpoints accept JSON input containing heart rate and BVP and return anomaly status, anomaly score, confidence level, and a personalized recommendation.
Insights or Next Steps
The current implementation provides a solid foundation for real-time health monitoring using Isolation Forest. Exploring deep learning models for time series anomaly detection could potentially capture more complex temporal patterns.
Integrating additional health metrics like blood oxygen, activity levels, and more precise HRV data (Interbeat Interval) is crucial for a more comprehensive and accurate health assessment system.
