import os
from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model #type: ignore
from statsmodels.tsa.arima.model import ARIMAResults
from datetime import datetime 

app = Flask(__name__)

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'lstm'))
XGB_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'xgboost'))
ARIMA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'arima'))

print("Loading AI Models into memory...")
try:
    # Load LSTM & Scaler
    lstm_model = load_model(os.path.join(LSTM_DIR, 'lstm_model.keras'))
    scaler = joblib.load(os.path.join(LSTM_DIR, 'scaler.pkl'))
    
    # Load XGBoost
    xgb_model = joblib.load(os.path.join(XGB_DIR, 'xgb_model.pkl'))
    
    # Load ARIMA 
    arima_model = ARIMAResults.load(os.path.join(ARIMA_DIR, 'arima_model.pkl'))
    
    print(" All 3 Models loaded")
except Exception as e:
    print(f" Error loading one or more models: {e}")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Online", "message": "Multi-Model API is running."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing "features" array.'}), 400
            
        features = data['features']
        if len(features) != 48:
            return jsonify({'error': f'Expected exactly 48 values, got {len(features)}.'}), 400

        # LSTM
        input_array = np.array(features).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)
        lstm_input = scaled_input.reshape(1, 48, 1)
        lstm_scaled_pred = lstm_model.predict(lstm_input, verbose=0)
        lstm_pred = float(scaler.inverse_transform(lstm_scaled_pred)[0][0])

        # XGBOOST
        lag_1 = features[-1] 
        lag_48 = features[0] 
        rolling_mean_48 = np.mean(features) 
        
        # Simulate real-time inference using the current system clock
        now = datetime.now()
        
        xgb_engineered_features = [
            lag_1, 
            lag_48, 
            rolling_mean_48, 
            now.hour, 
            now.weekday(), # Monday=0, Sunday=6 etc
            now.month
        ]
        
        xgb_input = np.array(xgb_engineered_features).reshape(1, -1)
        xgb_pred = float(xgb_model.predict(xgb_input)[0])

        # ARIMA 
        updated_arima = arima_model.apply(features)
        
        raw_forecast = np.asarray(updated_arima.forecast(steps=1))
        arima_pred = float(raw_forecast[0])

        return jsonify({
            'lstm_prediction_mw': lstm_pred,
            'xgb_prediction_mw': xgb_pred,
            'arima_prediction_mw': arima_pred
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

"""
References:
  https://www.geeksforgeeks.org/python/flask-creating-rest-apis/
  https://www.geeksforgeeks.org/javascript/json/
  
"""