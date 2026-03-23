import os
from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# --- Load Final Trained Model & Scaler ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'lstm'))

print("Loading LSTM model and scaler into memory...")
try:
    model = load_model(os.path.join(LSTM_DIR, 'lstm_model.keras'))
    scaler = joblib.load(os.path.join(LSTM_DIR, 'scaler.pkl'))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

# --- Base Route (Test Server) ---
@app.route('/', methods=['GET'])
def home():
    """Base route to check if the API is running."""
    return jsonify({
        "status": "Online",
        "message": "Energy Forecasting API is running. Ready for predictions!"
    })

# --- Prediction Endpoint (Step 7.2) ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives 48 half-hour intervals, processes them, and returns a forecast."""
    try:
        # 1. Accept JSON input
        data = request.get_json()
        
        # Validation: Ensure 'features' key exists
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing "features" array in JSON payload.'}), 400
            
        features = data['features']
        
        # Validation: Ensure exactly 48 values are provided
        if len(features) != 48:
            return jsonify({'error': f'Expected exactly 48 values, but got {len(features)}.'}), 400

        # 2. Convert input into NumPy array and scale it
        # Reshape to a column vector (48, 1) for the scaler
        input_array = np.array(features).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)
        
        # 3. Reshape for the LSTM model: (batch_size=1, sequence_length=48, num_features=1)
        lstm_input = scaled_input.reshape(1, 48, 1)
        
        # 4. Run model prediction
        scaled_prediction = model.predict(lstm_input)
        
        # 5. Inverse transform to get the real Megawatt (MW) value back
        real_prediction = scaler.inverse_transform(scaled_prediction)
        
        # 6. Return prediction as JSON response
        return jsonify({
            'forecasted_demand_mw': float(real_prediction[0][0])
        })

    except Exception as e:
        # Catch any unexpected errors (prevents the server from crashing)
        return jsonify({'error': str(e)}), 500

# Run the server
if __name__ == '__main__':
    app.run(debug=True, port=5000)