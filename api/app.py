import os
from flask import Flask, request, jsonify
import joblib
from tensorflow.keras.models import load_model 

# Import our helper function from utils.py
from utils import prepare_input_data

app = Flask(__name__)

# --- Load Models on Startup ---
# Use absolute paths so Flask doesn't get confused about where it's running from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'lstm'))

print("Loading LSTM model and scaler...")
model = load_model(os.path.join(LSTM_DIR, 'lstm_model.keras'))
scaler = joblib.load(os.path.join(LSTM_DIR, 'scaler.pkl'))
print("Models loaded successfully!")

# --- Define API Routes ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Energy Forecasting API is online."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" array.'}), 400
            
        features = data['features']
        
        if len(features) != 48:
            return jsonify({'error': f'Expected 48 half-hour values, got {len(features)}.'}), 400

        # Process the data using our utils function
        lstm_input = prepare_input_data(features, scaler)
        
        # Make the prediction (returns scaled value)
        scaled_pred = model.predict(lstm_input)
        
        # Inverse transform back to real Megawatts
        real_pred = scaler.inverse_transform(scaled_pred)
        
        return jsonify({
            'forecasted_demand_mw': float(real_pred[0][0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)