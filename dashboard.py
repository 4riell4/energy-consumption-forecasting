import streamlit as st
import requests
import pandas as pd
import os

# --- 7.4 Setup Streamlit Page ---
st.set_page_config(page_title="Energy Forecaster", page_icon="⚡", layout="wide")

# --- 7.6 Add Visual Elements (Sidebar) ---
st.sidebar.title("Model Insights")
st.sidebar.write("This dashboard connects to a Flask API serving an LSTM Deep Learning model.")

# Display the correlation matrix from your results folder if it exists
image_path = os.path.join("results", "correlation_matrix.png")
if os.path.exists(image_path):
    st.sidebar.image(image_path, caption="Feature Correlation Matrix")
else:
    st.sidebar.info("Run Model Evaluation notebooks to generate visual plots.")

# --- 7.5 Build Basic UI ---
st.title("Energy Demand Forecasting Dashboard")
st.write("Enter the past 24 hours of energy demand (exactly 48 half-hourly intervals in Megawatts) to predict the next interval.")

st.markdown("### 1. Input Data")
# Pre-fill with our 48 dummy values so testing is incredibly easy
default_input = "32000, 31800, 31750, 31500, 31200, 31000, 30800, 30500, 30200, 30000, 29800, 29500, 29300, 29000, 28800, 28500, 28200, 28000, 27800, 27500, 27300, 27000, 26800, 26500, 26200, 26000, 25800, 25500, 25300, 25000, 25200, 25500, 25800, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 30000, 30500, 31000, 31500, 32000, 32500, 33000"

user_input = st.text_area("Enter 48 comma-separated values (MW):", value=default_input, height=100)

# The Predict Button
if st.button("Predict Next Interval", type="primary"):
    try:
        # Convert the text input into a list of floats
        features = [float(x.strip()) for x in user_input.split(',')]
        
        # Validation
        if len(features) != 48:
            st.error(f"Expected exactly 48 values, but got {len(features)}. Please check your input.")
        else:
            # --- 7.6 Visual Element: Line Chart of Inputs ---
            st.markdown("### 2. Recent Demand Trend")
            chart_data = pd.DataFrame(features, columns=["Historical Demand (MW)"])
            st.line_chart(chart_data, color="#ffaa00")

            # --- API Call ---
            with st.spinner("LSTM Model is analysing temporal dynamics..."):
                response = requests.post('http://127.0.0.1:5000/predict', json={"features": features})
                
            # --- Display Results ---
            st.markdown("### 3. Forecast Result")
            if response.status_code == 200:
                prediction = response.json()['forecasted_demand_mw']
                st.success("Prediction generated successfully!")
                st.metric(label="Forecasted Demand (Next 30 Mins)", value=f"{prediction:,.2f} MW")
            else:
                st.error(f"API Error {response.status_code}: {response.json().get('error', 'Unknown Error')}")
                
    except ValueError:
        st.error("Invalid input format. Please ensure all values are numbers separated by commas.")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Please ensure your Flask server is running in another terminal.")