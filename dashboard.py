import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import json
from PIL import Image

st.set_page_config(page_title="Energy Forecaster", layout="wide")

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check both potential results locations
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ALT_RESULTS_DIR = os.path.join(BASE_DIR, "models", "results")

def find_file(filename):
    """Checks main results and models/results for a file."""
    path1 = os.path.join(RESULTS_DIR, filename)
    path2 = os.path.join(ALT_RESULTS_DIR, filename)
    if os.path.exists(path1): return path1
    if os.path.exists(path2): return path2
    return None

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Generate Forecast", "Data Analytics & Patterns", "Model Evaluation"]
)

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "Home":
    st.title("The Energy Demand Forecaster")
    
    st.markdown("### Project Details")
    st.write("**Student:** Ariella Amor")
    st.write("**ID:** 220019040")
    st.write("**Course:** BSc Computer Science")
    
    st.markdown("---")
    st.markdown("### Project Aim")
    st.write("To design a predictive tool that applies recent advancements in data analytics and deep learning to address challenges of forecasting energy consumption. Accurate predictions are essential for grid management, cost reduction, and sustainability goals.")

    st.markdown("---")
    st.markdown("### Model Overview")
    st.write("""
    * **ARIMA:** A statistical baseline that uses linear trends.
    * **XGBoost:** A machine learning model using engineered features like rolling averages.
    * **LSTM:** A deep learning model that learns complex temporal dependencies directly from sequences.
    """)

# ==========================================
# PAGE 2: GENERATE FORECAST
# ==========================================
elif page == "Generate Forecast":
    st.title("Generate a Multi-Model Forecast")
    
    with st.expander("How to use this page"):
        st.write("Input exactly 48 comma-separated values representing the last 24 hours of demand (MW). The models will predict the 49th value.")

    default_data = "32000, 31800, 31750, 31500, 31200, 31000, 30800, 30500, 30200, 30000, 29800, 29500, 29300, 29000, 28800, 28500, 28200, 28000, 27800, 27500, 27300, 27000, 26800, 26500, 26200, 26000, 25800, 25500, 25300, 25000, 25200, 25500, 25800, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 30000, 30500, 31000, 31500, 32000, 32500, 33000"
    user_input = st.text_area("Input 48 values (MW):", value=default_data, height=100)
    
    if st.button("Generate Comparative Forecast", type="primary"):
        try:
            features = [float(x.strip()) for x in user_input.split(',')]
            if len(features) != 48:
                st.error(f"Expected 48 values, got {len(features)}.")
            else:
                response = requests.post('http://127.0.0.1:5000/predict', json={"features": features})
                if response.status_code == 200:
                    res = response.json()
                    # Visualizing Result Trend
                    st.markdown("### Forecast Visualization")
                    forecast_data = pd.DataFrame({
                        "Model": ["LSTM", "XGBoost", "ARIMA"],
                        "Predicted MW": [res['lstm_prediction_mw'], res['xgb_prediction_mw'], res['arima_prediction_mw']]
                    })
                    st.bar_chart(data=forecast_data, x="Model", y="Predicted MW", color="#4C72B0")
                    #st.success("Predictions Generated")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("LSTM", f"{res['lstm_prediction_mw']:,.2f} MW")
                    c2.metric("XGBoost", f"{res['xgb_prediction_mw']:,.2f} MW")
                    c3.metric("ARIMA", f"{res['arima_prediction_mw']:,.2f} MW")
                    
                    st.markdown("### Explanation")
                    st.write("These predictions represent the estimated energy demand for the next 30-minute interval. Differences arise from how each model interprets the trend: ARIMA looks at linear momentum, while LSTM analyzes the sequence of volatility.")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# ==========================================
# PAGE 3: DATA ANALYTICS & PATTERNS
# ==========================================
elif page == "Data Analytics & Patterns":
    st.title("Data Analytics & Patterns")
    st.info("Insights derived from the National Grid dataset (Jan 2009 - Feb 2025)")

    # Helper for loading and plotting
    def plot_forecast_window(pred_file, actual_file, model_label, line_color):
        p_path = find_file(pred_file)
        a_path = find_file(actual_file)
        
        if p_path and a_path:
            # Load 72 hours (144 half-hour steps)
            actuals = np.load(a_path)[:144].flatten()
            preds = np.load(p_path)[:144].flatten()
            
            df = pd.DataFrame({
                "Actual Demand": actuals,
                f"{model_label} Forecast": preds
            })
            
            # Using Hex codes for guaranteed brightness
            # Orange for Actual, Model-specific color for Forecast
            st.line_chart(df, color=["#FF8C00", line_color]) 
        else:
            st.warning(f"Data for {model_label} not found in results folder.")

    # 1. ARIMA
    st.markdown("### 1. ARIMA: Statistical Baseline (72-Hour Window)")
    col1, col2 = st.columns([2, 1])
    with col1:
        plot_forecast_window("arima_preds.npy", "actual_values.npy", "ARIMA", "#00BFFF")
    with col2:
        st.write("**Analysis:**")
        st.write("The ARIMA model establishes a linear baseline. It captures the general cyclical trend but often 'smooths out' the sharp peaks, missing the high-volatility spikes seen in the orange Actual Demand line.")

    st.markdown("---")

    # 2. XGBoost
    st.markdown("### 2. XGBoost: Feature-Driven Performance (72-Hour Window)")
    col3, col4 = st.columns([2, 1])
    with col3:
        plot_forecast_window("xgb_preds.npy", "actual_values.npy", "XGBoost", "#32CD32")
    with col4:
        st.write("**Analysis:**")
        st.write("By using specific time-lags, XGBoost reacts much faster to changes. Note how the green line tracks the orange actuals more closely during rapid morning ramps compared to the ARIMA model.")

    st.markdown("---")

    # 3. LSTM
    st.markdown("### 3. LSTM: Deep Learning Sequence Performance (72-Hour Window)")
    col5, col6 = st.columns([2, 1])
    with col5:
        plot_forecast_window("lstm_preds.npy", "actual_values.npy", "LSTM", "#FF00FF")
    with col6:
        st.write("**Analysis:**")
        st.write("The LSTM captures the 'momentum' of the grid. It shows the highest precision in following the intricate 'wiggles' of the orange line, especially during the evening peak demand periods.")

# ==========================================
# PAGE 4: MODEL EVALUATION
# ==========================================
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    
    # Blue info box as requested
    st.info("Insights derived from the National Grid dataset (Jan 2009 - Feb 2025)")

    def load_metrics(m_name):
        f_path = find_file(f"{m_name}_metrics.json")
        if f_path:
            with open(f_path, 'r') as f:
                data = json.load(f)
                return {k.lower(): v for k, v in data.items()}
        return None

    l_m, x_m, a_m = load_metrics("lstm"), load_metrics("xgb"), load_metrics("arima")

    if l_m and x_m and a_m:
        st.markdown("### 1. Performance Metrics Comparison (Table)")
        
        # Table for exact figures
        comparison_df = pd.DataFrame({
            "Metric": ["MAE (MW)", "RMSE (MW)", "MAPE (%)"],
            "LSTM": [l_m['mae'], l_m['rmse'], l_m['mape']*100],
            "XGBoost": [x_m['mae'], x_m['rmse'], x_m['mape']*100],
            "ARIMA": [a_m['mae'], a_m['rmse'], a_m['mape']*100]
        }).set_index("Metric")
        st.table(comparison_df)

        st.markdown("### 2. Model Performance Comparison: RMSE vs MAE vs MAPE")
        # Creating three side-by-side columns to match Figure 141
        chart_col1, chart_col2, chart_col3 = st.columns(3)
        
        with chart_col1:
            st.markdown("##### RMSE (MW)")
            rmse_data = pd.DataFrame({
                "Model": ["LSTM", "XGBoost", "ARIMA"],
                "Value": [l_m['rmse'], x_m['rmse'], a_m['rmse']]
            }).set_index("Model")
            st.bar_chart(rmse_data, color="#4C72B0") 

        with chart_col2:
            st.markdown("##### MAE (MW)")
            mae_data = pd.DataFrame({
                "Model": ["LSTM", "XGBoost", "ARIMA"],
                "Value": [l_m['mae'], x_m['mae'], a_m['mae']]
            }).set_index("Model")
            st.bar_chart(mae_data, color="#55A868") 

        with chart_col3:
            st.markdown("##### MAPE (%)")
            mape_data = pd.DataFrame({
                "Model": ["LSTM", "XGBoost", "ARIMA"],
                "Value": [l_m['mape']*100, x_m['mape']*100, a_m['mape']*100]
            }).set_index("Model")
            st.bar_chart(mape_data, color="#C44E52")

        st.write("**Analysis of Metrics:**")
        st.write("""
        The quantitative evaluation reveals a stark performance divide between traditional linear statistical methods and advanced non-linear algorithms. \n
        Both XGBoost and LSTM achieved an approximate 96% reduction in error compared to the ARIMA baseline. \n
        While XGBoost was slightly more accurate on average (lowest MAE), the LSTM achieved a superior RMSE, indicating it was marginally better at minimizing large deviations during extreme peak spikes.
        """)

    else:
        st.error("Required metric JSON files not found in the results folder.")

    st.markdown("---")

    # 72-Hour Comparative Time-Series (The missing graph!)
    st.markdown("### 3. Time-Series Comparison: 72-Hour Demand Forecast")
    
    actual_p = find_file("actual_values.npy")
    lstm_p = find_file("lstm_preds.npy")
    xgb_p = find_file("xgb_preds.npy")
    arima_p = find_file("arima_preds.npy")

    if all([actual_p, lstm_p, xgb_p, arima_p]):
        # Load 144 points (72 hours) for direct comparison
        combined_data = pd.DataFrame({
            "Actual Demand": np.load(actual_p)[:144].flatten(),
            "LSTM": np.load(lstm_p)[:144].flatten(),
            "XGBoost": np.load(xgb_p)[:144].flatten(),
            "ARIMA": np.load(arima_p)[:144].flatten()
        })
        
        st.line_chart(combined_data, color=["#FF8C00", "#FF00FF", "#32CD32", "#00BFFF"])
        
        st.write("**Analysis of Forecast Behavior:**")
        st.write("""
        Visual inspection of the 72-hour window reinforces the quantitative metrics. \n
        The ARIMA model (Blue) establishes a central trajectory but completely fails to anticipate daily human energy cycles, rapidly smoothing into a flattened mean. \n
        Conversely, both XGBoost (Green) and LSTM (Magenta) successfully track the non-linear trend, adhering precisely to high-frequency evening peaks and early morning troughs.
        """)
    else:
        st.warning("One or more prediction files (.npy) are missing. Unable to generate comparison plot.")

    st.markdown("---")
    
    st.markdown("### Model Comparison Summary")
    st.write("""
    * **LSTM:** Outperformed other models in RMSE by utilizing its internal memory cells to retain deep sequential dependencies across the 48-step input arrays[cite: 132, 144].
    * **XGBoost:** Achieved the lowest overall MAPE by leveraging engineered temporal markers and sequence lags to capture non-linear relationships with high efficiency[cite: 101, 143].
    * **ARIMA:** Limited by its inherent linearity, the baseline struggled to model multiple overlapping seasonalities without becoming computationally unfeasible[cite: 90, 167].
    """)