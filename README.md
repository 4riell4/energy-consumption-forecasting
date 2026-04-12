# Energy Consumption Forecaster

## Project Overview
This project is a multi-model forecasting system designed to predict short-term regional energy consumption. It compares:

- Statistical Model: ARIMA  
- Machine Learning Model: XGBoost  
- Deep Learning Model: LSTM  

The system consists of:
- A Flask REST API backend for model inference  
- A Streamlit dashboard frontend for user interaction  


## Project Structure

The repository is organised exactly as follows:

- `api/` — Contains the Flask REST API backend application serving the model inference.  
- `data/` — Contains the raw and processed historical datasets used to train and evaluate the models.  
- `models/` — Stores the trained models, data scalers, and their specific training notebooks.
- `notebooks/` — Contains the core research pipeline Jupyter notebooks (exploratory data analysis, feature engineering, and model evaluation).  
- `results/` — Holds the exported output metrics (`.json`), prediction arrays (`.npy`), and visual assets used dynamically by the frontend dashboard.  
- `dashboard.py` — The Streamlit frontend application.  
- `test_api.py` — A script to verify the Flask API endpoints independently without the UI.

## Installation Instructions

Follow these steps to run the project locally:

### 1. Extract the Project
Unzip the project folder and open it in your IDE or terminal.

### 2. Prerequisites
- Python **3.8+**

### 3. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv
```

#### Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```


## How to Run the Project

You need **two terminals running at the same time**.

### 1. Start the Backend API

```bash
cd api
python app.py
```

Wait until the models finish loading before continuing.


### 2. Start the Frontend Dashboard

In a second terminal:

```bash
streamlit run dashboard.py
```

The app will open in your browser at:

```
http://localhost:8501
```


### Optional: Test the API

```bash
python test_api.py
```
 
