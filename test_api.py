import requests

# 48 dummy values 
dummy_features = [
    32000, 31800, 31750, 31500, 31200, 31000, 30800, 30500, 
    30200, 30000, 29800, 29500, 29300, 29000, 28800, 28500, 
    28200, 28000, 27800, 27500, 27300, 27000, 26800, 26500, 
    26200, 26000, 25800, 25500, 25300, 25000, 25200, 25500, 
    25800, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 
    29500, 30000, 30500, 31000, 31500, 32000, 32500, 33000
]

# Define the exact URL for the predict endpoint
url = 'http://127.0.0.1:5000/predict'

# Package the data into a JSON dictionary 
payload = {"features": dummy_features}

print("Sending test request to Flask API...")

try:
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("Success! API Response:")
        print(response.json())
    else:
        print(f"Error {response.status_code}:")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("Could not connect. Is the Flask server running in another terminal?")