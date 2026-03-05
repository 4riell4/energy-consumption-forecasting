import numpy as np

def prepare_input_data(raw_features, scaler, sequence_length=48):
    """
    Takes a list of raw demand values, scales them, 
    and reshapes them for the LSTM model.
    """
    # 1. Convert list to numpy array and reshape to a column: (48, 1)
    input_array = np.array(raw_features).reshape(-1, 1)
    
    # 2. Scale the data using your fitted scaler
    scaled_input = scaler.transform(input_array)
    
    # 3. Reshape for the LSTM: (batch_size=1, sequence_length=48, num_features=1)
    lstm_input = scaled_input.reshape(1, sequence_length, 1)
    
    return lstm_input