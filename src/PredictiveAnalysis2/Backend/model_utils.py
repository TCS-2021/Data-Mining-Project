"""
model_utils.py file for Streamlit application
"""

import numpy as np
from tensorflow import keras

def create_sequences(data, sequence_length, feature_index):
    """Create LSTM sequences from time series data"""
    x_seq, y_seq = [], []
    for i in range(len(data) - sequence_length):
        window = data[i:i + sequence_length]
        target = data[i + sequence_length, feature_index]
        if not np.any(np.isnan(window)) and not np.isnan(target):
            x_seq.append(window)
            y_seq.append(target)
    return np.array(x_seq), np.array(y_seq)


def build_lstm_model(input_shape):
    """Build a simple stacked LSTM model"""
    keras.backend.clear_session()
    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(25),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def build_bidirectional_lstm(sequence_length):
    """Build improved bidirectional LSTM model"""
    model = keras.Sequential([
        keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True),
            input_shape=(sequence_length, 1)),
        keras.layers.Dropout(0.3),
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def rolling_forecast(model, normalized_data, sequence_length):
    """Perform rolling forecast using trained model"""
    predictions = []
    if len(normalized_data) < sequence_length:
        return predictions

    last_sequence = normalized_data[:sequence_length].reshape(1, sequence_length, -1)

    for i in range(sequence_length, len(normalized_data)):
        prediction = model.predict(last_sequence, verbose=0)[0][0]
        predictions.append(prediction)
        next_input = normalized_data[i].reshape(1, 1, -1)
        last_sequence = np.append(last_sequence[:, 1:, :], next_input, axis=1)

    return predictions
