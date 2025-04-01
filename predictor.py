import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import logging

logging.basicConfig(level=logging.INFO)

# Load XGBoost model
def load_xgb_model(path='models/omnimarket_xgb.pkl'):
    try:
        if os.path.exists(path):
            return joblib.load(path)
        raise FileNotFoundError(f'XGBoost model not found at {path}')
    except Exception as e:
        logging.exception('Error loading XGBoost model')
        return None

# Load LSTM model
def load_lstm_model(path='models/omnimarket_lstm.h5'):
    try:
        if os.path.exists(path):
            return load_model(path)
        raise FileNotFoundError(f'LSTM model not found at {path}')
    except Exception as e:
        logging.exception('Error loading LSTM model')
        return None

# Load preprocessor
def load_preprocessor(path='models/preprocessor.pkl'):
    try:
        if os.path.exists(path):
            return joblib.load(path)
        raise FileNotFoundError(f'Preprocessor not found at {path}')
    except Exception as e:
        logging.exception('Error loading preprocessor')
        return None

# Predict with XGBoost
def predict_with_xgb(model, preprocessor, df_features):
    try:
        if model is None or preprocessor is None:
            raise ValueError('Model or preprocessor not loaded')
        X = preprocessor.transform(df_features)
        predictions = model.predict(X)
        if len(predictions) == 0:
            raise ValueError('Prediction output is empty')
        return predictions
    except Exception as e:
        logging.exception('Error during XGBoost prediction')
        return []

# Predict with LSTM
def predict_with_lstm(model, df_features):
    try:
        if model is None:
            raise ValueError('LSTM model not loaded')
        sequence_length = 30
        if len(df_features) < sequence_length:
            raise ValueError('Insufficient data for LSTM sequence input')
        X_seq = np.array([df_features[i-sequence_length:i] for i in range(sequence_length, len(df_features))])
        predictions = model.predict(X_seq).flatten()
        if len(predictions) == 0:
            raise ValueError('Prediction output is empty')
        return predictions
    except Exception as e:
        logging.exception('Error during LSTM prediction')
        return []
