import pandas as pd
import numpy as np
import logging
from data.loader import load_data
from core.predictor import (
    load_xgb_model,
    load_lstm_model,
    load_preprocessor,
    predict_with_xgb,
    predict_with_lstm
)

logging.basicConfig(level=logging.INFO)

def run_prediction(symbol='AAPL', model_choice='xgb'):
    try:
        df = load_data(symbol)
        if df is None or df.empty or 'close' not in df.columns:
            logging.warning('Insufficient data for prediction.')
            return []

        # Feature Engineering
        df['MA_7'] = df['close'].rolling(7).mean()
        df['RSI'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
        df['Volatility'] = df['close'].pct_change().rolling(30).std()
        df.dropna(inplace=True)

        if df.empty:
            logging.warning('All rows dropped after feature engineering.')
            return []

        features = ['open', 'high', 'low', 'volume', 'MA_7', 'RSI', 'Volatility']
        df.columns = [col.lower() for col in df.columns]  # Standardize column names
        X = df[features]
        results = []

        if model_choice == 'xgb':
            model = load_xgb_model()
            preprocessor = load_preprocessor()
            preds = predict_with_xgb(model, preprocessor, X)
        elif model_choice == 'lstm':
            model = load_lstm_model()
            preds = predict_with_lstm(model, X.values)
        else:
            raise ValueError('Unknown model choice.')

        # Validate predictions
        if preds is None or len(preds) == 0:
            logging.error('Empty prediction result.')
            return []

        actual = df['close'].values[-len(preds):]
        mae = np.mean(np.abs(actual - preds))
        direction_acc = np.mean(np.sign(actual[1:] - actual[:-1]) == np.sign(preds[1:] - preds[:-1])) * 100

        results.append({
            'model': model_choice,
            'mae': mae,
            'direction_accuracy': direction_acc
        })

    except Exception as e:
        logging.exception('Error during prediction')
        return []

    return results
