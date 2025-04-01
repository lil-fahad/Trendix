from dotenv import load_dotenv
import os
import logging
import requests
import pandas as pd

# Load .env locally
load_dotenv()

# Attempt to use Streamlit secrets if available
try:
    import streamlit as st
    def get_polygon_api_key():
        return st.secrets["POLYGON_API_KEY"]
except ImportError:
    def get_polygon_api_key():
        return os.getenv("POLYGON_API_KEY", "demo")

logging.basicConfig(level=logging.INFO)

def fetch_polygon_data(symbol, from_date='2022-01-01', to_date='2024-12-31', interval='day'):
    try:
        api_key = get_polygon_api_key()
        if not api_key or api_key == "demo":
            logging.warning('POLYGON_API_KEY is not set. Using demo key.')

        url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{interval}/{from_date}/{to_date}?adjusted=true&sort=asc&apiKey={api_key}'
        response = requests.get(url)

        if response.status_code != 200:
            logging.error(f'HTTP error {response.status_code} while fetching data for {symbol}')
            return pd.DataFrame()

        try:
            data = response.json().get('results', [])
            if not data:
                logging.warning(f'No data found for {symbol} in response.')
                return pd.DataFrame()

            df = pd.DataFrame(data)
            if 't' not in df.columns:
                logging.error('Missing timestamp column in data.')
                return pd.DataFrame()

            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('Date', inplace=True)
            df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logging.exception('Error parsing API response')
            return pd.DataFrame()
    except Exception as e:
        logging.exception("Unexpected error occurred in fetch_polygon_data")
        return pd.DataFrame()
