"""
data_utils.py file for Streamlit application
"""

import pandas as pd
import yfinance as yf
import streamlit as st

def load_full_data(file):
    """Load and preprocess historical stock data from CSV"""
    data_frame = pd.read_csv(file, parse_dates=['Date'], dayfirst=True)
    data_frame.columns = data_frame.columns.str.strip()
    data_frame['Date'] = pd.to_datetime(data_frame['Date'], format='%d-%m-%Y', errors='coerce')
    data_frame = data_frame.dropna(subset=['Date']).sort_values('Date')
    data_frame.set_index('Date', inplace=True)

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data_frame[numeric_cols] = data_frame[numeric_cols].apply(pd.to_numeric, errors='coerce')
    data_frame = data_frame.dropna(subset=numeric_cols)

    return data_frame[['Stock Name'] + numeric_cols]

@st.cache_data(ttl=300)
def get_realtime_data():
    """Fetch recent weekly NIFTY50 data"""
    ticker_data = yf.Ticker("^NSEI")
    return ticker_data.history(period="5y", interval="1wk")


@st.cache_data(ttl=300)
def get_realtime_daily_data():
    """Fetch recent daily NIFTY50 data"""
    ticker_data = yf.Ticker("^NSEI")
    return ticker_data.history(period="2y", interval="1d")

@st.cache_data
def load_and_cache_file(uploaded_file):
    """Cache uploaded CSV file content"""
    return load_full_data(uploaded_file)
