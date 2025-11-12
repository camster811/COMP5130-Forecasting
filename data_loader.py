"""
Data Loading Module for SPY Stock Price Forecasting

This module handles fetching and loading historical SPY stock data.
SPY is an ETF that tracks the S&P 500 index.
"""

import yfinance as yf
import pandas as pd
import os


def fetch_spy_data(save_path="data/spy_data.csv"):
    """
    Fetch historical SPY data (last 10 years) using yfinance.

    Args:
        save_path (str): Path to save the downloaded data

    Returns:
        pd.DataFrame: Historical SPY data with columns: Open, High, Low, Volume, Dividends, Stock Splits, Capital Gains
    """
    print("Fetching SPY data from yfinance...")

    ticker = yf.Ticker("SPY")
    data = ticker.history(period="10y")  # Last 10 years

    if data.empty:
        print("Failed to fetch data from yfinance.")
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data.to_csv(save_path)

    print(f"Data saved to {save_path}")
    return data


def load_data(data_path="data/spy_data.csv"):
    """
    Load SPY data from CSV file.

    Args:
        data_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded data
    """
    if not os.path.exists(data_path):
        fetch_spy_data(save_path=data_path)

    data = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")

    return data


def validate_data(data):
    """
    Basic validation of the loaded SPY data.

    Args:
        data (pd.DataFrame): Data to validate

    Returns:
        bool: True if data is valid
        pd.DataFrame: Cleaned data
    """
    print("\n=== Validating SPY Data ===")

    cleaned_data = data.copy()

    # Remove unnecessary columns Dividends, Stock Splits, Capital Gains
    cleaned_data = cleaned_data.drop(
        columns=["Dividends", "Stock Splits", "Capital Gains"], errors="ignore"
    )

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_columns:
        if col not in cleaned_data.columns:
            print(f"Missing required column: {col}")
            return False, cleaned_data

    if cleaned_data.isnull().values.any():
        import logging

        logging.warning("Data contains missing values.")
        return False, cleaned_data

    if len(cleaned_data) == 0:
        print("Data is empty.")
        return False, cleaned_data

    if not cleaned_data.index.is_monotonic_increasing:
        print("Data index is not sorted by date.")
        cleaned_data = cleaned_data.sort_index()

    print("Data validation passed.")

    return True, cleaned_data


if __name__ == "__main__":
    print("=== Testing Data Loader Module ===")
    print("Testing data loading and validation...")

    data = load_data()

    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    is_valid, cleaned_data = validate_data(data)
    if is_valid:
        print("Data is valid.")
        print(f"Final data shape: {cleaned_data.shape}")
        print(f"Columns: {list(cleaned_data.columns)}")
    else:
        print("Data validation failed.")
