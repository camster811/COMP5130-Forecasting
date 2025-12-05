"""
Data Loading Module for Stock Price Forecasting

This module handles fetching and loading historical stock data.
"""

import yfinance as yf
import pandas as pd
import os


def fetch_spy_data(save_path="data/aal_data.csv", ticker="AAL", period="5y"):
    """
    Fetch historical stock data using yfinance.
    """
    print(f"Fetching {ticker} data from yfinance (period: {period})...")

    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(period=period)

    if data.empty:
        print(f"Failed to fetch {ticker} data from yfinance.")
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data.to_csv(save_path)

    print(f"Data saved to {save_path}")
    print(f"Period: {period} ({len(data)} trading days)")
    return data


def load_data(data_path="data/aal_data.csv", ticker="AAL"):
    """
    Load stock data from CSV file, or fetch if not exists.
    """
    if not os.path.exists(data_path):
        fetch_spy_data(save_path=data_path, ticker=ticker)

    data = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")

    return data


def validate_data(data):
    """
    Basic validation of the loaded stock data.
    """
    import logging

    logging.basicConfig(level=logging.INFO)
    print("\n=== Validating Stock Data ===")

    cleaned_data = data.copy()

    # Remove unneeded columns Dividends, Stock Splits, Capital Gains
    cleaned_data = cleaned_data.drop(
        columns=["Dividends", "Stock Splits", "Capital Gains"], errors="ignore"
    )

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_columns:
        if col not in cleaned_data.columns:
            print(f"Missing required column: {col}")
            return False, cleaned_data

    # Check for and handle missing values
    if cleaned_data.isnull().values.any():
        missing_count = cleaned_data.isnull().sum().sum()
        logging.warning(
            f"Data contains {missing_count} missing values. Attempting forward fill."
        )
        cleaned_data = cleaned_data.fillna(method="ffill").fillna(method="bfill")

        # If still has NaN after fill, fail validation
        if cleaned_data.isnull().values.any():
            logging.error("Unable to handle all missing values.")
            return False, cleaned_data
        print(f"Fixed {missing_count} missing values using forward/backward fill")

    # Check for duplicate dates
    if cleaned_data.index.duplicated().any():
        dup_count = cleaned_data.index.duplicated().sum()
        logging.warning(f"Duplicate dates found: {dup_count}. Keeping last occurrence.")
        cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep="last")]
        print(f"Removed {dup_count} duplicate dates")

    if len(cleaned_data) == 0:
        print("Data is empty.")
        return False, cleaned_data

    # Sort by date
    if not cleaned_data.index.is_monotonic_increasing:
        print("Data index is not sorted by date. Sorting...")
        cleaned_data = cleaned_data.sort_index()

    print("Data validation passed.")

    return True, cleaned_data


# Testing
if __name__ == "__main__":
    print("Testing Data Loader Module")
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
