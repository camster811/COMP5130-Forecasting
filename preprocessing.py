"""
Preprocessing Module for SPY Stock Price Data

Stock price data is typically clean, but we may need:
- Feature engineering (e.g., returns, moving averages)
- Train/test split for time-series
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def feature_engineering(data):
    """
    Create additional features from raw price data.

    For stock forecasting, you might want:
    - Log returns
    - Moving averages
    - Volatility measures
    - Technical indicators

    Args:
        data (pd.DataFrame): Preprocessed data

    Returns:
        pd.DataFrame: Data with additional features
    """
    # Implement feature engineering
    print("\n=== Performing Feature Engineering ===")

    # Add log returns
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))

    # Add moving averages
    data["MA_20"] = data["Close"].rolling(window=20).mean()
    data["MA_50"] = data["Close"].rolling(window=50).mean()

    # Add volatility
    data["volatility"] = data["log_return"].rolling(window=7).std()

    return data


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """
    Split time-series data into train/validation/test sets.

    For time-series, we cannot random split - must maintain temporal order.

    Args:
        data (pd.DataFrame): Preprocessed data
        train_ratio (float): Proportion for training
        val_ratio (float): Proportion for validation

    Returns:
        tuple: (train_data, val_data, test_data)
    """
    print("\n=== Splitting Data into Train/Val/Test ===")

    # Implement time-series split
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    # Return splits
    return (train_data, val_data, test_data)


if __name__ == "__main__":
    from data_loader import load_data, validate_data

    print("=== Testing Preprocessing Module ===")

    raw_data = load_data()
    is_valid, cleaned_data = validate_data(raw_data)
    if not is_valid:
        print("Data validation failed.")
        exit(1)

    processed_data = feature_engineering(cleaned_data)
    train, val, test = split_data(processed_data)
    print(
        f"Train shape: {train.shape}, Val shape: {val.shape}, Test shape: {test.shape}"
    )
    print(f"Columns after feature engineering: {processed_data.columns.tolist()}")
    print(f"Sample data:\n{processed_data.tail()}")
