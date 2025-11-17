"""
Preprocessing Module for SPY Stock Price Data
"""

import pandas as pd
import numpy as np


def feature_engineering(data):
    """
    Create features from raw price data.
    - Log returns
    - Moving averages
    - Volatility measures
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

    # Remove NaN values created by rolling windows
    data_before = len(data)
    data = data.dropna()
    data_after = len(data)
    print(
        f"Removed {data_before - data_after} rows with NaN values (from rolling windows)"
    )

    return data


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """
    Split time-series data into train/validation/test sets.

    For time-series, we cannot random split - must maintain temporal order.
    """
    print("\n=== Splitting Data into Train/Val/Test ===")

    # Ensure data is sorted by date
    if not data.index.is_monotonic_increasing:
        print("Data not sorted by date. Sorting...")
        data = data.sort_index()

    # Implement time-series split
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    # Validation: Check for data leakage
    if len(val_data) > 0 and train_data.index[-1] >= val_data.index[0]:
        raise ValueError("Data leakage detected: train and validation overlap!")
    if len(test_data) > 0 and val_data.index[-1] >= test_data.index[0]:
        raise ValueError("Data leakage detected: validation and test overlap!")

    print(
        f"Train: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} samples)"
    )
    print(
        f"Val:   {val_data.index[0] if len(val_data) > 0 else 'N/A'} to {val_data.index[-1] if len(val_data) > 0 else 'N/A'} ({len(val_data)} samples)"
    )
    print(
        f"Test:  {test_data.index[0] if len(test_data) > 0 else 'N/A'} to {test_data.index[-1] if len(test_data) > 0 else 'N/A'} ({len(test_data)} samples)"
    )
    print("No data leakage")

    # Return splits
    return (train_data, val_data, test_data)


def ensure_datetime_index(data):
    """Ensure data has naive DatetimeIndex with proper frequency"""
    # Convert to datetime, forcing UTC
    data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)

    # Sort by date
    data = data.sort_index()

    # Infer the frequency without filling gaps
    try:
        data.index.freq = pd.infer_freq(data.index)
    except (ValueError, TypeError):
        print("Could not infer frequency; data may be irregular.")
        pass

    return data


if __name__ == "__main__":
    from data_loader import load_data, validate_data

    print("=== Testing Preprocessing Module ===")

    raw_data = load_data()
    is_valid, cleaned_data = validate_data(raw_data)
    if not is_valid:
        print("Data validation failed.")
        exit(1)

    processed_data = feature_engineering(cleaned_data)
    processed_data = ensure_datetime_index(processed_data)
    train, val, test = split_data(processed_data)
    print(
        f"Train shape: {train.shape}, Val shape: {val.shape}, Test shape: {test.shape}"
    )
    print(f"Columns after feature engineering: {processed_data.columns.tolist()}")
    print(f"Sample data:\n{processed_data.tail()}")
