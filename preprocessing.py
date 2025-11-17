"""
Preprocessing Module for SPY Stock Price Data
"""

import pandas as pd
import numpy as np


def feature_engineering(data):
    """
    Create features from raw price data for Prophet regressors.
    """
    # Implement feature engineering
    print("\n=== Performing Feature Engineering ===")

    # Log returns for volatility calculation
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))

    # 1. MOMENTUM: Lagged returns (past 5 days)
    # This captures recent momentum without data leakage
    data["momentum_5d"] = data["Close"].pct_change(periods=5)

    # 2. RSI: Relative Strength Index (14-day)
    # Momentum oscillator: 0-100, >70 overbought, <30 oversold
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["rsi"] = 100 - (100 / (1 + rs))

    # 3. VOLATILITY: Rolling standard deviation of returns
    data["volatility"] = data["log_return"].rolling(window=20).std()

    # 4. VOLUME MOMENTUM: If volume data exists
    if "Volume" in data.columns:
        # Volume ratio: current vs 20-day average
        data["volume_ratio"] = data["Volume"] / data["Volume"].rolling(window=20).mean()

    # 5. PRICE BANDS: Bollinger Band position
    ma_20 = data["Close"].rolling(window=20).mean()
    std_20 = data["Close"].rolling(window=20).std()
    data["bb_position"] = (data["Close"] - ma_20) / (2 * std_20)  # -1 to +1 roughly

    # Remove NaN values created by rolling windows
    data_before = len(data)
    data = data.dropna()
    data_after = len(data)
    print(
        f"Removed {data_before - data_after} rows with NaN values (from rolling windows)"
    )

    print(
        "Created features: momentum_5d, rsi, volatility, bb_position"
        + (", volume_ratio" if "Volume" in data.columns else "")
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
