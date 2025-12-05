"""
Evaluation Module for Time-Series Forecasting Models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate forecasting metrics including trading-specific ones.
    """
    metrics = {}

    # Basic forecasting metrics
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))

    # Avoid division by zero
    try:
        metrics["MAPE"] = mean_absolute_percentage_error(y_true, y_pred) * 100
    except Exception as e:
        print(f"Error calculating MAPE: {e}")
        # Fallback calculation
        mape_vals = np.abs((y_true - y_pred) / y_true)
        mape_vals = mape_vals[np.isfinite(mape_vals)]  # Remove inf/nan
        metrics["MAPE"] = np.mean(mape_vals) * 100 if len(mape_vals) > 0 else np.inf

    # Directional accuracy
    if len(y_true) > 1:
        actual_direction = np.sign(y_true.diff().dropna())
        pred_direction = np.sign(pd.Series(y_pred).diff().dropna())
        # Align the two series
        min_len = min(len(actual_direction), len(pred_direction))
        if min_len > 0:
            metrics["Directional_Accuracy"] = (
                np.mean(
                    actual_direction.values[-min_len:]
                    == pred_direction.values[-min_len:]
                )
                * 100
            )
        else:
            metrics["Directional_Accuracy"] = 0.0
    else:
        metrics["Directional_Accuracy"] = 0.0

    # Theil's U statistic
    if len(y_true) > 1:
        try:
            naive_forecast = y_true.shift(1).dropna()
            actual_naive = y_true.iloc[1:]
            # Align indices
            common_idx = actual_naive.index.intersection(naive_forecast.index)
            if len(common_idx) > 0:
                mse_model = np.mean((y_pred - y_true) ** 2)
                mse_naive = np.mean(
                    (
                        naive_forecast.loc[common_idx].values
                        - actual_naive.loc[common_idx].values
                    )
                    ** 2
                )
                if mse_naive > 0:
                    metrics["Theils_U"] = np.sqrt(mse_model) / np.sqrt(mse_naive)
                else:
                    metrics["Theils_U"] = np.inf
            else:
                metrics["Theils_U"] = np.inf
        except Exception as e:
            print(f"Error calculating Theil's U statistic: {e}")
            metrics["Theils_U"] = np.inf
    else:
        metrics["Theils_U"] = np.inf

    # Trading-specific metrics
    # Hit Rate: Percentage of profitable trades if trading on predictions
    if len(y_true) > 1:
        try:
            # Calculate returns
            actual_returns = y_true.pct_change().dropna()
            # Trading signal: 1 if predict up, -1 if predict down
            pred_series = pd.Series(y_pred, index=y_true.index[: len(y_pred)])
            pred_changes = pred_series.diff().dropna()
            signals = np.sign(pred_changes)

            # Align signals with returns
            common_idx = actual_returns.index.intersection(signals.index)
            if len(common_idx) > 1:
                aligned_returns = actual_returns.loc[common_idx]
                aligned_signals = signals.loc[common_idx]

                # Calculate trade returns
                trade_returns = aligned_signals * aligned_returns
                metrics["Hit_Rate"] = np.mean(trade_returns > 0) * 100

                # Sharpe Ratio (annualized assuming 252 trading days)
                if np.std(trade_returns) > 0:
                    metrics["Sharpe_Ratio"] = (
                        np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
                    )
                else:
                    metrics["Sharpe_Ratio"] = 0.0
            else:
                metrics["Hit_Rate"] = 0.0
                metrics["Sharpe_Ratio"] = 0.0
        except Exception as e:
            print(f"Error calculating trading metrics: {e}")
            metrics["Hit_Rate"] = 0.0
            metrics["Sharpe_Ratio"] = 0.0
    else:
        metrics["Hit_Rate"] = 0.0
        metrics["Sharpe_Ratio"] = 0.0

    # Print summary
    print(f"\n{model_name} Metrics:")
    print("=" * 50)
    print(f"  MAE:                   ${metrics['MAE']:.4f}")
    print(f"  RMSE:                  ${metrics['RMSE']:.4f}")
    print(f"  MAPE:                  {metrics['MAPE']:.2f}%")
    print(f"  Directional Accuracy:  {metrics['Directional_Accuracy']:.2f}%")
    print(f"  Theil's U:             {metrics['Theils_U']:.4f}")
    print(f"  Hit Rate:              {metrics['Hit_Rate']:.2f}%")
    print(f"  Sharpe Ratio:          {metrics['Sharpe_Ratio']:.4f}")
    print("=" * 50)

    return metrics


def plot_predictions(y_true, y_pred_arima, y_pred_prophet, title="Forecast Comparison"):
    """
    Plot actual vs predicted values for both models.
    """
    plt.figure(figsize=(15, 8))

    plt.plot(y_true.index, y_true.values, label="Actual", color="black", linewidth=2)

    # Plot ARIMA predictions
    plt.plot(
        y_pred_arima.index,
        y_pred_arima.values,
        label="ARIMA Forecast",
        color="blue",
        linestyle="--",
    )

    # Plot Prophet predictions
    plt.plot(
        y_pred_prophet.index,
        y_pred_prophet.values,
        label="Prophet Forecast",
        color="red",
        linestyle="-.",
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()
    pass


def compare_models(
    y_true, predictions_dict, metrics_to_compare=["MAE", "RMSE", "MAPE"]
):
    """
    Compare performance of multiple models.
    """
    results = {}

    for model_name, y_pred in predictions_dict.items():
        # Align predictions with actuals
        common_idx = y_true.index.intersection(y_pred.index)
        y_true_aligned = y_true.loc[common_idx]
        y_pred_aligned = y_pred.loc[common_idx]

        metrics = calculate_metrics(y_true_aligned, y_pred_aligned, model_name)
        results[model_name] = metrics

    # Create comparison dataframe
    comparison_df = pd.DataFrame(results).T

    # Print comparison
    print("\nModel Comparison:")
    print(comparison_df[metrics_to_compare])

    # Find best model for each metric
    print("\nBest models:")
    for metric in metrics_to_compare:
        best_model = comparison_df[metric].idxmin()
        best_value = comparison_df[metric].min()
        print(f"{metric}: {best_model} ({best_value:.4f})")

    # Return comparison
    return comparison_df


def residual_analysis(y_true, y_pred, model_name="Model"):
    """
    Perform residual analysis for model diagnostics.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Residual Analysis - {model_name}", fontsize=16)

    # Residuals over time
    axes[0, 0].plot(residuals.index, residuals.values)
    axes[0, 0].axhline(y=0, color="red", linestyle="--")
    axes[0, 0].set_title("Residuals Over Time")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Residual")

    # Residual distribution
    axes[0, 1].hist(residuals, bins=50, alpha=0.7)
    axes[0, 1].axvline(x=0, color="red", linestyle="--")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].set_xlabel("Residual Value")
    axes[0, 1].set_ylabel("Frequency")

    # Q-Q plot for normality
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")

    # Autocorrelation plot
    from pandas.plotting import autocorrelation_plot

    autocorrelation_plot(residuals, ax=axes[1, 1])
    axes[1, 1].set_title("Residual Autocorrelation")

    plt.tight_layout()
    plt.show()

    # Statistical tests
    print(f"\nResidual Statistics for {model_name}:")
    print(f"Mean: {residuals.mean():.6f}")
    print(f"Std: {residuals.std():.6f}")
    print(f"Skewness: {residuals.skew():.6f}")
    print(f"Kurtosis: {residuals.kurtosis():.6f}")

    # Normality test
    _, p_value = stats.shapiro(residuals)
    print(f"Shapiro-Wilk normality test p-value: {p_value:.6f}")
    pass


def forecast_accuracy(y_true, y_pred, confidence_level=0.95):
    """
    Calculate forecast accuracy measures including prediction intervals.
    """
    # Implement forecast accuracy measures
    accuracy = {}

    # Bias measures
    residuals = y_true - y_pred
    accuracy["Mean_Error"] = residuals.mean()
    accuracy["Mean_Absolute_Error"] = abs(residuals).mean()
    accuracy["Root_Mean_Squared_Error"] = np.sqrt((residuals**2).mean())

    # Percentage errors
    percentage_errors = residuals / y_true * 100
    accuracy["Mean_Absolute_Percentage_Error"] = abs(percentage_errors).mean()

    return accuracy


def evaluate_models(
    test_data, trained_models, train_data=None, val_data=None, forecast_horizon=None
):
    """
    Complete evaluation pipeline for both models with component analysis.
    """
    print("Evaluating Models on Test Data...")

    # Handle both DataFrame and Series input
    if isinstance(test_data, pd.DataFrame):
        test_data_full = test_data.dropna()
        test_data_close = (
            test_data_full["Close"]
            if "Close" in test_data_full.columns
            else test_data_full.iloc[:, 0]
        )
    else:
        test_data_close = test_data.dropna()
        test_data_full = None

    print(f"Original test data: {len(test_data)} samples")
    print(f"After removing NaN: {len(test_data_close)} samples")

    # Set forecast horizon to cleaned test data length if not specified
    if forecast_horizon is None or forecast_horizon > len(test_data_close):
        forecast_horizon = len(test_data_close)

    print(f"Forecast horizon: {forecast_horizon} periods")

    # Generate forecasts
    predictions = {}

    # ARIMA forecast
    if trained_models["arima"] is not None:
        # Generate ARIMA forecast
        arima_forecast = trained_models["arima"].forecast(steps=forecast_horizon)

        # Create indexed Series
        predictions["ARIMA"] = pd.Series(
            arima_forecast.values
            if hasattr(arima_forecast, "values")
            else arima_forecast,
            index=test_data_close.index[:forecast_horizon],
            name="ARIMA",
        )
        print(
            f"ARIMA forecast range: ${predictions['ARIMA'].min():.2f} to ${predictions['ARIMA'].max():.2f}"
        )
    else:
        print("ARIMA model not available for evaluation")
        predictions["ARIMA"] = pd.Series(dtype=float)

    # Prophet forecast
    if trained_models["prophet"] is not None:
        # Create future dataframe with only test period dates
        future_df = pd.DataFrame({"ds": test_data_close.index[:forecast_horizon]})

        # Add regressors if they were used during training
        prophet_model = trained_models["prophet"]
        if (
            hasattr(prophet_model.model, "extra_regressors")
            and prophet_model.model.extra_regressors
        ):
            # If test_data_full has the features use them
            if test_data_full is not None:
                for regressor_name in prophet_model.model.extra_regressors.keys():
                    if regressor_name in test_data_full.columns:
                        # Use actual test data values
                        future_df[regressor_name] = (
                            test_data_full[regressor_name]
                            .iloc[:forecast_horizon]
                            .values
                        )

        # Get predictions for future dates
        prophet_forecast = trained_models["prophet"].model.predict(future_df)

        predictions["Prophet"] = pd.Series(
            prophet_forecast["yhat"].values,
            index=test_data_close.index[:forecast_horizon],
            name="Prophet",
        )
        print(
            f"Prophet forecast range: ${predictions['Prophet'].min():.2f} to ${predictions['Prophet'].max():.2f}"
        )
    else:
        print("Prophet model not available for evaluation")
        predictions["Prophet"] = pd.Series(dtype=float)

    # Get test values for comparison
    actual_values = test_data_close.iloc[:forecast_horizon]

    # Compare models
    comparison = compare_models(actual_values, predictions)

    # Plot predictions
    if (
        len(actual_values) > 0
        and not predictions["ARIMA"].empty
        and not predictions["Prophet"].empty
    ):
        plot_predictions(actual_values, predictions["ARIMA"], predictions["Prophet"])

        # Residual analysis for each model
        for model_name, preds in predictions.items():
            if not preds.empty:
                residual_analysis(actual_values, preds, model_name)
    else:
        print("Insufficient data for plotting and residual analysis")

    # ARIMA
    print("ARIMA COEFFICIENT INTERPRETATION")
    print("=" * 70)

    if trained_models.get("arima") is not None:
        interpretation = trained_models["arima"].interpret_coefficients()

        if "error" not in interpretation:
            print(f"\nModel Order: ARIMA{interpretation['order']}")
            print(f"   AIC: {interpretation['aic']:.2f}")
            print(f"   BIC: {interpretation['bic']:.2f}")

            if interpretation["ar_coefficients"]:
                print(f"\nAutoregressive (AR) Coefficients:")
                for ar in interpretation["ar_coefficients"]:
                    print(
                        f"      Lag {ar['lag']}: {ar['coefficient']:+.4f} (p={ar['p_value']:.4f}) {ar['significance']}"
                    )

            if interpretation["ma_coefficients"]:
                print(f"\n   Moving Average (MA) Coefficients:")
                for ma in interpretation["ma_coefficients"]:
                    print(
                        f"      Lag {ma['lag']}: {ma['coefficient']:+.4f} (p={ma['p_value']:.4f}) {ma['significance']}"
                    )

            print(f"\n Other Insights:")
            for insight in interpretation["insights"]:
                print(f"   • {insight}")

    # Prophet
    print("\n" + "=" * 70)
    print("PROPHET COMPONENT DECOMPOSITION")

    if trained_models.get("prophet") is not None:
        try:
            # Get forecast for component analysis
            full_forecast = trained_models["prophet"].predict(
                periods=len(test_data_close), freq="B", train_data=train_data
            )

            # Save component plot
            comp_path = trained_models["prophet"].plot_components_to_file(
                full_forecast, "prophet_components.png"
            )
            print(f"\nProphet components plot saved to: {comp_path}")

            # Interpret components
            comp_interpretation = trained_models["prophet"].interpret_components(
                full_forecast
            )

            if "error" not in comp_interpretation:
                if "trend" in comp_interpretation:
                    trend_info = comp_interpretation["trend"]
                    print(f"\nTrend Analysis:")
                    print(f"   Start: ${trend_info['start_value']:.2f}")
                    print(f"   End: ${trend_info['end_value']:.2f}")
                    print(
                        f"   Change: {trend_info['percent_change']:+.2f}% {trend_info['direction']}"
                    )

                if "yearly_seasonality" in comp_interpretation:
                    yearly = comp_interpretation["yearly_seasonality"]
                    print(f"\nYearly Seasonality:")
                    print(f"   Amplitude: ${yearly['amplitude']:.2f}")
                    print(f"   {yearly['interpretation']}")

                if "monthly_seasonality" in comp_interpretation:
                    monthly = comp_interpretation["monthly_seasonality"]
                    print(f"\nMonthly Seasonality (Custom):")
                    print(f"   Amplitude: ${monthly['amplitude']:.2f}")
                    print(f"   {monthly['interpretation']}")

        except Exception as e:
            print(f"Could not generate Prophet component analysis: {e}")

    return comparison


# Testing
if __name__ == "__main__":
    from preprocessing import split_data, feature_engineering, ensure_datetime_index
    from data_loader import load_data
    from train import train_models

    data = load_data()
    processed = feature_engineering(data)
    processed = ensure_datetime_index(processed)
    train, val, test = split_data(processed)

    models = train_models(train["Close"], val["Close"])
    evaluation_results = evaluate_models(test["Close"], models)
    print("Evaluation completed")
    pass
