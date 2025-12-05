"""
Model Definitions for Time-Series Forecasting
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
import warnings

warnings.filterwarnings("ignore")


class ARIMAModel:
    """
    ARIMA Model for time-series forecasting.

    ARIMA(p,d,q) where:
    - p: autoregressive order
    - d: differencing order (for stationarity)
    - q: moving average order
    """

    def __init__(self, p=5, d=1, q=0):
        self.p = p
        self.d = d
        self.q = q
        self.model = None
        self.fitted_model = None

    def fit(self, train_data):
        """
        Fit ARIMA model to training data.
        """

        self.model = ARIMA(train_data, order=(self.p, self.d, self.q))
        self.fitted_model = self.model.fit()
        pass

    def predict(self, steps=30):
        """
        Generate forecasts from fitted model.
        """

        if self.fitted_model is None:
            raise ValueError("Model must be fitted before predicting")

        forecast = self.fitted_model.forecast(steps)
        # Ensure we return a pandas Series
        try:
            forecast = pd.Series(forecast, name="forecast")
        except Exception as e:
            print(f"Error creating forecast Series: {e}")
            forecast = pd.Series(np.asarray(forecast))
        return forecast

    def forecast(self, steps=30):
        """
        Alias for predict method to match statsmodels ARIMAResults interface.
        """
        return self.predict(steps)

    def get_summary(self):
        """
        Get model summary and diagnostics.
        """
        if self.fitted_model is None:
            return "Model not fitted yet"

        return self.fitted_model.summary()

    def interpret_coefficients(self):
        """
        Interpret ARIMA coefficients and provide insights.
        """
        if self.fitted_model is None:
            return {"error": "Model not fitted yet"}

        interpretation = {
            "order": (self.p, self.d, self.q),
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
        }

        # Get parameter values
        params = self.fitted_model.params
        pvalues = self.fitted_model.pvalues

        # Interpret AR coefficients
        ar_coeffs = []
        for i in range(1, self.p + 1):
            param_name = f"ar.L{i}"
            if param_name in params.index:
                coef = params[param_name]
                pval = pvalues[param_name]
                significant = "Significant" if pval < 0.05 else "SNot significant"
                ar_coeffs.append(
                    {
                        "lag": i,
                        "coefficient": coef,
                        "p_value": pval,
                        "significance": significant,
                    }
                )
        interpretation["ar_coefficients"] = ar_coeffs

        # Interpret MA coefficients
        ma_coeffs = []
        for i in range(1, self.q + 1):
            param_name = f"ma.L{i}"
            if param_name in params.index:
                coef = params[param_name]
                pval = pvalues[param_name]
                significant = "Significant" if pval < 0.05 else "Not significant"
                ma_coeffs.append(
                    {
                        "lag": i,
                        "coefficient": coef,
                        "p_value": pval,
                        "significance": significant,
                    }
                )
        interpretation["ma_coefficients"] = ma_coeffs

        insights = []

        if self.d > 0:
            insights.append(
                f"Series was differenced {self.d} time(s) to achieve stationarity"
            )

        if self.p > 0:
            total_ar_effect = sum([abs(c["coefficient"]) for c in ar_coeffs])
            if total_ar_effect > 0.5:
                insights.append(
                    "Strong autoregressive behavior - past prices heavily influence future"
                )
            else:
                insights.append(
                    "Weak autoregressive behavior - limited memory of past prices"
                )

        if self.q > 0:
            total_ma_effect = sum([abs(c["coefficient"]) for c in ma_coeffs])
            if total_ma_effect > 0.5:
                insights.append(
                    "Strong moving average component - shocks persist over time"
                )
            else:
                insights.append("Weak moving average component - shocks fade quickly")

        interpretation["insights"] = insights

        return interpretation


class ProphetModel:
    """
    Facebook Prophet for time-series forecasting.
    """

    def __init__(
        self,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        seasonality_mode="multiplicative",
    ):
        """
        Initialize Prophet model optimized for stock trading.
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.model = None

    def fit(self, train_data):
        """
        Fit Prophet model to training data.
        """
        # Handle both Series and DataFrame input
        if isinstance(train_data, pd.Series):
            # Convert timezone-aware index to naive (Prophet requirement)
            dates = train_data.index
            if hasattr(dates, "tz") and dates.tz is not None:
                dates = dates.tz_convert("UTC").tz_localize(None)

            prophet_data = pd.DataFrame({"ds": dates, "y": train_data.values})
            has_features = False
        else:
            # Convert timezone-aware index to naive (Prophet requirement)
            dates = train_data.index
            if hasattr(dates, "tz") and dates.tz is not None:
                dates = dates.tz_convert("UTC").tz_localize(None)

            prophet_data = pd.DataFrame({"ds": dates, "y": train_data["Close"].values})
            has_features = True

            # Add calendar features as regressors
            # These are always known in advance
            if "day_of_week" in train_data.columns:
                prophet_data["day_of_week"] = train_data["day_of_week"].values
            if "month" in train_data.columns:
                prophet_data["month"] = train_data["month"].values
            if "quarter" in train_data.columns:
                prophet_data["quarter"] = train_data["quarter"].values
            if "is_month_end" in train_data.columns:
                prophet_data["is_month_end"] = train_data["is_month_end"].values

        # Create Prophet model with stock-optimized parameters
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            interval_width=0.95,
        )

        # Add calendar features as regressors
        if has_features:
            if "day_of_week" in prophet_data.columns:
                self.model.add_regressor("day_of_week")
            if "month" in prophet_data.columns:
                self.model.add_regressor("month")
            if "quarter" in prophet_data.columns:
                self.model.add_regressor("quarter")
            if "is_month_end" in prophet_data.columns:
                self.model.add_regressor("is_month_end")
        # Fit the model
        self.model.fit(prophet_data)

    def predict(self, periods=30, freq="B", train_data=None):
        """
        Generate forecasts from fitted model.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before predicting")

        # Create future dataframe with business day frequency for stocks
        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        # Add calendar features for future predictions
        if train_data is not None and isinstance(train_data, pd.DataFrame):
            # Extract calendar features from future dates
            future_dates = pd.to_datetime(future["ds"])

            if "day_of_week" in self.model.extra_regressors:
                future["day_of_week"] = future_dates.dt.dayofweek

            if "month" in self.model.extra_regressors:
                future["month"] = future_dates.dt.month

            if "quarter" in self.model.extra_regressors:
                future["quarter"] = future_dates.dt.quarter

            if "is_month_end" in self.model.extra_regressors:
                future["is_month_end"] = future_dates.dt.is_month_end.astype(int)

        forecast = self.model.predict(future)
        return forecast

    def get_components(self, forecast):
        """
        Get forecast components (trend, seasonality, etc.)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting components")

        return self.model.plot_components(forecast)

    def plot_components_to_file(self, forecast, save_path="prophet_components.png"):
        """
        Plot and save Prophet components (trend, seasonality) to file.
        """
        import matplotlib.pyplot as plt

        if self.model is None:
            raise ValueError("Model must be fitted before plotting components")

        fig = self.model.plot_components(forecast)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def interpret_components(self, forecast):
        """
        Interpret Prophet forecast components.
        """
        if self.model is None:
            return {"error": "Model not fitted yet"}

        interpretation = {}

        # Trend analysis
        trend = forecast["trend"]
        trend_start = trend.iloc[0]
        trend_end = trend.iloc[-1]
        trend_change = ((trend_end - trend_start) / trend_start) * 100

        interpretation["trend"] = {
            "start_value": trend_start,
            "end_value": trend_end,
            "percent_change": trend_change,
            "direction": "Upward" if trend_change > 0 else "Downward",
        }

        # Yearly seasonality analysis (if available)
        if "yearly" in forecast.columns:
            yearly = forecast["yearly"]
            yearly_amplitude = yearly.max() - yearly.min()
            interpretation["yearly_seasonality"] = {
                "amplitude": yearly_amplitude,
                "peak_month": yearly.idxmax().month
                if hasattr(yearly.idxmax(), "month")
                else "N/A",
                "interpretation": "Strong yearly pattern"
                if yearly_amplitude > trend.mean() * 0.1
                else "Weak yearly pattern",
            }

        # Weekly seasonality analysis (if available)
        if "weekly" in forecast.columns:
            weekly = forecast["weekly"]
            weekly_amplitude = weekly.max() - weekly.min()
            interpretation["weekly_seasonality"] = {
                "amplitude": weekly_amplitude,
                "interpretation": "Strong weekly pattern"
                if weekly_amplitude > trend.mean() * 0.05
                else "Weak weekly pattern",
            }

        return interpretation


def create_model(model_type="arima", **kwargs):
    """
    Factory function to create model instances.
    """
    if model_type.lower() == "arima":
        return ARIMAModel(**kwargs)
    elif model_type.lower() == "prophet":
        return ProphetModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Testing
if __name__ == "__main__":
    from preprocessing import feature_engineering, split_data, ensure_datetime_index
    from data_loader import load_data, validate_data
    from train import optimize_prophet_params

    data = load_data()
    is_valid, cleaned_data = validate_data(data)
    if not is_valid:
        print("Data validation failed.")
        exit(1)
    processed = feature_engineering(cleaned_data)
    processed = ensure_datetime_index(processed)
    train, val, test = split_data(processed)

    print("Finding optimal ARIMA parameters...")
    # d = find_differencing_order(train["Close"])
    print(f"Suggested differencing order d: {d}")
    # p, q = find_optimal_pq(train["Close"], d)
    print(f"Optimal (p,q): ({p},{q})")

    # Test ARIMA
    print("Testing ARIMA Model...")
    # arima_model = create_model("arima", p=p, d=d, q=q)
    arima_model.fit(train["Close"])
    arima_forecast = arima_model.predict(steps=len(test))
    print("ARIMA forecast:\n", arima_forecast)

    print("Finding optimal Prophet parameters...")
    best_params = optimize_prophet_params(train["Close"], val["Close"])

    # Test Prophet
    print("Testing Prophet Model...")
    prophet_model = create_model("prophet", **best_params)
    prophet_model.fit(train["Close"])
    prophet_forecast = prophet_model.predict(periods=len(test))
    prophet_predictions = prophet_forecast["yhat"].iloc[-len(test) :]
    print("Prophet forecast:\n", prophet_predictions)  # Show only test period forecasts
    pass
