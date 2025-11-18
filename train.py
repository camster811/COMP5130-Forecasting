"""
Training Module for Time-Series Forecasting Models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
import itertools
import warnings

# Suppress all warnings including ValueWarning from statsmodels
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="statsmodels")


def train_arima(
    train_data, p_range=range(0, 6), d_range=range(0, 3), q_range=range(0, 6)
):
    """
    Train ARIMA model with parameter optimization.
    """
    from statsmodels.tsa.arima.model import ARIMA

    best_aic = float("inf")
    best_params = None
    best_model = None

    # Grid search over p,d,q parameters
    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(train_data, order=(p, d, q))
            # Try to fit with default settings, handle convergence warnings
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = model.fit()
                aic = fitted.aic
            except Exception as e:
                print(f"ARIMA({p},{d},{q}) fitting failed. {e}")
                continue  # Skip if fitting fails

            if aic < best_aic:
                best_aic = aic
                best_params = (p, d, q)
                best_model = fitted
        except Exception as e:
            print(f"Error in ARIMA parameter search: {e}")
            continue

    if best_model is None:
        raise ValueError(
            "No suitable ARIMA model found. Try different parameter ranges."
        )

    return best_model, best_params, best_aic


def train_prophet(train_data, param_grid=None):
    """
    Train Prophet model with optional parameter tuning.
    """
    from models import ProphetModel

    if param_grid is None:
        # Use default parameters
        model = ProphetModel()
    else:
        # Use provided parameters
        model = ProphetModel(**param_grid)

    model.fit(train_data)
    return model


def optimize_prophet_params(train_data, val_data):
    """
    Grid search to find optimal Prophet parameters based on validation MAE.
    """
    from models import ProphetModel

    # Extract Close column for validation comparison
    if isinstance(val_data, pd.DataFrame):
        val_close = val_data["Close"]
    else:
        val_close = val_data

    # Reduced grid - keep at least one seasonality ON for stock data
    param_grid = {
        "changepoint_prior_scale": [0.01, 0.05, 0.1],
        "seasonality_prior_scale": [0.1, 1.0, 10.0],
        "yearly_seasonality": [True],  # Always keep yearly seasonality for stocks
        "weekly_seasonality": [True, False],  # Can toggle weekly
    }

    best_mae = float("inf")
    best_params = None

    print("  Testing Prophet parameter combinations...")
    for params in ParameterGrid(param_grid):
        try:
            model = ProphetModel(
                yearly_seasonality=params["yearly_seasonality"],
                weekly_seasonality=params["weekly_seasonality"],
                changepoint_prior_scale=params["changepoint_prior_scale"],
                seasonality_prior_scale=params["seasonality_prior_scale"],
            )
            model.fit(train_data)

            # Create future dates for validation period - need to add regressors
            future_df = pd.DataFrame({"ds": val_close.index})

            # Add ALL regressor values from validation data
            # Prophet REQUIRES all regressors to be present in the future dataframe
            if isinstance(val_data, pd.DataFrame):
                # Add each regressor that was used during training
                for regressor_name in model.model.extra_regressors.keys():
                    # Calendar features are always present in val_data
                    if regressor_name in val_data.columns:
                        future_df[regressor_name] = val_data[regressor_name].values

            forecast = model.model.predict(future_df)
            predictions = forecast["yhat"]

            mae = np.mean(np.abs(predictions.values - val_close.values))

            if mae < best_mae:
                best_mae = mae
                best_params = params
                print(f"    New best: MAE={mae:.4f}, params={params}")
        except Exception as e:
            print(f"    Skipped params {params}: {e}")
            continue

    print(f"Optimal Prophet parameters: {best_params} with MAE={best_mae:.4f}")
    return best_params


def train_models(train_data, val_data=None):
    """
    Train both ARIMA and Prophet models.
    """
    models = {}

    # Extract Close price for ARIMA (univariate model)
    if isinstance(train_data, pd.DataFrame):
        arima_train = train_data["Close"]
    else:
        arima_train = train_data

    # Train ARIMA using grid search
    print("Training ARIMA...")
    try:
        arima_fitted, arima_params, arima_aic = train_arima(arima_train)

        # Wrap in our ARIMAModel class for interpretation methods
        from models import ARIMAModel

        arima_wrapper = ARIMAModel(
            p=arima_params[0], d=arima_params[1], q=arima_params[2]
        )
        arima_wrapper.model = arima_fitted.model
        arima_wrapper.fitted_model = arima_fitted

        models["arima"] = arima_wrapper
        print(f"Best ARIMA params: {arima_params}, AIC: {arima_aic:.2f}")
    except Exception as e:
        print(f"ARIMA training failed: {e}")
        import traceback

        traceback.print_exc()
        models["arima"] = None

    # Train Prophet with full DataFrame (includes regressors)
    print("Training Prophet...")
    try:
        if val_data is not None:
            # Use optimized parameters if validation data available
            best_params = optimize_prophet_params(train_data, val_data)
            prophet_model = train_prophet(train_data, best_params)
            print("Prophet trained with optimized parameters")
        else:
            # Use default parameters
            prophet_model = train_prophet(train_data)
            print("Prophet trained with default parameters")

        models["prophet"] = prophet_model
    except Exception as e:
        print(f"Prophet training failed: {e}")
        models["prophet"] = None

    return models


if __name__ == "__main__":
    pass
