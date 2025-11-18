"""
Main Script for Time-Series Forecasting: ARIMA vs Prophet

This script orchestrates the entire forecasting pipeline:
1. Data loading and preprocessing
2. Model training (ARIMA and Prophet)
3. Model evaluation and comparison
4. Results visualization and reporting

Default stock: AAL (American Airlines) - chosen for strong seasonal patterns
Airlines exhibit clear seasonality ideal for demonstrating time-series forecasting.
"""

import pandas as pd


def main():
    """
    Main execution function for the forecasting project.
    """
    print(
        "=== Time-Series Forecasting: ARIMA vs Prophet (AAL - American Airlines) ===\n"
    )

    try:
        # Import modules
        from data_loader import load_data, validate_data
        from preprocessing import feature_engineering, ensure_datetime_index, split_data
        from train import train_models
        from evaluate import evaluate_models

        # Step 1: Data Loading
        print("Step 1: Loading stock data...")
        data = load_data()

        is_valid, cleaned_data = validate_data(data)
        if not is_valid:
            print("Data validation failed. Exiting.")
            return

        print(f"Data loaded successfully. Shape: {cleaned_data.shape}")
        print(
            f"   Date range: {cleaned_data.index.min()} to {cleaned_data.index.max()}"
        )
        print(
            f"   Price range: ${cleaned_data['Close'].min():.2f} - ${cleaned_data['Close'].max():.2f}"
        )
        print("   Target: Close price")

        # Step 2: Preprocessing
        print("\nStep 2: Preprocessing data...")
        processed_data = feature_engineering(cleaned_data)
        processed_data = ensure_datetime_index(processed_data)
        train_data, val_data, test_data = split_data(processed_data)

        print("Preprocessing completed:")
        print(f"   Train set: {len(train_data)} samples")
        print(f"   Validation set: {len(val_data)} samples")
        print(f"   Test set: {len(test_data)} samples")

        # Step 3: Model Training
        print("\nStep 3: Training models...")
        # Pass full DataFrames to use engineered features
        trained_models = train_models(train_data, val_data)

        if trained_models["arima"] is None or trained_models["prophet"] is None:
            print("A model failed to train. Exiting.")
            return

        # Step 4: Model Evaluation
        print("\nStep 4: Evaluating models...")
        # Pass full test_data DataFrame so Prophet can access regressor features
        evaluation_results = evaluate_models(
            test_data, trained_models, train_data, val_data
        )

        # Step 5: Generate Report
        print("\nStep 5: Generating final report...")
        generate_report(
            cleaned_data, processed_data, trained_models, evaluation_results
        )

    except Exception as e:
        print(f"\nError in main execution: {e}")
        import traceback

        traceback.print_exc()


def generate_report(raw_data, processed_data, trained_models, evaluation_results):
    """
    Generate comprehensive project report.
    """
    print("ARIMA vs Prophet Stock Forecasting")
    print("=" * 70)

    overall_best = "N/A"

    # 1. Data Summary
    print("\nDATA SUMMARY")
    print("-" * 30)
    print(f"   • Total observations: {len(raw_data)}")
    print(f"   • Date range: {raw_data.index.min()} to {raw_data.index.max()}")
    print("   • Target variable: Stock Close price")
    print(
        f"   • Price range: ${raw_data['Close'].min():.2f} - ${raw_data['Close'].max():.2f}"
    )
    print(f"   • Features after engineering: {processed_data.shape[1]}")

    # 2. Preprocessing Summary
    print("\nPREPROCESSING SUMMARY")
    print("-" * 30)
    missing_vals = processed_data.isnull().sum().sum()
    print("   • Date indexing: Applied")
    print(
        f"   • Missing values: {'None' if missing_vals == 0 else f'⚠️  {missing_vals} found'}"
    )
    print("   • Feature engineering: day_of_week, month, quarter, is_month_end")
    print("   • Train/Val/Test split: 80/10/10 ratio")
    print("   • Prophet regressors: Uses calendar features")

    # 3. Model Summary
    print("\nMODEL TRAINING SUMMARY")
    print("-" * 30)
    for model_name, model in trained_models.items():
        if model is not None:
            status = "Trained successfully"
            if model_name == "arima" and hasattr(model, "params"):
                params = model.params
                p, d, q = (
                    params["ar.L1"],
                    params["ma.L1"],
                    0,
                )  # Extract from fitted model
                try:
                    q = params["ma.L1"]
                except Exception as e:
                    print(f"Error extracting MA parameter: {e}")
                    q = 0
                print(f"   • ARIMA: {status} (p={p:.0f}, d={d:.0f}, q={q:.0f})")
            elif model_name == "prophet":
                print(f"   • Prophet: {status} (optimized hyperparameters)")
        else:
            print(f"   • {model_name.title()}: Training failed")

    # 4. Evaluation Results
    print("\nEVALUATION RESULTS")
    print("-" * 30)
    if evaluation_results is not None:
        print(evaluation_results)

        # Find best model for each metric
        print("\nBEST MODELS BY METRIC:")
        for metric in evaluation_results.columns:
            best_model = evaluation_results[metric].idxmin()
            best_value = evaluation_results[metric].min()
            print(f"   • {metric}: {best_model} ({best_value:.4f})")

        # Determine overall best model based on weighted importance
        # For trading: accuracy (MAE, RMSE, MAPE) and directional prediction matter most
        print("\nOVERALL BEST MODEL DETERMINATION:")

        # Count wins in key accuracy metrics (lower is better)
        accuracy_metrics = ["MAE", "RMSE", "MAPE", "Theils_U"]
        accuracy_wins = {}
        for model in evaluation_results.index:
            wins = sum(
                1
                for metric in accuracy_metrics
                if metric in evaluation_results.columns
                and evaluation_results.loc[model, metric]
                == evaluation_results[metric].min()
            )
            accuracy_wins[model] = wins

        # Check directional accuracy (higher is better)
        if "Directional_Accuracy" in evaluation_results.columns:
            dir_acc_best = evaluation_results["Directional_Accuracy"].idxmax()
            dir_acc_values = evaluation_results["Directional_Accuracy"]
            print(
                f"   • Directional Accuracy: {dir_acc_best} ({dir_acc_values[dir_acc_best]:.2f}% vs {dir_acc_values[dir_acc_values.index != dir_acc_best].values[0]:.2f}%)"
            )

        # Check MAE (most interpretable metric)
        if "MAE" in evaluation_results.columns:
            mae_best = evaluation_results["MAE"].idxmin()
            mae_values = evaluation_results["MAE"]
            improvement = (mae_values.max() - mae_values.min()) / mae_values.max() * 100
            print(
                f"   • MAE Winner: {mae_best} (${mae_values[mae_best]:.2f} vs ${mae_values[mae_values.index != mae_best].values[0]:.2f}, {improvement:.1f}% better)"
            )

        # Overall determination: Prophet wins if it has better MAE AND directional accuracy
        prophet_better_mae = (
            evaluation_results.loc["Prophet", "MAE"]
            < evaluation_results.loc["ARIMA", "MAE"]
            if "MAE" in evaluation_results.columns
            else False
        )
        prophet_better_dir = (
            evaluation_results.loc["Prophet", "Directional_Accuracy"]
            > evaluation_results.loc["ARIMA", "Directional_Accuracy"]
            if "Directional_Accuracy" in evaluation_results.columns
            else False
        )

        if prophet_better_mae and prophet_better_dir:
            overall_best = "Prophet"
            print(f"   • Decision: PROPHET (superior in both accuracy and direction)")
        elif prophet_better_mae:
            overall_best = "Prophet"
            print(
                f"   • Decision: PROPHET (superior accuracy despite weaker directional prediction)"
            )
        elif prophet_better_dir:
            overall_best = "ARIMA"
            print(
                f"   • Decision: ARIMA (better directional prediction despite higher errors)"
            )
        else:
            overall_best = "ARIMA"
            print(f"   • Decision: ARIMA (superior in both metrics)")

        print(f"\n>>> OVERALL BEST MODEL: {overall_best.upper()} <<<")
    else:
        print("No evaluation results available")
        overall_best = "N/A"

    # 5. Key Insights
    print("\nINSIGHTS")
    print("-" * 40)

    if evaluation_results is not None:
        prophet_better = overall_best == "Prophet"
        print(f"   • {overall_best} showed better overall performance")

        # Provide specific insights based on results
        if "MAE" in evaluation_results.columns:
            mae_diff = abs(
                evaluation_results.loc["Prophet", "MAE"]
                - evaluation_results.loc["ARIMA", "MAE"]
            )
            print(f"   • MAE difference: ${mae_diff:.2f} in favor of {overall_best}")

        if "Directional_Accuracy" in evaluation_results.columns:
            dir_diff = abs(
                evaluation_results.loc["Prophet", "Directional_Accuracy"]
                - evaluation_results.loc["ARIMA", "Directional_Accuracy"]
            )
            dir_winner = evaluation_results["Directional_Accuracy"].idxmax()
            print(
                f"   • Directional accuracy: {dir_diff:.1f}% advantage to {dir_winner}"
            )

        print(
            "   • Stock prices appear to follow momentum patterns (Prophet's regressor features help)"
            if prophet_better
            else "   • Stock prices show mean-reverting behavior (ARIMA's AR terms capture this)"
        )

    # Save report to file
    report_content = f"""
TIME-SERIES FORECASTING PROJECT REPORT

Generated on: {pd.Timestamp.now()}

ARIMA vs Prophet Forecasting of AAL Stock Prices

DATA SUMMARY
- Observations: {len(raw_data)}
- Date Range: {raw_data.index.min()} to {raw_data.index.max()}
- Target: AAL Close Price
- Price Range: ${raw_data["Close"].min():.2f} - ${raw_data["Close"].max():.2f}

PREPROCESSING
- Features: {processed_data.shape[1]} (OHLCV + engineered features)
- Missing Values: {"None" if missing_vals == 0 else str(missing_vals)}
- Split: 80% train, 10% validation, 10% test

MODELS TRAINED
"""

    for model_name, model in trained_models.items():
        status = "SUCCESS" if model is not None else "FAILED"
        report_content += f"- {model_name.upper()}: {status}\n"

    if evaluation_results is not None:
        report_content += f"""

EVALUATION RESULTS
{evaluation_results.to_string()}

BEST MODEL: {overall_best.upper() if "overall_best" in locals() else "N/A"}
"""

    report_content += """
"""

    with open("forecasting_report.txt", "w") as f:
        f.write(report_content)

    print("\nReport saved to 'forecasting_report.txt'")


if __name__ == "__main__":
    main()
