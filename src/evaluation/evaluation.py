import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.train_models import (
    create_linear_model,
    create_tree_model,
    create_nonlinear_model,
    train_model,
    predict,
    evaluate_model,
)


DATA_DIR = Path("data/features")
OUTPUT_DIR = Path("data/evaluation")
PLOTS_DIR = OUTPUT_DIR / "plots"

TOP_ERRORS = 20


def load_data(data_directory=DATA_DIR):
    data_directory = Path(data_directory)

    X_train = pd.read_csv(data_directory / "x_train.csv")
    X_test = pd.read_csv(data_directory / "x_test.csv")
    y_train = pd.read_csv(data_directory / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(data_directory / "y_test.csv").squeeze("columns")

    return X_train, X_test, y_train, y_test

def create_models(X_train):
    return {
        "Linear Regression": create_linear_model(X_train),
        "Random Forest": create_tree_model(X_train),
        "Nonlinear Model": create_nonlinear_model(X_train),
    }

def train_and_predict_models(models, X_train, y_train, X_test):
    predictions = {}

    for name, model in models.items():
        trained_model = train_model(model, X_train, y_train)
        predictions[name] = predict(trained_model, X_test)

    return predictions

def calculate_comparison_table(y_test, predictions):
    results = []

    for name, model_predictions in predictions.items():
        metrics = evaluate_model(y_test, model_predictions)

        results.append(
            {
                "Model": name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

    return pd.DataFrame(results).sort_values(
        by="RMSE",
        ascending=True,
    ).reset_index(drop=True)

def save_comparison_table(results, output_directory=OUTPUT_DIR):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_directory / "model_comparison.csv",
        index=False,
    )

def create_plot_directory(output_directory=PLOTS_DIR):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory

def get_filename(model_name, suffix):
    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    return f"{filename}_{suffix}.png"

def plot_actual_vs_predicted(y_test, predictions, output_directory=PLOTS_DIR):
    output_directory = create_plot_directory(output_directory)

    for model_name, model_predictions in predictions.items():
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, model_predictions, alpha=0.6)

        minimum = min(y_test.min(), model_predictions.min())
        maximum = max(y_test.max(), model_predictions.max())

        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
        )

        plt.xlabel("Actual SalePrice")
        plt.ylabel("Predicted SalePrice")
        plt.title(f"Actual vs Predicted - {model_name}")
        plt.tight_layout()
        plt.savefig(
            output_directory / get_filename(
                model_name,
                "actual_vs_predicted",
            ),
            dpi=150,
        )
        plt.close()

def calculate_errors(y_test, predictions):
    errors = {}

    for model_name, model_predictions in predictions.items():
        model_errors = pd.DataFrame(
            {
                "actual": y_test.to_numpy(),
                "predicted": model_predictions,
            }
        )

        model_errors["error"] = (
            model_errors["predicted"] - model_errors["actual"]
        )
        model_errors["absolute_error"] = model_errors["error"].abs()
        model_errors["relative_error_percent"] = np.where(
            model_errors["actual"] != 0,
            (
                model_errors["absolute_error"]
                / model_errors["actual"].abs()
            )
            * 100,
            np.nan,
        )

        errors[model_name] = model_errors

    return errors

def plot_error_distribution(errors, output_directory=PLOTS_DIR):
    output_directory = create_plot_directory(output_directory)

    for model_name, model_errors in errors.items():
        plt.figure(figsize=(8, 6))
        plt.hist(model_errors["error"], bins=30)
        plt.axvline(0, linestyle="--")
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title(f"Error Distribution - {model_name}")
        plt.tight_layout()
        plt.savefig(
            output_directory / get_filename(
                model_name,
                "error_distribution",
            ),
            dpi=150,
        )
        plt.close()

def plot_residual_analysis(errors, output_directory=PLOTS_DIR):
    output_directory = create_plot_directory(output_directory)

    for model_name, model_errors in errors.items():
        plt.figure(figsize=(8, 6))
        plt.scatter(
            model_errors["predicted"],
            model_errors["error"],
            alpha=0.6,
        )
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted SalePrice")
        plt.ylabel("Residual")
        plt.title(f"Residual Analysis - {model_name}")
        plt.tight_layout()
        plt.savefig(
            output_directory / get_filename(
                model_name,
                "residuals",
            ),
            dpi=150,
        )
        plt.close()

def plot_model_comparison(results, output_directory=PLOTS_DIR):
    output_directory = create_plot_directory(output_directory)

    for metric in ["MAE", "RMSE", "R2"]:
        plt.figure(figsize=(9, 6))
        plt.bar(results["Model"], results[metric])
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.title(f"Model Comparison - {metric}")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(
            output_directory / f"model_comparison_{metric.lower()}.png",
            dpi=150,
        )
        plt.close()

def numeric_feature_deviations(row, X_train):
    explanations = []
    important_columns = [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "TotalBsmtSF",
        "YearBuilt",
        "1stFlrSF",
        "FullBath",
        "TotRmsAbvGrd",
    ]

    for column in important_columns:
        if column not in X_train.columns:
            continue

        train_series = pd.to_numeric(
            X_train[column],
            errors="coerce",
        )
        value = pd.to_numeric(
            pd.Series([row[column]]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(value):
            continue

        median = train_series.median()
        std = train_series.std()

        if pd.isna(median) or pd.isna(std) or std == 0:
            continue

        deviation = (value - median) / std

        if abs(deviation) >= 2:
            direction = "high" if deviation > 0 else "low"
            explanations.append(
                f"{column} is unusually {direction} compared with the training data"
            )

    return explanations

def categorical_feature_deviations(row, X_train):
    explanations = []
    important_columns = [
        "Neighborhood",
        "BldgType",
        "HouseStyle",
        "ExterQual",
        "KitchenQual",
    ]

    for column in important_columns:
        if column not in X_train.columns:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        train_counts = X_train[column].value_counts(normalize=True)
        frequency = train_counts.get(value, 0)

        if frequency < 0.05:
            explanations.append(
                f"{column}={value} is relatively uncommon in the training data"
            )

    return explanations

def explain_prediction_error(row, X_train):
    explanations = []
    explanations.extend(numeric_feature_deviations(row, X_train))
    explanations.extend(categorical_feature_deviations(row, X_train))

    if not explanations:
        explanations.append(
            "No strong feature-level anomaly was detected using the available explanatory variables"
        )

    return "; ".join(explanations[:4])

def identify_largest_errors(
    X_test,
    y_test,
    predictions,
    X_train,
    top_n=TOP_ERRORS,
):
    output = []

    for model_name, model_predictions in predictions.items():
        model_data = X_test.copy()
        model_data.insert(0, "test_index", X_test.index)
        model_data["actual_price"] = y_test.to_numpy()
        model_data["predicted_price"] = model_predictions
        model_data["error"] = (
            model_data["predicted_price"] - model_data["actual_price"]
        )
        model_data["absolute_error"] = model_data["error"].abs()
        model_data["relative_error_percent"] = np.where(
            model_data["actual_price"] != 0,
            (
                model_data["absolute_error"]
                / model_data["actual_price"].abs()
            )
            * 100,
            np.nan,
        )
        model_data["model"] = model_name
        model_data["explanation"] = model_data.apply(
            lambda row: explain_prediction_error(row, X_train),
            axis=1,
        )

        output.append(
            model_data.nlargest(top_n, "absolute_error")
        )

    return pd.concat(output, ignore_index=True)

def save_largest_errors(largest_errors, output_directory=OUTPUT_DIR):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    largest_errors.to_csv(
        output_directory / "largest_prediction_errors.csv",
        index=False,
    )

def save_summary(results, largest_errors, output_directory=OUTPUT_DIR):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_with_lowest_RMSE": results.loc[
            results["RMSE"].idxmin(), "Model"
        ],
        "model_with_lowest_MAE": results.loc[
            results["MAE"].idxmin(), "Model"
        ],
        "model_with_highest_R2": results.loc[
            results["R2"].idxmax(), "Model"
        ],
        "number_of_largest_error_rows": len(largest_errors),
    }

    with open(
        output_directory / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

def run_evaluation():
    X_train, X_test, y_train, y_test = load_data()
    models = create_models(X_train)

    predictions = train_and_predict_models(
        models,
        X_train,
        y_train,
        X_test,
    )

    results = calculate_comparison_table(y_test, predictions)
    errors = calculate_errors(y_test, predictions)

    largest_errors = identify_largest_errors(
        X_test,
        y_test,
        predictions,
        X_train,
    )

    save_comparison_table(results)
    save_largest_errors(largest_errors)
    save_summary(results, largest_errors)

    plot_actual_vs_predicted(y_test, predictions)
    plot_error_distribution(errors)
    plot_residual_analysis(errors)
    plot_model_comparison(results)


if __name__ == "__main__":
    run_evaluation()
