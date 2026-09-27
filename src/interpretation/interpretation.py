from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.train_models import (
    create_linear_model,
    create_tree_model,
    create_nonlinear_model,
    train_model,
)


DATA_DIR = Path("data/features")
OUTPUT_DIR = Path("data/interpretation")
PLOTS_DIR = OUTPUT_DIR / "plots"
TOP_N = 20


def load_data(data_directory=DATA_DIR):
    data_directory = Path(data_directory)

    X_train = pd.read_csv(data_directory / "x_train.csv")
    X_test = pd.read_csv(data_directory / "x_test.csv")
    y_train = pd.read_csv(data_directory / "y_train.csv").squeeze("columns")

    return X_train, X_test, y_train

def create_models(X_train):
    return {
        "Linear Regression": create_linear_model(X_train),
        "Random Forest": create_tree_model(X_train),
        "Nonlinear Model": create_nonlinear_model(X_train),
    }

def train_models(models, X_train, y_train):
    trained_models = {}

    for name, model in models.items():
        trained_models[name] = train_model(model, X_train, y_train)

    return trained_models

def get_preprocessed_feature_names(model):
    preprocessor = model.named_steps["preprocessing"]
    feature_names = preprocessor.get_feature_names_out()

    return np.array(feature_names)

def clean_feature_names(feature_names):
    cleaned_names = []

    for name in feature_names:
        name = name.replace("num__", "")
        name = name.replace("cat__", "")
        cleaned_names.append(name)

    return np.array(cleaned_names)

def extract_linear_coefficients(model):
    estimator = model.named_steps["model"]
    feature_names = get_preprocessed_feature_names(model)
    feature_names = clean_feature_names(feature_names)

    coefficients = np.asarray(estimator.coef_).ravel()

    result = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(coefficients),
    })

    return result.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

def extract_tree_importances(model):
    estimator = model.named_steps["model"]
    feature_names = get_preprocessed_feature_names(model)
    feature_names = clean_feature_names(feature_names)

    importances = np.asarray(estimator.feature_importances_)

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    return result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

def aggregate_categorical_importance(result):
    result = result.copy()

    result["base_feature"] = result["feature"].str.split("_").str[0]

    if "importance" in result.columns:
        result = (
            result.groupby("base_feature", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    elif "absolute_coefficient" in result.columns:
        grouped = (
            result.groupby("base_feature", as_index=False)
            .agg(
                absolute_coefficient=("absolute_coefficient", "sum"),
                coefficient=("coefficient", "sum"),
            )
            .sort_values("absolute_coefficient", ascending=False)
            .reset_index(drop=True)
        )
        return grouped

    return result

def calculate_eda_relationships(X_train, y_train):
    numeric_columns = X_train.select_dtypes(include=np.number).columns

    correlations = (
        X_train[numeric_columns]
        .corrwith(y_train)
        .dropna()
        .abs()
        .sort_values(ascending=False)
        .reset_index()
    )

    correlations.columns = ["feature", "absolute_correlation"]

    return correlations

def compare_with_eda(feature_importance, eda_relationships):
    importance = feature_importance.copy()
    eda = eda_relationships.copy()

    importance["feature_base"] = (
        importance["feature"]
        .str.replace(r"^[^_]+__", "", regex=True)
        .str.split("_")
        .str[0]
    )

    eda["feature_base"] = eda["feature"]

    top_importance = importance.head(TOP_N)[["feature_base"]].drop_duplicates()
    top_eda = eda.head(TOP_N)[["feature_base"]].drop_duplicates()

    comparison = pd.DataFrame({
        "feature": sorted(
            set(top_importance["feature_base"])
            | set(top_eda["feature_base"])
        )
    })

    comparison["important_in_model"] = comparison["feature"].isin(
        set(top_importance["feature_base"])
    )
    comparison["strong_target_correlation_in_eda"] = comparison["feature"].isin(
        set(top_eda["feature_base"])
    )

    comparison["interpretation"] = np.select(
        [
            comparison["important_in_model"]
            & comparison["strong_target_correlation_in_eda"],
            comparison["important_in_model"],
            comparison["strong_target_correlation_in_eda"],
        ],
        [
            "Confirmed by both model importance and EDA correlation",
            "Important for the model but not among the strongest linear EDA correlations",
            "Strong EDA relationship but not among the most important model features",
        ],
        default="Not among the strongest results from either analysis",
    )

    return comparison

def plot_linear_coefficients(coefficients, output_directory=PLOTS_DIR):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    data = coefficients.head(TOP_N).sort_values("absolute_coefficient")

    plt.figure(figsize=(10, 8))
    plt.barh(data["feature"], data["coefficient"])
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.title("Top Linear Regression Coefficients")
    plt.tight_layout()
    plt.savefig(
        output_directory / "linear_regression_coefficients.png",
        dpi=150,
    )
    plt.close()

def plot_tree_importances(importances, output_directory=PLOTS_DIR):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    data = importances.head(TOP_N).sort_values("importance")

    plt.figure(figsize=(10, 8))
    plt.barh(data["feature"], data["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(
        output_directory / "random_forest_importances.png",
        dpi=150,
    )
    plt.close()

def plot_eda_comparison(comparison, output_directory=PLOTS_DIR):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    data = comparison[
        comparison["important_in_model"]
        | comparison["strong_target_correlation_in_eda"]
    ].copy()

    data["model"] = data["important_in_model"].astype(int)
    data["eda"] = data["strong_target_correlation_in_eda"].astype(int)

    data = data.sort_values(
        ["model", "eda", "feature"],
        ascending=[True, True, True],
    ).tail(TOP_N)

    positions = np.arange(len(data))
    width = 0.35

    plt.figure(figsize=(12, 8))
    plt.barh(positions - width / 2, data["model"], width, label="Model")
    plt.barh(positions + width / 2, data["eda"], width, label="EDA")
    plt.yticks(positions, data["feature"])
    plt.xlabel("Presence in Top Results")
    plt.ylabel("Feature")
    plt.title("Model Importance vs EDA Relationship")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory / "model_vs_eda.png",
        dpi=150,
    )
    plt.close()

def save_results(
    linear_coefficients,
    tree_importances,
    nonlinear_importances,
    eda_relationships,
    comparison,
    output_directory=OUTPUT_DIR,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    linear_coefficients.to_csv(
        output_directory / "linear_coefficients.csv",
        index=False,
    )

    tree_importances.to_csv(
        output_directory / "random_forest_importances.csv",
        index=False,
    )

    nonlinear_importances.to_csv(
        output_directory / "nonlinear_model_importances.csv",
        index=False,
    )

    eda_relationships.to_csv(
        output_directory / "eda_target_correlations.csv",
        index=False,
    )

    comparison.to_csv(
        output_directory / "model_vs_eda_comparison.csv",
        index=False,
    )


def run_interpretation():
    X_train, X_test, y_train = load_data()

    models = create_models(X_train)
    trained_models = train_models(models, X_train, y_train)

    linear_coefficients = extract_linear_coefficients(
        trained_models["Linear Regression"]
    )

    tree_importances = extract_tree_importances(
        trained_models["Random Forest"]
    )

    nonlinear_importances = extract_tree_importances(
        trained_models["Nonlinear Model"]
    )

    eda_relationships = calculate_eda_relationships(
        X_train,
        y_train,
    )

    comparison = compare_with_eda(
        tree_importances,
        eda_relationships,
    )

    save_results(
        linear_coefficients,
        tree_importances,
        nonlinear_importances,
        eda_relationships,
        comparison,
    )

    plot_linear_coefficients(linear_coefficients)
    plot_tree_importances(tree_importances)
    plot_eda_comparison(comparison)


if __name__ == "__main__":
    run_interpretation()
