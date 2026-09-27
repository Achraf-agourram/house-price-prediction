import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.models.cross_validation import create_kfold, cross_validate_model
from src.models.train_models import (
    create_tree_model,
    train_model,
    predict,
    evaluate_model,
)

DATA_DIR = Path("data/features")
OUTPUT_DIR = Path("results")
RANDOM_STATE = 12
CV_FOLDS = 5


def load_data(data_directory=DATA_DIR):
    data_directory = Path(data_directory)

    X_train = pd.read_csv(data_directory / "x_train.csv")
    X_test = pd.read_csv(data_directory / "x_test.csv")
    y_train = pd.read_csv(data_directory / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(data_directory / "y_test.csv").squeeze("columns")

    return X_train, X_test, y_train, y_test

def create_param_grid():
    return {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": [1.0, "sqrt"],
    }

def create_grid_search(model, cv=None, param_grid=None):
    if cv is None:
        cv = create_kfold(CV_FOLDS, RANDOM_STATE)

    if param_grid is None:
        param_grid = create_param_grid()

    return GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

def get_cv_results(grid_search):
    results = pd.DataFrame(grid_search.cv_results_)

    return results[
        [
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ].sort_values("rank_test_score")


def save_results(
    best_params,
    baseline_cv,
    optimized_cv,
    baseline_test,
    optimized_test,
    cv_results,
    output_directory=OUTPUT_DIR,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame(
        [
            {
                "Version": "Before optimization",
                **baseline_cv,
            },
            {
                "Version": "After optimization",
                **optimized_cv,
            },
        ]
    )

    comparison.to_csv(
        output_directory / "cross_validation_comparison.csv",
        index=False,
    )

    test_comparison = pd.DataFrame(
        [
            {
                "Version": "Before optimization",
                **baseline_test,
            },
            {
                "Version": "After optimization",
                **optimized_test,
            },
        ]
    )

    test_comparison.to_csv(
        output_directory / "test_comparison.csv",
        index=False,
    )

    cv_results.to_csv(
        output_directory / "grid_search_results.csv",
        index=False,
    )

    with open(
        output_directory / "best_params.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(best_params, file, indent=4)


def run_optimization(X_train, y_train, X_test, y_test):
    cv = create_kfold(CV_FOLDS, RANDOM_STATE)

    baseline_model = create_tree_model(X_train)

    baseline_cv = cross_validate_model(
        baseline_model,
        X_train,
        y_train,
        cv=cv,
    )

    baseline_model = train_model(
        baseline_model,
        X_train,
        y_train,
    )

    baseline_predictions = predict(
        baseline_model,
        X_test,
    )

    baseline_test = evaluate_model(
        y_test,
        baseline_predictions,
    )

    base_model_for_search = create_tree_model(X_train)

    grid_search = create_grid_search(
        base_model_for_search,
        cv=cv,
        param_grid=create_param_grid(),
    )

    grid_search.fit(X_train, y_train)

    optimized_model = grid_search.best_estimator_

    optimized_cv = cross_validate_model(
        optimized_model,
        X_train,
        y_train,
        cv=cv,
    )

    optimized_predictions = predict(
        optimized_model,
        X_test,
    )

    optimized_test = evaluate_model(
        y_test,
        optimized_predictions,
    )

    grid_results = get_cv_results(grid_search)

    return {
        "best_params": grid_search.best_params_,
        "best_cv_rmse": -grid_search.best_score_,
        "baseline_cv": baseline_cv,
        "optimized_cv": optimized_cv,
        "baseline_test": baseline_test,
        "optimized_test": optimized_test,
        "grid_results": grid_results,
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    results = run_optimization(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    save_results(
        best_params=results["best_params"],
        baseline_cv=results["baseline_cv"],
        optimized_cv=results["optimized_cv"],
        baseline_test=results["baseline_test"],
        optimized_test=results["optimized_test"],
        cv_results=results["grid_results"],
    )

    print("Best parameters:")
    print(results["best_params"])

    print(
        f"Best cross-validation RMSE: "
        f"{results['best_cv_rmse']:.2f}"
    )

    print("\nCross-validation before optimization:")
    print(results["baseline_cv"])

    print("\nCross-validation after optimization:")
    print(results["optimized_cv"])

    print("\nTest performance before optimization:")
    print(results["baseline_test"])

    print("\nTest performance after optimization:")
    print(results["optimized_test"])