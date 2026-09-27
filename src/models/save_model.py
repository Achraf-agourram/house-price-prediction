import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

from src.models.train_models import create_tree_model


DATA_DIR = Path("data/features")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
SCHEMA_PATH = MODEL_DIR / "input_schema.json"

RANDOM_STATE = 12
CV_FOLDS = 5


def load_training_data(data_directory=DATA_DIR):
    data_directory = Path(data_directory)

    X_train = pd.read_csv(data_directory / "x_train.csv")
    y_train = pd.read_csv(data_directory / "y_train.csv").squeeze("columns")

    return X_train, y_train


def create_param_grid():
    return {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": [1.0, "sqrt"],
    }


def train_optimized_model(X_train, y_train):
    model = create_tree_model(X_train)

    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=create_param_grid(),
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def build_input_schema(X_train):
    schema = {"features": []}

    for column in X_train.columns:
        series = X_train[column]

        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")

            minimum = numeric_series.min()
            maximum = numeric_series.max()
            median = numeric_series.median()

            schema["features"].append({
                "name": column,
                "type": "numeric",
                "dtype": str(series.dtype),
                "min": None if pd.isna(minimum) else float(minimum),
                "max": None if pd.isna(maximum) else float(maximum),
                "median": None if pd.isna(median) else float(median),
            })
        else:
            values = (
                series.dropna()
                .astype(str)
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            schema["features"].append({
                "name": column,
                "type": "categorical",
                "dtype": str(series.dtype),
                "values": values,
                "default": values[0] if values else "",
            })

    return schema


def save_model(model, model_path=MODEL_PATH):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def save_schema(schema, schema_path=SCHEMA_PATH):
    schema_path = Path(schema_path)
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    with open(schema_path, "w", encoding="utf-8") as file:
        json.dump(schema, file, indent=4, ensure_ascii=False)


def save_training_info(best_params, model_directory=MODEL_DIR):
    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)

    with open(
        model_directory / "training_info.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "best_params": best_params,
                "cv_folds": CV_FOLDS,
                "random_state": RANDOM_STATE,
            },
            file,
            indent=4,
        )


def train_and_save():
    X_train, y_train = load_training_data()

    model, best_params = train_optimized_model(
        X_train,
        y_train,
    )

    schema = build_input_schema(X_train)

    save_model(model)
    save_schema(schema)
    save_training_info(best_params)

    return {
        "model_path": str(MODEL_PATH),
        "schema_path": str(SCHEMA_PATH),
        "best_params": best_params,
    }


if __name__ == "__main__":
    result = train_and_save()

    print(f"Model saved to: {result['model_path']}")
    print(f"Schema saved to: {result['schema_path']}")
    print("Best parameters:")
    print(result["best_params"])
