import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from src.models.train_models import (
    create_linear_model,
    create_tree_model,
    create_nonlinear_model,
)

CV_FOLDS = 5
RANDOM_STATE = 12


def create_kfold(n_splits=CV_FOLDS, random_state=RANDOM_STATE):
    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def cross_validate_model(model, X, y, cv=None):
    if cv is None:
        cv = create_kfold()

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2",
        },
        n_jobs=-1,
        return_train_score=False,
    )

    return {
        "MAE_mean": -scores["test_MAE"].mean(),
        "MAE_std": scores["test_MAE"].std(),
        "RMSE_mean": -scores["test_RMSE"].mean(),
        "RMSE_std": scores["test_RMSE"].std(),
        "R2_mean": scores["test_R2"].mean(),
        "R2_std": scores["test_R2"].std(),
    }


def compare_models(models, X, y, cv=None):
    results = []

    for model_name, model in models.items():
        scores = cross_validate_model(model, X, y, cv=cv)
        results.append({
            "Model": model_name,
            **scores,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    X_train = pd.read_csv("data/features/x_train.csv")
    y_train = pd.read_csv("data/features/y_train.csv").squeeze("columns")

    models = {
        "Linear Regression": create_linear_model(X_train),
        "Random Forest": create_tree_model(X_train),
        "Nonlinear Model": create_nonlinear_model(X_train),
    }

    results = compare_models(models, X_train, y_train)

    print(results.to_string(index=False))