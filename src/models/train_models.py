import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from src.features.preprocessing import create_preprocessor


def create_linear_model(X_train):
    preprocessor = create_preprocessor(X_train)

    model = Pipeline([("preprocessing", preprocessor), ("model", LinearRegression())])

    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

def evaluate_model(y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# print(evaluate_model(pd.read_csv("data/features/y_test.csv"), predict(train_model(create_linear_model(pd.read_csv("data/features/x_train.csv")), pd.read_csv("data/features/x_train.csv"), pd.read_csv("data/features/y_train.csv")), pd.read_csv("data/features/x_test.csv"))))
