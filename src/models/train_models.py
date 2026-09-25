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

