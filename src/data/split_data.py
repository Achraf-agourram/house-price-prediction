from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"

TEST_SIZE = 0.20
RANDOM_STATE = 42

def split_features_target(df):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{TARGET_COLUMN} is not in the dataset.")

    x = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN]

    return x, y

def split_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    return (x_train, x_test, y_train, y_test)

