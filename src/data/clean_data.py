import numpy as np
import pandas as pd


ABSENCE_CATEGORICAL_COLUMNS = [
    "Alley",
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
]

ABSENCE_NUMERICAL_COLUMNS = [
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
]


def clean_text_columns(df):

    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        df[column] = df[column].apply(lambda value: value.strip() if isinstance(value, str) else value)
        df[column] = df[column].replace("", np.nan)

    return df

def handle_absence_values(df):

    for column in ABSENCE_CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("None")

    for column in ABSENCE_NUMERICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    return df

