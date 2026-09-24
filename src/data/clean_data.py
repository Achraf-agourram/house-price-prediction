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

def fix_invalid_values(df):

    if "SalePrice" in df.columns:
        df.loc[df["SalePrice"] <= 0, "SalePrice"] = np.nan

    if "LotArea" in df.columns:
        df.loc[df["LotArea"] <= 0, "LotArea"] = np.nan

    if "GrLivArea" in df.columns:
        df.loc[df["GrLivArea"] <= 0, "GrLivArea"] = np.nan

    if "OverallQual" in df.columns:
        df.loc[~df["OverallQual"].between(1, 10), "OverallQual"] = np.nan

    if "OverallCond" in df.columns:
        df.loc[~df["OverallCond"].between(1, 10), "OverallCond"] = np.nan

    if "MoSold" in df.columns:
        df.loc[~df["MoSold"].between(1, 12), "MoSold"] = np.nan

    return df

def fix_year_values(df):

    if "YearBuilt" in df.columns:
        invalid_year_built = (df["YearBuilt"] > df["YrSold"])

        df.loc[invalid_year_built, "YearBuilt"] = np.nan

    if "YearRemodAdd" in df.columns:
        invalid_remodel_year = (df["YearRemodAdd"] > df["YrSold"])

        df.loc[invalid_remodel_year, "YearRemodAdd"] = np.nan

    if "GarageYrBlt" in df.columns:
        invalid_garage_year = (df["GarageYrBlt"] > df["YrSold"])

        df.loc[invalid_garage_year, "GarageYrBlt"] = np.nan

    return df

def remove_duplicates(df):
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    return df

