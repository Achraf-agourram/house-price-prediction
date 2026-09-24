import pandas as pd

def create_total_sf(df):

    df["TotalSF"] = (df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"])

    return df

def create_total_bathrooms(df):

    df["TotalBathrooms"] = (df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])

    return df

