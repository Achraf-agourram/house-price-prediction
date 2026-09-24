import pandas as pd

def create_total_sf(df):

    df["TotalSF"] = (df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"])

    return df

def create_total_bathrooms(df):

    df["TotalBathrooms"] = (df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])

    return df

def create_house_age(df):

    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"])

    return df

def create_remod_age(df):

    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"])

    return df

def create_total_porch_sf(df):

    df["TotalPorchSF"] = (df["WoodDeckSF"] + df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"])

    return df

def create_total_indoor_sf(df):

    df["TotalIndoorSF"] = (df["GrLivArea"] + df["TotalBsmtSF"])

    return df

def create_garage_age(df):

    df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"])

    return df

