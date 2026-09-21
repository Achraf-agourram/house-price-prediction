from pathlib import Path

import pandas as pd


TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    df = pd.read_csv(path)

    required_columns = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError("Missing required columns")

    return df