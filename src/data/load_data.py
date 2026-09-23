from pathlib import Path
import pandas as pd


def load_data(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    required_columns = ["Id", "SalePrice"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' is missing.")

    return df