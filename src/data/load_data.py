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


def save_data(df, path, file):
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / file

    df.to_csv(output_path, index=False)

    return df