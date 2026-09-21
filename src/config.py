from pathlib import Path

from dotenv import load_dotenv
import os


PROJECT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_DIR / ".env")

DATA_PATH = PROJECT_DIR / os.getenv(
    "DATA_PATH",
    "data/raw/House_Prices.csv",
)

RAW_DESCRIPTION_PATH = PROJECT_DIR / os.getenv(
    "RAW_DESCRIPTION_PATH",
    "data/raw/data_description.txt",
)

PROCESSED_DATA_DIR = PROJECT_DIR / "data/processed"
FEATURES_DATA_DIR = PROJECT_DIR / "data/features"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "results/reports"
METRICS_DIR = PROJECT_DIR / "results/metrics"

for directory in [PROCESSED_DATA_DIR, FEATURES_DATA_DIR, MODELS_DIR, REPORTS_DIR, METRICS_DIR,]:
    directory.mkdir(parents=True, exist_ok=True)