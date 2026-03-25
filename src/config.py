from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "archive (4)"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TRAIN_FILE = "train - Walmart Sales Forecast.csv"
TEST_FILE = "test - Walmart Sales Forecast.csv"
STORES_FILE = "stores - Walmart Sales Forecast.csv"
FEATURES_FILE = "features - Walmart Sales Forecast.csv"

RANDOM_STATE = 42
