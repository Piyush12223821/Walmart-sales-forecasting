from io import StringIO
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import FEATURES_FILE, STORES_FILE, TEST_FILE, TRAIN_FILE


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSVs robustly, including files with carriage-return line endings only."""
    df = pd.read_csv(path, na_values=["NA", ""])
    if len(df.columns) == 1 or len(df) <= 1:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        if "\r" in raw_text:
            normalized = raw_text.replace("\r", "\n")
            df = pd.read_csv(StringIO(normalized), na_values=["NA", ""])
    return df


def _clean_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    if "IsHoliday" in out.columns:
        out["IsHoliday"] = (
            out["IsHoliday"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )
    return out


def load_and_merge_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load source files and create merged train/test tables.

    Explicitly keeps the required file names from the project specification.
    """
    train = pd.read_csv(data_dir / TRAIN_FILE, na_values=["NA", ""])
    test = pd.read_csv(data_dir / TEST_FILE, na_values=["NA", ""])
    stores = _read_csv_robust(data_dir / STORES_FILE)
    features = pd.read_csv(data_dir / FEATURES_FILE, na_values=["NA", ""])

    train = _clean_common_columns(train)
    test = _clean_common_columns(test)
    stores = _clean_common_columns(stores)
    features = _clean_common_columns(features)

    train = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    train = train.merge(stores, on="Store", how="left")

    test = test.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    test = test.merge(stores, on="Store", how="left")

    return train, test, stores, features
