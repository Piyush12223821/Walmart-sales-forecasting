from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    iso_week = out["Date"].dt.isocalendar()
    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    out["week"] = iso_week.week.astype(int)
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["quarter"] = out["Date"].dt.quarter
    out["is_month_start"] = out["Date"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["Date"].dt.is_month_end.astype(int)
    return out


def add_store_intelligence(
    train_df: pd.DataFrame,
    df: pd.DataFrame,
    average_sales_map: Dict[int, float] | None = None,
    size_thresholds: Tuple[float, float] | None = None,
    type_encoding: Dict[str, int] | None = None,
) -> Tuple[pd.DataFrame, Dict[int, float], Tuple[float, float], Dict[str, int]]:
    out = df.copy()

    if average_sales_map is None:
        average_sales_map = train_df.groupby("Store")["Weekly_Sales"].mean().to_dict()
    out["average_sales_per_store"] = out["Store"].map(average_sales_map)

    if "Size" in out.columns and out["Size"].notna().any():
        if size_thresholds is None:
            q1 = train_df["Size"].quantile(0.33)
            q2 = train_df["Size"].quantile(0.67)
            size_thresholds = (q1, q2)

        def size_bucket(size_value: float) -> str:
            if pd.isna(size_value):
                return "Unknown"
            if size_value <= size_thresholds[0]:
                return "Small"
            if size_value <= size_thresholds[1]:
                return "Medium"
            return "Large"

        out["store_size_category"] = out["Size"].apply(size_bucket)
    else:
        size_thresholds = size_thresholds or (0.0, 0.0)
        out["store_size_category"] = "Unknown"

    if "Type" in out.columns and out["Type"].notna().any():
        if type_encoding is None:
            known_types = sorted(train_df["Type"].dropna().astype(str).unique().tolist())
            type_encoding = {label: idx for idx, label in enumerate(known_types)}
        out["store_type_encoding"] = (
            out["Type"].astype(str).map(type_encoding).fillna(-1).astype(int)
        )
    else:
        type_encoding = type_encoding or {}
        out["store_type_encoding"] = -1

    return out, average_sales_map, size_thresholds, type_encoding


def build_group_history(train_df: pd.DataFrame) -> pd.DataFrame:
    sorted_train = train_df.sort_values(["Store", "Dept", "Date"]).copy()
    grp = sorted_train.groupby(["Store", "Dept"], observed=True)["Weekly_Sales"]

    history = (
        sorted_train.groupby(["Store", "Dept"], observed=True)
        .tail(60)
        .groupby(["Store", "Dept"], observed=True)["Weekly_Sales"]
        .apply(list)
        .reset_index(name="history")
    )

    baseline = grp.mean().reset_index(name="group_mean_sales")
    history = history.merge(baseline, on=["Store", "Dept"], how="left")
    return history


def _get_lag(values: List[float], lag: int, default_value: float) -> float:
    if len(values) >= lag:
        return float(values[-lag])
    return float(default_value)


def _rolling(values: List[float], window: int, default_mean: float) -> Tuple[float, float]:
    if not values:
        return float(default_mean), 0.0
    tail = values[-window:] if len(values) >= window else values
    arr = np.array(tail, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)


def add_lag_and_behavior_features_train(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["Store", "Dept", "Date"]).copy()
    g = out.groupby(["Store", "Dept"], observed=True)

    out["lag_1"] = g["Weekly_Sales"].shift(1)
    out["lag_4"] = g["Weekly_Sales"].shift(4)
    out["lag_52"] = g["Weekly_Sales"].shift(52)

    out["rolling_mean_4"] = g["Weekly_Sales"].transform(
        lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
    )
    out["rolling_std_4"] = g["Weekly_Sales"].transform(
        lambda s: s.shift(1).rolling(window=4, min_periods=2).std()
    )

    # Backward-compatible aliases required by project checklist.
    out["rolling_mean"] = out["rolling_mean_4"]
    out["rolling_std"] = out["rolling_std_4"]

    # Sales growth is based on lagged sales only to prevent target leakage.
    out["sales_growth_rate"] = np.where(
        out["lag_4"].abs() > 1e-9,
        (out["lag_1"] - out["lag_4"]) / out["lag_4"].abs(),
        0.0,
    )

    out["holiday_effect_last_year"] = out["lag_52"] * out["IsHoliday"].astype(int)

    return out


def add_lag_and_behavior_features_test(test_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    out = test_df.sort_values(["Store", "Dept", "Date"]).copy()
    hist_map = {
        (row.Store, row.Dept): (row.history, row.group_mean_sales)
        for row in history_df.itertuples(index=False)
    }

    lag_1_vals = []
    lag_4_vals = []
    lag_52_vals = []
    roll_mean_vals = []
    roll_std_vals = []

    for row in out.itertuples(index=False):
        key = (row.Store, row.Dept)
        if key in hist_map:
            hist, mean_sales = hist_map[key]
        else:
            hist, mean_sales = [], 0.0

        lag_1 = _get_lag(hist, 1, mean_sales)
        lag_4 = _get_lag(hist, 4, mean_sales)
        lag_52 = _get_lag(hist, 52, mean_sales)
        rolling_mean_4, rolling_std_4 = _rolling(hist, 4, mean_sales)

        lag_1_vals.append(lag_1)
        lag_4_vals.append(lag_4)
        lag_52_vals.append(lag_52)
        roll_mean_vals.append(rolling_mean_4)
        roll_std_vals.append(rolling_std_4)

    out["lag_1"] = lag_1_vals
    out["lag_4"] = lag_4_vals
    out["lag_52"] = lag_52_vals
    out["rolling_mean_4"] = roll_mean_vals
    out["rolling_std_4"] = roll_std_vals

    # Backward-compatible aliases required by project checklist.
    out["rolling_mean"] = out["rolling_mean_4"]
    out["rolling_std"] = out["rolling_std_4"]

    out["sales_growth_rate"] = np.where(
        out["lag_4"].abs() > 1e-9,
        (out["lag_1"] - out["lag_4"]) / out["lag_4"].abs(),
        0.0,
    )
    out["holiday_effect_last_year"] = out["lag_52"] * out["IsHoliday"].astype(int)
    return out


def fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        out[col] = out[col].fillna(out[col].median())

    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].fillna("Unknown")
    return out
