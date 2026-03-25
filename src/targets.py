from typing import Dict, Tuple

import numpy as np
import pandas as pd


def build_target_artifacts(reference_df: pd.DataFrame) -> Dict:
    ref = reference_df.copy()

    holiday_stats = (
        ref.groupby(["Store", "Dept", "IsHoliday"], observed=True)["Weekly_Sales"]
        .mean()
        .unstack(fill_value=np.nan)
        .rename(columns={0: "non_holiday_sales", 1: "holiday_sales"})
        .reset_index()
    )
    holiday_stats["holiday_impact_score"] = np.where(
        holiday_stats["non_holiday_sales"].abs() > 1e-9,
        (holiday_stats["holiday_sales"] - holiday_stats["non_holiday_sales"]) / holiday_stats["non_holiday_sales"] * 100,
        0.0,
    )
    holiday_impact_map = {
        (row.Store, row.Dept): float(row.holiday_impact_score)
        for row in holiday_stats.itertuples(index=False)
    }

    store_avg = ref.groupby("Store", observed=True)["Weekly_Sales"].mean()
    q1 = store_avg.quantile(0.33)
    q2 = store_avg.quantile(0.67)

    def perf_bucket(v: float) -> str:
        if v <= q1:
            return "Low"
        if v <= q2:
            return "Medium"
        return "High"

    store_perf_map = {int(store): perf_bucket(avg) for store, avg in store_avg.items()}

    trend_map = {}
    trend_threshold = ref["Weekly_Sales"].std() * 0.02
    for key, group in ref.sort_values("Date").groupby(["Store", "Dept"], observed=True):
        recent = group["Weekly_Sales"].tail(8).values
        if len(recent) < 3:
            trend = "Stable"
        else:
            x = np.arange(len(recent), dtype=float)
            slope = np.polyfit(x, recent, 1)[0]
            if slope > trend_threshold:
                trend = "Increasing"
            elif slope < -trend_threshold:
                trend = "Decreasing"
            else:
                trend = "Stable"
        trend_map[key] = trend

    artifacts = {
        "holiday_impact_map": holiday_impact_map,
        "store_performance_map": store_perf_map,
        "demand_trend_map": trend_map,
        "store_perf_thresholds": (float(q1), float(q2)),
    }
    return artifacts


def build_multi_targets(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    artifacts = build_target_artifacts(train_df)
    out = apply_auxiliary_targets(train_df, artifacts)
    return out, artifacts


def apply_auxiliary_targets(test_df: pd.DataFrame, artifacts: Dict) -> pd.DataFrame:
    out = test_df.copy()
    holiday_impact_map = artifacts.get("holiday_impact_map", {})
    store_perf_map = artifacts.get("store_performance_map", {})
    trend_map = artifacts.get("demand_trend_map", {})

    out["holiday_impact_score"] = out.apply(
        lambda r: holiday_impact_map.get((r["Store"], r["Dept"]), 0.0), axis=1
    )
    out["store_performance"] = out["Store"].map(store_perf_map).fillna("Medium")
    out["demand_trend"] = out.apply(
        lambda r: trend_map.get((r["Store"], r["Dept"]), "Stable"), axis=1
    )
    return out
