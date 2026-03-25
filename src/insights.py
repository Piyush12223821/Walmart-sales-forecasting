from typing import Dict

import numpy as np
import pandas as pd


def generate_business_insights(df: pd.DataFrame) -> Dict[str, object]:
    out = {}

    if "Type" in df.columns and df["Type"].notna().any():
        type_perf = df.groupby("Type", observed=True)["Weekly_Sales"].mean().sort_values(ascending=False)
        out["best_performing_store_type"] = type_perf.index[0]
        out["store_type_average_sales"] = type_perf.round(2).to_dict()
    else:
        out["best_performing_store_type"] = "Unavailable"
        out["store_type_average_sales"] = {}

    holiday_means = df.groupby("IsHoliday", observed=True)["Weekly_Sales"].mean()
    non_holiday = float(holiday_means.get(0, np.nan))
    holiday = float(holiday_means.get(1, np.nan))
    if pd.notna(non_holiday) and abs(non_holiday) > 1e-9 and pd.notna(holiday):
        holiday_impact_pct = (holiday - non_holiday) / non_holiday * 100
    else:
        holiday_impact_pct = np.nan
    out["holiday_impact_percent"] = round(float(holiday_impact_pct), 4) if pd.notna(holiday_impact_pct) else None

    if "Temperature" in df.columns:
        corr = df[["Temperature", "Weekly_Sales"]].corr().iloc[0, 1]
        out["temperature_vs_sales_correlation"] = round(float(corr), 4) if pd.notna(corr) else None
    else:
        out["temperature_vs_sales_correlation"] = None

    monthly = (
        df.assign(month=df["Date"].dt.month)
        .groupby("month", observed=True)["Weekly_Sales"]
        .mean()
        .round(2)
    )
    out["monthly_trend_average_sales"] = {int(k): float(v) for k, v in monthly.items()}

    return out
