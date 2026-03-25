import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from src.data_loader import load_and_merge_data
from src.feature_engineering import (
    add_calendar_features,
    add_lag_and_behavior_features_train,
    add_store_intelligence,
    build_group_history,
    fill_missing_features,
)
from src.insights import generate_business_insights
from src.modeling import evaluate_arima_baseline, time_split, train_regression_models
from src.targets import apply_auxiliary_targets, build_target_artifacts


def _extract_feature_importance(trained_pipeline, feature_cols: list[str]) -> pd.DataFrame:
    try:
        preprocessor = trained_pipeline.named_steps["preprocessor"]
        model = trained_pipeline.named_steps["model"]
        transformed_names = preprocessor.get_feature_names_out(feature_cols)

        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.ravel(model.coef_)
        else:
            return pd.DataFrame(columns=["feature", "importance"])

        imp_df = pd.DataFrame({"feature": transformed_names, "importance": values})
        imp_df["importance"] = imp_df["importance"].abs()
        return imp_df.sort_values("importance", ascending=False).head(20)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train, test, stores, features = load_and_merge_data(DATA_DIR)

    train = add_calendar_features(train)
    test = add_calendar_features(test)

    train = add_lag_and_behavior_features_train(train)

    history_df = build_group_history(train)

    train, avg_sales_map, size_thresholds, type_encoding = add_store_intelligence(
        train_df=train,
        df=train,
    )

    train = fill_missing_features(train)

    feature_cols = [
        "Store",
        "Dept",
        "IsHoliday",
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment",
        "Type",
        "Size",
        "year",
        "month",
        "week",
        "dayofweek",
        "quarter",
        "is_month_start",
        "is_month_end",
        "lag_1",
        "lag_4",
        "lag_52",
        "rolling_mean_4",
        "rolling_std_4",
        "sales_growth_rate",
        "holiday_effect_last_year",
        "store_size_category",
        "store_type_encoding",
        "average_sales_per_store",
    ]

    feature_cols = [c for c in feature_cols if c in train.columns]

    model_train_df, model_valid_df = time_split(train, date_col="Date", valid_frac=0.2)

    # Build auxiliary target mappings strictly from training split only.
    target_artifacts = build_target_artifacts(model_train_df)
    model_train_df = apply_auxiliary_targets(model_train_df, target_artifacts)
    model_valid_df = apply_auxiliary_targets(model_valid_df, target_artifacts)

    comparison_df, fitted_models, valid_preds = train_regression_models(
        model_train_df,
        model_valid_df,
        feature_cols,
        target_col="Weekly_Sales",
    )

    arima_row = evaluate_arima_baseline(train)
    comparison_df = pd.concat([comparison_df, pd.DataFrame([arima_row])], ignore_index=True)
    comparison_df = comparison_df.sort_values("RMSE", na_position="last").reset_index(drop=True)

    best_row = comparison_df[comparison_df["Model"] != "ARIMA"].sort_values("RMSE").iloc[0]
    best_model_name = best_row["Model"]
    best_model = fitted_models[best_model_name]

    feature_importance_df = _extract_feature_importance(best_model, feature_cols)

    insights = generate_business_insights(train)

    comparison_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    valid_with_preds = model_valid_df[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
    for col in valid_preds.columns:
        valid_with_preds[col] = valid_preds[col]
    valid_with_preds.to_csv(OUTPUTS_DIR / "validation_predictions.csv", index=False)

    feature_importance_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    with open(OUTPUTS_DIR / "business_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)

    artifacts = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "feature_cols": feature_cols,
        "average_sales_map": avg_sales_map,
        "size_thresholds": size_thresholds,
        "type_encoding": type_encoding,
        "history_df": history_df,
        "target_artifacts": target_artifacts,
    }
    joblib.dump(artifacts, MODELS_DIR / "forecast_artifacts.joblib")

    print("Training complete")
    print(f"Best model: {best_model_name}")
    print(f"Comparison saved: {OUTPUTS_DIR / 'model_comparison.csv'}")
    print(f"Artifacts saved: {MODELS_DIR / 'forecast_artifacts.joblib'}")


if __name__ == "__main__":
    main()
