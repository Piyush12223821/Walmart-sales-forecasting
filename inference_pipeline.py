from pathlib import Path

import joblib
import pandas as pd

from src.config import DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from src.data_loader import load_and_merge_data
from src.feature_engineering import (
    add_calendar_features,
    add_lag_and_behavior_features_test,
    add_store_intelligence,
    fill_missing_features,
)
from src.targets import apply_auxiliary_targets


def _validate_and_order_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Inference schema mismatch: missing required feature columns: "
            + ", ".join(missing)
        )

    ordered = df.loc[:, feature_cols]
    if ordered.columns.tolist() != list(feature_cols):
        raise ValueError("Inference schema mismatch: feature column order does not match training artifacts")
    return ordered


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = joblib.load(MODELS_DIR / "forecast_artifacts.joblib")
    model = artifacts["best_model"]
    feature_cols = artifacts["feature_cols"]
    avg_sales_map = artifacts["average_sales_map"]
    size_thresholds = artifacts["size_thresholds"]
    type_encoding = artifacts["type_encoding"]
    history_df = artifacts["history_df"]
    target_artifacts = artifacts["target_artifacts"]

    train, test, stores, features = load_and_merge_data(DATA_DIR)

    train = add_calendar_features(train)
    test = add_calendar_features(test)

    test = add_lag_and_behavior_features_test(test, history_df)

    test, _, _, _ = add_store_intelligence(
        train_df=train,
        df=test,
        average_sales_map=avg_sales_map,
        size_thresholds=size_thresholds,
        type_encoding=type_encoding,
    )

    test = apply_auxiliary_targets(test, target_artifacts)
    test = fill_missing_features(test)

    feature_matrix = _validate_and_order_features(test, feature_cols)
    preds = model.predict(feature_matrix)

    submission = test[["Store", "Dept", "Date", "IsHoliday"]].copy()
    submission["Predicted_Weekly_Sales"] = preds
    submission["demand_trend"] = test["demand_trend"]
    submission["store_performance"] = test["store_performance"]
    submission["holiday_impact_score"] = test["holiday_impact_score"]

    submission.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

    print("Inference complete")
    print(f"Predictions saved: {OUTPUTS_DIR / 'predictions.csv'}")


if __name__ == "__main__":
    main()
