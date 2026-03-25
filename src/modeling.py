from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.config import RANDOM_STATE


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-9
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def time_split(df: pd.DataFrame, date_col: str = "Date", valid_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(df[date_col].dropna().unique()))
    split_idx = max(int(len(unique_dates) * (1 - valid_frac)), 1)
    cutoff_date = unique_dates[min(split_idx, len(unique_dates) - 1)]

    train_df = df[df[date_col] < cutoff_date].copy()
    valid_df = df[df[date_col] >= cutoff_date].copy()

    if train_df.empty or valid_df.empty:
        train_df, valid_df = train_test_split(df, test_size=valid_frac, random_state=RANDOM_STATE)
    return train_df, valid_df


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def train_regression_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "Weekly_Sales",
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], pd.DataFrame]:
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col].values

    preprocessor = _build_preprocessor(X_train)

    model_specs = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=20),
        "Random Forest": RandomForestRegressor(
            n_estimators=80,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=18,
        ),
        "KNN": KNeighborsRegressor(n_neighbors=8, weights="distance"),
    }

    try:
        from xgboost import XGBRegressor

        model_specs["XGBoost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except Exception:
        pass

    fitted_models: Dict[str, Pipeline] = {}
    rows = []
    valid_predictions = pd.DataFrame(index=valid_df.index)

    for model_name, estimator in model_specs.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_valid)

        mae = mean_absolute_error(y_valid, preds)
        rmse = float(np.sqrt(mean_squared_error(y_valid, preds)))
        mape = _mape(y_valid, preds)

        rows.append(
            {
                "Model": model_name,
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
                "MAPE": round(float(mape), 4),
            }
        )

        valid_predictions[f"pred_{model_name}"] = preds
        fitted_models[model_name] = pipeline

    comparison_df = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    return comparison_df, fitted_models, valid_predictions


def evaluate_arima_baseline(df: pd.DataFrame) -> Dict[str, float]:
    """ARIMA baseline is evaluated on aggregated total sales by week."""
    aggregated = (
        df.groupby("Date", observed=True)["Weekly_Sales"].sum().sort_index()
    )
    split_idx = max(int(len(aggregated) * 0.8), 1)
    train_series = aggregated.iloc[:split_idx]
    valid_series = aggregated.iloc[split_idx:]

    if len(valid_series) == 0:
        return {"Model": "ARIMA", "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(train_series, order=(2, 1, 2))
        result = model.fit()
        forecast = result.forecast(steps=len(valid_series))

        y_true = valid_series.values
        y_pred = forecast.values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = _mape(y_true, y_pred)
    except Exception:
        mae, rmse, mape = np.nan, np.nan, np.nan

    return {
        "Model": "ARIMA",
        "MAE": round(float(mae), 4) if not np.isnan(mae) else np.nan,
        "RMSE": round(float(rmse), 4) if not np.isnan(rmse) else np.nan,
        "MAPE": round(float(mape), 4) if not np.isnan(mape) else np.nan,
    }
