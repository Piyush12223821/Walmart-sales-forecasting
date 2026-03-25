# An Intelligent Multi-Model Sales Forecasting and Holiday Impact Analysis System Using Machine Learning

## Project Overview
This final semester project builds a complete end-to-end forecasting system on the Walmart sales dataset. It includes:

- Multi-model weekly sales forecasting (Linear Regression, Decision Tree, Random Forest, XGBoost, KNN, ARIMA baseline)
- Holiday impact analysis
- Store performance banding
- Demand trend tagging
- Business insights generation
- Streamlit dashboard with white Bento-grid UI

## Dataset
Place all required files in the data folder:

- `archive (4)/train - Walmart Sales Forecast.csv`
- `archive (4)/test - Walmart Sales Forecast.csv`
- `archive (4)/stores - Walmart Sales Forecast.csv`
- `archive (4)/features - Walmart Sales Forecast.csv`

The loader handles carriage-return-only files (notably `stores`) safely.

## Modular Structure

- `src/config.py`: paths and constants
- `src/data_loader.py`: robust loading and merges
- `src/feature_engineering.py`: calendar, lag, rolling, behavioral, and store-intelligence features
- `src/targets.py`: multi-target derivation (holiday impact, performance bands, demand trend)
- `src/modeling.py`: training and evaluation utilities
- `src/insights.py`: business insight generation
- `train_pipeline.py`: training workflow and artifact creation
- `inference_pipeline.py`: inference workflow on test set
- `app.py`: Streamlit dashboard

## Feature Engineering Included

### Calendar Features
- `year`, `month`, `week`, `dayofweek`
- `quarter`
- `is_month_start`, `is_month_end`

### Grouped Lag and Rolling
- `lag_1`, `lag_4`, `lag_52`
- `rolling_mean_4`, `rolling_std_4`

### Behavioral Features
- `sales_growth_rate` (lag-based, leakage-safe)
- `holiday_effect_last_year`

### Store Intelligence
- `store_size_category`
- `store_type_encoding`
- `average_sales_per_store`

## Multi Targets

- Target 1: `Weekly_Sales` prediction (regression)
- Target 2: `holiday_impact_score` (% holiday lift vs non-holiday baseline)
- Target 3: `store_performance` (High/Medium/Low by store quantiles)
- Target 4: `demand_trend` (Increasing/Stable/Decreasing)

## Training + Inference Pipeline

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train models and generate artifacts
```bash
python train_pipeline.py
```

Outputs generated:
- `models/forecast_artifacts.joblib`
- `outputs/model_comparison.csv`
- `outputs/validation_predictions.csv`
- `outputs/feature_importance.csv`
- `outputs/business_insights.json`

### 3) Run inference on test set
```bash
python inference_pipeline.py
```

Output:
- `outputs/predictions.csv`

### 4) Launch dashboard
```bash
streamlit run app.py
```

## Evaluation Metrics
Model comparison table includes:

- MAE
- RMSE
- MAPE

ARIMA is included as an aggregated weekly-sales baseline for time-series comparison.

## Leakage Controls

- Lag/rolling features for training use shifted values only
- Sales growth uses lagged values only
- Time-based split is used for validation
- Test inference uses historical train-only group history
- Weekly_Sales is never used in inference feature creation

## Dashboard UI (White Bento Grid)

Rows in app layout:
1. Sales Forecast | Model Comparison
2. Holiday Impact | Store Performance
3. Demand Trend | Feature Importance
4. Business Insights | Monthly Trend

Style:
- White cards
- Rounded corners
- Soft shadows
- Minimal design
- Hover animation
- Responsive for desktop and mobile

## Notes
- Uses only uploaded dataset files.
- No placeholder values are injected into forecasting outputs.
- End-to-end runnable workflow from training to dashboard.
