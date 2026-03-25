import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"

st.set_page_config(
    page_title="Retail Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

      :root {
        --app-bg: #f8fafc;
        --card-bg: #ffffff;
        --card-border: #e2e8f0;
        --title: #0f172a;
        --subtitle: #64748b;
        --muted: #94a3b8;
      }

      html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
      }

      html, body {
        color-scheme: light !important;
        background: var(--app-bg) !important;
        color: var(--title) !important;
      }

      [data-testid="stSidebar"] {
        display: none;
      }

      [data-testid="collapsedControl"] {
        display: none;
      }

      [data-testid="stAppViewContainer"], .stApp {
        background: var(--app-bg) !important;
        color: var(--title) !important;
      }

      [data-testid="stHeader"] {
        background: #ffffff;
        border-bottom: 1px solid var(--card-border);
      }

      [data-testid="stAppViewContainer"] p,
      [data-testid="stAppViewContainer"] label,
      [data-testid="stAppViewContainer"] span {
        color: var(--title);
      }

      .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
      }

      .hero {
        text-align: center;
        margin-bottom: 1.25rem;
      }

      .hero-title {
        color: var(--title);
        font-size: 32px;
        line-height: 1.2;
        letter-spacing: -0.01em;
        font-weight: 600;
        margin: 0 0 20px 0;
      }

      .hero-subtitle {
        color: var(--subtitle);
        font-size: 0.95rem;
        margin-top: 0;
      }

      .section-label {
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        color: var(--muted);
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        font-weight: 600;
      }

      .card {
        background: var(--card-bg);
        border-radius: 20px;
        border: 1px solid var(--card-border);
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.04);
        padding: 24px;
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }

      .card:hover {
        transform: translateY(-4px);
        box-shadow: 0px 16px 30px rgba(0, 0, 0, 0.08);
      }

      div[data-testid="stVerticalBlock"]:has(.card-hook) {
        background: var(--card-bg);
        border-radius: 20px;
        border: 1px solid var(--card-border);
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.04);
        padding: 24px;
        height: 100%;
      }

      div[data-testid="stVerticalBlock"]:has(.card-hook) .card-hook {
        display: none;
      }

      .card-title {
        color: var(--title);
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin-bottom: 0.2rem;
      }

      .card-subtitle {
        color: var(--subtitle);
        font-size: 0.86rem;
        margin-bottom: 1rem;
      }

      .metric-value {
        color: var(--title);
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.02em;
      }

      .muted {
        color: var(--subtitle);
        font-size: 0.86rem;
        margin-top: 0.35rem;
      }

      .pill {
        margin-top: 0.85rem;
        display: inline-block;
        border: 1px solid var(--card-border);
        background: #f9fbff;
        border-radius: 999px;
        color: #334155;
        padding: 0.3rem 0.65rem;
        font-size: 0.76rem;
      }

      .insights ul {
        margin: 0;
        padding-left: 1.1rem;
      }

      .insights li {
        color: #334155;
        margin-bottom: 0.45rem;
        font-size: 0.9rem;
      }

      .insight-list {
        margin: 0;
        padding-left: 1.1rem;
      }

      .insight-list li {
        color: #1f2937 !important;
        margin-bottom: 0.45rem;
        font-size: 0.9rem;
      }

      .holiday-metric {
        color: #0f172a !important;
      }

      .holiday-indicator {
        color: #334155 !important;
      }

      .spacer {
        height: 1rem;
      }

      div[data-testid="stForm"] {
        border: 0;
        padding: 0;
        margin: 0;
      }

      div[data-testid="stForm"] [data-baseweb="select"] > div,
      div[data-testid="stForm"] [data-baseweb="input"] > div,
      div[data-testid="stForm"] [data-baseweb="base-input"] {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
      }

      div[data-testid="stForm"] input,
      div[data-testid="stForm"] [data-baseweb="input"] input,
      div[data-testid="stForm"] [data-baseweb="select"] input {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
      }

      div[data-testid="stForm"] input::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
      }

      div[data-testid="stForm"] [data-testid="stWidgetLabel"] p,
      div[data-testid="stForm"] [data-testid="stWidgetLabel"] span,
      div[data-testid="stForm"] label {
        color: #475569 !important;
        opacity: 1 !important;
      }

      div[data-testid="stForm"] [data-testid="stToggle"] label p,
      div[data-testid="stForm"] [data-testid="stDateInput"] label p,
      div[data-testid="stForm"] [data-testid="stSelectbox"] label p {
        color: #475569 !important;
        opacity: 1 !important;
      }

      div[data-testid="stForm"] [data-baseweb="switch"] > div {
        border: 1px solid #cbd5e1;
      }

      div[data-testid="stForm"] button[kind="secondaryFormSubmit"] {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
      }

      div[data-testid="stForm"] button[kind="secondaryFormSubmit"]:hover {
        background: #f8fafc !important;
        border: 1px solid #94a3b8 !important;
      }

      [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        color: #111827 !important;
      }

      [data-testid="stDataFrame"] table {
        background-color: #ffffff !important;
      }

      [data-testid="stDataFrame"] th {
        background-color: #f9fafb !important;
        color: #111827 !important;
      }

      [data-testid="stDataFrame"] td {
        color: #111827 !important;
      }

      .model-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 8px;
      }

      .model-table th {
        text-align: left;
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0 10px 6px 10px;
      }

      .model-table td {
        background: #f9fbff;
        border-top: 1px solid var(--card-border);
        border-bottom: 1px solid var(--card-border);
        color: #1f2937;
        font-size: 0.86rem;
        padding: 8px 10px;
      }

      .model-table td:first-child {
        border-left: 1px solid var(--card-border);
        border-radius: 10px 0 0 10px;
      }

      .model-table td:last-child {
        border-right: 1px solid var(--card-border);
        border-radius: 0 10px 10px 0;
      }

      .best-row td {
        background: #eef6ff;
      }

      @media (max-width: 991px) {
        .block-container {
          padding-top: 1.2rem;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

pred_path = OUTPUTS_DIR / "predictions.csv"
comp_path = OUTPUTS_DIR / "model_comparison.csv"
insight_path = OUTPUTS_DIR / "business_insights.json"
feat_imp_path = OUTPUTS_DIR / "feature_importance.csv"

if not pred_path.exists() or not comp_path.exists() or not insight_path.exists():
    st.error("Required outputs are missing. Run train_pipeline.py and inference_pipeline.py first.")
    st.stop()

pred = pd.read_csv(pred_path)
pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce")
comp = pd.read_csv(comp_path)
comp_sorted = comp.copy().sort_values("RMSE", ascending=True).reset_index(drop=True)
model_used = comp_sorted.iloc[0]["Model"] if not comp_sorted.empty else "N/A"

if not comp_sorted.empty and pd.notna(comp_sorted.iloc[0]["RMSE"]):
  rmse_min = float(comp_sorted["RMSE"].min())
  rmse_max = float(comp_sorted["RMSE"].max())
  if rmse_max > rmse_min:
    confidence_pct = 100.0 * (1.0 - ((float(comp_sorted.iloc[0]["RMSE"]) - rmse_min) / (rmse_max - rmse_min)))
  else:
    confidence_pct = 100.0
  confidence_text = f"{confidence_pct:.1f}% (validation RMSE proxy)"
else:
  confidence_text = "Not available"

with open(insight_path, "r", encoding="utf-8") as f:
    insights = json.load(f)

feature_imp = (
    pd.read_csv(feat_imp_path)
    if feat_imp_path.exists()
    else pd.DataFrame(columns=["feature", "importance"])
)

stores = sorted(pred["Store"].dropna().astype(int).unique().tolist())
depts = sorted(pred["Dept"].dropna().astype(int).unique().tolist())

min_date = pred["Date"].dropna().min()
default_date = min_date.date() if pd.notna(min_date) else pd.Timestamp.today().date()

if "store_sel" not in st.session_state:
    st.session_state.store_sel = stores[0] if stores else 1
if "dept_sel" not in st.session_state:
    st.session_state.dept_sel = depts[0] if depts else 1
if "date_sel" not in st.session_state:
    st.session_state.date_sel = default_date
if "holiday_sel" not in st.session_state:
    st.session_state.holiday_sel = False

st.markdown(
    """
    <div class="hero">
      <p class="hero-title">Retail Intelligence</p>
      <p class="hero-subtitle">Retail forecasting and decision intelligence dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Resolve selected record from prediction output.
sel_store = st.session_state.store_sel
sel_dept = st.session_state.dept_sel
sel_date = st.session_state.date_sel
sel_holiday = st.session_state.holiday_sel

filtered = pred[(pred["Store"] == sel_store) & (pred["Dept"] == sel_dept)].copy()
if filtered.empty:
    selected_row = pred.iloc[0]
else:
    exact = filtered[
        (filtered["Date"].dt.date == sel_date)
        & (filtered["IsHoliday"].astype(int) == int(sel_holiday))
    ]
    if exact.empty:
        nearest_idx = (filtered["Date"] - pd.Timestamp(sel_date)).abs().idxmin()
        selected_row = filtered.loc[nearest_idx]
    else:
        selected_row = exact.iloc[0]

pred_sales = float(selected_row["Predicted_Weekly_Sales"])
selected_date = pd.to_datetime(selected_row["Date"], errors="coerce")
selected_date_text = selected_date.strftime("%b %d, %Y") if pd.notna(selected_date) else "N/A"


# Shared metrics for cards.
subset = pred[(pred["Store"] == sel_store) & (pred["Dept"] == sel_dept)].copy()
holiday_subset = subset[subset["IsHoliday"].astype(int) == 1]
non_holiday_subset = subset[subset["IsHoliday"].astype(int) == 0]

if not holiday_subset.empty and not non_holiday_subset.empty:
    holiday_mean = float(holiday_subset["Predicted_Weekly_Sales"].mean())
    non_holiday_mean = float(non_holiday_subset["Predicted_Weekly_Sales"].mean())
    holiday_delta_pct = ((holiday_mean - non_holiday_mean) / max(abs(non_holiday_mean), 1e-9)) * 100
else:
    holiday_delta_pct = float(selected_row.get("holiday_impact_score", 0.0))

trend_label = str(selected_row.get("demand_trend", "Stable"))
if trend_label == "Increasing":
    trend_indicator = "^ Increasing"
elif trend_label == "Decreasing":
    trend_indicator = "v Decreasing"
else:
    trend_indicator = "- Stable"

store_perf = str(selected_row.get("store_performance", "Medium"))
perf_score = {"High": 90, "Medium": 62, "Low": 36}.get(store_perf, 62)

store_rank_map = pred.groupby("Store")["Predicted_Weekly_Sales"].mean().rank(ascending=False, method="dense")
store_rank = int(store_rank_map.get(sel_store, 0))
total_stores = int(store_rank_map.shape[0])


st.markdown('<div class="section-label">Section 1</div>', unsafe_allow_html=True)
row_1_top_left, row_1_top_right = st.columns(2, gap="large")

with row_1_top_left:
    st.markdown('<div class="card-hook"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Sales Forecast Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Weekly sales prediction form and forecast output.</div>', unsafe_allow_html=True)

    with st.form("predict_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            store_sel = st.selectbox(
                "Store",
                stores,
                index=stores.index(st.session_state.store_sel) if st.session_state.store_sel in stores else 0,
            )
        with c2:
            dept_sel = st.selectbox(
                "Dept",
                depts,
                index=depts.index(st.session_state.dept_sel) if st.session_state.dept_sel in depts else 0,
            )

        c3, c4 = st.columns(2)
        with c3:
            holiday_sel = st.toggle("Holiday Week", value=st.session_state.holiday_sel)
        with c4:
            date_sel = st.date_input("Date", value=st.session_state.date_sel)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        st.session_state.store_sel = store_sel
        st.session_state.dept_sel = dept_sel
        st.session_state.holiday_sel = holiday_sel
        st.session_state.date_sel = date_sel
        st.rerun()

    st.markdown(f'<div class="metric-value">${pred_sales:,.0f}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="muted">Store {sel_store} | Dept {sel_dept} | Date {selected_date_text} | Holiday {"Yes" if int(selected_row["IsHoliday"]) == 1 else "No"}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="muted">Model used: {model_used}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="muted">Prediction confidence: {confidence_text}</div>', unsafe_allow_html=True)

with row_1_top_right:
    st.markdown('<div class="card-hook"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Validation summary with best model highlighted.</div>', unsafe_allow_html=True)

    comp_view = comp_sorted.copy()
    rmse_max = float(comp_view["RMSE"].max()) if not comp_view.empty else 1.0
    rmse_max = rmse_max if rmse_max > 0 else 1.0
    comp_view["Accuracy"] = ((1 - (comp_view["RMSE"] / rmse_max)) * 100).clip(lower=0, upper=100)

    best_model = comp_view.iloc[0]["Model"] if not comp_view.empty else "N/A"

    table_html = [
        '<table class="model-table">',
        '<thead><tr><th>Model</th><th>RMSE</th><th>Accuracy</th></tr></thead>',
        '<tbody>',
    ]

    for idx, row in comp_view.head(6).iterrows():
        row_class = ' class="best-row"' if idx == 0 else ""
        table_html.append(
            f'<tr{row_class}><td>{row["Model"]}</td><td>{float(row["RMSE"]):,.2f}</td><td>{float(row["Accuracy"]):,.1f}%</td></tr>'
        )

    table_html.extend(['</tbody>', '</table>'])
    st.markdown("".join(table_html), unsafe_allow_html=True)
    st.markdown(f'<span class="pill">Best model: {best_model}</span>', unsafe_allow_html=True)

row_1_bottom_left, row_1_bottom_right = st.columns(2, gap="large")

with row_1_bottom_left:
    st.markdown('<div class="card-hook"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Holiday Impact</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Estimated sales lift during holiday periods.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value holiday-metric">{holiday_delta_pct:+.2f}%</div>', unsafe_allow_html=True)
    indicator = "Holiday sales uplift" if holiday_delta_pct >= 0 else "Holiday sales decline"
    st.markdown(f'<div class="muted holiday-indicator">Indicator: {indicator}</div>', unsafe_allow_html=True)

with row_1_bottom_right:
    st.markdown('<div class="card-hook"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Demand Trend</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Trend class generated from model outputs.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{trend_indicator}</div>', unsafe_allow_html=True)

    if len(subset) >= 8:
        start_avg = float(subset.sort_values("Date")["Predicted_Weekly_Sales"].head(4).mean())
        end_avg = float(subset.sort_values("Date")["Predicted_Weekly_Sales"].tail(4).mean())
        trend_pct = ((end_avg - start_avg) / max(abs(start_avg), 1e-9)) * 100
    else:
        trend_pct = 0.0

    st.markdown(f'<div class="muted">Recent change: {trend_pct:+.2f}%</div>', unsafe_allow_html=True)


st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Section 2</div>', unsafe_allow_html=True)
row_2_top_left, row_2_top_right = st.columns(2, gap="large")

with row_2_top_left:
    st.markdown('<div class="card-hook"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Store Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Current store performance class and rank.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{store_perf}</div>', unsafe_allow_html=True)
    st.progress(perf_score / 100.0)
    st.markdown(f'<div class="muted">Store rank: {store_rank} / {total_stores}</div>', unsafe_allow_html=True)

with row_2_top_right:
    st.markdown('<div class="card-hook"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Top features from trained model artifacts.</div>', unsafe_allow_html=True)

    if not feature_imp.empty:
      label_map = {
        "num_lag_1": "Last Week Sales",
        "num_lag_4": "4 Week Avg Sales",
        "num_lag_52": "Last Year Sales",
        "num_rolling_mean_4": "Rolling Average",
        "num_rolling_std_4": "Sales Volatility",
        "num_rolling_mean": "Rolling Average",
        "num_rolling_std": "Sales Volatility",
        "num_week": "Week of Year",
        "num_Dept": "Department",
        "num_dept": "Department",
        "num_holiday_effect_last_year": "Holiday Impact",
        "num_holiday_effect": "Holiday Impact",
      }
      importance_df = feature_imp.sort_values("importance", ascending=False).head(8).copy()
      importance_df["feature"] = (
        importance_df["feature"]
        .astype(str)
        .str.replace("__", "_", regex=False)
        .replace(label_map)
      )
      importance_df = importance_df.rename(columns={"feature": "Feature", "importance": "Importance"})
      st.dataframe(importance_df, use_container_width=True, hide_index=True)
      st.markdown(
            """
            <style>
            [data-testid="stDataFrame"] {
                background-color: white !important;
                color: black !important;
            }

            [data-testid="stDataFrame"] table {
                background-color: white !important;
            }

            [data-testid="stDataFrame"] th {
                background-color: #f9fafb !important;
                color: #111827 !important;
            }

            [data-testid="stDataFrame"] td {
                color: #111827 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Feature importance file is unavailable.")

row_2_bottom_left, row_2_bottom_right = st.columns(2, gap="large")

with row_2_bottom_left:
    st.markdown('<div class="card-hook"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Business Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Model-driven summary from insight pipeline.</div>', unsafe_allow_html=True)

    best_store_type = insights.get("best_performing_store_type", "Unavailable")
    holiday_impact = insights.get("holiday_impact_percent", "N/A")
    temp_corr = insights.get("temperature_vs_sales_correlation", "N/A")

    st.markdown(
        f"""
        <ul class="insight-list">
          <li>Best performing store type: {best_store_type}</li>
          <li>Holiday impact: {holiday_impact}%</li>
          <li>Temperature correlation: {temp_corr}</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

with row_2_bottom_right:
    st.markdown('<div class="card-hook"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Monthly Trend</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Average predicted sales across months.</div>', unsafe_allow_html=True)

    monthly = insights.get("monthly_trend_average_sales", {})
    if monthly:
        monthly_df = pd.DataFrame(
            {"Month": list(monthly.keys()), "Average Sales": list(monthly.values())}
        )
        monthly_df["Month"] = monthly_df["Month"].astype(int)
        monthly_df = monthly_df.sort_values("Month")

        st.line_chart(monthly_df.set_index("Month"), use_container_width=True)

        peak = monthly_df.loc[monthly_df["Average Sales"].idxmax()]
        st.markdown(
            f'<span class="pill">Peak month: {int(peak["Month"])} | Avg sales: ${float(peak["Average Sales"]):,.0f}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Monthly trend data is unavailable.")
