"""
Streamlit Web Application: Part 1 HR Salary Prediction Dashboard
================================================================
Optional Streamlit UI for users who prefer the Streamlit framework.

Run via:
  streamlit run src/salary_streamlit_app.py
"""

import os
import json
import pandas as pd
import numpy as np
try:
    import streamlit as st
    import matplotlib.pyplot as plt
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def main():
    if not HAS_STREAMLIT:
        print("Streamlit is not installed in this environment. Run with built-in UI: python src/salary_ui.py")
        return

    st.set_page_config(
        page_title="HR Salary Prediction Audit",
        page_icon="💼",
        layout="wide"
    )

    st.title("💼 Part 1: HR Salary Predictive Analytics Audit")
    st.markdown("Supervised Machine Learning with Linear Regression ($Y = MX + C$) & Data Hygiene Pipeline")

    audit_path = os.path.join(REPORTS_DIR, "salary_prediction_audit.json")
    clean_csv_path = os.path.join(REPORTS_DIR, "cleaned_hr_dataset.csv")

    if not os.path.exists(audit_path) or not os.path.exists(clean_csv_path):
        st.warning("Reports not found. Generating pipeline data...")
        from salary_prediction_project import run_salary_pipeline
        run_salary_pipeline(output_dir=REPORTS_DIR)

    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    df_clean = pd.read_csv(clean_csv_path)

    # Top KPI Metrics
    slope_m = audit['model_results']['slope_m']
    intercept_c = audit['model_results']['intercept_c']
    r2_pct = audit['model_results']['metrics']['r2_percentage']
    mae = audit['model_results']['metrics']['mae']
    rmse = audit['model_results']['metrics']['rmse']
    pred_15 = audit['candidate_15_yr_prediction']['predicted_annual_salary']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Regression Equation", f"Y = {slope_m:,.0f}X + {intercept_c:,.0f}", f"${slope_m:,.0f}/yr")
    col2.metric("Accuracy (R² Score)", f"{r2_pct}%", "≥95% Budget Grade" if r2_pct >= 95 else "Below Threshold")
    col3.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}", f"RMSE: ${rmse:,.2f}")
    col4.metric("15-Yr Candidate Target", f"${pred_15:,.2f}", f"${pred_15/12:,.2f}/mo")

    st.markdown("---")

    # Interactive Candidate Prediction Sidebar / Column
    st.subheader("🎯 Interactive Candidate Salary Predictor")
    left_col, right_col = st.columns([1, 1])

    with left_col:
        exp_input = st.slider("Candidate Experience (Years)", min_value=0.0, max_value=30.0, value=15.0, step=0.5)
        live_pred = (slope_m * exp_input) + intercept_c
        margin = 1.96 * rmse

        st.info(f"""
        **Calculation Details:**
        - **Formula:** $\\text{{Salary}} = ({slope_m:,.2f} \\times {exp_input:.1f}) + {intercept_c:,.2f}$
        - **Predicted Annual Salary:** **${live_pred:,.2f}**
        - **Predicted Monthly Salary:** **${live_pred / 12:,.2f}**
        - **95% Budget Confidence Range:** **${max(0, live_pred - margin):,.2f}** – **${live_pred + margin:,.2f}**
        """)

    with right_col:
        st.markdown("#### 🧹 Data Hygiene & Cleaning Funnel")
        clean_meta = audit['cleaning_audit']
        st.write(f"- **Initial Raw Records:** `{clean_meta['initial_rows']}`")
        st.write(f"- **Duplicates Removed:** `{clean_meta['duplicates_removed']}`")
        st.write(f"- **Null Values Handled:** `{clean_meta['imputation_method']}` (Filled with column medians)")
        st.write(f"- **Outliers Eliminated:** `{clean_meta['outliers_removed']}` records removed outside IQR bounds")
        st.write(f"- **Certified Clean Records:** `{clean_meta['final_rows']}` ({clean_meta['retention_pct']}% retention)")

    st.markdown("---")

    # Chart Section
    st.subheader("📈 Regression Trendline & Visual Analytics")
    reg_img = os.path.join(REPORTS_DIR, "salary_regression_plot.png")
    hygiene_img = os.path.join(REPORTS_DIR, "data_cleaning_audit.png")
    residuals_img = os.path.join(REPORTS_DIR, "model_residuals_plot.png")

    tab1, tab2, tab3, tab4 = st.tabs(["Regression Fit (Y=MX+C)", "Data Hygiene Audit", "Error Residuals", "Clean Data Table"])
    with tab1:
        if os.path.exists(reg_img):
            st.image(reg_img, use_container_width=True)
    with tab2:
        if os.path.exists(hygiene_img):
            st.image(hygiene_img, use_container_width=True)
    with tab3:
        if os.path.exists(residuals_img):
            st.image(residuals_img, use_container_width=True)
    with tab4:
        st.dataframe(df_clean, use_container_width=True)

if __name__ == '__main__':
    main()
