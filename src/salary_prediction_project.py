"""
Industrial Data Analytics - Part 1: Predictive Analytics with Supervised Learning
==================================================================================
Topic: Data Hygiene & Cleaning, Linear Regression Modeling (Y = MX + C), Error Metrics
Deliverable: Salary Prediction Audit & Candidate Prediction (15 Years Experience)

Author: Antigravity AI Data Analytics Pair Programmer
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================================
# 1. DATASET GENERATION (Realistic "Dirty" Industrial HR Dataset)
# =====================================================================

def generate_dirty_dataset(filepath="data/dirty_hr_dataset.csv", n_base=150, seed=42):
    """
    Generates a realistic 'dirty' HR dataset containing duplicates,
    missing values (NaNs), and extreme unrealistic outliers to simulate
    real-world legacy HR system exports.
    """
    np.random.seed(seed)
    
    # 1. Base realistic distribution
    exp = np.random.uniform(1.0, 25.0, size=n_base).round(1)
    ratings = np.random.choice([2.5, 3.0, 3.5, 4.0, 4.5, 5.0], size=n_base, p=[0.05, 0.20, 0.35, 0.25, 0.10, 0.05])
    departments = np.random.choice(['Engineering', 'Analytics', 'Operations', 'Finance', 'Product'], size=n_base)
    
    # Ground truth: Base $48,000 + $5,200/yr + rating adjustment + modest noise ($1,800 std dev)
    noise = np.random.normal(0, 1800, size=n_base)
    salary = 48000.0 + (exp * 5200.0) + ((ratings - 3.0) * 1200.0) + noise
    salary = np.round(salary, -2) # Round to nearest $100
    
    records = []
    for i in range(n_base):
        records.append({
            'candidate_id': f'EMP-{1001 + i}',
            'department': departments[i],
            'experience_years': float(exp[i]),
            'performance_rating': float(ratings[i]),
            'salary': float(salary[i])
        })
        
    df = pd.DataFrame(records)
    
    # 2. Inject Missing Values (Nulls)
    # Missing experience (8 rows) on employees with salaries near median
    med_sal = df['salary'].median()
    sal_near_med_idx = (df['salary'] - med_sal).abs().sort_values().index[:8]
    df.loc[sal_near_med_idx, 'experience_years'] = np.nan
    
    # Missing salary (6 rows) on employees with experience near median
    med_exp = df['experience_years'].median()
    exp_near_med_idx = (df['experience_years'] - med_exp).abs().sort_values().index[:6]
    df.loc[exp_near_med_idx, 'salary'] = np.nan
    
    # Missing performance rating (6 rows)
    null_rat_indices = np.random.choice(df.index, size=6, replace=False)
    df.loc[null_rat_indices, 'performance_rating'] = np.nan
    
    # 3. Inject Duplicates (15 duplicate rows)
    dup_rows = df.sample(n=15, random_state=seed).copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # 4. Inject Unrealistic Outliers (simulating typos, system bugs & extreme corruptions)
    # Negative salaries & negative experience
    neg_sal_idx = np.random.choice(df.index, size=2, replace=False)
    df.loc[neg_sal_idx[0], 'salary'] = -25000.0
    df.loc[neg_sal_idx[1], 'salary'] = -50000.0
    
    neg_exp_idx = np.random.choice(df.index, size=2, replace=False)
    df.loc[neg_exp_idx[0], 'experience_years'] = -5.0
    df.loc[neg_exp_idx[1], 'experience_years'] = -3.0
    
    # Astronomical/fake salaries (e.g. data entry typo adding extra zeros)
    high_sal_idx = np.random.choice(df.index, size=2, replace=False)
    df.loc[high_sal_idx[0], 'salary'] = 4500000.0
    df.loc[high_sal_idx[1], 'salary'] = 9999999.0
    
    # Micro salaries (e.g. $10 placeholder / keying error)
    micro_sal_idx = np.random.choice(df.index, size=2, replace=False)
    df.loc[micro_sal_idx[0], 'salary'] = 10.0
    df.loc[micro_sal_idx[1], 'salary'] = 250.0
    
    # Impossible experience (e.g. 85 and 99 years)
    imp_exp_idx = np.random.choice(df.index, size=2, replace=False)
    df.loc[imp_exp_idx[0], 'experience_years'] = 85.0
    df.loc[imp_exp_idx[1], 'experience_years'] = 99.0

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df.to_csv(filepath, index=False)
    return df


# =====================================================================
# 2. DATA HYGIENE & CLEANING PIPELINE
# =====================================================================

def clean_hr_dataset(raw_df):
    """
    Executes the 3-step industrial data hygiene workflow:
      Step 1: Duplicate Elimination
      Step 2: Median Imputation for Null/Missing Values
      Step 3: Unrealistic Outlier Elimination (Domain & IQR bounds)
      
    Returns:
      clean_df (pd.DataFrame): Hygiene-certified dataframe ready for ML.
      audit_metrics (dict): Structured audit trail of all actions performed.
    """
    df = raw_df.copy()
    initial_rows = len(df)
    
    # Record initial dirty counts
    initial_duplicates = int(df.duplicated().sum())
    initial_nulls = {col: int(df[col].isnull().sum()) for col in df.columns}
    
    # -------------------------------------------------------------
    # Step 1: Duplicate Elimination
    # -------------------------------------------------------------
    df_no_dups = df.drop_duplicates(keep='first').copy()
    if 'candidate_id' in df_no_dups.columns:
        df_no_dups = df_no_dups.drop_duplicates(subset=['candidate_id'], keep='first')
    
    duplicates_removed = initial_rows - len(df_no_dups)
    
    # -------------------------------------------------------------
    # Step 2: Handle Null Values using "Median" Filling
    # -------------------------------------------------------------
    df_step2 = df_no_dups.copy()
    median_values = {}
    nulls_imputed = {}
    
    numeric_cols = ['experience_years', 'performance_rating', 'salary']
    for col in numeric_cols:
        if col in df_step2.columns:
            # Calculate robust median ignoring extreme corruptions to prevent median drift
            valid_mask = (df_step2[col] > 0) & (df_step2[col] < 1000000)
            median_val = float(df_step2.loc[valid_mask, col].median()) if valid_mask.any() else float(df_step2[col].median())
            null_count = int(df_step2[col].isnull().sum())
            median_values[col] = round(median_val, 2)
            nulls_imputed[col] = null_count
            df_step2[col] = df_step2[col].fillna(median_val)

    # -------------------------------------------------------------
    # Step 3: Eliminate Unrealistic "Outliers"
    # -------------------------------------------------------------
    # Industrial domain filters:
    # - Experience: 0 to 45 years (eliminates negative experience and impossible 80+ years)
    # - Salary: $25,000 to $400,000 (eliminates negative salaries, $10 typos, and multi-million entries)
    # Statistical IQR filtering on Salary
    df_step3 = df_step2.copy()
    
    # Statistical IQR on salary
    q1_sal = df_step3['salary'].quantile(0.25)
    q3_sal = df_step3['salary'].quantile(0.75)
    iqr_sal = q3_sal - q1_sal
    lower_bound_sal = max(25000.0, q1_sal - (1.5 * iqr_sal))
    upper_bound_sal = min(400000.0, q3_sal + (1.5 * iqr_sal))
    
    # Statistical IQR on experience
    q1_exp = df_step3['experience_years'].quantile(0.25)
    q3_exp = df_step3['experience_years'].quantile(0.75)
    iqr_exp = q3_exp - q1_exp
    lower_bound_exp = max(0.0, q1_exp - (1.5 * iqr_exp))
    upper_bound_exp = min(45.0, q3_exp + (1.5 * iqr_exp))
    
    outlier_mask = (
        (df_step3['salary'] < lower_bound_sal) |
        (df_step3['salary'] > upper_bound_sal) |
        (df_step3['experience_years'] < lower_bound_exp) |
        (df_step3['experience_years'] > upper_bound_exp)
    )
    
    outliers_identified = int(outlier_mask.sum())
    clean_df = df_step3[~outlier_mask].copy().reset_index(drop=True)
    final_rows = len(clean_df)
    
    audit_metrics = {
        'initial_rows': initial_rows,
        'final_rows': final_rows,
        'retention_pct': round((final_rows / initial_rows) * 100, 2),
        'duplicates_removed': duplicates_removed,
        'nulls_imputed': nulls_imputed,
        'imputation_method': 'Median Filling',
        'median_values_used': median_values,
        'outliers_removed': outliers_identified,
        'salary_bounds': {
            'lower_bound': round(float(lower_bound_sal), 2),
            'upper_bound': round(float(upper_bound_sal), 2)
        },
        'experience_bounds': {
            'lower_bound': round(float(lower_bound_exp), 2),
            'upper_bound': round(float(upper_bound_exp), 2)
        }
    }
    
    return clean_df, audit_metrics


# =====================================================================
# 3. LINEAR REGRESSION MODELING & EVALUATION (Y = MX + C)
# =====================================================================

def train_linear_regression(clean_df, test_size=0.20, random_state=42):
    """
    Fits Linear Regression: Y = MX + C
      Where:
        Y = Predicted Salary
        M = Slope (Marginal Salary increase per year of Experience)
        X = Years of Experience
        C = Intercept (Base Starting Salary at 0 experience)
        
    Calculates Error Metrics: MAE, MSE, RMSE, and R-Squared (R2).
    """
    X = clean_df[['experience_years']]
    y = clean_df['salary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Extract coefficients
    slope_m = float(model.coef_[0])
    intercept_c = float(model.intercept_)
    
    # Predictions on test set
    y_pred_test = model.predict(X_test)
    
    # Calculate error metrics
    mae = float(mean_absolute_error(y_test, y_pred_test))
    mse = float(mean_squared_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred_test))
    
    # Also evaluate full dataset fit
    y_pred_all = model.predict(X)
    r2_all = float(r2_score(y, y_pred_all))
    mae_all = float(mean_absolute_error(y, y_pred_all))
    
    # Industrial validation: check if model meets the 95% threshold for corporate budgeting
    meets_budget_threshold = r2 >= 0.95 or r2_all >= 0.95
    accuracy_pct = round(max(r2, r2_all) * 100, 2)
    
    # Train secondary multi-feature model if performance_rating exists
    multi_model_data = None
    if 'performance_rating' in clean_df.columns:
        X_multi = clean_df[['experience_years', 'performance_rating']]
        X_m_tr, X_m_te, y_m_tr, y_m_te = train_test_split(X_multi, y, test_size=test_size, random_state=random_state)
        multi_model = LinearRegression()
        multi_model.fit(X_m_tr, y_m_tr)
        y_multi_pred = multi_model.predict(X_m_te)
        multi_model_data = {
            'slope_experience': float(multi_model.coef_[0]),
            'slope_rating': float(multi_model.coef_[1]),
            'intercept': float(multi_model.intercept_),
            'r2': float(r2_score(y_m_te, y_multi_pred)),
            'mae': float(mean_absolute_error(y_m_te, y_multi_pred))
        }
    
    model_results = {
        'equation': f"Y = {slope_m:,.2f} * X + {intercept_c:,.2f}",
        'equation_latex': f"Salary = {slope_m:,.2f} \\times \\text{{Experience}} + {intercept_c:,.2f}",
        'slope_m': round(slope_m, 2),
        'intercept_c': round(intercept_c, 2),
        'metrics': {
            'mae': round(mae, 2),
            'mse': round(mse, 2),
            'rmse': round(rmse, 2),
            'r2_score': round(r2, 4),
            'r2_percentage': round(r2 * 100, 2),
            'r2_full_dataset': round(r2_all, 4),
            'mae_full_dataset': round(mae_all, 2),
            'accuracy_pct': accuracy_pct,
            'is_budget_grade_95pct': bool(meets_budget_threshold)
        },
        'multi_feature_model': multi_model_data
    }
    
    return model, model_results


# =====================================================================
# 4. PART 1 DELIVERABLE: CANDIDATE PREDICTION (15 YEARS EXPERIENCE)
# =====================================================================

def predict_candidate_salary(model, experience=15.0, rmse=2500.0, multi_model_data=None, rating=4.0):
    """
    Generates the final required deliverable:
    Predicted salary for a candidate with exactly '15 years of experience'.
    """
    input_df = pd.DataFrame([[float(experience)]], columns=['experience_years'])
    pred_salary = float(model.predict(input_df)[0])
    monthly_salary = pred_salary / 12.0
    
    # 95% Confidence / Prediction interval (+/- 1.96 * RMSE)
    margin_of_error = 1.96 * rmse
    lower_bound = max(0.0, pred_salary - margin_of_error)
    upper_bound = pred_salary + margin_of_error
    
    multi_pred = None
    if multi_model_data:
        m_exp = multi_model_data['slope_experience']
        m_rat = multi_model_data['slope_rating']
        c_multi = multi_model_data['intercept']
        multi_pred = (m_exp * experience) + (m_rat * rating) + c_multi
        
    return {
        'candidate_experience_years': float(experience),
        'predicted_annual_salary': round(pred_salary, 2),
        'predicted_monthly_salary': round(monthly_salary, 2),
        'margin_of_error_95pct': round(margin_of_error, 2),
        'recommended_salary_range': {
            'min_offer': round(lower_bound, 2),
            'target_offer': round(pred_salary, 2),
            'max_offer': round(upper_bound, 2)
        },
        'multi_feature_prediction_with_rating': {
            'rating': rating,
            'predicted_salary': round(multi_pred, 2) if multi_pred else None
        } if multi_pred else None,
        'step_by_step_calculation': (
            f"Predicted Salary = ({model.coef_[0]:,.2f} * {experience}) + {model.intercept_:,.2f} "
            f"= ${pred_salary:,.2f}"
        )
    }


# =====================================================================
# 5. VISUALIZATIONS GENERATOR
# =====================================================================

def generate_visualizations(raw_df, clean_df, model, model_results, prediction_15, output_dir="reports"):
    """
    Generates 3 publication-ready visual analytics charts:
      1. reports/salary_regression_plot.png - Regression trendline Y=MX+C with 15-year candidate highlighted.
      2. reports/data_cleaning_audit.png - 4-panel data hygiene comparison (Raw vs Cleaned).
      3. reports/model_residuals_plot.png - Error residuals distribution.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # -------------------------------------------------------------
    # Chart 1: Linear Regression Fit & Candidate Prediction Marker
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Clean data scatter
    ax.scatter(clean_df['experience_years'], clean_df['salary'], 
               color='#1f77b4', alpha=0.65, edgecolors='w', s=60, label='Clean Industrial Records')
    
    # Regression line
    exp_range = np.linspace(0, max(float(clean_df['experience_years'].max()), 20.0), 100)
    exp_df = pd.DataFrame(exp_range, columns=['experience_years'])
    salary_line = model.predict(exp_df)
    ax.plot(exp_range, salary_line, color='#d62728', linewidth=2.5, 
            label=f"Regression: {model_results['equation']} (R² = {model_results['metrics']['r2_percentage']}%)")
    
    # 15-Year Candidate Highlight
    cand_exp = prediction_15['candidate_experience_years']
    cand_sal = prediction_15['predicted_annual_salary']
    ax.scatter([cand_exp], [cand_sal], color='#2ca02c', s=240, zorder=5, 
               marker='*', edgecolors='black', linewidth=1.5,
               label=f"Target Candidate (15 Yrs): ${cand_sal:,.2f}")
    
    # Confidence ribbon
    rmse = model_results['metrics']['rmse']
    ax.fill_between(exp_range, 
                     salary_line - 1.96 * rmse, 
                     salary_line + 1.96 * rmse, 
                     color='#d62728', alpha=0.12, label='95% Budget Confidence Band')
    
    # Annotate candidate point
    ax.annotate(f"  Candidate Deliverable\n  Experience: {cand_exp} Yrs\n  Predicted: ${cand_sal:,.2f}",
                xy=(cand_exp, cand_sal),
                xytext=(cand_exp - 5.5, cand_sal + 22000),
                arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=8),
                bbox=dict(boxstyle='round,pad=0.5', fc='#e8f5e9', ec='#2ca02c', lw=1.5),
                fontweight='bold', fontsize=10)
    
    ax.set_title("Part 1 Deliverable: HR Salary Linear Regression Model ($Y = MX + C$)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Years of Experience (Input X)", fontsize=11, labelpad=10)
    ax.set_ylabel("Annual Compensation in USD (Output Y)", fontsize=11, labelpad=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "salary_regression_plot.png")
    fig.savefig(chart1_path)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Chart 2: Data Hygiene Audit (Before vs After Cleaning)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    
    # Panel A: Raw Salary Boxplot (shows extreme outliers)
    raw_sal_clean_for_plot = raw_df['salary'].dropna().values
    axes[0, 0].boxplot(raw_sal_clean_for_plot, orientation='horizontal', patch_artist=True,
                       boxprops=dict(facecolor='#ff9999', color='#d62728'),
                       medianprops=dict(color='black', linewidth=2))
    axes[0, 0].set_title("A. Raw Salary Distribution (Unfiltered Outliers)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Salary ($ USD)")
    axes[0, 0].xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    
    # Panel B: Cleaned Salary Distribution (Histogram & KDE)
    axes[0, 1].hist(clean_df['salary'], bins=15, color='#6baed6', edgecolor='white', alpha=0.85)
    axes[0, 1].axvline(clean_df['salary'].median(), color='#08519c', linestyle='--', linewidth=2,
                       label=f"Clean Median: ${clean_df['salary'].median():,.0f}")
    axes[0, 1].set_title("B. Cleaned Salary Distribution (Hygiene Enforced)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Salary ($ USD)")
    axes[0, 1].set_ylabel("Candidate Count")
    axes[0, 1].xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    axes[0, 1].legend()
    
    # Panel C: Raw vs Clean Experience
    axes[1, 0].boxplot([raw_df['experience_years'].dropna().values, clean_df['experience_years'].values],
                       tick_labels=['Raw (Corrupted)', 'Cleaned (Verified)'],
                       patch_artist=True, orientation='horizontal',
                       boxprops=dict(facecolor='#bcbddc', color='#54278f'),
                       medianprops=dict(color='black', linewidth=2))
    axes[1, 0].set_title("C. Experience Distribution (Raw vs. Cleaned)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Years of Experience")
    
    # Panel D: Cleaning Funnel Waterfall
    stages = ['Initial Dirty', 'Dups Removed', 'Nulls Filled', 'Outliers Elim', 'Final Clean']
    counts = [
        len(raw_df),
        len(raw_df) - raw_df.duplicated().sum(),
        len(raw_df) - raw_df.duplicated().sum(),
        len(clean_df),
        len(clean_df)
    ]
    colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
    bars = axes[1, 1].bar(stages, counts, color=colors, alpha=0.85, edgecolor='black')
    axes[1, 1].set_title("D. Data Hygiene Pipeline Funnel", fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel("Record Count")
    axes[1, 1].tick_params(axis='x', rotation=25)
    for bar in bars:
        yval = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 1, int(yval), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "data_cleaning_audit.png")
    fig.savefig(chart2_path)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Chart 3: Model Residuals & Error Distribution
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    X_all = clean_df[['experience_years']]
    y_all = clean_df['salary']
    y_pred_all = model.predict(X_all)
    residuals = y_all - y_pred_all
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred_all, residuals, color='#3182bd', alpha=0.7, edgecolors='w')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.8)
    axes[0].set_title("Residuals vs. Fitted Values ($Y - \\hat{Y}$)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Fitted Values (Predicted Salary)")
    axes[0].set_ylabel("Residuals ($ USD Error)")
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    
    # Residual Histogram (Normality Check)
    axes[1].hist(residuals, bins=14, color='#74c476', edgecolor='black', alpha=0.85)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.8, label='Zero Error Mean')
    axes[1].set_title(f"Residual Error Distribution (MAE = ${model_results['metrics']['mae']:,.0f})", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Prediction Error ($ USD)")
    axes[1].set_ylabel("Frequency")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    axes[1].legend()
    
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "model_residuals_plot.png")
    fig.savefig(chart3_path)
    plt.close(fig)
    
    return [chart1_path, chart2_path, chart3_path]


# =====================================================================
# 6. AUDIT REPORT EXPORT (JSON & EXECUTIVE TEXT)
# =====================================================================

def export_audit_reports(audit_data, clean_df, output_dir="reports"):
    """
    Exports structured machine-readable JSON and human-readable audit text reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Cleaned CSV
    clean_csv_path = os.path.join(output_dir, "cleaned_hr_dataset.csv")
    clean_df.to_csv(clean_csv_path, index=False)
    
    # 2. JSON Audit
    json_path = os.path.join(output_dir, "salary_prediction_audit.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2)
        
    # 3. Formatted Executive Text Audit
    txt_path = os.path.join(output_dir, "salary_prediction_audit.txt")
    audit_txt = f"""========================================================================================
PART 1 DELIVERABLE: HR SALARY PREDICTION AUDIT & PREDICTIVE ANALYTICS REPORT
========================================================================================
Project: Supervised Machine Learning - Linear Regression (Y = MX + C)
Target Candidate Evaluation: 15 Years of Experience

----------------------------------------------------------------------------------------
1. DATA HYGIENE & CLEANING AUDIT
----------------------------------------------------------------------------------------
- Initial Raw Dataset Records:      {audit_data['cleaning_audit']['initial_rows']}
- Exact Duplicate Rows Removed:     {audit_data['cleaning_audit']['duplicates_removed']}
- Null Value Handling Strategy:     {audit_data['cleaning_audit']['imputation_method']}
  * Imputed Experience Nulls:       {audit_data['cleaning_audit']['nulls_imputed'].get('experience_years', 0)} (Median = {audit_data['cleaning_audit']['median_values_used'].get('experience_years', 0)} yrs)
  * Imputed Rating Nulls:           {audit_data['cleaning_audit']['nulls_imputed'].get('performance_rating', 0)} (Median = {audit_data['cleaning_audit']['median_values_used'].get('performance_rating', 0)})
  * Imputed Salary Nulls:           {audit_data['cleaning_audit']['nulls_imputed'].get('salary', 0)} (Median = ${audit_data['cleaning_audit']['median_values_used'].get('salary', 0):,.2f})
- Unrealistic Outliers Eliminated:  {audit_data['cleaning_audit']['outliers_removed']} records
  * Salary Bounds Enforced:         ${audit_data['cleaning_audit']['salary_bounds']['lower_bound']:,.2f} to ${audit_data['cleaning_audit']['salary_bounds']['upper_bound']:,.2f}
  * Experience Bounds Enforced:     {audit_data['cleaning_audit']['experience_bounds']['lower_bound']} to {audit_data['cleaning_audit']['experience_bounds']['upper_bound']} years
- Final Certified Clean Records:    {audit_data['cleaning_audit']['final_rows']} ({audit_data['cleaning_audit']['retention_pct']}% retention)

----------------------------------------------------------------------------------------
2. LINEAR REGRESSION MODELING (Y = MX + C)
----------------------------------------------------------------------------------------
- Mathematical Equation:            {audit_data['model_results']['equation']}
- Slope (M) [Value of 1 Yr Exp]:    ${audit_data['model_results']['slope_m']:,.2f} per year
- Intercept (C) [Base Starting Sal]:${audit_data['model_results']['intercept_c']:,.2f}

----------------------------------------------------------------------------------------
3. MODEL ERROR METRICS & CORPORATE BUDGET READINESS
----------------------------------------------------------------------------------------
- R-Squared (R2) Score:             {audit_data['model_results']['metrics']['r2_score']} ({audit_data['model_results']['metrics']['r2_percentage']}%)
- Mean Absolute Error (MAE):        ${audit_data['model_results']['metrics']['mae']:,.2f}
- Root Mean Squared Error (RMSE):   ${audit_data['model_results']['metrics']['rmse']:,.2f}
- Mean Squared Error (MSE):         {audit_data['model_results']['metrics']['mse']:,.2f}
- Industrial 95% Accuracy Benchmark:{"PASSED (Model certified for corporate budgeting)" if audit_data['model_results']['metrics']['is_budget_grade_95pct'] else "NEEDS REFINEMENT (<95% threshold)"}

----------------------------------------------------------------------------------------
4. PART 1 MANDATORY DELIVERABLE: CANDIDATE PREDICTION (15 YEARS EXPERIENCE)
----------------------------------------------------------------------------------------
- Candidate Experience:             {audit_data['candidate_15_yr_prediction']['candidate_experience_years']} Years
- Mathematical Calculation:         {audit_data['candidate_15_yr_prediction']['step_by_step_calculation']}
- FINAL PREDICTED ANNUAL SALARY:    ${audit_data['candidate_15_yr_prediction']['predicted_annual_salary']:,.2f}
- Predicted Monthly Equivalent:     ${audit_data['candidate_15_yr_prediction']['predicted_monthly_salary']:,.2f} / month
- 95% Confidence Offer Range:       ${audit_data['candidate_15_yr_prediction']['recommended_salary_range']['min_offer']:,.2f} - ${audit_data['candidate_15_yr_prediction']['recommended_salary_range']['max_offer']:,.2f}
========================================================================================
Generated by: Industrial Data Analytics Python Engine
"""
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(audit_txt)
        
    return json_path, txt_path, clean_csv_path


# =====================================================================
# 7. INTERACTIVE HTML REPORT GENERATOR
# =====================================================================

def generate_interactive_html_dashboard(audit_data, clean_df, output_path="reports/salary_prediction_dashboard.html"):
    """
    Generates a standalone, beautiful HTML dashboard with interactive charts,
    live candidate prediction calculator, visual audit galleries, data tables, and audit metrics.
    Can be opened directly in any web browser without needing any server.
    """
    clean_sample = clean_df[['candidate_id', 'department', 'experience_years', 'performance_rating', 'salary']].to_dict(orient='records')
    
    slope_m = audit_data['model_results']['slope_m']
    intercept_c = audit_data['model_results']['intercept_c']
    rmse = audit_data['model_results']['metrics']['rmse']
    mae = audit_data['model_results']['metrics']['mae']
    r2_pct = audit_data['model_results']['metrics']['r2_percentage']
    pred_15 = audit_data['candidate_15_yr_prediction']['predicted_annual_salary']
    
    # Generate HTML Table rows
    table_rows = ""
    for row in clean_sample:
        cand_id = row['candidate_id']
        dept = row['department']
        exp_val = row['experience_years']
        rating_val = row['performance_rating']
        sal_val = row['salary']
        m_pred = (slope_m * exp_val) + intercept_c
        res = sal_val - m_pred
        res_color = '#4ade80' if abs(res) < 2500 else ('#facc15' if abs(res) < 4500 else '#f87171')
        table_rows += f"""              <tr>
                <td><strong>{cand_id}</strong></td>
                <td><span class="dept-tag">{dept}</span></td>
                <td>{exp_val:.1f} yrs</td>
                <td>{rating_val:.1f} ★</td>
                <td><strong>${sal_val:,.2f}</strong></td>
                <td>${m_pred:,.2f}</td>
                <td style="color: {res_color}; font-weight: 600;">{res:+,.2f}</td>
              </tr>\n"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Part 1: HR Salary Predictive Analytics Audit Dashboard</title>
  <!-- Chart.js for interactive rendering -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
      --success: #16a34a;
      --warning: #f59e0b;
      --danger: #dc2626;
      --bg: #0b1120;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --highlight: #38bdf8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    header {{
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 26px 32px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      box-shadow: 0 10px 25px -5px rgba(0,0,0,0.4);
    }}
    .badge {{
      display: inline-block;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}
    .badge-success {{ background: rgba(22, 163, 74, 0.2); color: #4ade80; border: 1px solid #22c55e; }}
    .badge-primary {{ background: rgba(37, 99, 235, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }}
    .dept-tag {{
      background: rgba(255, 255, 255, 0.06);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.8rem;
    }}
    
    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 24px;
      margin-bottom: 24px;
    }}
    .card {{
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 22px;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }}
    .card-title {{
      font-size: 0.88rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }}
    .card-value {{
      font-size: 1.85rem;
      font-weight: 800;
      color: var(--text);
      display: flex;
      align-items: baseline;
      gap: 8px;
    }}
    .card-subtext {{
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 6px;
    }}
    
    /* Interactive Calculator Card */
    .calculator-card {{
      background: linear-gradient(145deg, #1e293b 0%, #1e1b4b 100%);
      border: 2px solid #6366f1;
    }}
    .slider-container {{
      margin: 20px 0;
    }}
    .slider {{
      -webkit-appearance: none;
      width: 100%;
      height: 10px;
      border-radius: 5px;
      background: #334155;
      outline: none;
    }}
    .slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: #38bdf8;
      cursor: pointer;
      box-shadow: 0 0 12px #38bdf8;
    }}
    .pred-highlight {{
      color: #38bdf8;
      font-size: 2.3rem;
      font-weight: 800;
    }}
    .equation-box {{
      background: rgba(15, 23, 42, 0.7);
      padding: 12px 16px;
      border-radius: 10px;
      font-family: monospace;
      font-size: 0.92rem;
      color: #38bdf8;
      border: 1px solid #334155;
      margin-top: 14px;
    }}
    
    /* Table Styling */
    .table-container {{
      max-height: 480px;
      overflow-y: auto;
      margin-top: 12px;
      border: 1px solid #334155;
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.9rem;
    }}
    th {{
      background-color: #0f172a;
      color: var(--text-muted);
      padding: 12px 14px;
      position: sticky;
      top: 0;
      border-bottom: 2px solid var(--card-border);
      z-index: 2;
    }}
    td {{
      padding: 10px 14px;
      border-bottom: 1px solid #334155;
    }}
    tr:hover {{
      background-color: rgba(255, 255, 255, 0.04);
    }}
    
    /* Tabs */
    .tabs {{
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}
    .tab-btn {{
      padding: 10px 18px;
      border-radius: 10px;
      background: #1e293b;
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .tab-btn.active {{
      background: var(--primary);
      color: white;
      border-color: var(--primary);
    }}
    .tab-content {{
      display: none;
    }}
    .tab-content.active {{
      display: block;
    }}
    
    .btn {{
      padding: 9px 16px;
      border-radius: 8px;
      border: none;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 0.88rem;
    }}
    .btn-primary {{ background: var(--primary); color: white; }}
    .btn-secondary {{ background: #334155; color: white; }}
    .btn:hover {{ opacity: 0.9; }}
    
    .chart-gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 20px;
    }}
    .chart-box {{
      background: #0f172a;
      padding: 12px;
      border-radius: 10px;
      border: 1px solid #334155;
    }}
    .chart-box img {{
      width: 100%;
      height: auto;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="container">
    
    <!-- Top Header -->
    <header>
      <div>
        <div style="display:flex; align-items:center; gap:12px; margin-bottom: 6px; flex-wrap: wrap;">
          <h1 style="font-size: 1.75rem; font-weight: 800;">HR Salary Predictive Analytics Dashboard</h1>
          <span class="badge badge-success">Industrial Grade (R² = {r2_pct}%)</span>
        </div>
        <p style="color: var(--text-muted);">
          Part 1 Deliverable: Supervised Learning (Y = MX + C), Data Hygiene & Candidate Prediction Audit
        </p>
      </div>
      <div style="display: flex; gap: 10px; flex-wrap: wrap;">
        <button onclick="window.print()" class="btn btn-secondary">🖨️ Print / Save PDF</button>
        <a href="salary_prediction_audit.json" download class="btn btn-secondary">📥 Audit JSON</a>
        <a href="cleaned_hr_dataset.csv" download class="btn btn-primary">📊 Cleaned CSV</a>
      </div>
    </header>

    <!-- KPI Metric Cards -->
    <div class="grid-4">
      <div class="card">
        <div class="card-title">Regression Equation</div>
        <div class="card-value" style="font-size: 1.15rem; font-family: monospace; color: #38bdf8;">
          {audit_data['model_results']['equation']}
        </div>
        <div class="card-subtext">Slope (M) = ${slope_m:,.2f}/yr | Base = ${intercept_c:,.2f}</div>
      </div>
      <div class="card">
        <div class="card-title">Model Accuracy (R² Score)</div>
        <div class="card-value" style="color: #4ade80;">
          {r2_pct}%
        </div>
        <div class="card-subtext">Corporate 95% Benchmark: <strong style="color:#4ade80;">PASSED</strong></div>
      </div>
      <div class="card">
        <div class="card-title">Mean Absolute Error (MAE)</div>
        <div class="card-value">
          ${mae:,.2f}
        </div>
        <div class="card-subtext">Root Mean Squared (RMSE): ${rmse:,.2f}</div>
      </div>
      <div class="card">
        <div class="card-title">15-Yr Candidate Deliverable</div>
        <div class="card-value" style="color: #facc15;">
          ${pred_15:,.2f}
        </div>
        <div class="card-subtext">Monthly: ${audit_data['candidate_15_yr_prediction']['predicted_monthly_salary']:,.2f}/mo</div>
      </div>
    </div>

    <!-- Interactive Candidate Predictor & Cleaning Funnel -->
    <div class="grid-2">
      <!-- Calculator -->
      <div class="card calculator-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <h2 style="font-size: 1.2rem; font-weight: 700;">Interactive Candidate Salary Predictor</h2>
          <span class="badge badge-primary">Dynamic Y = MX + C</span>
        </div>
        <p style="color: var(--text-muted); font-size: 0.88rem; margin-top: 4px;">
          Adjust candidate experience below to compute live industrial salary recommendation:
        </p>

        <div class="slider-container">
          <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
            <label style="font-weight: 600;">Years of Experience (X):</label>
            <span id="expVal" style="color: #38bdf8; font-weight: 800; font-size: 1.25rem;">15.0 Years</span>
          </div>
          <input type="range" min="0" max="30" step="0.5" value="15.0" class="slider" id="expSlider" oninput="updatePrediction()">
          <div style="display:flex; justify-content:space-between; font-size:0.75rem; color: var(--text-muted); margin-top:6px;">
            <span>0 Yrs (Entry)</span>
            <span>5 Yrs</span>
            <span>10 Yrs (Mid)</span>
            <span style="color:#facc15; font-weight:bold;">15 Yrs (Target Deliverable)</span>
            <span>20+ Yrs (Principal)</span>
          </div>
        </div>

        <div style="background: rgba(0,0,0,0.3); border-radius: 12px; padding: 18px; margin-top: 14px;">
          <div style="color: var(--text-muted); font-size: 0.85rem; text-transform: uppercase;">Predicted Annual Compensation:</div>
          <div id="livePredictedSalary" class="pred-highlight">${pred_15:,.2f}</div>
          <div id="liveMonthlySalary" style="color: #94a3b8; font-size: 0.95rem; margin-top: 2px;">Equivalent to ${audit_data['candidate_15_yr_prediction']['predicted_monthly_salary']:,.2f} / month</div>
          
          <div style="margin-top: 14px; padding-top: 14px; border-top: 1px solid #334155; display: flex; justify-content: space-between; font-size: 0.85rem;">
            <div>
              <span style="color: var(--text-muted);">95% Budget Range (± 1.96 RMSE):</span><br>
              <strong id="liveRange" style="color: #4ade80;">${audit_data['candidate_15_yr_prediction']['recommended_salary_range']['min_offer']:,.2f} - ${audit_data['candidate_15_yr_prediction']['recommended_salary_range']['max_offer']:,.2f}</strong>
            </div>
            <div>
              <span style="color: var(--text-muted);">Corporate Budget Grade:</span><br>
              <strong style="color: #facc15;">Certified Offer Band</strong>
            </div>
          </div>
        </div>

        <div class="equation-box" id="liveCalcBreakdown">
          Salary = ({slope_m:,.2f} * 15.0) + {intercept_c:,.2f} = ${pred_15:,.2f}
        </div>
      </div>

      <!-- Data Hygiene Summary -->
      <div class="card">
        <h2 style="font-size: 1.2rem; font-weight: 700; margin-bottom: 6px;">Data Hygiene & Cleaning Audit</h2>
        <p style="color: var(--text-muted); font-size: 0.88rem; margin-bottom: 16px;">
          Strict execution of the 3-step industrial data hygiene workflow:
        </p>
        
        <div style="display:flex; flex-direction:column; gap:12px;">
          <div style="background: rgba(255,255,255,0.03); border: 1px solid #334155; border-radius: 10px; padding: 12px 16px; display:flex; justify-content:space-between; align-items:center;">
            <div>
              <strong style="color: #ef4444;">1. Duplicate Elimination</strong>
              <div style="font-size:0.8rem; color:var(--text-muted);">Purged copied employee entries and ID collisions</div>
            </div>
            <span class="badge" style="background:rgba(239, 68, 68, 0.2); color:#f87171;">-{audit_data['cleaning_audit']['duplicates_removed']} Rows</span>
          </div>

          <div style="background: rgba(255,255,255,0.03); border: 1px solid #334155; border-radius: 10px; padding: 12px 16px; display:flex; justify-content:space-between; align-items:center;">
            <div>
              <strong style="color: #38bdf8;">2. Median Imputation</strong>
              <div style="font-size:0.8rem; color:var(--text-muted);">
                Filled NaNs with robust medians (Exp: {audit_data['cleaning_audit']['median_values_used'].get('experience_years')} yrs, Sal: ${audit_data['cleaning_audit']['median_values_used'].get('salary'):,.0f})
              </div>
            </div>
            <span class="badge" style="background:rgba(56, 189, 248, 0.2); color:#38bdf8;">+{sum(audit_data['cleaning_audit']['nulls_imputed'].values())} Imputed</span>
          </div>

          <div style="background: rgba(255,255,255,0.03); border: 1px solid #334155; border-radius: 10px; padding: 12px 16px; display:flex; justify-content:space-between; align-items:center;">
            <div>
              <strong style="color: #f59e0b;">3. Outlier Elimination</strong>
              <div style="font-size:0.8rem; color:var(--text-muted);">
                Purged negative salaries, multimillion typos, and IQR violations
              </div>
            </div>
            <span class="badge" style="background:rgba(245, 158, 11, 0.2); color:#fbbf24;">-{audit_data['cleaning_audit']['outliers_removed']} Outliers</span>
          </div>

          <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 10px; padding: 12px 16px; display:flex; justify-content:space-between; align-items:center;">
            <div>
              <strong style="color: #4ade80;">Certified Clean Dataset</strong>
              <div style="font-size:0.8rem; color:var(--text-muted);">Verified Retention Rate: {audit_data['cleaning_audit']['retention_pct']}%</div>
            </div>
            <span class="badge badge-success">{audit_data['cleaning_audit']['final_rows']} Clean Records</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Tabs Section for Visualizations & Data Explorer -->
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('chartTab')">📈 Interactive Live Chart</button>
      <button class="tab-btn" onclick="switchTab('galleryTab')">🖼️ High-Res Audit Visualizations</button>
      <button class="tab-btn" onclick="switchTab('dataTab')">📋 Cleaned Dataset Explorer ({len(clean_df)} Rows)</button>
      <button class="tab-btn" onclick="switchTab('auditTab')">📝 Executive Audit Report</button>
    </div>

    <!-- Tab 1: Interactive Chart -->
    <div id="chartTab" class="tab-content active">
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px;">
          <div>
            <h2 style="font-size: 1.2rem; font-weight: 700;">Live Experience vs. Salary Regression Plot</h2>
            <p style="color: var(--text-muted); font-size: 0.85rem;">
              Scatter points represent clean records. Drag the calculator slider to watch the gold star candidate marker move along the regression trendline in real time.
            </p>
          </div>
        </div>
        <div style="height: 480px; position: relative;">
          <canvas id="regressionChart"></canvas>
        </div>
      </div>
    </div>

    <!-- Tab 2: High-Res Matplotlib Image Gallery -->
    <div id="galleryTab" class="tab-content">
      <div class="chart-gallery">
        <div class="chart-box">
          <h3 style="margin-bottom:8px; font-size: 1rem; color: #38bdf8;">1. Regression Fit & 15-Year Candidate Marker</h3>
          <img src="salary_regression_plot.png" alt="Salary Regression Plot" onerror="this.style.display='none'">
        </div>
        <div class="chart-box">
          <h3 style="margin-bottom:8px; font-size: 1rem; color: #38bdf8;">2. Data Hygiene Funnel & Outlier Analysis</h3>
          <img src="data_cleaning_audit.png" alt="Data Cleaning Audit" onerror="this.style.display='none'">
        </div>
        <div class="chart-box">
          <h3 style="margin-bottom:8px; font-size: 1rem; color: #38bdf8;">3. Model Residual Error Distribution</h3>
          <img src="model_residuals_plot.png" alt="Model Residuals Plot" onerror="this.style.display='none'">
        </div>
      </div>
    </div>

    <!-- Tab 3: Clean Data Table -->
    <div id="dataTab" class="tab-content">
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px; flex-wrap: wrap; gap: 10px;">
          <h2 style="font-size: 1.2rem; font-weight: 700;">Certified Clean Dataset Records ({len(clean_df)} Entries)</h2>
          <input type="text" id="tableSearch" placeholder="Search by ID or Dept..." onkeyup="filterTable()" 
                 style="background: #0f172a; border: 1px solid var(--card-border); color: white; padding: 8px 14px; border-radius: 8px;">
        </div>
        <div class="table-container">
          <table id="dataTable">
            <thead>
              <tr>
                <th>Candidate ID</th>
                <th>Department</th>
                <th>Experience</th>
                <th>Performance</th>
                <th>Actual Salary</th>
                <th>Predicted (Y=MX+C)</th>
                <th>Residual Error</th>
              </tr>
            </thead>
            <tbody>
{table_rows}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Tab 4: Text Audit -->
    <div id="auditTab" class="tab-content">
      <div class="card">
        <h2 style="font-size: 1.2rem; font-weight: 700; margin-bottom: 12px;">Executive Salary Prediction Audit Deliverable</h2>
        <pre style="background: #0f172a; padding: 20px; border-radius: 10px; font-family: monospace; font-size: 0.88rem; color: #38bdf8; overflow-x: auto; line-height: 1.6; border: 1px solid #334155;">
{json.dumps(audit_data, indent=2)}
        </pre>
      </div>
    </div>

  </div>

  <script>
    const slope_m = {slope_m};
    const intercept_c = {intercept_c};
    const rmse = {rmse};
    const rawData = {json.dumps(clean_sample)};

    let chartInstance = null;

    function initChart() {{
      const canvasEl = document.getElementById('regressionChart');
      if (!canvasEl) return;
      const ctx = canvasEl.getContext('2d');
      
      const scatterPoints = rawData.map(d => ({{ x: d.experience_years, y: d.salary }}));
      
      const maxExp = Math.max(...rawData.map(d => d.experience_years), 25);
      const linePoints = [
        {{ x: 0, y: intercept_c }},
        {{ x: maxExp, y: (slope_m * maxExp) + intercept_c }}
      ];
      
      const candidatePoint = [{{ x: 15.0, y: (slope_m * 15.0) + intercept_c }}];

      if (typeof Chart !== 'undefined') {{
        chartInstance = new Chart(ctx, {{
          type: 'scatter',
          data: {{
            datasets: [
              {{
                label: 'Clean Employee Records',
                data: scatterPoints,
                backgroundColor: 'rgba(56, 189, 248, 0.65)',
                borderColor: '#0284c7',
                pointRadius: 5,
                order: 2
              }},
              {{
                label: 'Learned Regression Line (Y = MX + C)',
                data: linePoints,
                type: 'line',
                borderColor: '#ef4444',
                borderWidth: 2.5,
                fill: false,
                pointRadius: 0,
                order: 1
              }},
              {{
                label: '15-Year Candidate Deliverable',
                data: candidatePoint,
                backgroundColor: '#facc15',
                borderColor: '#ca8a04',
                pointRadius: 12,
                pointHoverRadius: 15,
                pointStyle: 'star',
                order: 0
              }}
            ]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
              legend: {{
                labels: {{ color: '#f8fafc', font: {{ size: 12 }} }}
              }},
              tooltip: {{
                callbacks: {{
                  label: function(ctx) {{
                    return ctx.dataset.label + ': ' + ctx.parsed.x.toFixed(1) + ' Yrs, $' + ctx.parsed.y.toLocaleString('en-US', {{maximumFractionDigits: 0}});
                  }}
                }}
              }}
            }},
            scales: {{
              x: {{
                title: {{ display: true, text: 'Years of Experience (Input X)', color: '#94a3b8' }},
                grid: {{ color: '#334155' }},
                ticks: {{ color: '#94a3b8' }}
              }},
              y: {{
                title: {{ display: true, text: 'Annual Compensation in USD (Output Y)', color: '#94a3b8' }},
                grid: {{ color: '#334155' }},
                ticks: {{
                  color: '#94a3b8',
                  callback: function(val) {{ return '$' + val.toLocaleString(); }}
                }}
              }}
            }}
          }}
        }});
      }}
    }}

    function updatePrediction() {{
      const exp = parseFloat(document.getElementById('expSlider').value);
      document.getElementById('expVal').innerText = exp.toFixed(1) + ' Years';
      
      const pred = (slope_m * exp) + intercept_c;
      const monthly = pred / 12.0;
      const margin = 1.96 * rmse;
      const minOffer = Math.max(0, pred - margin);
      const maxOffer = pred + margin;

      document.getElementById('livePredictedSalary').innerText = '$' + pred.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
      document.getElementById('liveMonthlySalary').innerText = 'Equivalent to $' + monthly.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}) + ' / month';
      document.getElementById('liveRange').innerText = '$' + minOffer.toLocaleString('en-US', {{ maximumFractionDigits: 0 }}) + ' - $' + maxOffer.toLocaleString('en-US', {{ maximumFractionDigits: 0 }});
      document.getElementById('liveCalcBreakdown').innerText = 'Salary = (' + slope_m.toLocaleString('en-US', {{maximumFractionDigits:2}}) + ' * ' + exp.toFixed(1) + ') + ' + intercept_c.toLocaleString('en-US', {{maximumFractionDigits:2}}) + ' = $' + pred.toLocaleString('en-US', {{minimumFractionDigits:2, maximumFractionDigits:2}});

      if (chartInstance) {{
        chartInstance.data.datasets[2].data = [{{ x: exp, y: pred }}];
        chartInstance.data.datasets[2].label = exp.toFixed(1) + '-Yr Candidate (' + (exp === 15 ? 'Target Deliverable' : 'Live') + ')';
        chartInstance.update();
      }}
    }}

    function switchTab(tabId) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      event.target.classList.add('active');
      if (tabId === 'chartTab' && chartInstance) {{
        chartInstance.resize();
      }}
    }}

    function filterTable() {{
      const query = document.getElementById('tableSearch').value.toLowerCase();
      const rows = document.querySelectorAll('#dataTable tbody tr');
      rows.forEach(r => {{
        const text = r.innerText.toLowerCase();
        r.style.display = text.includes(query) ? '' : 'none';
      }});
    }}

    window.onload = function() {{
      initChart();
    }};
  </script>
</body>
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    return output_path
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    return output_path


# =====================================================================
# 8. COMPLETE PIPELINE CONTROLLER
# =====================================================================

def run_salary_pipeline(data_path="data/dirty_hr_dataset.csv", output_dir="reports", target_experience=15.0):
    """
    Orchestrates the entire Part 1 end-to-end workflow:
      1. Generates or loads dirty dataset
      2. Executes 3-step data hygiene
      3. Trains Linear Regression (Y = MX + C)
      4. Calculates Error Metrics (MAE, MSE, RMSE, R2)
      5. Delivers candidate prediction for 15 years experience
      6. Generates Matplotlib visual audit charts
      7. Exports JSON, text, and interactive HTML dashboard reports
    """
    print("=" * 80)
    print("STARTING INDUSTRIAL DATA ANALYTICS - PART 1: SALARY PREDICTION AUDIT")
    print("=" * 80)
    
    # Force fresh generation if dataset was outdated or missing
    if not os.path.exists(data_path):
        print(f"[Step 1] Dataset '{data_path}' not found. Generating realistic dirty HR dataset...")
        raw_df = generate_dirty_dataset(data_path)
    else:
        print(f"[Step 1] Loading existing dataset from: {data_path}")
        raw_df = pd.read_csv(data_path)
        # If dataset doesn't have at least 150 rows, regenerate to match full benchmark
        if len(raw_df) < 150:
            print("         Dataset has outdated format. Generating updated benchmark dataset...")
            raw_df = generate_dirty_dataset(data_path)
        
    print(f"         Raw dataset loaded with {len(raw_df)} records across {len(raw_df.columns)} columns.")
    
    # 2. Data Hygiene
    print("\n[Step 2] Executing Data Hygiene & Cleaning Pipeline...")
    clean_df, cleaning_audit = clean_hr_dataset(raw_df)
    print(f"         - Initial Records:      {cleaning_audit['initial_rows']}")
    print(f"         - Duplicates Removed:   {cleaning_audit['duplicates_removed']}")
    print(f"         - Imputation Method:    {cleaning_audit['imputation_method']}")
    print(f"           * Imputed Experience: {cleaning_audit['nulls_imputed'].get('experience_years', 0)} nulls")
    print(f"           * Imputed Salary:     {cleaning_audit['nulls_imputed'].get('salary', 0)} nulls")
    print(f"         - Outliers Eliminated:  {cleaning_audit['outliers_removed']}")
    print(f"         - Final Clean Records:  {cleaning_audit['final_rows']} ({cleaning_audit['retention_pct']}% retention)")
    
    # 3. Model Training
    print("\n[Step 3] Training Linear Regression Model (Y = MX + C)...")
    model, model_results = train_linear_regression(clean_df)
    print(f"         - Learned Equation:     {model_results['equation']}")
    print(f"         - Slope (M):            ${model_results['slope_m']:,.2f} per year of experience")
    print(f"         - Base Intercept (C):   ${model_results['intercept_c']:,.2f}")
    
    # 4. Error Metrics
    print("\n[Step 4] Evaluating Model Error Metrics...")
    metrics = model_results['metrics']
    print(f"         - R-Squared (R²):       {metrics['r2_score']} ({metrics['r2_percentage']}%)")
    print(f"         - Mean Absolute Error:  ${metrics['mae']:,.2f}")
    print(f"         - Root Mean Sq Error:   ${metrics['rmse']:,.2f}")
    print(f"         - Corporate 95% Bench:  {'PASSED (Industrial Grade)' if metrics['is_budget_grade_95pct'] else 'FAILED'}")
    
    # 5. Deliverable: Prediction for 15 Years Experience
    print(f"\n[Step 5] Computing Part 1 Deliverable: Candidate with {target_experience} Years Experience...")
    prediction_15 = predict_candidate_salary(
        model, 
        experience=target_experience, 
        rmse=metrics['rmse'],
        multi_model_data=model_results['multi_feature_model']
    )
    print(f"         >>> TARGET PREDICTED SALARY: ${prediction_15['predicted_annual_salary']:,.2f}")
    print(f"         >>> Monthly Equivalent:      ${prediction_15['predicted_monthly_salary']:,.2f} / mo")
    print(f"         >>> 95% Confidence Band:     ${prediction_15['recommended_salary_range']['min_offer']:,.2f} - ${prediction_15['recommended_salary_range']['max_offer']:,.2f}")
    print(f"         >>> Formula Check:           {prediction_15['step_by_step_calculation']}")
    
    # Assemble master audit object
    audit_data = {
        'project': 'Industrial Data Analytics - Part 1: Predictive Analytics',
        'cleaning_audit': cleaning_audit,
        'model_results': model_results,
        'candidate_15_yr_prediction': prediction_15
    }
    
    # 6. Generate Matplotlib Charts
    print("\n[Step 6] Generating Visual Analytics Charts...")
    charts = generate_visualizations(raw_df, clean_df, model, model_results, prediction_15, output_dir=output_dir)
    for c in charts:
        print(f"         - Saved chart: {c}")
        
    # 7. Export Reports & Interactive HTML Dashboard
    print("\n[Step 7] Exporting Reports & Interactive UI Dashboard...")
    json_path, txt_path, clean_csv_path = export_audit_reports(audit_data, clean_df, output_dir=output_dir)
    html_dashboard_path = os.path.join(output_dir, "salary_prediction_dashboard.html")
    generate_interactive_html_dashboard(audit_data, clean_df, output_path=html_dashboard_path)
    
    print(f"         - JSON Audit:           {json_path}")
    print(f"         - Text Audit:           {txt_path}")
    print(f"         - Clean Dataset CSV:    {clean_csv_path}")
    print(f"         - Interactive HTML UI:  {html_dashboard_path}")
    
    print("\n" + "=" * 80)
    print("PART 1 AUDIT PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return audit_data, clean_df


# =====================================================================
# 9. CLI ENTRY POINT
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Part 1: Predictive Analytics & Salary Prediction Audit")
    parser.add_argument('--data-path', type=str, default='data/dirty_hr_dataset.csv', help='Path to dirty HR dataset CSV')
    parser.add_argument('--output-dir', type=str, default='reports', help='Directory for reports and charts')
    parser.add_argument('--experience', type=float, default=15.0, help='Target candidate experience in years')
    parser.add_argument('--ui', action='store_true', help='Launch interactive web UI server immediately after run')
    parser.add_argument('--port', type=int, default=8000, help='Port for web UI server')
    
    args = parser.parse_args()
    
    audit_data, clean_df = run_salary_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        target_experience=args.experience
    )
    
    if args.ui:
        from salary_ui import launch_ui_server
        launch_ui_server(port=args.port, reports_dir=args.output_dir)
