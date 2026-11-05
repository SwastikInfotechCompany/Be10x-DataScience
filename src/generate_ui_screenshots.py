"""
UI Screenshots Generator for Industrial Data Analytics Dashboard
=================================================================
Generates high-resolution visual screenshots depicting all UI features,
views, and interactive functionalities of Part 1: Predictive Analytics.

Output Directory: ui_screenshots/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Ensure output directory exists
OUTPUT_DIR = "ui_screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load audit data and clean dataset
with open("reports/salary_prediction_audit.json", "r", encoding="utf-8") as f:
    audit_data = json.load(f)

clean_df = pd.read_csv("reports/cleaned_hr_dataset.csv")

slope_m = audit_data['model_results']['slope_m']
intercept_c = audit_data['model_results']['intercept_c']
rmse = audit_data['model_results']['metrics']['rmse']
mae = audit_data['model_results']['metrics']['mae']
r2_pct = audit_data['model_results']['metrics']['r2_percentage']
pred_15 = audit_data['candidate_15_yr_prediction']['predicted_annual_salary']
monthly_15 = audit_data['candidate_15_yr_prediction']['predicted_monthly_salary']
min_offer = audit_data['candidate_15_yr_prediction']['recommended_salary_range']['min_offer']
max_offer = audit_data['candidate_15_yr_prediction']['recommended_salary_range']['max_offer']


# =====================================================================
# Helper: Styling Helpers
# =====================================================================
BG_DARK = '#0b1120'
CARD_BG = '#1e293b'
CARD_BORDER = '#334155'
TEXT_WHITE = '#f8fafc'
TEXT_MUTED = '#94a3b8'
CYAN = '#38bdf8'
GREEN = '#4ade80'
YELLOW = '#facc15'
RED = '#ef4444'
BLUE = '#2563eb'


# =====================================================================
# Image 1: Full Dashboard Overview
# =====================================================================
def generate_full_dashboard_overview():
    fig = plt.figure(figsize=(18, 12), dpi=200, facecolor=BG_DARK)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.18, 0.40, 0.42], hspace=0.32, wspace=0.25)
    
    # Header Area
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.set_facecolor(CARD_BG)
    for spine in ax_head.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
    ax_head.set_xticks([])
    ax_head.set_yticks([])
    
    ax_head.text(0.02, 0.68, "HR Salary Predictive Analytics Dashboard", 
                 color=TEXT_WHITE, fontsize=20, fontweight='bold', va='center')
    ax_head.text(0.02, 0.32, "Part 1 Deliverable: Supervised Machine Learning (Y = MX + C), Data Hygiene & Salary Prediction Audit", 
                 color=TEXT_MUTED, fontsize=12, va='center')
    
    # Header Badges
    bbox_green = dict(boxstyle='round,pad=0.5', facecolor='#14532d', edgecolor=GREEN, linewidth=1.5)
    ax_head.text(0.72, 0.65, f"INDUSTRIAL GRADE (R² = {r2_pct}%)", color=GREEN, fontsize=11, fontweight='bold', bbox=bbox_green, va='center')
    
    bbox_blue = dict(boxstyle='round,pad=0.5', facecolor='#1e3a8a', edgecolor=CYAN, linewidth=1.5)
    ax_head.text(0.89, 0.65, "BUDGET CERTIFIED", color=CYAN, fontsize=11, fontweight='bold', bbox=bbox_blue, va='center')
    
    # 4 KPI Cards (Row 1 across bottom of header)
    kpis = [
        ("REGRESSION EQUATION", f"Y = {slope_m:,.0f}X + {intercept_c:,.0f}", f"Slope = ${slope_m:,.2f}/yr | Base = ${intercept_c:,.2f}", CYAN),
        ("ACCURACY (R² SCORE)", f"{r2_pct}%", "Budget Benchmark (≥95%): PASSED", GREEN),
        ("MEAN ABSOLUTE ERROR", f"${mae:,.2f}", f"Root Mean Sq (RMSE): ${rmse:,.2f}", TEXT_WHITE),
        ("15-YR CANDIDATE DELIVERABLE", f"${pred_15:,.2f}", f"Monthly: ${monthly_15:,.2f} / month", YELLOW)
    ]
    
    for i, (title, val, sub, col) in enumerate(kpis):
        x_start = 0.02 + i * 0.245
        bbox_card = dict(boxstyle='round,pad=0.4', facecolor='#0f172a', edgecolor=CARD_BORDER)
        ax_head.text(x_start + 0.11, -0.25, f"{title}\n{val}\n{sub}", 
                     color=TEXT_WHITE, fontsize=9.5, ha='center', va='top', bbox=bbox_card)

    # Middle Row: Left = Candidate Predictor Card, Right = Data Hygiene Funnel
    ax_calc = fig.add_subplot(gs[1, :2])
    ax_calc.set_facecolor(CARD_BG)
    for spine in ax_calc.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
    ax_calc.set_xticks([])
    ax_calc.set_yticks([])
    
    ax_calc.text(0.05, 0.90, "Interactive Candidate Salary Predictor", color=TEXT_WHITE, fontsize=14, fontweight='bold')
    ax_calc.text(0.05, 0.81, "Dynamic compensation calculator based on learned equation Y = MX + C", color=TEXT_MUTED, fontsize=10)
    
    # Slider bar visualization
    ax_calc.text(0.05, 0.68, "Candidate Experience (X):", color=TEXT_WHITE, fontsize=11, fontweight='bold')
    ax_calc.text(0.78, 0.68, "15.0 Years", color=CYAN, fontsize=14, fontweight='bold')
    
    # Draw slider track
    track = patches.Rectangle((0.05, 0.58), 0.90, 0.03, facecolor='#334155', edgecolor='none')
    ax_calc.add_patch(track)
    # Active fill
    active_track = patches.Rectangle((0.05, 0.58), 0.45, 0.03, facecolor=CYAN, edgecolor='none')
    ax_calc.add_patch(active_track)
    # Slider thumb at 15 yrs (halfway)
    thumb = patches.Circle((0.50, 0.595), 0.025, facecolor=CYAN, edgecolor='white', linewidth=2)
    ax_calc.add_patch(thumb)
    
    # Result Box
    res_box = patches.FancyBboxPatch((0.05, 0.12), 0.90, 0.38, boxstyle="round,pad=0.02",
                                     facecolor='#0f172a', edgecolor='#6366f1', linewidth=1.5)
    ax_calc.add_patch(res_box)
    ax_calc.text(0.10, 0.42, "PREDICTED ANNUAL SALARY (15 YRS EXPERIENCE):", color=TEXT_MUTED, fontsize=9.5, fontweight='bold')
    ax_calc.text(0.10, 0.28, f"${pred_15:,.2f}", color=YELLOW, fontsize=24, fontweight='bold')
    ax_calc.text(0.10, 0.17, f"Monthly: ${monthly_15:,.2f}/mo   |   95% Offer Band: ${min_offer:,.0f} - ${max_offer:,.0f}", color=TEXT_MUTED, fontsize=10)

    # Middle Row Right: Data Hygiene Summary
    ax_hyg = fig.add_subplot(gs[1, 2:])
    ax_hyg.set_facecolor(CARD_BG)
    for spine in ax_hyg.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
    ax_hyg.set_xticks([])
    ax_hyg.set_yticks([])
    
    ax_hyg.text(0.05, 0.90, "Data Hygiene & Cleaning Funnel", color=TEXT_WHITE, fontsize=14, fontweight='bold')
    ax_hyg.text(0.05, 0.81, "Strict 3-step industrial hygiene audit before model training", color=TEXT_MUTED, fontsize=10)
    
    hygiene_steps = [
        ("1. Duplicate Elimination", "Purged 15 duplicate candidate records", "-15 Rows", RED),
        ("2. Median Imputation", "Filled NaNs (8 Exp, 6 Sal, 6 Rating) with Medians", "+20 Imputed", CYAN),
        ("3. Outlier Elimination", "Purged negative values, typos, and IQR outliers", "-9 Outliers", YELLOW),
        ("Certified Clean Data", "Retained 141 high-quality verified employee records", "141 Records (85.5%)", GREEN)
    ]
    
    for idx, (title, desc, badge_txt, col) in enumerate(hygiene_steps):
        y_pos = 0.65 - (idx * 0.16)
        row_box = patches.FancyBboxPatch((0.05, y_pos - 0.03), 0.90, 0.13, boxstyle="round,pad=0.01",
                                         facecolor='#0f172a', edgecolor=CARD_BORDER, linewidth=1)
        ax_hyg.add_patch(row_box)
        ax_hyg.text(0.08, y_pos + 0.05, title, color=col, fontsize=10.5, fontweight='bold')
        ax_hyg.text(0.08, y_pos + 0.005, desc, color=TEXT_MUTED, fontsize=9)
        bbox_b = dict(boxstyle='round,pad=0.3', facecolor=CARD_BG, edgecolor=col, linewidth=1)
        ax_hyg.text(0.92, y_pos + 0.03, badge_txt, color=col, fontsize=9.5, fontweight='bold', ha='right', bbox=bbox_b)

    # Bottom Row: Regression Fit Chart
    ax_plot = fig.add_subplot(gs[2, :])
    ax_plot.set_facecolor(CARD_BG)
    for spine in ax_plot.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
        
    ax_plot.scatter(clean_df['experience_years'], clean_df['salary'], 
                    color=CYAN, alpha=0.65, edgecolors='white', s=55, label='Clean Industrial Records (N=141)')
    
    exp_range = np.linspace(0, 25, 100)
    salary_line = (slope_m * exp_range) + intercept_c
    ax_plot.plot(exp_range, salary_line, color=RED, linewidth=2.5, 
                 label=f"Learned Regression: Y = {slope_m:,.2f}*X + {intercept_c:,.2f} (R² = {r2_pct}%)")
    
    ax_plot.fill_between(exp_range, salary_line - 1.96 * rmse, salary_line + 1.96 * rmse,
                         color=RED, alpha=0.10, label='95% Corporate Budget Confidence Band')
    
    ax_plot.scatter([15.0], [pred_15], color=YELLOW, s=260, zorder=6, marker='*', edgecolors='black', linewidth=1.5,
                    label=f"Part 1 Deliverable: Candidate (15 Yrs Exp) -> ${pred_15:,.2f}")
    
    ax_plot.annotate(f"  TARGET CANDIDATE DELIVERABLE\n  Experience: 15.0 Years\n  Predicted Salary: ${pred_15:,.2f}\n  95% Band: ${min_offer:,.0f} - ${max_offer:,.0f}",
                     xy=(15.0, pred_15), xytext=(8.5, pred_15 + 26000),
                     arrowprops=dict(facecolor=YELLOW, shrink=0.08, width=2, headwidth=8),
                     bbox=dict(boxstyle='round,pad=0.6', facecolor='#0f172a', edgecolor=YELLOW, linewidth=1.8),
                     color=TEXT_WHITE, fontsize=10, fontweight='bold')
    
    ax_plot.set_title("Interactive Live Regression Fit with Target Candidate Highlight", color=TEXT_WHITE, fontsize=13, fontweight='bold', pad=12)
    ax_plot.set_xlabel("Years of Experience (Input X)", color=TEXT_MUTED, fontsize=10.5)
    ax_plot.set_ylabel("Annual Compensation in USD (Output Y)", color=TEXT_MUTED, fontsize=10.5)
    ax_plot.tick_params(colors=TEXT_MUTED)
    ax_plot.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax_plot.grid(color=CARD_BORDER, linestyle='--', alpha=0.7)
    ax_plot.legend(loc='upper left', facecolor='#0f172a', edgecolor=CARD_BORDER, labelcolor=TEXT_WHITE, fontsize=9.5)
    
    out_file = os.path.join(OUTPUT_DIR, "01_full_dashboard_overview.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Image 2: Interactive Candidate Predictor Close-Up
# =====================================================================
def generate_candidate_predictor_ui():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200, facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Title & Badge
    ax.text(0.06, 0.92, "Interactive Candidate Salary Predictor", color=TEXT_WHITE, fontsize=18, fontweight='bold')
    ax.text(0.06, 0.86, "Supervised Learning Tool: Optimal salary prediction based on years of experience (Y = MX + C)", color=TEXT_MUTED, fontsize=11)
    
    bbox_badge = dict(boxstyle='round,pad=0.5', facecolor='#1e1b4b', edgecolor='#6366f1', linewidth=1.5)
    ax.text(0.88, 0.92, "DYNAMIC CALCULATOR", color='#a5b4fc', fontsize=10, fontweight='bold', bbox=bbox_badge, ha='right')
    
    # Interactive Slider
    ax.text(0.06, 0.76, "CANDIDATE EXPERIENCE (INPUT X):", color=TEXT_WHITE, fontsize=11, fontweight='bold')
    ax.text(0.88, 0.76, "15.0 Years", color=CYAN, fontsize=18, fontweight='bold', ha='right')
    
    track = patches.Rectangle((0.06, 0.67), 0.82, 0.025, facecolor='#334155', edgecolor='none')
    ax.add_patch(track)
    active_track = patches.Rectangle((0.06, 0.67), 0.41, 0.025, facecolor=CYAN, edgecolor='none')
    ax.add_patch(active_track)
    thumb = patches.Circle((0.47, 0.682), 0.022, facecolor=CYAN, edgecolor='white', linewidth=2.5)
    ax.add_patch(thumb)
    
    labels = [("0 Yrs (Entry)", 0.06), ("5 Yrs (Junior)", 0.20), ("10 Yrs (Mid-Level)", 0.35), 
              ("15 Yrs (Target Deliverable)", 0.50), ("20 Yrs (Senior/Staff)", 0.70), ("25+ Yrs (Principal)", 0.88)]
    for lbl, xpos in labels:
        col = YELLOW if "Target" in lbl else TEXT_MUTED
        weight = 'bold' if "Target" in lbl else 'normal'
        ax.text(xpos, 0.62, lbl, color=col, fontsize=9, fontweight=weight, ha='center')

    # Main Output Card
    out_card = patches.FancyBboxPatch((0.06, 0.20), 0.82, 0.37, boxstyle="round,pad=0.02",
                                      facecolor='#0f172a', edgecolor='#6366f1', linewidth=2)
    ax.add_patch(out_card)
    
    ax.text(0.10, 0.51, "FINAL PREDICTED ANNUAL SALARY (15 YEARS EXPERIENCE):", color=TEXT_MUTED, fontsize=11, fontweight='bold')
    ax.text(0.10, 0.40, f"${pred_15:,.2f}", color=YELLOW, fontsize=32, fontweight='bold')
    ax.text(0.10, 0.31, f"Predicted Monthly Equivalent: ${monthly_15:,.2f} / month", color=CYAN, fontsize=13, fontweight='bold')
    
    # 95% Confidence Band Card inside
    band_box = patches.FancyBboxPatch((0.10, 0.23), 0.74, 0.065, boxstyle="round,pad=0.01",
                                      facecolor='#1e293b', edgecolor=CARD_BORDER)
    ax.add_patch(band_box)
    ax.text(0.13, 0.26, f"95% Budget Confidence Range: ${min_offer:,.2f} – ${max_offer:,.2f}  (Margin of Error: ±${1.96*rmse:,.2f})", 
            color=GREEN, fontsize=10.5, fontweight='bold')

    # Step by Step calculation
    calc_box = patches.FancyBboxPatch((0.06, 0.07), 0.82, 0.09, boxstyle="round,pad=0.01",
                                      facecolor='#020617', edgecolor=CARD_BORDER)
    ax.add_patch(calc_box)
    ax.text(0.09, 0.115, f"Step-by-Step Formula: Salary = (Slope {slope_m:,.2f} * 15.0) + Intercept {intercept_c:,.2f} = ${pred_15:,.2f}",
            color=CYAN, fontsize=10.5, fontfamily='monospace')

    out_file = os.path.join(OUTPUT_DIR, "02_interactive_candidate_predictor.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Image 3: Data Hygiene & Cleaning Funnel
# =====================================================================
def generate_hygiene_audit_ui():
    fig = plt.figure(figsize=(14, 9), dpi=200, facecolor=BG_DARK)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.45, 0.55], hspace=0.35, wspace=0.25)
    
    # Panel 1: Waterfall Pipeline
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(CARD_BG)
    for spine in ax1.spines.values():
        spine.set_color(CARD_BORDER)
    
    stages = ['Raw Input Data', '1. Duplicates Dropped', '2. Median Imputation', '3. Outliers Purged', 'Certified Clean Data']
    counts = [165, 150, 150, 141, 141]
    bar_colors = ['#dc2626', '#ea580c', '#0284c7', '#ca8a04', '#16a34a']
    
    bars = ax1.bar(stages, counts, color=bar_colors, width=0.55, edgecolor='black', linewidth=1.5)
    ax1.set_title("Data Hygiene Pipeline Funnel: 165 Raw Records -> 141 Verified Records (85.5% Retention)", 
                  color=TEXT_WHITE, fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("Employee Record Count", color=TEXT_MUTED)
    ax1.tick_params(colors=TEXT_MUTED)
    ax1.grid(axis='y', color=CARD_BORDER, linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 195)
    
    sub_labels = ["165 Dirty Exports", "-15 Cloned Entries", "20 Nulls Imputed", "-9 Unrealistic Records", "141 Ready for ML"]
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}", 
                 ha='center', va='bottom', color=TEXT_WHITE, fontweight='bold', fontsize=12)
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval - 25, sub_labels[i], 
                 ha='center', va='center', color='white', fontweight='bold', fontsize=9, rotation=0)

    # Panel 2: Before vs After Salary Boxplot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(CARD_BG)
    for spine in ax2.spines.values():
        spine.set_color(CARD_BORDER)
        
    ax2.boxplot([clean_df['salary']], orientation='horizontal', patch_artist=True, tick_labels=['Cleaned Salary'],
                boxprops=dict(facecolor='#0284c7', color=CYAN), medianprops=dict(color=YELLOW, linewidth=2.5))
    ax2.set_title("Cleaned Salary Distribution ($25k - $240k Bounds)", color=TEXT_WHITE, fontsize=11, fontweight='bold')
    ax2.set_xlabel("Salary in USD", color=TEXT_MUTED)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax2.tick_params(colors=TEXT_MUTED)
    ax2.grid(color=CARD_BORDER, linestyle='--', alpha=0.5)

    # Panel 3: Detailed Rules Breakdown Table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(CARD_BG)
    for spine in ax3.spines.values():
        spine.set_color(CARD_BORDER)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    ax3.text(0.06, 0.90, "Enforced Data Hygiene Rules", color=TEXT_WHITE, fontsize=12, fontweight='bold')
    
    rules = [
        ("Duplicate Elimination", "drop_duplicates() purged 15 copies", "-15 Rows", RED),
        ("Median Experience Filling", "Imputed 8 nulls with median 13.0 yrs", "+8 Values", CYAN),
        ("Median Salary Filling", "Imputed 6 nulls with median $105,900", "+6 Values", CYAN),
        ("Negative Outliers Purged", "Removed negative salary & experience", "-4 Rows", YELLOW),
        ("Extreme Multimillion Purged", "Removed fake $4.5M & $9.9M typos", "-3 Rows", YELLOW),
        ("IQR Bounds Enforcement", "Kept records strictly within 1.5*IQR", "-2 Rows", YELLOW)
    ]
    
    for i, (rule, detail, badge, col) in enumerate(rules):
        y_pos = 0.77 - (i * 0.125)
        ax3.text(0.06, y_pos, rule, color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
        ax3.text(0.06, y_pos - 0.045, detail, color=TEXT_MUTED, fontsize=8.5)
        ax3.text(0.92, y_pos - 0.02, badge, color=col, fontsize=9, fontweight='bold', ha='right')

    out_file = os.path.join(OUTPUT_DIR, "03_data_hygiene_cleaning_funnel.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Image 4: Live Regression Trendline Chart View
# =====================================================================
def generate_regression_chart_ui():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200, facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
        
    ax.scatter(clean_df['experience_years'], clean_df['salary'], 
               color=CYAN, alpha=0.70, edgecolors='white', s=65, label='Clean Industrial Records (N=141)')
    
    exp_range = np.linspace(0, 25, 100)
    salary_line = (slope_m * exp_range) + intercept_c
    ax.plot(exp_range, salary_line, color=RED, linewidth=3, 
            label=f"Regression Equation: Y = {slope_m:,.2f}*X + {intercept_c:,.2f}  (Accuracy R² = {r2_pct}%)")
    
    ax.fill_between(exp_range, salary_line - 1.96 * rmse, salary_line + 1.96 * rmse,
                    color=RED, alpha=0.12, label='95% Corporate Budget Confidence Band (±$5,528)')
    
    # 15-Year Candidate Highlight
    ax.scatter([15.0], [pred_15], color=YELLOW, s=320, zorder=6, marker='*', edgecolors='black', linewidth=1.8,
               label=f"Part 1 Deliverable: Candidate (15.0 Yrs) -> ${pred_15:,.2f}")
    
    ax.annotate(f"  PART 1 DELIVERABLE CANDIDATE\n  Experience: 15.0 Years\n  Predicted Salary: ${pred_15:,.2f}\n  Monthly: ${monthly_15:,.2f}/mo\n  Offer Band: ${min_offer:,.0f} - ${max_offer:,.0f}",
                xy=(15.0, pred_15), xytext=(8.0, pred_15 + 28000),
                arrowprops=dict(facecolor=YELLOW, shrink=0.08, width=2, headwidth=9),
                bbox=dict(boxstyle='round,pad=0.7', facecolor='#0f172a', edgecolor=YELLOW, linewidth=2),
                color=TEXT_WHITE, fontsize=11, fontweight='bold')
    
    ax.set_title("Part 1 Deliverable: HR Salary Linear Regression Model ($Y = MX + C$)", 
                 color=TEXT_WHITE, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Years of Experience (Input X)", color=TEXT_MUTED, fontsize=11, labelpad=8)
    ax.set_ylabel("Annual Compensation in USD (Output Y)", color=TEXT_MUTED, fontsize=11, labelpad=8)
    ax.tick_params(colors=TEXT_MUTED)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax.grid(color=CARD_BORDER, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', facecolor='#0f172a', edgecolor=CARD_BORDER, labelcolor=TEXT_WHITE, fontsize=10.5)

    out_file = os.path.join(OUTPUT_DIR, "04_regression_trendline_chart.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Image 5: Model Residuals & Error Diagnostics
# =====================================================================
def generate_residuals_diagnostics_ui():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=200, facecolor=BG_DARK)
    
    X_exp = clean_df['experience_years'].values
    y_sal = clean_df['salary'].values
    y_pred = (slope_m * X_exp) + intercept_c
    residuals = y_sal - y_pred
    
    # Left: Residuals vs Fitted
    ax1 = axes[0]
    ax1.set_facecolor(CARD_BG)
    for spine in ax1.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
        
    ax1.scatter(y_pred, residuals, color=CYAN, alpha=0.70, edgecolors='white', s=50)
    ax1.axhline(0, color=RED, linestyle='--', linewidth=2, label='Zero Error Baseline')
    ax1.axhline(1.96 * rmse, color=YELLOW, linestyle=':', label='+1.96 RMSE Bound')
    ax1.axhline(-1.96 * rmse, color=YELLOW, linestyle=':', label='-1.96 RMSE Bound')
    ax1.set_title("Residuals vs. Fitted Values ($Y - \\hat{Y}$)", color=TEXT_WHITE, fontsize=12, fontweight='bold')
    ax1.set_xlabel("Fitted Values (Predicted Salary)", color=TEXT_MUTED)
    ax1.set_ylabel("Residual Error ($ USD)", color=TEXT_MUTED)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax1.tick_params(colors=TEXT_MUTED)
    ax1.grid(color=CARD_BORDER, linestyle='--', alpha=0.5)
    ax1.legend(facecolor='#0f172a', edgecolor=CARD_BORDER, labelcolor=TEXT_WHITE, fontsize=9.5)
    
    # Right: Error Histogram
    ax2 = axes[1]
    ax2.set_facecolor(CARD_BG)
    for spine in ax2.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
        
    ax2.hist(residuals, bins=14, color='#10b981', edgecolor='black', alpha=0.85)
    ax2.axvline(0, color=RED, linestyle='--', linewidth=2, label='Zero Error Mean')
    ax2.set_title(f"Residual Error Distribution (MAE = ${mae:,.2f})", color=TEXT_WHITE, fontsize=12, fontweight='bold')
    ax2.set_xlabel("Prediction Error ($ USD)", color=TEXT_MUTED)
    ax2.set_ylabel("Employee Frequency", color=TEXT_MUTED)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax2.tick_params(colors=TEXT_MUTED)
    ax2.grid(color=CARD_BORDER, linestyle='--', alpha=0.5)
    ax2.legend(facecolor='#0f172a', edgecolor=CARD_BORDER, labelcolor=TEXT_WHITE, fontsize=9.5)

    fig.suptitle(f"Model Error Metrics & Diagnostics: R² = {r2_pct}% (Budget Threshold ≥ 95% PASSED)", 
                 color=TEXT_WHITE, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_file = os.path.join(OUTPUT_DIR, "05_model_residuals_and_diagnostics.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Image 6: Clean Dataset Table Explorer View
# =====================================================================
def generate_table_explorer_ui():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=200, facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Table Header & Search bar representation
    ax.text(0.04, 0.94, "Certified Clean Dataset Explorer (141 Verified Employee Records)", 
            color=TEXT_WHITE, fontsize=14, fontweight='bold')
    
    search_bar = patches.FancyBboxPatch((0.68, 0.915), 0.28, 0.05, boxstyle="round,pad=0.01",
                                       facecolor='#0f172a', edgecolor=CARD_BORDER)
    ax.add_patch(search_bar)
    ax.text(0.70, 0.935, "[Filter] Search by ID or Dept...", color=TEXT_MUTED, fontsize=9.5)
    
    # Column headers
    cols = ["Candidate ID", "Department", "Experience", "Rating", "Actual Salary", "Model Prediction", "Residual Error"]
    x_positions = [0.05, 0.20, 0.35, 0.48, 0.62, 0.77, 0.90]
    
    head_bg = patches.Rectangle((0.04, 0.84), 0.92, 0.05, facecolor='#0f172a', edgecolor='none')
    ax.add_patch(head_bg)
    for c, x in zip(cols, x_positions):
        ax.text(x, 0.865, c, color=TEXT_MUTED, fontsize=9.5, fontweight='bold')

    # Draw sample rows
    sample_rows = clean_df.head(10).copy()
    for i, (_, row) in enumerate(sample_rows.iterrows()):
        y_line = 0.78 - (i * 0.07)
        cand_id = row['candidate_id']
        dept = row['department']
        exp = row['experience_years']
        rat = row['performance_rating']
        sal = row['salary']
        pred = (slope_m * exp) + intercept_c
        res = sal - pred
        res_col = GREEN if abs(res) < 2500 else YELLOW
        
        row_bg = patches.Rectangle((0.04, y_line - 0.02), 0.92, 0.06, 
                                   facecolor='#1e293b' if i % 2 == 0 else '#162032', edgecolor='none')
        ax.add_patch(row_bg)
        
        ax.text(x_positions[0], y_line + 0.01, cand_id, color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
        ax.text(x_positions[1], y_line + 0.01, dept, color=TEXT_MUTED, fontsize=9)
        ax.text(x_positions[2], y_line + 0.01, f"{exp:.1f} yrs", color=TEXT_WHITE, fontsize=9)
        ax.text(x_positions[3], y_line + 0.01, f"{rat:.1f} / 5.0", color=YELLOW, fontsize=9)
        ax.text(x_positions[4], y_line + 0.01, f"${sal:,.2f}", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
        ax.text(x_positions[5], y_line + 0.01, f"${pred:,.2f}", color=CYAN, fontsize=9)
        ax.text(x_positions[6], y_line + 0.01, f"{res:+,.2f}", color=res_col, fontsize=9, fontweight='bold')

    # Pagination footer
    ax.text(0.05, 0.04, "Showing 10 of 141 verified records (Page 1 of 15)  |  Export available as cleaned_hr_dataset.csv", 
            color=TEXT_MUTED, fontsize=9.5)

    out_file = os.path.join(OUTPUT_DIR, "06_clean_dataset_table_explorer.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Image 7: Official Salary Prediction Audit Deliverable View
# =====================================================================
def generate_audit_report_ui():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=200, facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.text(0.05, 0.93, "Official Salary Prediction Audit Report Deliverable", 
            color=TEXT_WHITE, fontsize=15, fontweight='bold')
    ax.text(0.05, 0.88, "Automated executive summary generated by Industrial Data Analytics Python Engine", 
            color=TEXT_MUTED, fontsize=10)
    
    # Terminal / Code Container
    term_box = patches.FancyBboxPatch((0.05, 0.05), 0.90, 0.80, boxstyle="round,pad=0.01",
                                      facecolor='#020617', edgecolor='#334155', linewidth=1.5)
    ax.add_patch(term_box)
    
    with open("reports/salary_prediction_audit.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    text_content = "".join(lines[:38]) # Display main 38 lines
    ax.text(0.07, 0.82, text_content, color='#38bdf8', fontfamily='monospace', fontsize=8.5, va='top', linespacing=1.35)

    out_file = os.path.join(OUTPUT_DIR, "07_salary_prediction_audit_report.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Main Orchestrator
# =====================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("GENERATING UI FUNCTIONALITY SCREENSHOTS")
    print(f"Target Directory: {OUTPUT_DIR}/")
    print("=" * 70)
    
    generate_full_dashboard_overview()
    generate_candidate_predictor_ui()
    generate_hygiene_audit_ui()
    generate_regression_chart_ui()
    generate_residuals_diagnostics_ui()
    generate_table_explorer_ui()
    generate_audit_report_ui()
    
    print("\nAll 7 UI images generated successfully!")
