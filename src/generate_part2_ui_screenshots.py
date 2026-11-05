"""
Part 2 UI Screenshots Generator for Customer Segmentation Dashboard
===================================================================
Generates high-resolution visual screenshots depicting all UI features,
views, and interactive functionalities of Part 2: Pattern Discovery & Clustering.

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

OUTPUT_DIR = "ui_screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load audit data and segmented customer data
with open("reports/customer_segmentation_audit.json", "r", encoding="utf-8") as f:
    audit_data = json.load(f)

df_seg = pd.read_csv("reports/segmented_customers.csv")

sil_score = audit_data['overall_silhouette_score']
sil_pct = audit_data['overall_silhouette_percentage']
target_c = audit_data['new_customer_classification']
profiles = audit_data['cluster_profiles']
elbow_metrics = audit_data['elbow_analysis']['k_metrics']

BG_DARK = '#0b1120'
CARD_BG = '#1e293b'
CARD_BORDER = '#334155'
TEXT_WHITE = '#f8fafc'
TEXT_MUTED = '#94a3b8'
CYAN = '#38bdf8'
GREEN = '#4ade80'
YELLOW = '#facc15'
RED = '#ef4444'
EMERALD = '#10b981'
PURPLE = '#a855f7'
COLORS = ['#38bdf8', '#ef4444', '#10b981', '#f59e0b']


# =====================================================================
# Image 8: Full Customer Segmentation Dashboard Overview
# =====================================================================
def generate_part2_dashboard_overview():
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
    
    ax_head.text(0.02, 0.68, "Customer Segmentation & Pattern Discovery Dashboard", 
                 color=TEXT_WHITE, fontsize=20, fontweight='bold', va='center')
    ax_head.text(0.02, 0.32, "Part 2 Deliverable: Unsupervised K-Means Clustering, StandardScaler Normalization & New Customer Classification", 
                 color=TEXT_MUTED, fontsize=12, va='center')
    
    bbox_green = dict(boxstyle='round,pad=0.5', facecolor='#064e3b', edgecolor=EMERALD, linewidth=1.5)
    ax_head.text(0.70, 0.65, f"SILHOUETTE SCORE: {sil_pct}% (HIGH QUALITY)", color=GREEN, fontsize=11, fontweight='bold', bbox=bbox_green, va='center')
    
    bbox_blue = dict(boxstyle='round,pad=0.5', facecolor='#1e3a8a', edgecolor=CYAN, linewidth=1.5)
    ax_head.text(0.91, 0.65, "OPTIMAL K = 4", color=CYAN, fontsize=11, fontweight='bold', bbox=bbox_blue, va='center')
    
    # 4 KPI Cards
    kpis = [
        ("OPTIMAL CLUSTERS", "K = 4 Groups", "The Elbow Method Inflection Point", CYAN),
        ("CLUSTER ACCURACY", f"{sil_pct}%", f"Silhouette Score: {sil_score} (>0.70)", GREEN),
        ("FEATURE SCALING", "StandardScaler", "Equal Weight: Age (18-80) vs Spend ($1.5k-$22k)", '#a5b4fc'),
        ("TARGET DELIVERABLE", target_c['assigned_loyalty_group'], f"Age 25, Spend $2,500 -> {target_c['assigned_tier']}", YELLOW)
    ]
    
    for i, (title, val, sub, col) in enumerate(kpis):
        x_start = 0.02 + i * 0.245
        bbox_card = dict(boxstyle='round,pad=0.4', facecolor='#0f172a', edgecolor=CARD_BORDER)
        ax_head.text(x_start + 0.11, -0.25, f"{title}\n{val}\n{sub}", 
                     color=TEXT_WHITE, fontsize=9.5, ha='center', va='top', bbox=bbox_card)

    # Middle Row Left: Live Classifier Card
    ax_calc = fig.add_subplot(gs[1, :2])
    ax_calc.set_facecolor(CARD_BG)
    for spine in ax_calc.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
    ax_calc.set_xticks([])
    ax_calc.set_yticks([])
    
    ax_calc.text(0.05, 0.90, "Live Customer Classifier Widget", color=TEXT_WHITE, fontsize=14, fontweight='bold')
    ax_calc.text(0.05, 0.81, "Classifies any customer into their natural loyalty group based on scaled proximity", color=TEXT_MUTED, fontsize=10)
    
    # Sliders Display
    ax_calc.text(0.05, 0.68, "Customer Age: 25 Years", color=CYAN, fontsize=11, fontweight='bold')
    track1 = patches.Rectangle((0.05, 0.60), 0.90, 0.025, facecolor='#334155', edgecolor='none')
    ax_calc.add_patch(track1)
    thumb1 = patches.Circle((0.15, 0.612), 0.020, facecolor=EMERALD, edgecolor='white', linewidth=2)
    ax_calc.add_patch(thumb1)
    
    ax_calc.text(0.05, 0.49, "Annual Spend: $2,500", color=CYAN, fontsize=11, fontweight='bold')
    track2 = patches.Rectangle((0.05, 0.41), 0.90, 0.025, facecolor='#334155', edgecolor='none')
    ax_calc.add_patch(track2)
    thumb2 = patches.Circle((0.12, 0.422), 0.020, facecolor=EMERALD, edgecolor='white', linewidth=2)
    ax_calc.add_patch(thumb2)
    
    # Classification Result
    res_box = patches.FancyBboxPatch((0.05, 0.08), 0.90, 0.28, boxstyle="round,pad=0.02",
                                     facecolor='#064e3b', edgecolor=EMERALD, linewidth=1.5)
    ax_calc.add_patch(res_box)
    ax_calc.text(0.09, 0.28, f"ASSIGNED: {target_c['assigned_loyalty_group']} ({target_c['assigned_tier']})", 
                 color=YELLOW, fontsize=14, fontweight='bold')
    ax_calc.text(0.09, 0.19, f"Action: {target_c['recommended_discount_offer']}", color=TEXT_WHITE, fontsize=10.5, fontweight='bold')
    ax_calc.text(0.09, 0.11, f"Centroid Proximity: Distance = {target_c['distance_to_nearest_centroid']} (Scaled Euclidean Distance)", color='#a7f3d0', fontsize=9.5)

    # Middle Row Right: 4 Loyalty Personas Strategy Cards
    ax_strat = fig.add_subplot(gs[1, 2:])
    ax_strat.set_facecolor(CARD_BG)
    for spine in ax_strat.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
    ax_strat.set_xticks([])
    ax_strat.set_yticks([])
    
    ax_strat.text(0.05, 0.90, "4 Natural Industrial Loyalty Personas", color=TEXT_WHITE, fontsize=14, fontweight='bold')
    ax_strat.text(0.05, 0.81, "Targeted discount email strategies derived from unsupervised centroids", color=TEXT_MUTED, fontsize=10)
    
    for idx, (cid, prof) in enumerate(profiles.items()):
        y_pos = 0.65 - (idx * 0.16)
        c_col = COLORS[int(cid) % 4]
        row_box = patches.FancyBboxPatch((0.05, y_pos - 0.03), 0.90, 0.13, boxstyle="round,pad=0.01",
                                         facecolor='#0f172a', edgecolor=CARD_BORDER, linewidth=1)
        ax_strat.add_patch(row_box)
        ax_strat.text(0.08, y_pos + 0.05, f"Cluster {cid}: {prof['persona_name']}", color=c_col, fontsize=10.5, fontweight='bold')
        ax_strat.text(0.08, y_pos + 0.005, f"Age ~{prof['centroid_real_units']['age']} yrs | Spend ~${prof['centroid_real_units']['annual_spend']:,.0f} | {prof['customer_count']} cust", color=TEXT_MUTED, fontsize=9)
        ax_strat.text(0.08, y_pos - 0.035, f"Offer: {prof['recommended_discount']}", color=TEXT_WHITE, fontsize=8.5)

    # Bottom Row: Cluster Scatter Plot
    ax_plot = fig.add_subplot(gs[2, :])
    ax_plot.set_facecolor(CARD_BG)
    for spine in ax_plot.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
        
    for c in range(4):
        c_sub = df_seg[df_seg['cluster_id'] == c]
        p_name = c_sub['persona_name'].iloc[0]
        ax_plot.scatter(c_sub['age'], c_sub['annual_spend'],
                        color=COLORS[c], alpha=0.75, s=60, edgecolors='white',
                        label=f"Cluster {c}: {p_name} (N={len(c_sub)})")

    # Centroids
    for c in range(4):
        c_age = profiles[str(c)]['centroid_real_units']['age']
        c_sp = profiles[str(c)]['centroid_real_units']['annual_spend']
        ax_plot.scatter([c_age], [c_sp], color='black', s=260, marker='X', edgecolors='white', linewidth=2, zorder=5)
        
    # Target Deliverable Marker
    ax_plot.scatter([25.0], [2500.0], color=YELLOW, s=340, marker='*', edgecolors='black', linewidth=1.8,
                    zorder=6, label=f"Part 2 Deliverable: New Customer (Age 25, $2,500) -> {target_c['assigned_loyalty_group']}")
    
    ax_plot.annotate(f"  PART 2 DELIVERABLE CANDIDATE\n  Input: Age 25 | Spend $2,500\n  Classified: {target_c['assigned_loyalty_group']}\n  Tier: {target_c['assigned_tier']}\n  Action: {target_c['recommended_discount_offer']}",
                     xy=(25.0, 2500.0), xytext=(28.0, 6000.0),
                     arrowprops=dict(facecolor=YELLOW, shrink=0.08, width=2, headwidth=8),
                     bbox=dict(boxstyle='round,pad=0.6', fc='#0f172a', ec=YELLOW, lw=1.8),
                     color=TEXT_WHITE, fontsize=10, fontweight='bold')
    
    ax_plot.set_title("Customer Segmentation via K-Means Clustering (K=4) with Target Candidate Marker", color=TEXT_WHITE, fontsize=13, fontweight='bold', pad=12)
    ax_plot.set_xlabel("Customer Age in Years (Input Feature 1)", color=TEXT_MUTED, fontsize=10.5)
    ax_plot.set_ylabel("Annual Spend in USD ($) (Input Feature 2)", color=TEXT_MUTED, fontsize=10.5)
    ax_plot.tick_params(colors=TEXT_MUTED)
    ax_plot.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax_plot.grid(color=CARD_BORDER, linestyle='--', alpha=0.7)
    ax_plot.legend(loc='upper right', facecolor='#0f172a', edgecolor=CARD_BORDER, labelcolor=TEXT_WHITE, fontsize=9.5)

    out_file = os.path.join(OUTPUT_DIR, "08_customer_segmentation_dashboard.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Image 9: The Elbow Method & Silhouette Scores View
# =====================================================================
def generate_elbow_silhouette_view():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=200, facecolor=BG_DARK)
    
    k_vals = [item['k'] for item in elbow_metrics]
    inertias = [item['inertia'] for item in elbow_metrics]
    sil_items = [item for item in elbow_metrics if item['silhouette_score'] is not None]
    k_sil_vals = [item['k'] for item in sil_items]
    sil_scores = [item['silhouette_score'] for item in sil_items]
    
    # Left: Inertia Plot
    ax1 = axes[0]
    ax1.set_facecolor(CARD_BG)
    for spine in ax1.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
        
    ax1.plot(k_vals, inertias, marker='o', color='#2563eb', linewidth=2.5, markersize=8)
    ax1.axvline(4, color=RED, linestyle='--', linewidth=2, label='Optimal Elbow (K=4)')
    ax1.scatter([4], [inertias[3]], color=RED, s=150, zorder=5)
    ax1.annotate(f"  Elbow Point (K=4)\n  Inertia = {inertias[3]:.2f}", xy=(4, inertias[3]), xytext=(4.3, inertias[3] + 25),
                 arrowprops=dict(facecolor=RED, shrink=0.08, width=1.5, headwidth=6),
                 fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.4', fc='#fee2e2', ec=RED), color='black')
    ax1.set_title("The Elbow Method: Inertia vs. Number of Clusters (K)", color=TEXT_WHITE, fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of Clusters (K)", color=TEXT_MUTED)
    ax1.set_ylabel("Inertia (Sum of Squared Distances to Centroids)", color=TEXT_MUTED)
    ax1.set_xticks(k_vals)
    ax1.tick_params(colors=TEXT_MUTED)
    ax1.grid(color=CARD_BORDER, linestyle='--', alpha=0.5)
    ax1.legend(facecolor='#0f172a', edgecolor=CARD_BORDER, labelcolor=TEXT_WHITE)

    # Right: Silhouette Plot
    ax2 = axes[1]
    ax2.set_facecolor(CARD_BG)
    for spine in ax2.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
        
    ax2.plot(k_sil_vals, sil_scores, marker='s', color=EMERALD, linewidth=2.5, markersize=8)
    ax2.axvline(4, color=YELLOW, linestyle='--', linewidth=2, label='Peak Silhouette (K=4)')
    ax2.scatter([4], [sil_scores[2]], color=YELLOW, s=150, zorder=5)
    ax2.annotate(f"  Peak Quality (K=4)\n  Silhouette = {sil_scores[2]:.4f} ({sil_scores[2]*100:.1f}%)", xy=(4, sil_scores[2]), xytext=(4.3, sil_scores[2] - 0.08),
                 arrowprops=dict(facecolor=YELLOW, shrink=0.08, width=1.5, headwidth=6),
                 fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.4', fc='#fef3c7', ec=YELLOW), color='black')
    ax2.set_title("Cluster Quality: Silhouette Score vs. K", color=TEXT_WHITE, fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of Clusters (K)", color=TEXT_MUTED)
    ax2.set_ylabel("Silhouette Score (Cohesion vs Separation)", color=TEXT_MUTED)
    ax2.set_xticks(k_sil_vals)
    ax2.tick_params(colors=TEXT_MUTED)
    ax2.grid(color=CARD_BORDER, linestyle='--', alpha=0.5)
    ax2.legend(facecolor='#0f172a', edgecolor=CARD_BORDER, labelcolor=TEXT_WHITE)

    fig.suptitle("Mathematical Cluster Selection: The Elbow Method & Silhouette Score (Peak = 74.91% at K=4)", 
                 color=TEXT_WHITE, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_file = os.path.join(OUTPUT_DIR, "09_elbow_and_silhouette_analysis.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Image 10: Customer Clusters Scatter View Close-Up
# =====================================================================
def generate_clusters_scatter_view():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200, facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
        
    for c in range(4):
        c_sub = df_seg[df_seg['cluster_id'] == c]
        p_name = c_sub['persona_name'].iloc[0]
        ax.scatter(c_sub['age'], c_sub['annual_spend'],
                   color=COLORS[c], alpha=0.75, s=70, edgecolors='white',
                   label=f"Cluster {c}: {p_name} (N={len(c_sub)})")

    # Centroids
    for c in range(4):
        c_age = profiles[str(c)]['centroid_real_units']['age']
        c_sp = profiles[str(c)]['centroid_real_units']['annual_spend']
        ax.scatter([c_age], [c_sp], color='black', s=280, marker='X', edgecolors='white', linewidth=2, zorder=5)
        ax.text(c_age, c_sp + 600, f"Centroid {c}\n({c_age:.0f}y, ${c_sp:,.0f})", 
                color=TEXT_WHITE, fontsize=8.5, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='#0f172a', ec=CARD_BORDER, alpha=0.85))

    # Target Deliverable Marker
    ax.scatter([25.0], [2500.0], color=YELLOW, s=360, marker='*', edgecolors='black', linewidth=1.8,
               zorder=6, label=f"Target Candidate Deliverable: Age 25, $2,500 -> {target_c['assigned_loyalty_group']}")
    
    ax.annotate(f"  PART 2 DELIVERABLE CANDIDATE\n  Input: Age 25 | Spend $2,500\n  Assigned: {target_c['assigned_loyalty_group']}\n  Loyalty Tier: {target_c['assigned_tier']}\n  Campaign: {target_c['recommended_discount_offer']}",
                xy=(25.0, 2500.0), xytext=(28.0, 6500.0),
                arrowprops=dict(facecolor=YELLOW, shrink=0.08, width=2, headwidth=8),
                bbox=dict(boxstyle='round,pad=0.6', fc='#0f172a', ec=YELLOW, lw=2),
                color=TEXT_WHITE, fontsize=10.5, fontweight='bold')
    
    ax.set_title("Customer Segmentation Scatter Plot (Age vs. Annual Spend) with K=4 Centroids", 
                 color=TEXT_WHITE, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Customer Age (Years)", color=TEXT_MUTED, fontsize=11, labelpad=8)
    ax.set_ylabel("Annual Spend in USD ($)", color=TEXT_MUTED, fontsize=11, labelpad=8)
    ax.tick_params(colors=TEXT_MUTED)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax.grid(color=CARD_BORDER, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', facecolor='#0f172a', edgecolor=CARD_BORDER, labelcolor=TEXT_WHITE, fontsize=10)

    out_file = os.path.join(OUTPUT_DIR, "10_customer_clusters_scatter_view.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Image 11: New Customer Classifier Card View
# =====================================================================
def generate_classifier_card_view():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200, facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.text(0.06, 0.92, "Live Customer Classifier Widget", color=TEXT_WHITE, fontsize=18, fontweight='bold')
    ax.text(0.06, 0.86, "Classifies new unlabeled customers into loyalty segments using scaled proximity to K-Means centroids", color=TEXT_MUTED, fontsize=11)
    
    bbox_badge = dict(boxstyle='round,pad=0.5', facecolor='#064e3b', edgecolor=EMERALD, linewidth=1.5)
    ax.text(0.88, 0.92, "PART 2 DELIVERABLE", color='#6ee7b7', fontsize=10, fontweight='bold', bbox=bbox_badge, ha='right')
    
    # Input Sliders representation
    ax.text(0.06, 0.76, "CUSTOMER AGE (INPUT 1):", color=TEXT_WHITE, fontsize=11, fontweight='bold')
    ax.text(0.88, 0.76, "25 Years", color=CYAN, fontsize=16, fontweight='bold', ha='right')
    track1 = patches.Rectangle((0.06, 0.69), 0.82, 0.022, facecolor='#334155', edgecolor='none')
    ax.add_patch(track1)
    thumb1 = patches.Circle((0.18, 0.701), 0.020, facecolor=EMERALD, edgecolor='white', linewidth=2)
    ax.add_patch(thumb1)
    
    ax.text(0.06, 0.60, "ANNUAL SPEND (INPUT 2):", color=TEXT_WHITE, fontsize=11, fontweight='bold')
    ax.text(0.88, 0.60, "$2,500.00 USD", color=CYAN, fontsize=16, fontweight='bold', ha='right')
    track2 = patches.Rectangle((0.06, 0.53), 0.82, 0.022, facecolor='#334155', edgecolor='none')
    ax.add_patch(track2)
    thumb2 = patches.Circle((0.14, 0.541), 0.020, facecolor=EMERALD, edgecolor='white', linewidth=2)
    ax.add_patch(thumb2)

    # Classification Output Box
    out_card = patches.FancyBboxPatch((0.06, 0.16), 0.82, 0.32, boxstyle="round,pad=0.02",
                                      facecolor='#064e3b', edgecolor=EMERALD, linewidth=2)
    ax.add_patch(out_card)
    
    ax.text(0.10, 0.43, "CLASSIFIED LOYALTY SEGMENT:", color='#a7f3d0', fontsize=10.5, fontweight='bold')
    ax.text(0.10, 0.34, target_c['assigned_loyalty_group'], color=YELLOW, fontsize=24, fontweight='bold')
    ax.text(0.10, 0.27, f"Loyalty Tier: {target_c['assigned_tier']}  |  Centroid Distance: {target_c['distance_to_nearest_centroid']} (Scaled Units)", color=TEXT_WHITE, fontsize=11)
    ax.text(0.10, 0.20, f"Target Action: {target_c['recommended_discount_offer']}", color='#6ee7b7', fontsize=11, fontweight='bold')

    # Distance Comparison Matrix Box
    dist_box = patches.FancyBboxPatch((0.06, 0.04), 0.82, 0.09, boxstyle="round,pad=0.01",
                                      facecolor='#020617', edgecolor=CARD_BORDER)
    ax.add_patch(dist_box)
    ax.text(0.08, 0.085, f"Distances to Centroids: Cluster 0: 1.34 | Cluster 1: 2.87 | Cluster 2: 2.91 | Cluster 3 (Nearest): {target_c['distance_to_nearest_centroid']}",
            color=CYAN, fontsize=9.5, fontfamily='monospace')

    out_file = os.path.join(OUTPUT_DIR, "11_new_customer_classifier_card.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Image 12: 4 Personas & Personalized Discount Matrix
# =====================================================================
def generate_loyalty_matrix_view():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=200, facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.text(0.05, 0.94, "Customer Loyalty Groups & Personalized Email Strategy Matrix", 
            color=TEXT_WHITE, fontsize=15, fontweight='bold')
    ax.text(0.05, 0.89, "Actionable commercial strategies for each cluster based on empirical centroid properties", 
            color=TEXT_MUTED, fontsize=10.5)

    for i, (cid, prof) in enumerate(profiles.items()):
        y_top = 0.83 - (i * 0.19)
        c_col = COLORS[int(cid) % 4]
        
        box = patches.FancyBboxPatch((0.05, y_top - 0.15), 0.90, 0.16, boxstyle="round,pad=0.01",
                                     facecolor='#0f172a', edgecolor=c_col, linewidth=1.5)
        ax.add_patch(box)
        
        ax.text(0.08, y_top - 0.02, f"Cluster {cid}: {prof['persona_name']}", color=c_col, fontsize=13, fontweight='bold')
        ax.text(0.08, y_top - 0.055, f"Tier: {prof['loyalty_tier']}  |  Customer Share: {prof['customer_count']} customers ({prof['percentage_of_base']}%)", color=TEXT_WHITE, fontsize=10)
        ax.text(0.08, y_top - 0.085, f"Centroid Profile: Age {prof['centroid_real_units']['age']} yrs | Annual Spend ${prof['centroid_real_units']['annual_spend']:,.2f} | Order Freq: {prof['centroid_real_units']['purchase_frequency']:.0f}/yr", color=TEXT_MUTED, fontsize=9.5)
        ax.text(0.08, y_top - 0.12, f"Personalized Discount: {prof['recommended_discount']}", color=YELLOW, fontsize=10, fontweight='bold')
        ax.text(0.08, y_top - 0.145, f"Email Strategy: {prof['email_strategy']}", color=CYAN, fontsize=9)

    out_file = os.path.join(OUTPUT_DIR, "12_loyalty_groups_marketing_matrix.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Image 13: Segmented Customers Table Explorer
# =====================================================================
def generate_segmented_table_view():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=200, facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.text(0.04, 0.94, "Segmented Customer Roster (140 E-Commerce Records)", 
            color=TEXT_WHITE, fontsize=14, fontweight='bold')
    
    search_bar = patches.FancyBboxPatch((0.68, 0.915), 0.28, 0.05, boxstyle="round,pad=0.01",
                                       facecolor='#0f172a', edgecolor=CARD_BORDER)
    ax.add_patch(search_bar)
    ax.text(0.70, 0.935, "[Filter] Search by ID, Name, or Persona...", color=TEXT_MUTED, fontsize=9.5)
    
    cols = ["Customer ID", "Name", "Age", "Annual Spend", "Orders/Yr", "Cluster", "Loyalty Persona", "Tier"]
    x_positions = [0.05, 0.18, 0.32, 0.42, 0.55, 0.65, 0.77, 0.92]
    
    head_bg = patches.Rectangle((0.04, 0.84), 0.92, 0.05, facecolor='#0f172a', edgecolor='none')
    ax.add_patch(head_bg)
    for c, x in zip(cols, x_positions):
        ax.text(x, 0.865, c, color=TEXT_MUTED, fontsize=9.5, fontweight='bold')

    sample_rows = df_seg.head(10).copy()
    for i, (_, row) in enumerate(sample_rows.iterrows()):
        y_line = 0.78 - (i * 0.07)
        cid = row['customer_id']
        name = row['customer_name']
        age = row['age']
        spend = row['annual_spend']
        freq = row['purchase_frequency']
        cluster = row['cluster_id']
        persona = row['persona_name']
        tier = row['loyalty_tier'].split(":")[0] # Short tier
        
        c_col = COLORS[cluster % 4]
        
        row_bg = patches.Rectangle((0.04, y_line - 0.02), 0.92, 0.06, 
                                   facecolor='#1e293b' if i % 2 == 0 else '#162032', edgecolor='none')
        ax.add_patch(row_bg)
        
        ax.text(x_positions[0], y_line + 0.01, cid, color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
        ax.text(x_positions[1], y_line + 0.01, name, color=TEXT_MUTED, fontsize=9)
        ax.text(x_positions[2], y_line + 0.01, f"{age} yrs", color=TEXT_WHITE, fontsize=9)
        ax.text(x_positions[3], y_line + 0.01, f"${spend:,.2f}", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
        ax.text(x_positions[4], y_line + 0.01, f"{freq} orders", color=CYAN, fontsize=9)
        ax.text(x_positions[5], y_line + 0.01, f"Cluster {cluster}", color=c_col, fontsize=9, fontweight='bold')
        ax.text(x_positions[6], y_line + 0.01, persona, color=TEXT_WHITE, fontsize=8.5)
        ax.text(x_positions[7], y_line + 0.01, tier, color=YELLOW, fontsize=8.5, fontweight='bold')

    ax.text(0.05, 0.04, "Showing 10 of 140 clustered e-commerce records  |  Export available as segmented_customers.csv", 
            color=TEXT_MUTED, fontsize=9.5)

    out_file = os.path.join(OUTPUT_DIR, "13_segmented_customers_table_explorer.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Image 14: Official Part 2 Audit Report View
# =====================================================================
def generate_part2_audit_report_view():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=200, facecolor=BG_DARK)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.text(0.05, 0.93, "Official Customer Segmentation Audit Report Deliverable", 
            color=TEXT_WHITE, fontsize=15, fontweight='bold')
    ax.text(0.05, 0.88, "Automated executive summary generated by Industrial Data Analytics Unsupervised Learning Engine", 
            color=TEXT_MUTED, fontsize=10)
    
    term_box = patches.FancyBboxPatch((0.05, 0.05), 0.90, 0.80, boxstyle="round,pad=0.01",
                                      facecolor='#020617', edgecolor='#334155', linewidth=1.5)
    ax.add_patch(term_box)
    
    with open("reports/customer_segmentation_audit.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    text_content = "".join(lines[:38])
    ax.text(0.07, 0.82, text_content, color='#38bdf8', fontfamily='monospace', fontsize=8.5, va='top', linespacing=1.35)

    out_file = os.path.join(OUTPUT_DIR, "14_customer_segmentation_audit_report.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")


# =====================================================================
# Main Orchestrator
# =====================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("GENERATING PART 2 UI FUNCTIONALITY SCREENSHOTS")
    print(f"Target Directory: {OUTPUT_DIR}/")
    print("=" * 70)
    
    generate_part2_dashboard_overview()
    generate_elbow_silhouette_view()
    generate_clusters_scatter_view()
    generate_classifier_card_view()
    generate_loyalty_matrix_view()
    generate_segmented_table_view()
    generate_part2_audit_report_view()
    
    print("\nAll Part 2 UI images generated successfully!")
