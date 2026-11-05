# AI-Augmented Industrial Data Analytics & Machine Learning

A production-grade reference project for industrial data analytics, predictive modeling with supervised learning (Part 1), pattern discovery with unsupervised clustering (Part 2), and automated interactive reporting.

---

## 🎯 Part 1: Predictive Analytics with Supervised Learning

Part 1 implements an end-to-end supervised machine learning pipeline designed to train a corporate compensation model on real-world "dirty" HR records, enforce rigorous data hygiene, evaluate generalization error, and deliver a certified **Salary Prediction Audit**.

### Implementation Guide & Architecture

| Topic / Skill | Implementation in Pipeline | Real-World Industrial Use Case |
| :--- | :--- | :--- |
| **Data Hygiene & Cleaning** | 1. Purges duplicates (`drop_duplicates`)<br>2. Imputes missing entries using **Column Median Filling** (`fillna(median)`)<br>3. Eliminates unrealistic **Outliers** via domain limits and statistical IQR bounds ($Q1 - 1.5 \times IQR$ to $Q3 + 1.5 \times IQR$). | Cleans raw HR system exports to prevent salary predictions from being skewed by typos, fake entries, or missing data. |
| **Linear Regression Modeling** | Applies $Y = MX + C$:<br>• $X$ = Years of Experience<br>• $Y$ = Annual Compensation (USD)<br>• $M$ = Marginal salary gain per year of experience<br>• $C$ = Base starting salary | Provides an explainable HR compensation engine that prices new hires equitably based on verified market experience. |
| **Error Metrics & Evaluation** | Computes **MAE**, **MSE**, **RMSE**, and **$R^2$ Score**.<br>Validates against the corporate 95% budget accuracy benchmark ($R^2 \ge 0.95$). | Determines whether an ML compensation model is statistically sound before committing corporate hiring budgets. |

### 📋 Part 1 Deliverable: Salary Prediction Audit Summary
- **Target Candidate**: Candidate with **15 Years of Experience** ($X = 15.0$)
- **Learned Regression Equation**: $\text{Salary} = 5,176.59 \times \text{Experience} + 48,024.45$
- **Step-by-Step Calculation**: $\text{Salary} = (5,176.59 \times 15.0) + 48,024.45 = \mathbf{\$125,673.29}$
- **Predicted Monthly Salary**: **\$10,472.77 / month**
- **95% Corporate Confidence Range**: **\$120,145.34 – \$131,201.23** (Margin of error: $\pm 1.96 \times \text{RMSE}$)
- **Model Evaluation Scores**: $R^2 = \mathbf{99.57\%}$ (Exceeds 95% threshold — **PASSED**), $\text{MAE} = \mathbf{\$2,267.79}$, $\text{RMSE} = \mathbf{\$2,820.38}$
- **Data Hygiene Audit**: 165 Raw Records $\rightarrow$ 15 Duplicates Purged $\rightarrow$ 20 Nulls Imputed with Median $\rightarrow$ 9 Outliers Eliminated $\rightarrow$ 141 Clean Records (85.5% retention)

---

## 🔍 Part 2: Pattern Discovery with Unsupervised Clustering

Part 2 shifts to discovery on unlabeled e-commerce customer data, employing feature normalization, optimal cluster identification via the Elbow Method, K-Means centroid optimization, and silhouette cohesion checks to deliver an interactive **Customer Segmentation Dashboard**.

### Implementation Guide & Architecture

| Topic / Skill | Implementation in Pipeline | Real-World Industrial Use Case |
| :--- | :--- | :--- |
| **Feature Scaling (`StandardScaler`)** | Standardizes feature distributions to zero mean and unit variance ($\mu=0, \sigma=1$). | Eliminates scale disparity between Age (18–80 yrs) and Spend (\$1.5k–\$22k) so that Age carries equal weight in Euclidean distance calculations. |
| **The Elbow Method** | Evaluates **Inertia** (sum of squared centroid distances) across $K = 1 \dots 8$. Locates the mathematical inflection elbow point. | Empirically confirms that the customer cohort divides into exactly **4 natural groups**. |
| **K-Means Clustering** | Optimizes $K=4$ cluster centroids and assigns every customer to their nearest behavioral group. | Categorizes 140 e-commerce customers into actionable personas to send personalized discount email campaigns. |
| **Cluster Quality Check** | Calculates **Silhouette Scores** across cluster counts and samples. | Validates cluster separation and cohesion ($0.7491 > 0.70$ threshold — **High Separation Quality**). |

### 📋 Part 2 Deliverable: New Customer Classification Summary
- **Target Candidate Input**: New customer with **Age = 25 Years**, **Annual Spend = \$2,500.00**
- **Normalized Feature Vector**: $[\text{Age}_{\text{scaled}} = -1.2569, \text{Spend}_{\text{scaled}} = -1.0493]$
- **Assigned Loyalty Group**: **Cluster 3 — Young Digital Savers** (Tier 4: Growth / Emerging)
- **Distance to Centroid**: **0.0852** (very close proximity to centroid at Age 24.1, Spend \$2,875)
- **Recommended Marketing Offer**: **25% Flash Discount on Trending & Gadgets (Cart > \$50)**
- **Personalized Email Strategy**: Automated push notifications, weekend flash promotions, student/early-career perks, and social referral rewards.

### 👥 The 4 Natural Industrial Customer Personas ($K=4$)

| Cluster ID | Persona Name | Loyalty Tier | Share | Centroid Profile | Targeted Marketing Action |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **Cluster 0** | **Mid-Age Practical Buyers** | Tier 3: Silver Family | 25.0% | Age 42.1 yrs \| Spend \$4,439 \| 16 orders/yr | **Buy-2-Get-1-Free** on Pantry & Home Essentials; bulk family bundle emails. |
| **Cluster 1** | **Affluent VIP Champions** | Tier 1: Platinum | 25.0% | Age 44.2 yrs \| Spend \$17,870 \| 39 orders/yr | **Exclusive 15% VIP Luxury Perk**; personal concierge & early product access. |
| **Cluster 2** | **Mature High-Value Loyalists** | Tier 2: Gold Loyalty | 25.0% | Age 65.2 yrs \| Spend \$9,774 \| 24 orders/yr | **Flat \$75 Loyalty Reward** on orders > \$300; anniversary & catalog rewards. |
| **Cluster 3** | **Young Digital Savers** | Tier 4: Emerging | 25.0% | Age 24.1 yrs \| Spend \$2,875 \| 14 orders/yr | **25% Flash Discount** on Tech & Gadgets; mobile app flash alerts. |

---

## 🖥️ Interactive User Interface (UI) Options

### Option 1: Standalone Browser Dashboards (Zero Setup Required)
Simply double-click or open directly in your web browser:
- **Part 1 Dashboard (Salary Prediction)**:
  ```
  reports/salary_prediction_dashboard.html
  ```
- **Part 2 Dashboard (Customer Segmentation)**:
  ```
  reports/customer_segmentation_dashboard.html
  ```
  - **Live Customer Classifier Widget**: Drag Age (18–75) and Spend (\$1.5k–\$22k) sliders to dynamically classify any customer into their loyalty group, calculate centroid distance, and display personalized marketing actions.
  - **Interactive Cluster Scatter Plot**: Dynamic Canvas/Chart.js scatter showing the 4 colored clusters, centroids, and a movable customer star marker.
  - **The Elbow Method & Silhouette Tab**: Interactive curve showing Inertia drop and peak Silhouette score (74.91%).
  - **Searchable Customer Explorer**: Filter 140 clustered customer records by ID, Name, or Persona.

### Option 2: Unified Local Python Web UI Server
Launch the unified web server:
```powershell
python src/salary_ui.py --port 8000
```
- Main Portal: `http://localhost:8000`
- Part 1 Salary UI: `http://localhost:8000/part1`
- Part 2 Segmentation UI: `http://localhost:8000/part2`
- REST APIs:
  - `GET /api/part1/predict?experience=15`
  - `GET /api/part2/classify?age=25&spend=2500`

---

## 📸 Generated UI Screenshots (`ui_screenshots/`)

All 14 high-resolution visual screenshots are saved in [`ui_screenshots/`](file:///d:/Be10x/Projects/Assignments/IndustrialDataAnalytics/ui_screenshots):

### Part 1: Predictive Analytics UI
- `01_full_dashboard_overview.png` – Master full-dashboard UI overview.
- `02_interactive_candidate_predictor.png` – Interactive candidate predictor (15-yr deliverable calculation).
- `03_data_hygiene_cleaning_funnel.png` – 3-step data hygiene pipeline funnel & boxplots.
- `04_regression_trendline_chart.png` – Interactive regression fit with target candidate highlight.
- `05_model_residuals_and_diagnostics.png` – Model residual error diagnostics & normality distribution.
- `06_clean_dataset_table_explorer.png` – Certified clean dataset table explorer.
- `07_salary_prediction_audit_report.png` – Executive salary prediction audit report deliverable.

### Part 2: Customer Segmentation UI
- `08_customer_segmentation_dashboard.png` – Full master segmentation dashboard overview.
- `09_elbow_and_silhouette_analysis.png` – The Elbow Method (Inertia) & Silhouette Score (74.91% peak).
- `10_customer_clusters_scatter_view.png` – 4 customer clusters scatter plot with centroids & target customer.
- `11_new_customer_classifier_card.png` – Live classifier widget for Candidate (Age 25, Spend \$2,500).
- `12_loyalty_groups_marketing_matrix.png` – 4 Customer personas and personalized discount strategies.
- `13_segmented_customers_table_explorer.png` – 140-record searchable customer explorer.
- `14_customer_segmentation_audit_report.png` – Official Part 2 segmentation audit report deliverable.

---

## 🚀 How to Run the Project

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Run Part 1: Salary Prediction Audit Pipeline
python src/salary_prediction_project.py

# 3. Run Part 2: Customer Segmentation Pipeline
python src/customer_segmentation_project.py

# 4. Generate all UI Screenshots
python src/generate_ui_screenshots.py
python src/generate_part2_ui_screenshots.py

# 5. Launch Unified Local UI Server
python src/salary_ui.py --port 8000
```
