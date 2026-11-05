"""Clean, join, analyze, and chart the ShopEase retail CSV dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FILES = ("orders.csv", "products.csv", "customers.csv")


def load_and_clean(data_dir: Path) -> tuple[pd.DataFrame, int]:
    missing_files = [name for name in FILES if not (data_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing input CSV file(s) in {data_dir}: {', '.join(missing_files)}"
        )

    orders = pd.read_csv(data_dir / "orders.csv")
    products = pd.read_csv(data_dir / "products.csv")
    customers = pd.read_csv(data_dir / "customers.csv")
    required = {
        "orders": {"order_id", "customer_id", "product_id", "order_date", "quantity", "discount_pct"},
        "products": {"product_id", "product_name", "category", "unit_price", "unit_cost"},
        "customers": {"customer_id", "customer_name", "city", "region"},
    }
    for name, frame in (("orders", orders), ("products", products), ("customers", customers)):
        absent = required[name] - set(frame.columns)
        if absent:
            raise ValueError(f"{name}.csv is missing required columns: {', '.join(sorted(absent))}")

    discounts = pd.to_numeric(orders["discount_pct"], errors="coerce")
    missing_discounts = int(discounts.isna().sum())
    orders["discount_pct"] = discounts.fillna(0)
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="raise")
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="raise")
    products["unit_price"] = pd.to_numeric(products["unit_price"], errors="raise")
    products["unit_cost"] = pd.to_numeric(products["unit_cost"], errors="raise")

    if orders["order_id"].duplicated().any():
        raise ValueError("orders.csv must contain unique order_id values.")
    if products["product_id"].duplicated().any():
        raise ValueError("products.csv must contain unique product_id values.")
    if customers["customer_id"].duplicated().any():
        raise ValueError("customers.csv must contain unique customer_id values.")
    if (orders["quantity"] <= 0).any():
        raise ValueError("Order quantities must be positive.")
    if ((orders["discount_pct"] < 0) | (orders["discount_pct"] > 1)).any():
        raise ValueError("discount_pct must be between 0 and 1, expressed as a fraction.")
    if ((products["unit_price"] < 0) | (products["unit_cost"] < 0)).any():
        raise ValueError("Product prices and costs must not be negative.")

    for frame, columns in (
        (products, ["product_name", "category"]),
        (customers, ["customer_name", "city", "region"]),
    ):
        for column in columns:
            frame[column] = frame[column].fillna("Unknown").astype(str).str.strip()
            frame.loc[frame[column].eq(""), column] = "Unknown"

    master = orders.merge(
        products, on="product_id", how="left", validate="many_to_one", indicator="_product_match"
    )
    master = master.merge(
        customers, on="customer_id", how="left", validate="many_to_one", indicator="_customer_match"
    )
    unmatched_products = int(master["_product_match"].eq("left_only").sum())
    unmatched_customers = int(master["_customer_match"].eq("left_only").sum())
    if unmatched_products or unmatched_customers:
        raise ValueError(
            "Unmatched order references: "
            f"{unmatched_products} product(s), {unmatched_customers} customer(s)."
        )
    master = master.drop(columns=["_product_match", "_customer_match"])
    master["revenue"] = master["quantity"] * master["unit_price"] * (1 - master["discount_pct"])
    master["profit"] = master["revenue"] - master["quantity"] * master["unit_cost"]
    master["year"] = master["order_date"].dt.year
    master["month"] = master["order_date"].dt.month
    master["quarter"] = master["order_date"].dt.quarter
    master["month_name"] = master["order_date"].dt.strftime("%B")
    master["day_of_week"] = master["order_date"].dt.day_name()
    return master, missing_discounts


def write_dashboard(master: pd.DataFrame, missing_discounts: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    years = sorted(master["year"].unique())
    if len(years) != 1:
        raise ValueError(
            f"Retail Annual Summary expects one calendar year; found: {', '.join(map(str, years))}."
        )
    year = int(years[0])

    top_customers = (
        master.groupby(["customer_id", "customer_name", "city", "region"], as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "lifetime_value"})
        .sort_values(["lifetime_value", "customer_id"], ascending=[False, True])
        .head(10)
    )
    category_summary = (
        master.groupby("category", as_index=False)
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"), units_sold=("quantity", "sum"))
        .sort_values("profit", ascending=False)
    )
    monthly_revenue = (
        master.groupby("month", as_index=False)["revenue"].sum().sort_values("month")
    )
    weekday_counts = master.groupby("day_of_week").size()
    busiest_day = str(weekday_counts.idxmax())
    correlation = float(master["discount_pct"].corr(master["revenue"]))
    if not np.isfinite(correlation):
        correlation = float("nan")

    master.to_csv(output_dir / "master_table.csv", index=False)
    top_customers.to_csv(output_dir / "top_10_customers.csv", index=False)
    category_summary.to_csv(output_dir / "category_profitability.csv", index=False)
    monthly_revenue.to_csv(output_dir / "monthly_revenue.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(master["discount_pct"] * 100, master["revenue"], alpha=0.75, color="#277da1")
    ax.set(title="ShopEase Revenue vs. Discount", xlabel="Discount (%)", ylabel="Net revenue per order line")
    ax.grid(alpha=0.25)
    ax.text(
        0.04,
        0.95,
        f"Pearson correlation: {correlation:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "revenue_vs_discount.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ordered_categories = category_summary.sort_values("profit")
    ax.barh(ordered_categories["category"], ordered_categories["profit"], color="#43aa8b")
    ax.set(title=f"ShopEase Category Profit — {year}", xlabel="Profit", ylabel="Product category")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "category_profitability.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(monthly_revenue["month"], monthly_revenue["revenue"], marker="o", color="#f8961e")
    ax.set(
        title=f"ShopEase Monthly Revenue — {year}",
        xlabel="Month",
        ylabel="Net revenue",
        xticks=range(1, 13),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_revenue.png", dpi=160)
    plt.close(fig)

    best_category = category_summary.iloc[0]
    summary = [
        f"ShopEase Retail Annual Summary — {year}",
        f"Total net revenue: {master['revenue'].sum():.2f}",
        f"Total profit: {master['profit'].sum():.2f}",
        f"Best-profit category: {best_category['category']} ({best_category['profit']:.2f})",
        f"Busiest order weekday: {busiest_day}",
        f"Revenue/discount Pearson correlation: {correlation:.3f}",
        f"Missing discounts filled with 0: {missing_discounts}",
        "",
        "Top 10 customers by lifetime value:",
        top_customers.to_string(index=False, formatters={"lifetime_value": "{:.2f}".format}),
        "",
        "Category profitability:",
        category_summary.to_string(index=False, formatters={"revenue": "{:.2f}".format, "profit": "{:.2f}".format}),
    ]
    (output_dir / "annual_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary[:7]))
    print(f"\nDashboard files saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Folder containing orders.csv, products.csv, and customers.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports",
        help="Folder to receive the master table, summaries, and charts",
    )
    args = parser.parse_args()
    master, missing_discounts = load_and_clean(args.data_dir)
    write_dashboard(master, missing_discounts, args.output_dir)


if __name__ == "__main__":
    main()
