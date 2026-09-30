import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.descriptive_analysis import save_table

logger = logging.getLogger(__name__)


def top_products_by_quantity(df_positive: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    logger.info(f"Computing top {top_n} products by quantity sold")
    prod = df_positive.groupby(["StockCode", "Description"], dropna=False).agg(
        total_quantity=("Quantity", "sum"),
        lines=("InvoiceNo", "count"),
        revenue=("TotalAmount", "sum"),
    ).reset_index()
    prod = prod.sort_values("total_quantity", ascending=False).head(top_n).reset_index(drop=True)
    prod["Description"] = prod["Description"].fillna("(no description)").astype(str).str[:60]
    save_table(prod.round(2), f"09_top_{top_n}_products_quantity.csv")
    logger.info(f"Top 1 product (qty): {prod.loc[0, 'Description'][:50]} = {int(prod.loc[0, 'total_quantity']):,}")
    return prod


def top_products_by_revenue(df_positive: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    logger.info(f"Computing top {top_n} products by revenue")
    prod = df_positive.groupby(["StockCode", "Description"], dropna=False).agg(
        revenue=("TotalAmount", "sum"),
        total_quantity=("Quantity", "sum"),
        lines=("InvoiceNo", "count"),
    ).reset_index()
    prod["avg_unit_price"] = prod["revenue"] / prod["total_quantity"].replace(0, np.nan)
    prod = prod.sort_values("revenue", ascending=False).head(top_n).reset_index(drop=True)
    prod["Description"] = prod["Description"].fillna("(no description)").astype(str).str[:60]
    save_table(prod.round(2), f"10_top_{top_n}_products_revenue.csv")
    logger.info(f"Top 1 product (rev): {prod.loc[0, 'Description'][:50]} = {prod.loc[0, 'revenue']:,.2f}")
    return prod


def compare_quantity_vs_revenue_rankings(top_qty: pd.DataFrame, top_rev: pd.DataFrame) -> pd.DataFrame:
    logger.info("Comparing quantity-based vs revenue-based product rankings")
    qty_set = set(top_qty["StockCode"].tolist())
    rev_set = set(top_rev["StockCode"].tolist())
    overlap = qty_set & rev_set
    only_qty = qty_set - rev_set
    only_rev = rev_set - qty_set
    comparison = pd.DataFrame([
        {"metric": "Products in top by quantity", "value": len(qty_set)},
        {"metric": "Products in top by revenue", "value": len(rev_set)},
        {"metric": "Products in BOTH top lists", "value": len(overlap)},
        {"metric": "Only in quantity top", "value": len(only_qty)},
        {"metric": "Only in revenue top", "value": len(only_rev)},
    ])
    save_table(comparison, "11_qty_vs_revenue_ranking_comparison.csv")
    logger.info(f"Overlap: {len(overlap)} products in both top lists; "
                f"Quantity-only: {len(only_qty)}; Revenue-only: {len(only_rev)}")
    return comparison


def product_revenue_distribution(df_positive: pd.DataFrame) -> Dict:
    logger.info("Analyzing product revenue distribution")
    prod_rev = df_positive.groupby("StockCode")["TotalAmount"].sum().sort_values(ascending=False)
    total_rev = prod_rev.sum()
    cum_pct = (prod_rev.cumsum() / total_rev * 100)
    top1_pct = cum_pct.iloc[int(len(prod_rev) * 0.01)] if len(prod_rev) >= 100 else None
    top10_pct = cum_pct.iloc[int(len(prod_rev) * 0.10)]
    top20_pct = cum_pct.iloc[int(len(prod_rev) * 0.20)]
    gini_value = None
    if len(prod_rev) > 0:
        pr_pos = prod_rev[prod_rev > 0]
        if len(pr_pos) >= 2:
            n = len(pr_pos)
            sorted_y = np.sort(pr_pos.values)
            cum_x = np.arange(1, n + 1) / n
            cum_y = np.cumsum(sorted_y) / sorted_y.sum()
            lorenz = np.concatenate([[0.0], cum_y])
            xs = np.concatenate([[0.0], cum_x])
            area_under_lorenz = np.trapezoid(lorenz, xs)
            gini_value = round(float(1.0 - 2.0 * area_under_lorenz), 4)
    summary = pd.DataFrame([
        {"metric": "Total unique products (positive sales)", "value": int(len(prod_rev))},
        {"metric": "Top 1% products share of revenue (%)", "value": round(top1_pct, 2) if top1_pct else None},
        {"metric": "Top 10% products share of revenue (%)", "value": round(top10_pct, 2)},
        {"metric": "Top 20% products share of revenue (%)", "value": round(top20_pct, 2)},
        {"metric": "Gini coefficient estimate", "value": gini_value},
    ])
    save_table(summary, "12_product_revenue_concentration.csv")
    logger.info(f"Top 20% of products = {top20_pct:.1f}% of revenue")
    return {"distribution": prod_rev, "summary": summary}


def run_product_analysis(df_positive: pd.DataFrame) -> Dict:
    logger.info("=" * 60)
    logger.info("RUNNING PRODUCT ANALYSIS")
    logger.info("=" * 60)
    top_qty = top_products_by_quantity(df_positive, top_n=20)
    top_rev = top_products_by_revenue(df_positive, top_n=20)
    comparison = compare_quantity_vs_revenue_rankings(top_qty.head(10), top_rev.head(10))
    rev_dist = product_revenue_distribution(df_positive)
    logger.info("PRODUCT ANALYSIS COMPLETE")
    return {
        "top_quantity": top_qty,
        "top_revenue": top_rev,
        "ranking_comparison": comparison,
        "revenue_distribution": rev_dist,
    }
