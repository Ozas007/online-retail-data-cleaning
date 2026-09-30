import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.descriptive_analysis import save_table

logger = logging.getLogger(__name__)


def top_countries_by_transactions(df_full: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    logger.info(f"Computing top {top_n} countries by transaction-line count")
    country = df_full.groupby("Country", dropna=False).agg(
        lines=("InvoiceNo", "count"),
        unique_invoices=("InvoiceNo", "nunique"),
        unique_customers=("CustomerID", "nunique"),
        unique_products=("StockCode", "nunique"),
    ).reset_index()
    country["lines_pct"] = (country["lines"] / len(df_full) * 100).round(3)
    country = country.sort_values("lines", ascending=False).reset_index(drop=True)
    save_table(country.head(top_n), f"13_top_{top_n}_countries_transactions.csv")
    logger.info(f"Top country: {country.loc[0, 'Country']} = {int(country.loc[0, 'lines']):,} lines "
                f"({country.loc[0, 'lines_pct']:.2f}%)")
    return country


def country_revenue(df_positive: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    logger.info(f"Computing revenue by country (top {top_n})")
    rev = df_positive.groupby("Country", dropna=False).agg(
        revenue=("TotalAmount", "sum"),
        lines=("InvoiceNo", "count"),
        avg_unit_price=("UnitPrice", "mean"),
        avg_total_amount=("TotalAmount", "mean"),
        unique_customers=("CustomerID", "nunique"),
    ).reset_index()
    total_rev = rev["revenue"].sum()
    rev["revenue_pct"] = (rev["revenue"] / total_rev * 100).round(3)
    rev = rev.sort_values("revenue", ascending=False).reset_index(drop=True)
    save_table(rev.round(2).head(top_n), f"14_top_{top_n}_countries_revenue.csv")
    logger.info(f"Top country (revenue): {rev.loc[0, 'Country']} = {rev.loc[0, 'revenue']:,.2f} "
                f"({rev.loc[0, 'revenue_pct']:.2f}%)")
    return rev


def compare_country_rankings(country_lines: pd.DataFrame, country_rev: pd.DataFrame) -> pd.DataFrame:
    logger.info("Comparing country rankings: transaction count vs revenue")
    top10_lines = country_lines.head(10)["Country"].tolist()
    top10_rev = country_rev.head(10)["Country"].tolist()
    overlap = set(top10_lines) & set(top10_rev)
    comp = pd.DataFrame([
        {"metric": "Countries in top 10 by lines", "value": len(top10_lines)},
        {"metric": "Countries in top 10 by revenue", "value": len(top10_rev)},
        {"metric": "Countries in BOTH top 10 lists", "value": len(overlap)},
        {"metric": "Only in lines top 10", "value": len(set(top10_lines) - set(top10_rev))},
        {"metric": "Only in revenue top 10", "value": len(set(top10_rev) - set(top10_lines))},
    ])
    save_table(comp, "15_country_ranking_comparison.csv")
    logger.info(f"Top-10 overlap: {len(overlap)} countries")
    return comp


def run_geographic_analysis(df_full: pd.DataFrame, df_positive: pd.DataFrame) -> Dict:
    logger.info("=" * 60)
    logger.info("RUNNING GEOGRAPHIC ANALYSIS")
    logger.info("=" * 60)
    country_lines = top_countries_by_transactions(df_full, top_n=40)
    country_rev_df = country_revenue(df_positive, top_n=40)
    comp = compare_country_rankings(country_lines, country_rev_df)
    logger.info("GEOGRAPHIC ANALYSIS COMPLETE")
    return {
        "country_transactions": country_lines,
        "country_revenue": country_rev_df,
        "ranking_comparison": comp,
    }
