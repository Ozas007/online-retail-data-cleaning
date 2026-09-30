import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.descriptive_analysis import save_table

logger = logging.getLogger(__name__)


def monthly_transaction_volume(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing monthly transaction volume")
    monthly = df.groupby("YearMonth").agg(
        transaction_lines=("InvoiceNo", "count"),
        unique_invoices=("InvoiceNo", "nunique"),
        unique_customers=("CustomerID", "nunique"),
        total_quantity=("Quantity", "sum"),
    ).reset_index().sort_values("YearMonth")
    save_table(monthly, "05_monthly_transaction_volume.csv")
    logger.info(f"Monthly rows: {len(monthly)} | Range: {monthly['YearMonth'].iloc[0]} to {monthly['YearMonth'].iloc[-1]}")
    return monthly


def monthly_revenue(df_positive: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing monthly revenue")
    rev = df_positive.groupby("YearMonth").agg(
        revenue=("TotalAmount", "sum"),
        avg_order_value=("TotalAmount", "mean"),
        median_order_value=("TotalAmount", "median"),
        lines=("InvoiceNo", "count"),
        unique_invoices=("InvoiceNo", "nunique"),
    ).reset_index().sort_values("YearMonth")
    rev["revenue_per_invoice"] = rev["revenue"] / rev["unique_invoices"]
    save_table(rev.round(2), "06_monthly_revenue.csv")
    best_idx = rev["revenue"].idxmax()
    worst_idx = rev["revenue"].idxmin()
    logger.info(f"Highest revenue month: {rev.loc[best_idx, 'YearMonth']} = {rev.loc[best_idx, 'revenue']:,.2f}")
    logger.info(f"Lowest revenue month: {rev.loc[worst_idx, 'YearMonth']} = {rev.loc[worst_idx, 'revenue']:,.2f}")
    return rev


def transactions_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing transactions by hour of day")
    hourly = df.groupby("Hour").agg(
        lines=("InvoiceNo", "count"),
        unique_invoices=("InvoiceNo", "nunique"),
    ).reset_index().sort_values("Hour")
    save_table(hourly, "07_transactions_by_hour.csv")
    peak_idx = hourly["lines"].idxmax()
    low_idx = hourly["lines"].idxmin()
    logger.info(f"Peak hour: {int(hourly.loc[peak_idx, 'Hour'])}:00 with {int(hourly.loc[peak_idx, 'lines']):,} lines")
    logger.info(f"Lowest hour: {int(hourly.loc[low_idx, 'Hour'])}:00 with {int(hourly.loc[low_idx, 'lines']):,} lines")
    return hourly


def transactions_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing transactions by day of week")
    weekday_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = df.groupby(["DayOfWeek", "DayOfWeekName"]).agg(
        lines=("InvoiceNo", "count"),
        unique_invoices=("InvoiceNo", "nunique"),
    ).reset_index()
    daily["DayOfWeekName"] = pd.Categorical(daily["DayOfWeekName"], categories=weekday_order, ordered=True)
    daily = daily.sort_values("DayOfWeekName")
    save_table(daily, "08_transactions_by_weekday.csv")
    logger.info(f"Weekday pattern captured, {len(daily)} distinct days represented")
    return daily


def run_temporal_analysis(df_full: pd.DataFrame, df_positive: pd.DataFrame) -> Dict:
    logger.info("=" * 60)
    logger.info("RUNNING TEMPORAL ANALYSIS")
    logger.info("=" * 60)
    monthly_vol = monthly_transaction_volume(df_full)
    monthly_rev = monthly_revenue(df_positive)
    hourly = transactions_by_hour(df_full)
    weekday = transactions_by_weekday(df_full)
    logger.info("TEMPORAL ANALYSIS COMPLETE")
    return {
        "monthly_volume": monthly_vol,
        "monthly_revenue": monthly_rev,
        "hourly": hourly,
        "weekday": weekday,
    }
