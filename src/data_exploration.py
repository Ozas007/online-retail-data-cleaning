import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import TABLES_DIR

logger = logging.getLogger(__name__)


def save_table(df: pd.DataFrame, filename: str, output_dir: Path = TABLES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Saved table to: {filepath}")
    return filepath


def basic_info(df: pd.DataFrame) -> Dict:
    logger.info("Generating basic dataset information")
    info = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }
    logger.info(f"Rows: {info['rows']}, Columns: {info['columns']}, Memory: {info['memory_usage_mb']} MB")
    return info


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating numeric summary statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        logger.warning("No numeric columns found")
        return pd.DataFrame()
    summary = df[numeric_cols].describe().transpose()
    summary["column"] = summary.index
    summary = summary.reset_index(drop=True)
    cols = ["column"] + [c for c in summary.columns if c != "column"]
    summary = summary[cols]
    logger.info(f"Numeric summary for {len(numeric_cols)} columns:\n{summary.head()}")
    return summary


def categorical_summary(df: pd.DataFrame, top_n: int = 10) -> Dict[str, pd.DataFrame]:
    logger.info("Generating categorical value frequencies")
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    summaries: Dict[str, pd.DataFrame] = {}
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).head(top_n).reset_index()
        vc.columns = [col, "count"]
        vc["percentage"] = (vc["count"] / len(df) * 100).round(2)
        summaries[col] = vc
        logger.info(f"  {col}: {df[col].nunique(dropna=True)} unique values")
    return summaries


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating missing values summary")
    missing = df.isna().sum()
    pct = (missing / len(df)) * 100
    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing.values,
        "missing_percentage": pct.values.round(4),
        "non_missing_count": df.notna().sum().values,
    }).sort_values("missing_count", ascending=False).reset_index(drop=True)
    return summary


def duplicate_summary(df: pd.DataFrame) -> Dict:
    logger.info("Summarizing duplicate records")
    dup_mask = df.duplicated(keep=False)
    dup_count = int(df.duplicated().sum())
    dup_rows = df[dup_mask].sort_values(list(df.columns)).head(20).copy()
    summary = {
        "total_duplicate_rows": dup_count,
        "duplicate_percentage": round(dup_count / len(df) * 100, 4),
        "duplicate_examples": dup_rows,
    }
    logger.info(f"Exact duplicates: {dup_count} ({summary['duplicate_percentage']}%)")
    return summary


def country_distribution(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating country distribution")
    if "Country" not in df.columns:
        return pd.DataFrame()
    country = df["Country"].value_counts(dropna=False).reset_index()
    country.columns = ["Country", "transaction_count"]
    country["percentage"] = (country["transaction_count"] / len(df) * 100).round(2)
    return country


def invoice_monthly_volume(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating monthly transaction volume")
    if "InvoiceDate" not in df.columns:
        return pd.DataFrame()
    temp = df.copy()
    temp["InvoiceDate"] = pd.to_datetime(temp["InvoiceDate"], errors="coerce")
    temp["year_month"] = temp["InvoiceDate"].dt.to_period("M")
    monthly = temp.groupby("year_month").agg(
        invoice_count=("InvoiceNo", "nunique"),
        transaction_count=("InvoiceNo", "count"),
        total_quantity=("Quantity", "sum"),
    ).reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)
    return monthly


def run_full_exploration(df: pd.DataFrame, save_outputs: bool = True) -> Dict:
    logger.info("=" * 60)
    logger.info("RUNNING EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    results = {}
    results["basic_info"] = basic_info(df)
    results["data_types"] = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
    })
    results["numeric_summary"] = numeric_summary(df)
    results["categorical_summary"] = categorical_summary(df)
    results["missing_values"] = missing_values_summary(df)
    results["duplicates"] = duplicate_summary(df)
    results["country_distribution"] = country_distribution(df)
    results["monthly_volume"] = invoice_monthly_volume(df)
    results["head"] = df.head(10).copy()
    results["tail"] = df.tail(10).copy()

    if save_outputs:
        save_table(results["missing_values"], "missing_values.csv")
        save_table(results["numeric_summary"], "numeric_summary.csv")
        save_table(results["data_types"], "data_types.csv")
        save_table(results["country_distribution"], "country_distribution.csv")
        save_table(results["monthly_volume"], "monthly_volume.csv")
        if len(results["duplicates"]["duplicate_examples"]) > 0:
            save_table(results["duplicates"]["duplicate_examples"], "duplicate_examples.csv")

    logger.info("=" * 60)
    logger.info("EXPLORATION COMPLETE")
    logger.info("=" * 60)
    return results
