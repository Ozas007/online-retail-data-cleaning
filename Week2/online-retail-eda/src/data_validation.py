import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CANCELLATION_PREFIX, CATEGORICAL_COLUMNS,
    DATETIME_COLUMNS, NUMERIC_COLUMNS,
)

logger = logging.getLogger(__name__)


def check_data_types(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Checking data types of all columns")
    dtype_report = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Non-Null Count": df.notna().sum().values,
    })
    return dtype_report


def count_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Counting missing values per column")
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    missing_report = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing.values,
        "Missing Percentage (%)": missing_pct.values.round(2),
    }).sort_values("Missing Count", ascending=False).reset_index(drop=True)
    logger.info(f"Missing values summary:\n{missing_report.to_string(index=False)}")
    return missing_report


def count_duplicates(df: pd.DataFrame) -> Tuple[int, float]:
    logger.info("Counting exact duplicate rows")
    dup_count = int(df.duplicated().sum())
    dup_pct = (dup_count / len(df)) * 100
    logger.info(f"Found {dup_count} exact duplicate rows ({dup_pct:.2f}% of dataset)")
    return dup_count, dup_pct


def count_unique_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Counting unique values per column")
    unique_counts = df.nunique(dropna=True)
    unique_report = pd.DataFrame({
        "Column": df.columns,
        "Unique Count": unique_counts.values,
        "Unique Percentage (%)": (unique_counts.values / len(df) * 100).round(2),
    })
    return unique_report


def check_invalid_numeric(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Dict]:
    if columns is None:
        columns = [c for c in NUMERIC_COLUMNS if c in df.columns]
    report = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        nan_count = max(0, int(series.isna().sum() - df[col].isna().sum()))
        report[col] = {
            "non_numeric_count": nan_count,
            "negative_count": int((series < 0).sum()),
            "zero_count": int((series == 0).sum()),
            "positive_count": int((series > 0).sum()),
        }
        logger.info(f"  {col}: {report[col]}")
    return report


def check_date_parsing(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Dict]:
    if columns is None:
        columns = [c for c in DATETIME_COLUMNS if c in df.columns]
    report = {}
    for col in columns:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        unparseable = max(0, int(parsed.isna().sum() - df[col].isna().sum()))
        report[col] = {
            "unparseable_count": unparseable,
            "min_date": str(parsed.min()) if parsed.notna().any() else None,
            "max_date": str(parsed.max()) if parsed.notna().any() else None,
        }
        logger.info(f"  {col}: {report[col]}")
    return report


def run_full_validation(df: pd.DataFrame) -> Dict:
    logger.info("=" * 60)
    logger.info("RUNNING FULL DATA VALIDATION")
    logger.info("=" * 60)
    results = {
        "shape": df.shape,
        "data_types": check_data_types(df),
        "missing_values": count_missing_values(df),
        "duplicates": count_duplicates(df),
        "unique_values": count_unique_values(df),
        "invalid_numeric": check_invalid_numeric(df),
        "date_parsing": check_date_parsing(df),
        "n_invoices": int(df["InvoiceNo"].nunique()) if "InvoiceNo" in df.columns else 0,
        "n_customers": int(df["CustomerID"].nunique()) if "CustomerID" in df.columns else 0,
        "n_products": int(df["StockCode"].nunique()) if "StockCode" in df.columns else 0,
        "n_countries": int(df["Country"].nunique()) if "Country" in df.columns else 0,
    }
    logger.info(f"  Invoices: {results['n_invoices']:,} | Customers: {results['n_customers']:,} | "
                f"Products: {results['n_products']:,} | Countries: {results['n_countries']}")
    logger.info("VALIDATION COMPLETE")
    return results
