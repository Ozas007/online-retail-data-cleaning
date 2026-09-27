import logging
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from src.config import (
    CANCELLATION_PREFIX,
    CATEGORICAL_COLUMNS,
    DATETIME_COLUMNS,
    EXPECTED_COLUMNS,
    NUMERIC_COLUMNS,
)

logger = logging.getLogger(__name__)


def check_data_types(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Checking data types of all columns")
    dtype_report = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Non-Null Count": df.notna().sum().values,
    })
    logger.info(f"\n{dtype_report.to_string(index=False)}")
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
    logger.info(f"Unique values summary:\n{unique_report.to_string(index=False)}")
    return unique_report


def check_invalid_numeric(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Dict]:
    logger.info("Checking for invalid numeric values")
    if columns is None:
        columns = [c for c in NUMERIC_COLUMNS if c in df.columns]

    report = {}
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        nan_count = int(series.isna().sum() - df[col].isna().sum())
        neg_count = int((series < 0).sum())
        zero_count = int((series == 0).sum())
        report[col] = {
            "non_numeric_count": nan_count if nan_count > 0 else 0,
            "negative_count": neg_count,
            "zero_count": zero_count,
        }
        logger.info(f"  {col}: {report[col]}")
    return report


def check_date_parsing(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Dict]:
    logger.info("Checking datetime parsing")
    if columns is None:
        columns = [c for c in DATETIME_COLUMNS if c in df.columns]

    report = {}
    for col in columns:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        unparseable = int(parsed.isna().sum() - df[col].isna().sum())
        min_date = parsed.min() if parsed.notna().any() else None
        max_date = parsed.max() if parsed.notna().any() else None
        report[col] = {
            "unparseable_count": unparseable if unparseable > 0 else 0,
            "min_date": str(min_date),
            "max_date": str(max_date),
        }
        logger.info(f"  {col}: {report[col]}")
    return report


def check_negative_quantities(df: pd.DataFrame) -> Tuple[int, int, int]:
    logger.info("Checking negative and zero quantities")
    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    neg_qty = int((qty < 0).sum())
    zero_qty = int((qty == 0).sum())
    pos_qty = int((qty > 0).sum())
    logger.info(f"Negative quantities: {neg_qty}, Zero quantities: {zero_qty}, Positive quantities: {pos_qty}")
    return neg_qty, zero_qty, pos_qty


def check_negative_prices(df: pd.DataFrame) -> Tuple[int, int, int]:
    logger.info("Checking negative and zero prices")
    price = pd.to_numeric(df["UnitPrice"], errors="coerce")
    neg_price = int((price < 0).sum())
    zero_price = int((price == 0).sum())
    pos_price = int((price > 0).sum())
    logger.info(f"Negative prices: {neg_price}, Zero prices: {zero_price}, Positive prices: {pos_price}")
    return neg_price, zero_price, pos_price


def check_suspicious_invoices(df: pd.DataFrame) -> Dict[str, int]:
    logger.info("Checking for suspicious invoice numbers")
    invoices = df["InvoiceNo"].astype(str)
    cancellation = int(invoices.str.startswith(CANCELLATION_PREFIX, na=False).sum())
    empty_inv = int(invoices.str.strip().isin(["", "nan", "None"]).sum())
    non_standard_mask = (
        ~invoices.str.replace(f"^{CANCELLATION_PREFIX}?", "", regex=True).str.match(r"^\d+$", na=False) &
        ~invoices.str.strip().isin(["", "nan", "None"])
    )
    non_numeric = int(non_standard_mask.sum())
    report = {
        "cancellation_invoices": cancellation,
        "empty_invoice_numbers": empty_inv,
        "non_standard_invoice_format": non_numeric,
    }
    logger.info(f"  Suspicious invoices: {report}")
    return report


def check_blank_strings(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, int]:
    logger.info("Checking for blank/empty strings")
    if columns is None:
        columns = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    report = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].astype(str).str.strip()
        blanks = int(series.isin(["", "nan", "None", "NaN"]).sum())
        report[col] = blanks
        logger.info(f"  {col}: {blanks} blank/empty values")
    return report


def run_full_validation(df: pd.DataFrame) -> Dict:
    logger.info("=" * 60)
    logger.info("RUNNING FULL DATA VALIDATION")
    logger.info("=" * 60)

    results = {}
    results["shape"] = df.shape
    results["data_types"] = check_data_types(df)
    results["missing_values"] = count_missing_values(df)
    results["duplicates"] = count_duplicates(df)
    results["unique_values"] = count_unique_values(df)
    results["invalid_numeric"] = check_invalid_numeric(df)
    results["date_parsing"] = check_date_parsing(df)
    results["negative_quantities"] = check_negative_quantities(df)
    results["negative_prices"] = check_negative_prices(df)
    results["suspicious_invoices"] = check_suspicious_invoices(df)
    results["blank_strings"] = check_blank_strings(df)

    logger.info("=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 60)
    return results
