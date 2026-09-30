import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import CANCELLATION_PREFIX, DATETIME_COLUMNS

logger = logging.getLogger(__name__)


def compute_total_amount(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing TotalAmount = Quantity * UnitPrice")
    df = df.copy()
    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["TotalAmount"] = qty * price
    logger.info(f"TotalAmount range: [{df['TotalAmount'].min():.2f}, {df['TotalAmount'].max():.2f}] | "
                f"Mean: {df['TotalAmount'].mean():.2f}")
    return df


def create_temporal_features(df: pd.DataFrame, datetime_col: str = "InvoiceDate") -> pd.DataFrame:
    logger.info("Extracting temporal features from InvoiceDate")
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col], errors="coerce")
    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Hour"] = dt.dt.hour
    df["DayOfWeek"] = dt.dt.dayofweek
    df["DayOfWeekName"] = dt.dt.day_name()
    df["YearMonth"] = dt.dt.to_period("M").astype(str)
    df["Date"] = dt.dt.date
    logger.info("Extracted: Year, Month, Day, Hour, DayOfWeek, DayOfWeekName, YearMonth, Date")
    return df


def flag_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Flagging cancellation invoices (InvoiceNo starting with 'C')")
    df = df.copy()
    invoices = df["InvoiceNo"].astype(str)
    df["IsCancellation"] = invoices.str.startswith(CANCELLATION_PREFIX, na=False)
    logger.info(f"Cancellation records: {int(df['IsCancellation'].sum()):,}")
    return df


def remove_exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    logger.info("Removing exact duplicate rows (keep first) — for Week 1 cleaning parity")
    before = len(df)
    df_deduped = df.drop_duplicates(keep="first").copy()
    removed = before - len(df_deduped)
    logger.info(f"  before={before:,}  removed={removed:,}  after={len(df_deduped):,}")
    return df_deduped, removed


def create_analytical_datasets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    logger.info("Creating separate analytical datasets (dedup first for Week 1 parity)")

    df_deduped, n_dup_removed = remove_exact_duplicates(df)

    full = df_deduped.copy()

    valid_price_mask = pd.to_numeric(full["UnitPrice"], errors="coerce") > 0
    valid_price = full[valid_price_mask].copy()
    logger.info(f"Valid-price dataset: {len(valid_price):,} rows (UnitPrice > 0, after dedup)")

    qty = pd.to_numeric(valid_price["Quantity"], errors="coerce")
    invoices = valid_price["InvoiceNo"].astype(str)
    is_cancel = invoices.str.startswith(CANCELLATION_PREFIX, na=False)
    if "IsCancellation" in valid_price.columns:
        is_cancel = valid_price["IsCancellation"]
    positive_sales = valid_price[(~is_cancel) & (qty > 0)].copy()
    logger.info(f"Positive-sales dataset: {len(positive_sales):,} rows (dedup, non-cancel, UnitPrice>0, Quantity>0)")

    return {
        "full": full,
        "valid_price": valid_price,
        "positive_sales": positive_sales,
    }


def numeric_summary(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    logger.info("Generating numeric descriptive statistics")
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = df[columns].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T
    stats["skewness"] = df[columns].skew()
    stats["kurtosis"] = df[columns].kurtosis()
    return stats.round(4).reset_index().rename(columns={"index": "column"})


def prepare_all(df_raw: pd.DataFrame) -> Dict:
    logger.info("=" * 60)
    logger.info("RUNNING DATA PREPARATION FOR EDA")
    logger.info("=" * 60)
    df = compute_total_amount(df_raw)
    df = create_temporal_features(df)
    df = flag_cancellations(df)
    datasets = create_analytical_datasets(df)
    datasets["engineered"] = datasets["full"]
    n_eng_cols = len(datasets["engineered"].columns)
    n_raw_cols = len(df_raw.columns)
    logger.info(f"Original UCI raw columns = {n_raw_cols}  |  After feature engineering = {n_eng_cols}")
    logger.info("DATA PREPARATION COMPLETE")
    return datasets
