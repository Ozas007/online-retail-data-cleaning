import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import CANCELLATION_PREFIX, CLEANED_DATA_DIR, TABLES_DIR
from src.data_validation import count_missing_values

logger = logging.getLogger(__name__)


def _log_step(
    cleaning_log: List[Dict],
    step: str,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    reason: str,
) -> None:
    cleaning_log.append({
        "step": step,
        "rows_before": len(df_before),
        "rows_removed": len(df_before) - len(df_after),
        "rows_after": len(df_after),
        "reason": reason,
    })
    logger.info(
        f"[{step}] Before: {len(df_before):,} | Removed: {len(df_before) - len(df_after):,} | After: {len(df_after):,} | {reason}"
    )


def remove_exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    logger.info("Removing exact duplicate rows")
    df_copy = df.copy()
    before = len(df_copy)
    df_deduped = df_copy.drop_duplicates(keep="first")
    removed = before - len(df_deduped)
    logger.info(f"Removed {removed} exact duplicate rows ({removed / before * 100:.2f}%)")
    return df_deduped, removed


def flag_cancellation_invoices(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Flagging cancellation invoices (starting with 'C')")
    df = df.copy()
    invoices = df["InvoiceNo"].astype(str)
    df["IsCancellation"] = invoices.str.startswith(CANCELLATION_PREFIX, na=False)
    cancellation_count = int(df["IsCancellation"].sum())
    logger.info(f"Flagged {cancellation_count} cancellation records ({cancellation_count / len(df) * 100:.2f}%)")
    return df


def flag_negative_quantities(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Flagging records with negative quantities")
    df = df.copy()
    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    df["IsNegativeQuantity"] = qty < 0
    df["IsZeroQuantity"] = qty == 0
    neg_count = int(df["IsNegativeQuantity"].sum())
    zero_count = int(df["IsZeroQuantity"].sum())
    logger.info(f"Negative quantities: {neg_count} | Zero quantities: {zero_count}")
    return df


def flag_zero_negative_prices(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Flagging records with zero or negative UnitPrice")
    df = df.copy()
    price = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["IsNegativePrice"] = price < 0
    df["IsZeroPrice"] = price == 0
    neg_count = int(df["IsNegativePrice"].sum())
    zero_count = int(df["IsZeroPrice"].sum())
    logger.info(f"Negative prices: {neg_count} | Zero prices: {zero_count}")
    return df


def remove_invalid_prices(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    logger.info("Removing records with UnitPrice <= 0 from quality-cleaned dataset")
    df = df.copy()
    before = len(df)
    price = pd.to_numeric(df["UnitPrice"], errors="coerce")
    mask_valid_price = price > 0
    df_clean = df[mask_valid_price].copy()
    removed = before - len(df_clean)
    logger.info(f"Removed {removed} records with UnitPrice <= 0")
    return df_clean, removed


def compute_total_amount(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing TotalAmount = Quantity * UnitPrice")
    df = df.copy()
    df["TotalAmount"] = (
        pd.to_numeric(df["Quantity"], errors="coerce") *
        pd.to_numeric(df["UnitPrice"], errors="coerce")
    )
    logger.info(
        f"TotalAmount range: {df['TotalAmount'].min():.2f} to {df['TotalAmount'].max():.2f} | "
        f"Mean: {df['TotalAmount'].mean():.2f}"
    )
    return df


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extracting datetime features from InvoiceDate")
    df = df.copy()
    dt = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Hour"] = dt.dt.hour
    df["DayOfWeek"] = dt.dt.dayofweek
    df["YearMonth"] = dt.dt.to_period("M").astype(str)
    logger.info("Extracted Year, Month, Day, Hour, DayOfWeek, YearMonth")
    return df


def create_positive_sales_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    logger.info("Creating positive-sales analytical dataset (excl. cancellations & non-positive qty)")
    df = df.copy()
    before = len(df)
    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    invoices = df["InvoiceNo"].astype(str)
    is_cancellation = invoices.str.startswith(CANCELLATION_PREFIX, na=False)
    mask = (~is_cancellation) & (qty > 0)
    if "IsCancellation" in df.columns:
        mask = (~df["IsCancellation"]) & (qty > 0)
    df_pos = df[mask].copy()
    removed = before - len(df_pos)
    logger.info(f"Positive-sales dataset: {len(df_pos):,} records (removed {removed:,})")
    return df_pos, removed


def cleaning_decision_table(
    df: pd.DataFrame,
    dup_removed: int,
    invalid_price_removed: int,
    positive_sales_removed: int,
) -> pd.DataFrame:
    logger.info("Generating cleaning decision table")
    neg_qty = int((pd.to_numeric(df["Quantity"], errors="coerce") < 0).sum())
    zero_neg_price = int((pd.to_numeric(df["UnitPrice"], errors="coerce") <= 0).sum())
    cancel_inv = int(df["InvoiceNo"].astype(str).str.startswith(CANCELLATION_PREFIX, na=False).sum())
    missing_cust = int(df["CustomerID"].isna().sum())
    missing_desc = int(df["Description"].isna().sum())

    table = pd.DataFrame([
        {
            "Issue": "Exact duplicate rows",
            "Count": dup_removed,
            "Treatment": "Removed (keep first)",
            "Reason": "Identical transaction rows are likely data-entry errors",
            "Impact": f"Removed {dup_removed} rows; reduces distortion in aggregations",
        },
        {
            "Issue": "UnitPrice <= 0",
            "Count": zero_neg_price,
            "Treatment": "Removed from quality-cleaned dataset; flagged",
            "Reason": "Invalid/unknown price prevents financial analysis",
            "Impact": f"Removed {invalid_price_removed} rows from cleaned set; preserved in flagged full set",
        },
        {
            "Issue": "Negative Quantity (cancellations/returns)",
            "Count": neg_qty,
            "Treatment": "Flagged (NOT removed); excluded from positive-sales dataset",
            "Reason": "InvoiceNo starting with 'C' indicates legitimate cancellations",
            "Impact": "Retained for return/cancellation analysis; excluded from sales-only analysis",
        },
        {
            "Issue": "Cancellation invoices (prefix 'C')",
            "Count": cancel_inv,
            "Treatment": "Flagged as IsCancellation; NOT auto-deleted",
            "Reason": "Cancellations are valid business events, not data errors",
            "Impact": "Analysts can filter based on use case",
        },
        {
            "Issue": "Missing CustomerID",
            "Count": missing_cust,
            "Treatment": "Retained; flagged as missing",
            "Reason": "Transaction itself is valid; customer-level analysis impossible",
            "Impact": "Customer-level studies (RFM, segmentation) must exclude these rows",
        },
        {
            "Issue": "Missing Description",
            "Count": missing_desc,
            "Treatment": "Retained as-is in all pipeline datasets; no arbitrary text imputation is applied",
            "Reason": "Description column contains 1,454 (0.27%) nulls; financial-level analysis does not require product labels, and imputation via StockCode is not performed unless explicitly requested",
            "Impact": "Product-level labelling studies may need a StockCode→Description lookup step; revenue/quantity aggregations are unaffected",
        },
        {
            "Issue": "Non-positive Quantity (non-cancellation)",
            "Count": positive_sales_removed,
            "Treatment": "Excluded from positive-sales analytical dataset",
            "Reason": "Positive-sales analysis requires actual revenue-generating transactions",
            "Impact": f"{positive_sales_removed} rows excluded from sales-only set",
        },
    ])
    return table


def build_cleaning_log_csv(cleaning_log: List[Dict], output_dir: Path = TABLES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(cleaning_log)
    filepath = output_dir / "cleaning_log.csv"
    log_df.to_csv(filepath, index=False)
    logger.info(f"Cleaning log saved to: {filepath}")
    return filepath


def save_cleaned_datasets(
    df_flagged: pd.DataFrame,
    df_clean: pd.DataFrame,
    df_positive_sales: pd.DataFrame,
    output_dir: Path = CLEANED_DATA_DIR,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["flagged"] = output_dir / "online_retail_flagged.csv"
    paths["cleaned"] = output_dir / "online_retail_cleaned.csv"
    paths["positive_sales"] = output_dir / "online_retail_positive_sales.csv"

    df_flagged.to_csv(paths["flagged"], index=False)
    logger.info(f"Saved flagged dataset ({len(df_flagged):,} rows) to: {paths['flagged']}")

    df_clean.to_csv(paths["cleaned"], index=False)
    logger.info(f"Saved quality-cleaned dataset ({len(df_clean):,} rows) to: {paths['cleaned']}")

    df_positive_sales.to_csv(paths["positive_sales"], index=False)
    logger.info(f"Saved positive-sales dataset ({len(df_positive_sales):,} rows) to: {paths['positive_sales']}")

    return paths


def run_full_cleaning(
    df_raw: pd.DataFrame,
    save_outputs: bool = True,
    cleaned_output_dir: Path = CLEANED_DATA_DIR,
    tables_output_dir: Path = TABLES_DIR,
) -> Dict:
    logger.info("=" * 60)
    logger.info("RUNNING DATA CLEANING PIPELINE")
    logger.info("=" * 60)

    cleaning_log: List[Dict] = []

    df = df_raw.copy()
    cleaning_log.append({
        "step": "Initial raw dataset",
        "rows_before": len(df),
        "rows_removed": 0,
        "rows_after": len(df),
        "reason": "Loaded from Excel source",
    })

    df, dup_removed = remove_exact_duplicates(df)
    _log_step(cleaning_log, "Remove exact duplicates", df_raw, df, "Identical rows are data-entry duplicates")

    df = flag_cancellation_invoices(df)
    _log_step(cleaning_log, "Flag cancellation invoices", df, df, "IsCancellation column added (no rows removed)")

    df = flag_negative_quantities(df)
    _log_step(cleaning_log, "Flag negative/zero quantities", df, df, "IsNegativeQuantity, IsZeroQuantity flags added")

    df = flag_zero_negative_prices(df)
    _log_step(cleaning_log, "Flag zero/negative prices", df, df, "IsNegativePrice, IsZeroPrice flags added")

    df_clean, invalid_price_removed = remove_invalid_prices(df)
    _log_step(cleaning_log, "Remove UnitPrice <= 0 (cleaned set only)", df, df_clean,
              "Invalid price for financial analysis")

    df_flagged_with_amount = compute_total_amount(df)
    df_clean_with_amount = compute_total_amount(df_clean)

    df_flagged_final = extract_datetime_features(df_flagged_with_amount)
    df_clean_final = extract_datetime_features(df_clean_with_amount)
    _log_step(cleaning_log, "Feature engineering (TotalAmount + datetime)", df_clean, df_clean_final,
              "Added engineered features")

    df_positive_sales, pos_sales_removed = create_positive_sales_dataset(df_clean_final)
    _log_step(cleaning_log, "Create positive-sales analytical dataset", df_clean_final, df_positive_sales,
              "Exclude cancellations + non-positive qty for sales analysis")

    decision_table = cleaning_decision_table(df_raw, dup_removed, invalid_price_removed, pos_sales_removed)

    outputs = {}
    if save_outputs:
        outputs = save_cleaned_datasets(df_flagged_final, df_clean_final, df_positive_sales, output_dir=cleaned_output_dir)
        build_cleaning_log_csv(cleaning_log, output_dir=tables_output_dir)
        tables_output_dir = Path(tables_output_dir)
        tables_output_dir.mkdir(parents=True, exist_ok=True)
        decision_path = tables_output_dir / "cleaning_decision_table.csv"
        decision_table.to_csv(decision_path, index=False)
        logger.info(f"Saved cleaning decision table to: {decision_path}")

    logger.info("=" * 60)
    logger.info("CLEANING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return {
        "flagged_dataset": df_flagged_final,
        "cleaned_dataset": df_clean_final,
        "positive_sales_dataset": df_positive_sales,
        "cleaning_log": cleaning_log,
        "decision_table": decision_table,
        "output_paths": outputs,
    }
