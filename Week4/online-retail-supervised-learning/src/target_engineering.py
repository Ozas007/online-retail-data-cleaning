import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CANCELLATION_PREFIX,
    PREDICTION_WINDOW_MONTHS,
    TEMPORAL_SPLIT_QUANTILE,
    RANDOM_SEED,
    TARGET_COLUMN,
    ID_COLUMN,
)

logger = logging.getLogger(__name__)


def _ensure_datetime(df: pd.DataFrame, col: str = "InvoiceDate") -> pd.Series:
    dt = pd.to_datetime(df[col], errors="coerce")
    return dt


def derive_dates_from_df(
    df: pd.DataFrame,
    datetime_col: str = "InvoiceDate",
    split_quantile: float = TEMPORAL_SPLIT_QUANTILE,
    prediction_window_months: int = PREDICTION_WINDOW_MONTHS,
) -> Dict[str, pd.Timestamp]:
    """
    Compute HISTORICAL_END, FUTURE_START, FUTURE_END from actual InvoiceDate values.

    The temporal split is derived from the actual data distribution rather than being
    hard-coded so that the pipeline works regardless of the exact dataset date range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a parseable InvoiceDate column.
    datetime_col : str
        Name of the datetime column.
    split_quantile : float
        Quantile (0-1) of unique InvoiceDate values used to place HISTORICAL_END.
        The default 0.75 corresponds to using the last ~25% of the time range as the
        prediction window, which for the UCI dataset (~13 months) is ~3 months.
    prediction_window_months : int
        Desired future window length (used as a fallback/verification).

    Returns
    -------
    Dict with keys:
        - HISTORICAL_START : earliest InvoiceDate (inclusive)
        - HISTORICAL_END   : last date used for features (exclusive upper bound for future)
        - FUTURE_START     : first date used for target building (inclusive, == HISTORICAL_END)
        - FUTURE_END       : latest InvoiceDate (inclusive)
    """
    logger.info("Deriving temporal split dates from InvoiceDate distribution")
    np.random.seed(RANDOM_SEED)

    dt = _ensure_datetime(df, datetime_col)
    valid_mask = dt.notna()
    if valid_mask.sum() == 0:
        raise ValueError(f"No valid dates found in column '{datetime_col}'")

    dt_valid = dt[valid_mask]
    min_date = pd.Timestamp(dt_valid.min())
    max_date = pd.Timestamp(dt_valid.max())

    unique_sorted = np.sort(dt_valid.drop_duplicates().values.astype("datetime64[ns]"))
    if len(unique_sorted) < 2:
        raise ValueError("Not enough unique dates to compute a temporal split")

    q_idx = int(np.clip(np.floor(split_quantile * len(unique_sorted)), 0, len(unique_sorted) - 1))
    historical_end = pd.Timestamp(unique_sorted[q_idx])

    if historical_end <= min_date:
        historical_end = min_date + (max_date - min_date) * split_quantile
        historical_end = pd.Timestamp(historical_end)

    future_start = historical_end
    future_end = max_date

    days_historical = (historical_end - min_date).days
    days_future = (future_end - future_start).days

    dates = {
        "HISTORICAL_START": min_date,
        "HISTORICAL_END": historical_end,
        "FUTURE_START": future_start,
        "FUTURE_END": future_end,
        "days_historical": days_historical,
        "days_future": days_future,
    }

    logger.info(
        f"  HISTORICAL_START = {min_date.date()}  |  "
        f"HISTORICAL_END = {historical_end.date()}  |  "
        f"FUTURE_END = {max_date.date()}"
    )
    logger.info(
        f"  Historical window: {days_historical} days  |  "
        f"Future window: {days_future} days  |  "
        f"Split quantile q={split_quantile}"
    )

    return dates


def _filter_positive_sales_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cancellations and non-positive sales rows; keep only positive sales lines.
    (Lightweight copy of the prep logic so target builder can work independently.)
    """
    df = df.copy()
    df["_InvoiceNo_str"] = df["InvoiceNo"].astype(str)
    df["_IsCancellation"] = df["_InvoiceNo_str"].str.startswith(CANCELLATION_PREFIX, na=False)
    df["_Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["_UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    mask = (
        (~df["_IsCancellation"])
        & (df["_Quantity"] > 0)
        & (df["_UnitPrice"] > 0)
        & (df["CustomerID"].notna())
    )
    out = df.loc[mask].copy()
    out.drop(columns=["_InvoiceNo_str", "_IsCancellation", "_Quantity", "_UnitPrice"], inplace=True)
    return out


def build_target(
    df_positive_sales: pd.DataFrame,
    historical_end: pd.Timestamp,
    future_start: pd.Timestamp,
    future_end: pd.Timestamp,
    datetime_col: str = "InvoiceDate",
) -> pd.DataFrame:
    """
    Build the customer-level binary target DataFrame.

    A customer receives FuturePurchased == 1 if they appear in the future window
    [future_start, future_end]; else 0. The set of customers in the output is
    exactly those customers that have any positive-sales transaction in the
    *historical* window (strictly before historical_end). This guarantees the
    target rows are aligned with the feature rows — no leakage of future-only
    customers into X.

    Parameters
    ----------
    df_positive_sales : pd.DataFrame
        DataFrame with at least [CustomerID, InvoiceDate]. Caller should normally
        run data_preparation first; internally a lightweight filter is also applied.
    historical_end : pd.Timestamp
        Cutoff: historical features use rows < historical_end; future target uses
        rows >= future_start (future_start == historical_end).
    future_start : pd.Timestamp
        Inclusive start of the prediction window (== historical_end per design).
    future_end : pd.Timestamp
        Inclusive end of the prediction window.
    datetime_col : str
        Name of datetime column.

    Returns
    -------
    pd.DataFrame with columns [CustomerID, FuturePurchased]. Row count equals the
    number of distinct customers with at least one historical transaction.
    """
    logger.info("Building customer-level FuturePurchased target (temporal split)")
    np.random.seed(RANDOM_SEED)

    historical_end = pd.Timestamp(historical_end)
    future_start = pd.Timestamp(future_start)
    future_end = pd.Timestamp(future_end)

    df = df_positive_sales.copy()
    df = _filter_positive_sales_transactions(df)
    df["_dt"] = _ensure_datetime(df, datetime_col)
    df = df[df["_dt"].notna()].copy()

    historical_mask = df["_dt"] < historical_end
    future_mask = (df["_dt"] >= future_start) & (df["_dt"] <= future_end)

    hist_customers = df.loc[historical_mask, ID_COLUMN].dropna().astype(float)
    future_customers = df.loc[future_mask, ID_COLUMN].dropna().astype(float)

    hist_unique = pd.Series(hist_customers.unique(), name=ID_COLUMN)
    future_unique_set = set(future_customers.unique())

    if hist_unique.empty:
        raise ValueError("No customers found in historical window — cannot build target")

    target_df = pd.DataFrame({ID_COLUMN: hist_unique.values})
    target_df[TARGET_COLUMN] = target_df[ID_COLUMN].isin(future_unique_set).astype(int)

    n_total = len(target_df)
    n_pos = int(target_df[TARGET_COLUMN].sum())
    n_neg = n_total - n_pos
    pct_pos = n_pos / n_total * 100.0 if n_total else 0.0

    logger.info(
        f"  Target customers: {n_total:,}  |  "
        f"Positive (future purchase): {n_pos:,} ({pct_pos:.1f}%)  |  "
        f"Negative (no future purchase): {n_neg:,} ({100-pct_pos:.1f}%)"
    )
    logger.info(
        f"  Historical transactions used: {int(historical_mask.sum()):,}  |  "
        f"Future transactions used: {int(future_mask.sum()):,}"
    )

    target_df = target_df.sort_values(ID_COLUMN).reset_index(drop=True)
    return target_df
