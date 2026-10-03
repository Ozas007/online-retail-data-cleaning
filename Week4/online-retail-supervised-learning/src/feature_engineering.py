import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    RANDOM_SEED,
    ID_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


def _mode(series: pd.Series):
    modes = series.mode(dropna=True)
    if modes.empty:
        return np.nan
    return modes.iloc[0]


def build_features(
    df_historical_transactions: pd.DataFrame,
    historical_end: pd.Timestamp,
    historical_start: Optional[pd.Timestamp] = None,
    datetime_col: str = "InvoiceDate",
) -> pd.DataFrame:
    """
    Build customer-level features from STRICTLY historical transactions only
    (InvoiceDate < historical_end). All aggregations use rows before the cutoff;
    no information from the future window enters the feature matrix.

    Features produced:
      - Recency: days from customer's last purchase to historical_end
      - Frequency: number of distinct invoices placed
      - Monetary: sum of line-item TotalAmount (Quantity * UnitPrice)
      - AOV: Monetary / Frequency (average order value)
      - UniqueProducts: nunique StockCode
      - AvgQuantity: mean Quantity across line items
      - TotalQuantity: sum of Quantity across line items
      - PurchaseFrequency: Frequency / days_in_historical_period (customer's orders per day)
      - Country: most common Country for the customer

    Parameters
    ----------
    df_historical_transactions : pd.DataFrame
        Transaction lines. Should be pre-filtered to the historical window by the
        caller; this function applies an additional strict filter (< historical_end)
        as a leakage guard.
    historical_end : pd.Timestamp
        Exclusive upper bound for InvoiceDate. Any row >= historical_end is dropped.
    historical_start : Optional[pd.Timestamp]
        Inclusive lower bound (used only to compute days_in_period for PurchaseFrequency).
        If None, the minimum observed InvoiceDate in the input is used.
    datetime_col : str
        Name of the datetime column.

    Returns
    -------
    pd.DataFrame with columns [CustomerID] + FEATURE_COLUMNS (in config order).
    One row per customer with at least one historical transaction.
    """
    logger.info("Building customer-level features from historical transactions")
    np.random.seed(RANDOM_SEED)

    historical_end = pd.Timestamp(historical_end)

    df = df_historical_transactions.copy()
    df["_dt"] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df[df["_dt"].notna()].copy()

    strict_hist_mask = df["_dt"] < historical_end
    n_leaked = int((~strict_hist_mask).sum())
    if n_leaked > 0:
        logger.warning(
            f"Dropped {n_leaked:,} transaction rows with InvoiceDate >= historical_end "
            f"({historical_end.date()}) — prevents future leakage into X"
        )
        df = df.loc[strict_hist_mask].copy()

    df = df[df[ID_COLUMN].notna()].copy()
    if df.empty:
        raise ValueError("No valid historical transactions after filtering — cannot build features")

    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["_TotalAmount"] = qty * price

    if historical_start is None:
        historical_start = df["_dt"].min()
    else:
        historical_start = pd.Timestamp(historical_start)

    days_in_period = max(1, int((historical_end - historical_start).days))
    logger.info(
        f"  Historical period: {historical_start.date()} -> {historical_end.date()} "
        f"({days_in_period} days)  |  Historical transactions: {len(df):,}"
    )

    recency_ref = historical_end

    features = (
        df.groupby(ID_COLUMN, dropna=False)
        .agg(
            Recency=("_dt", lambda x: (recency_ref - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("_TotalAmount", "sum"),
            UniqueProducts=("StockCode", "nunique"),
            AvgQuantity=("Quantity", "mean"),
            TotalQuantity=("Quantity", "sum"),
            Country=("Country", _mode),
        )
        .reset_index()
    )

    features["AOV"] = features["Monetary"] / features["Frequency"]
    features["AOV"] = features["AOV"].replace([np.inf, -np.inf], np.nan)

    features["PurchaseFrequency"] = features["Frequency"] / days_in_period
    features["PurchaseFrequency"] = features["PurchaseFrequency"].replace([np.inf, -np.inf], np.nan)

    numeric_cols = ["AvgQuantity", "TotalQuantity", "Monetary", "AOV", "Recency", "Frequency", "UniqueProducts", "PurchaseFrequency"]
    for c in numeric_cols:
        features[c] = pd.to_numeric(features[c], errors="coerce")

    features = features[[ID_COLUMN] + FEATURE_COLUMNS]

    logger.info(
        f"  Feature rows (customers): {len(features):,}  |  "
        f"Nulls: {features.isna().sum().sum()} total"
    )
    logger.debug(f"Features head:\n{features.head(3).to_string()}")
    return features
