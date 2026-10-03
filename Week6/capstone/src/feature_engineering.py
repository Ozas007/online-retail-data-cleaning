import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, ID_COLUMN, RANDOM_SEED

logger = logging.getLogger(__name__)


def build_rfm_features(
    df_positive_sales: pd.DataFrame,
    reference_date: Optional[datetime] = None,
) -> pd.DataFrame:
    logger.info("Building RFM features")
    np.random.seed(RANDOM_SEED)

    required_cols = [
        ID_COLUMN, "InvoiceNo", "InvoiceDate",
        "Quantity", "UnitPrice", "TotalAmount",
    ]
    missing_cols = [c for c in required_cols if c not in df_positive_sales.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df_positive_sales.empty:
        raise ValueError("Input DataFrame is empty")

    df = df_positive_sales.copy()
    before = len(df)
    df = df[df[ID_COLUMN].notna()]
    after = len(df)
    logger.info(f"Filtered out {before - after} rows with null {ID_COLUMN}")

    if df.empty:
        raise ValueError("No rows remain after filtering null CustomerID")

    dt = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["_dt"] = dt

    if reference_date is None:
        max_date = df["_dt"].max()
        reference_date = max_date + timedelta(days=1)
        logger.info(f"Derived reference_date: {reference_date} (from max InvoiceDate {max_date})")

    reference_ts = pd.Timestamp(reference_date)

    rfm = (
        df.groupby(ID_COLUMN)
        .agg(
            Recency=("_dt", lambda x: (reference_ts - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        )
        .reset_index()
    )

    logger.info(f"Computed RFM for {len(rfm)} customers")
    return rfm


def _mode(series: pd.Series):
    modes = series.mode(dropna=True)
    if modes.empty:
        return np.nan
    return modes.iloc[0]


def build_supervised_features(
    df_historical: pd.DataFrame,
    historical_end: pd.Timestamp,
    historical_start: Optional[pd.Timestamp] = None,
    datetime_col: str = "InvoiceDate",
) -> pd.DataFrame:
    logger.info("Building supervised customer-level features from historical transactions")
    np.random.seed(RANDOM_SEED)

    historical_end = pd.Timestamp(historical_end)

    df = df_historical.copy()
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

    ordered_cols = [ID_COLUMN] + [c for c in FEATURE_COLUMNS if c in features.columns]
    features = features[ordered_cols]

    logger.info(
        f"  Feature rows (customers): {len(features):,}  |  "
        f"Nulls: {features.isna().sum().sum()} total"
    )
    return features
