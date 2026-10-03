import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def compute_rfm(
    df_positive_sales: pd.DataFrame,
    reference_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) metrics for each customer.

    Parameters
    ----------
    df_positive_sales : pd.DataFrame
        DataFrame containing positive sales transactions. Required columns:
        CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice, TotalAmount.
    reference_date : Optional[datetime], optional
        Reference date for computing Recency. If None, defaults to
        max(InvoiceDate) + 1 day, derived from the data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - CustomerID : Unique customer identifier
        - Recency : (reference_date - max InvoiceDate).days — days since last purchase
        - Frequency : count of distinct InvoiceNo values — total number of orders
        - Monetary : sum of TotalAmount (positive sales only) — total revenue from customer

    Raises
    ------
    ValueError
        If required columns are missing or input DataFrame is empty.
    """
    logger.info("Starting RFM computation")
    np.random.seed(RANDOM_SEED)

    required_cols = [
        "CustomerID", "InvoiceNo", "InvoiceDate",
        "Quantity", "UnitPrice", "TotalAmount",
    ]
    missing_cols = [c for c in required_cols if c not in df_positive_sales.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df_positive_sales.empty:
        raise ValueError("Input DataFrame is empty")

    df = df_positive_sales.copy()
    before = len(df)
    df = df[df["CustomerID"].notna()]
    after = len(df)
    logger.info(f"Filtered out {before - after} rows with null CustomerID")

    if df.empty:
        raise ValueError("No rows remain after filtering null CustomerID")

    if reference_date is None:
        max_date = df["InvoiceDate"].max()
        reference_date = max_date + timedelta(days=1)
        logger.info(f"Derived reference_date: {reference_date} (from max InvoiceDate {max_date})")

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        )
        .reset_index()
    )

    logger.info(f"Computed RFM for {len(rfm)} customers")
    logger.debug(f"RFM head:\n{rfm.head()}")
    return rfm


def inspect_rfm_distributions(rfm_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Inspect RFM column distributions with describe, skewness, and kurtosis.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        DataFrame containing at least columns Recency, Frequency, Monetary.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary keyed by column name. Each value is a DataFrame row
        containing the describe() statistics plus skew and kurtosis.
    """
    logger.info("Inspecting RFM distributions")
    rfm_cols = ["Recency", "Frequency", "Monetary"]
    missing = [c for c in rfm_cols if c not in rfm_df.columns]
    if missing:
        raise ValueError(f"Missing RFM columns: {missing}")

    result: Dict[str, pd.DataFrame] = {}
    for col in rfm_cols:
        series = rfm_df[col]
        desc = series.describe().to_frame().T
        desc["skew"] = series.skew()
        desc["kurtosis"] = series.kurtosis()
        result[col] = desc
        logger.info(
            f"{col}: mean={desc['mean'].values[0]:.2f}, "
            f"skew={desc['skew'].values[0]:.2f}, kurtosis={desc['kurtosis'].values[0]:.2f}"
        )

    return result


def apply_log1p_transformation(
    rfm_df: pd.DataFrame,
    columns: List[str] = None,
) -> pd.DataFrame:
    """
    Apply log1p transformation to specified RFM columns.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Source DataFrame containing the columns to transform.
    columns : List[str], optional
        Columns to transform. Defaults to ["Recency", "Frequency", "Monetary"].

    Returns
    -------
    pd.DataFrame
        Copy of rfm_df with additional Log_<col> columns for each input column,
        computed using np.log1p.
    """
    logger.info("Applying log1p transformation")
    if columns is None:
        columns = ["Recency", "Frequency", "Monetary"]

    missing = [c for c in columns if c not in rfm_df.columns]
    if missing:
        raise ValueError(f"Missing columns for log transform: {missing}")

    result = rfm_df.copy()
    for col in columns:
        log_col = f"Log_{col}"
        result[log_col] = np.log1p(result[col])
        logger.info(
            f"Created {log_col}: original min/max = "
            f"{result[col].min():.2f}/{result[col].max():.2f}, "
            f"log min/max = {result[log_col].min():.2f}/{result[log_col].max():.2f}"
        )

    return result


def scale_rfm_features(
    rfm_df: pd.DataFrame,
    feature_cols: List[str] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize RFM features using sklearn StandardScaler (zero mean, unit variance).

    Parameters
    ----------
    rfm_df : pd.DataFrame
        DataFrame containing the feature columns.
    feature_cols : List[str], optional
        Columns to scale. Defaults to ["Recency", "Frequency", "Monetary"].
        Log columns (e.g. ["Log_Recency", "Log_Frequency", "Log_Monetary"])
        may also be supplied.

    Returns
    -------
    Tuple[np.ndarray, StandardScaler]
        - scaled_array : 2D numpy array of scaled features, shape (n_samples, len(feature_cols))
        - scaler_fitted : Fitted sklearn StandardScaler instance (useful for inverse transform)
    """
    logger.info("Scaling RFM features with StandardScaler")
    if feature_cols is None:
        feature_cols = ["Recency", "Frequency", "Monetary"]

    missing = [c for c in feature_cols if c not in rfm_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for scaling: {missing}")

    features = rfm_df[feature_cols].values
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(features)

    logger.info(
        f"Scaled {scaled_array.shape[0]} samples across {scaled_array.shape[1]} features. "
        f"Fitted means: {scaler.mean_}, stds: {scaler.scale_}"
    )
    return scaled_array, scaler
