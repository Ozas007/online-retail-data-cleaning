import logging
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import CANCELLATION_PREFIX, ID_COLUMN, RANDOM_SEED

logger = logging.getLogger(__name__)


def clean_dataset(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Starting full cleaning pipeline")
    np.random.seed(RANDOM_SEED)

    df = df_raw.copy()

    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["TotalAmount"] = qty * price

    dt = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["InvoiceDate"] = dt
    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Hour"] = dt.dt.hour
    df["DOW"] = dt.dt.dayofweek
    df["DOWName"] = dt.dt.day_name()
    df["YearMonth"] = dt.dt.to_period("M").astype(str)
    df["Date"] = dt.dt.date

    invoices = df["InvoiceNo"].astype(str)
    df["IsCancellation"] = invoices.str.startswith(CANCELLATION_PREFIX, na=False)

    before = len(df)
    df = df.drop_duplicates(keep="first").copy()
    logger.info(f"Deduplicated rows: before={before:,}  after={len(df):,}")

    valid_price = pd.to_numeric(df["UnitPrice"], errors="coerce") > 0
    qty_pos = pd.to_numeric(df["Quantity"], errors="coerce") > 0
    not_cancel = ~df["IsCancellation"]
    positive_sales = df.loc[valid_price & not_cancel & qty_pos].copy()
    logger.info(f"Positive sales (non-cancel, Qty>0, Price>0): {len(positive_sales):,}")

    positive_sales_cid = positive_sales.dropna(subset=[ID_COLUMN]).copy()
    logger.info(
        f"With non-null CustomerID: {len(positive_sales_cid):,}  "
        f"(dropped {len(positive_sales) - len(positive_sales_cid):,} anonymous rows)"
    )

    return df, positive_sales_cid
