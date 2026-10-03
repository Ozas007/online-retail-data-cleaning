import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, TABLES_DIR

logger = logging.getLogger(__name__)


def run_eda(df_clean: pd.DataFrame, df_positive: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    logger.info("Running EDA and building summary tables")
    np.random.seed(RANDOM_SEED)

    results: Dict[str, pd.DataFrame] = {}

    overview = pd.DataFrame({
        "metric": [
            "raw_rows",
            "unique_invoices",
            "unique_stockcodes",
            "unique_customers",
            "unique_countries",
            "date_min",
            "date_max",
            "total_revenue_positive",
        ],
        "value": [
            len(df_clean),
            df_clean["InvoiceNo"].nunique(),
            df_clean["StockCode"].nunique(),
            df_positive["CustomerID"].nunique() if "CustomerID" in df_positive.columns else df_clean["CustomerID"].nunique(),
            df_clean["Country"].nunique(),
            str(pd.to_datetime(df_clean["InvoiceDate"], errors="coerce").min().date()) if pd.to_datetime(df_clean["InvoiceDate"], errors="coerce").notna().any() else "",
            str(pd.to_datetime(df_clean["InvoiceDate"], errors="coerce").max().date()) if pd.to_datetime(df_clean["InvoiceDate"], errors="coerce").notna().any() else "",
            float(df_positive["TotalAmount"].sum()) if "TotalAmount" in df_positive.columns else float(pd.to_numeric(df_clean["Quantity"], errors="coerce").fillna(0).mul(pd.to_numeric(df_clean["UnitPrice"], errors="coerce").fillna(0)).sum()),
        ],
    })
    results["overview"] = overview

    missing = pd.DataFrame({
        "column": list(df_clean.columns),
        "missing_count": [int(df_clean[c].isna().sum()) for c in df_clean.columns],
        "missing_pct": [float(df_clean[c].isna().mean() * 100) for c in df_clean.columns],
    }).sort_values("missing_count", ascending=False).reset_index(drop=True)
    results["missingness"] = missing

    dt_col = pd.to_datetime(df_positive["InvoiceDate"], errors="coerce")
    ym = dt_col.dt.to_period("M").astype(str)
    monthly = df_positive.assign(YearMonth=ym).groupby("YearMonth").agg(
        revenue=("TotalAmount", "sum"),
        transactions=("InvoiceNo", "nunique"),
        items_sold=("Quantity", "sum"),
        unique_customers=("CustomerID", "nunique"),
    ).reset_index().sort_values("YearMonth").reset_index(drop=True)
    results["monthly_volume"] = monthly

    product_rev = df_positive.groupby(["StockCode", "Description"]).agg(
        revenue=("TotalAmount", "sum"),
        quantity_sold=("Quantity", "sum"),
        transactions=("InvoiceNo", "nunique"),
    ).reset_index().sort_values("revenue", ascending=False).head(10).reset_index(drop=True)
    results["top10_products_by_revenue"] = product_rev

    country_trx = df_positive.groupby("Country").agg(
        transactions=("InvoiceNo", "nunique"),
        revenue=("TotalAmount", "sum"),
        unique_customers=("CustomerID", "nunique"),
    ).reset_index().sort_values("transactions", ascending=False).head(10).reset_index(drop=True)
    results["top10_countries_by_transactions"] = country_trx

    numeric_cols = ["Quantity", "UnitPrice", "TotalAmount"]
    cols_available = [c for c in numeric_cols if c in df_positive.columns]
    corr_df = df_positive[cols_available].apply(pd.to_numeric, errors="coerce").corr()
    corr_melted = corr_df.reset_index().melt(id_vars="index", var_name="col2", value_name="correlation")
    corr_melted.columns = ["col1", "col2", "correlation"]
    results["numeric_correlation_matrix"] = corr_melted

    numeric_desc = df_positive[cols_available].apply(pd.to_numeric, errors="coerce").describe()
    numeric_desc = numeric_desc.reset_index().rename(columns={"index": "statistic"})
    results["numeric_summaries"] = numeric_desc

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for key, df_result in results.items():
        fname = f"eda_{key}.csv"
        path = TABLES_DIR / fname
        df_result.to_csv(path, index=False)
        logger.info(f"Saved EDA table {key} -> {path}")

    return results
