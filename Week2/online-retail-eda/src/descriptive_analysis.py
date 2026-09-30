import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.config import TABLES_DIR

logger = logging.getLogger(__name__)


def save_table(df: pd.DataFrame, filename: str, output_dir: Path = TABLES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Saved table to: {filepath}")
    return filepath


def dataset_overview(df: pd.DataFrame, raw_columns_count: int = None) -> Dict:
    logger.info("Generating dataset overview")
    cols_count = raw_columns_count if raw_columns_count is not None else len(df.columns)
    overview = {
        "rows": len(df),
        "columns": cols_count,
        "column_names": list(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "n_invoices": int(df["InvoiceNo"].nunique()),
        "n_customers": int(df["CustomerID"].nunique()),
        "n_products": int(df["StockCode"].nunique()),
        "n_countries": int(df["Country"].nunique()),
        "n_descriptions": int(df["Description"].nunique()),
    }
    overview_df = pd.DataFrame([
        {"Metric": k, "Value": (f"{v:,}" if isinstance(v, int) else (", ".join(v) if isinstance(v, list) else v))}
        for k, v in overview.items()
    ])
    save_table(overview_df, "01_dataset_overview.csv")
    return overview


def categorical_frequencies(df: pd.DataFrame, columns: list = None) -> Dict[str, pd.DataFrame]:
    logger.info("Computing categorical frequencies")
    if columns is None:
        columns = ["InvoiceNo", "StockCode", "Description", "Country"]
    results = {}
    for col in columns:
        if col not in df.columns:
            continue
        freq = df[col].value_counts(dropna=False).reset_index()
        freq.columns = [col, "count"]
        freq["percentage"] = (freq["count"] / len(df) * 100).round(3)
        freq["cumulative_percentage"] = freq["percentage"].cumsum().round(3)
        results[col] = freq
        if col == "Country":
            save_table(freq.head(20), "02_country_frequencies.csv")
        if col == "StockCode":
            save_table(freq.head(50), "02_stockcode_frequencies_top50.csv")
    return results


def transaction_amount_stats(df_positive: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing transaction amount statistics")
    stats = pd.DataFrame({
        "metric": ["count", "sum", "mean", "median", "std", "min", "max", "Q1", "Q3", "IQR", "P90", "P95", "P99"],
        "TotalAmount": [
            int(len(df_positive)),
            float(df_positive["TotalAmount"].sum()),
            float(df_positive["TotalAmount"].mean()),
            float(df_positive["TotalAmount"].median()),
            float(df_positive["TotalAmount"].std()),
            float(df_positive["TotalAmount"].min()),
            float(df_positive["TotalAmount"].max()),
            float(df_positive["TotalAmount"].quantile(0.25)),
            float(df_positive["TotalAmount"].quantile(0.75)),
            float(df_positive["TotalAmount"].quantile(0.75) - df_positive["TotalAmount"].quantile(0.25)),
            float(df_positive["TotalAmount"].quantile(0.90)),
            float(df_positive["TotalAmount"].quantile(0.95)),
            float(df_positive["TotalAmount"].quantile(0.99)),
        ],
        "Quantity": [
            int(len(df_positive)),
            float(df_positive["Quantity"].sum()),
            float(df_positive["Quantity"].mean()),
            float(df_positive["Quantity"].median()),
            float(df_positive["Quantity"].std()),
            float(df_positive["Quantity"].min()),
            float(df_positive["Quantity"].max()),
            float(df_positive["Quantity"].quantile(0.25)),
            float(df_positive["Quantity"].quantile(0.75)),
            float(df_positive["Quantity"].quantile(0.75) - df_positive["Quantity"].quantile(0.25)),
            float(df_positive["Quantity"].quantile(0.90)),
            float(df_positive["Quantity"].quantile(0.95)),
            float(df_positive["Quantity"].quantile(0.99)),
        ],
        "UnitPrice": [
            int(len(df_positive)),
            float(df_positive["UnitPrice"].sum()),
            float(df_positive["UnitPrice"].mean()),
            float(df_positive["UnitPrice"].median()),
            float(df_positive["UnitPrice"].std()),
            float(df_positive["UnitPrice"].min()),
            float(df_positive["UnitPrice"].max()),
            float(df_positive["UnitPrice"].quantile(0.25)),
            float(df_positive["UnitPrice"].quantile(0.75)),
            float(df_positive["UnitPrice"].quantile(0.75) - df_positive["UnitPrice"].quantile(0.25)),
            float(df_positive["UnitPrice"].quantile(0.90)),
            float(df_positive["UnitPrice"].quantile(0.95)),
            float(df_positive["UnitPrice"].quantile(0.99)),
        ],
    }).round(4)
    save_table(stats, "03_transaction_amount_stats.csv")
    return stats


def run_descriptive_analysis(df_full: pd.DataFrame, df_positive: pd.DataFrame, raw_columns_count: int = None) -> Dict:
    logger.info("=" * 60)
    logger.info("RUNNING DESCRIPTIVE ANALYSIS")
    logger.info("=" * 60)
    overview = dataset_overview(df_full, raw_columns_count=raw_columns_count)
    cat_freqs = categorical_frequencies(df_full)
    amount_stats = transaction_amount_stats(df_positive)
    num_summary = df_positive[["Quantity", "UnitPrice", "TotalAmount"]].describe().round(4).reset_index()
    save_table(num_summary, "04_numeric_summary.csv")
    logger.info("DESCRIPTIVE ANALYSIS COMPLETE")
    return {"overview": overview, "cat_freqs": cat_freqs, "amount_stats": amount_stats}
