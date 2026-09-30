import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.descriptive_analysis import save_table

logger = logging.getLogger(__name__)


def pearson_correlation(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    logger.info("Computing Pearson correlation matrix")
    if columns is None:
        columns = ["Quantity", "UnitPrice", "TotalAmount"]
    corr = df[columns].corr(method="pearson").round(4)
    save_table(corr.reset_index(), "16_correlation_matrix.csv")
    logger.info(f"Correlation matrix:\n{corr.to_string()}")
    return corr


def quantity_vs_totalamount(df_positive: pd.DataFrame, sample_size: int = 50000,
                            seed: int = 42) -> Dict:
    logger.info("Analyzing Quantity vs TotalAmount relationship")
    df_use = df_positive[["Quantity", "UnitPrice", "TotalAmount"]].copy()
    df_use = df_use[(df_use["Quantity"] > 0) & (df_use["TotalAmount"] > 0)].dropna()
    corr_p = float(df_use["Quantity"].corr(df_use["TotalAmount"], method="pearson"))
    corr_s = float(df_use["Quantity"].corr(df_use["TotalAmount"], method="spearman"))
    if len(df_use) > sample_size:
        df_sample = df_use.sample(n=sample_size, random_state=seed)
    else:
        df_sample = df_use
    stats = pd.DataFrame([
        {"metric": "Valid rows (qty>0, amount>0)", "value": int(len(df_use))},
        {"metric": "Pearson r (Quantity vs TotalAmount)", "value": round(corr_p, 4)},
        {"metric": "Spearman rho (Quantity vs TotalAmount)", "value": round(corr_s, 4)},
        {"metric": "Sampled rows for scatter plot", "value": int(len(df_sample))},
        {"metric": "Qty Q1/Q3 (for plot bounds)", "value": f"{df_use['Quantity'].quantile(0.01):.0f} / {df_use['Quantity'].quantile(0.99):.0f}"},
        {"metric": "Amt Q1/Q3 (for plot bounds)", "value": f"{df_use['TotalAmount'].quantile(0.01):.1f} / {df_use['TotalAmount'].quantile(0.99):.1f}"},
    ])
    save_table(stats, "17_quantity_vs_totalamount_stats.csv")
    logger.info(f"Pearson r(Qty,Amt)={corr_p:.4f}, Spearman={corr_s:.4f}")
    return {"correlation": corr_p, "spearman": corr_s, "scatter_sample": df_sample, "stats": stats, "full": df_use}


def unitprice_vs_quantity(df_positive: pd.DataFrame, sample_size: int = 50000,
                          seed: int = 42) -> Dict:
    logger.info("Analyzing UnitPrice vs Quantity relationship")
    df_use = df_positive[["Quantity", "UnitPrice", "TotalAmount"]].copy()
    df_use = df_use[(df_use["Quantity"] > 0) & (df_use["UnitPrice"] > 0)].dropna()
    corr_p = float(df_use["UnitPrice"].corr(df_use["Quantity"], method="pearson"))
    corr_s = float(df_use["UnitPrice"].corr(df_use["Quantity"], method="spearman"))
    if len(df_use) > sample_size:
        df_sample = df_use.sample(n=sample_size, random_state=seed)
    else:
        df_sample = df_use
    stats = pd.DataFrame([
        {"metric": "Valid rows (qty>0, price>0)", "value": int(len(df_use))},
        {"metric": "Pearson r (UnitPrice vs Quantity)", "value": round(corr_p, 4)},
        {"metric": "Spearman rho (UnitPrice vs Quantity)", "value": round(corr_s, 4)},
        {"metric": "Sampled rows for scatter plot", "value": int(len(df_sample))},
    ])
    save_table(stats, "18_unitprice_vs_quantity_stats.csv")
    logger.info(f"Pearson r(Price,Qty)={corr_p:.4f}, Spearman={corr_s:.4f}")
    return {"correlation": corr_p, "spearman": corr_s, "scatter_sample": df_sample, "stats": stats, "full": df_use}


def run_relationship_analysis(df_positive: pd.DataFrame) -> Dict:
    logger.info("=" * 60)
    logger.info("RUNNING RELATIONSHIP / CORRELATION ANALYSIS")
    logger.info("=" * 60)
    corr_matrix = pearson_correlation(df_positive)
    qty_vs_amt = quantity_vs_totalamount(df_positive)
    price_vs_qty = unitprice_vs_quantity(df_positive)
    logger.info("RELATIONSHIP ANALYSIS COMPLETE")
    return {
        "correlation_matrix": corr_matrix,
        "qty_vs_totalamount": qty_vs_amt,
        "unitprice_vs_quantity": price_vs_qty,
    }
