import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import IQR_MULTIPLIER, TABLES_DIR
from src.descriptive_analysis import save_table

logger = logging.getLogger(__name__)


def compute_iqr_bounds(series: pd.Series, multiplier: float = IQR_MULTIPLIER,
                       drop_negative: bool = True) -> Dict:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if drop_negative:
        clean = clean[clean > 0]
    q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
    iqr = q3 - q1
    return {
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower": q1 - multiplier * iqr,
        "upper": q3 + multiplier * iqr,
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "n": int(len(clean)),
    }


def iqr_outlier_summary(df_positive: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    logger.info("Running IQR outlier summary")
    if columns is None:
        columns = ["Quantity", "UnitPrice", "TotalAmount"]
    rows = []
    for col in columns:
        if col not in df_positive.columns:
            continue
        bounds = compute_iqr_bounds(df_positive[col], multiplier=IQR_MULTIPLIER, drop_negative=True)
        s = pd.to_numeric(df_positive[col], errors="coerce").dropna()
        s = s[s > 0]
        mask = (s < bounds["lower"]) | (s > bounds["upper"])
        count = int(mask.sum())
        pct = count / len(s) * 100 if len(s) > 0 else 0
        rows.append({
            "column": col, "Q1": bounds["Q1"], "Q3": bounds["Q3"], "IQR": bounds["IQR"],
            "lower_bound": bounds["lower"], "upper_bound": bounds["upper"],
            "outlier_count": count, "outlier_pct": round(pct, 3),
            "min": float(s.min()), "max": float(s.max()), "mean": bounds["mean"],
        })
    summary = pd.DataFrame(rows)
    save_table(summary.round(4), "19_iqr_outlier_summary.csv")
    logger.info(f"Outlier summary:\n{summary[['column','outlier_count','outlier_pct']].to_string(index=False)}")
    return summary


def concrete_anomaly_examples(df_full: pd.DataFrame, df_positive: pd.DataFrame,
                              n_examples: int = 5) -> Dict[str, pd.DataFrame]:
    logger.info("Collecting concrete anomaly examples from actual dataset")
    examples = {}

    high_qty = df_positive.sort_values("Quantity", ascending=False).head(n_examples)
    cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "TotalAmount", "InvoiceDate", "Country"]
    cols_available = [c for c in cols if c in df_full.columns]
    examples["largest_quantity"] = high_qty[cols_available].reset_index(drop=True)

    high_amount = df_positive.sort_values("TotalAmount", ascending=False).head(n_examples)
    examples["highest_totalamount"] = high_amount[cols_available].reset_index(drop=True)

    cancel_mask = df_full["InvoiceNo"].astype(str).str.startswith("C", na=False)
    cancels = df_full[cancel_mask].sort_values("Quantity", ascending=True).head(n_examples)
    examples["cancellation_examples"] = cancels[cols_available].reset_index(drop=True)

    zero_price_mask = pd.to_numeric(df_full["UnitPrice"], errors="coerce") <= 0
    zero_price = df_full[zero_price_mask].head(n_examples)
    examples["zero_or_negative_price"] = zero_price[cols_available].reset_index(drop=True)

    for name, df in examples.items():
        save_table(df.round(2), f"20_anomaly_{name}_examples.csv")
        logger.info(f"  {name}: {len(df)} examples saved")

    return examples


def percentile_analysis(df_positive: pd.DataFrame) -> pd.DataFrame:
    logger.info("Running percentile analysis for heavy-tail assessment")
    cols = ["Quantity", "UnitPrice", "TotalAmount"]
    percentiles = [0.5, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
    rows = []
    for col in cols:
        s = pd.to_numeric(df_positive[col], errors="coerce").dropna()
        s = s[s > 0]
        row = {"column": col, "mean": float(s.mean()), "median": float(s.median())}
        for p in percentiles:
            row[f"P{int(p*100)}"] = float(s.quantile(p))
        row["P99/P50_ratio"] = round(float(s.quantile(0.99) / s.median()), 2) if s.median() > 0 else None
        row["Max/P99_ratio"] = round(float(s.max() / s.quantile(0.99)), 2) if s.quantile(0.99) > 0 else None
        rows.append(row)
    pct = pd.DataFrame(rows)
    save_table(pct.round(4), "21_percentile_heavy_tail_analysis.csv")
    logger.info(f"Percentile analysis complete; heavy-tail ratios captured for {len(cols)} columns")
    return pct


def run_anomaly_analysis(df_full: pd.DataFrame, df_positive: pd.DataFrame) -> Dict:
    logger.info("=" * 60)
    logger.info("RUNNING ANOMALY / OUTLIER ANALYSIS")
    logger.info("=" * 60)
    iqr = iqr_outlier_summary(df_positive)
    examples = concrete_anomaly_examples(df_full, df_positive)
    pct = percentile_analysis(df_positive)
    logger.info("ANOMALY ANALYSIS COMPLETE")
    return {"iqr_summary": iqr, "examples": examples, "percentile_analysis": pct}
