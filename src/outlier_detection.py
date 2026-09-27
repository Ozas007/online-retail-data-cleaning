import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import IQR_MULTIPLIER, TABLES_DIR

logger = logging.getLogger(__name__)


def compute_iqr_bounds(
    series: pd.Series,
    multiplier: float = IQR_MULTIPLIER,
    drop_negative: bool = False,
) -> Dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if drop_negative:
        clean = clean[clean > 0]

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "n": int(len(clean)),
    }


def flag_iqr_outliers(
    df: pd.DataFrame,
    column: str,
    multiplier: float = IQR_MULTIPLIER,
    consider_positive_only: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float], int, float]:
    logger.info(f"Running IQR outlier detection on column: {column}")
    df = df.copy()
    numeric = pd.to_numeric(df[column], errors="coerce")

    bounds = compute_iqr_bounds(numeric, multiplier=multiplier, drop_negative=consider_positive_only)

    if consider_positive_only:
        mask_non_positive = numeric <= 0
        outlier_mask = (
            (~mask_non_positive) &
            (
                (numeric < bounds["lower_bound"]) |
                (numeric > bounds["upper_bound"])
            )
        )
    else:
        outlier_mask = (numeric < bounds["lower_bound"]) | (numeric > bounds["upper_bound"])

    flag_col = f"{column}_IsOutlier"
    df[flag_col] = outlier_mask.fillna(False)

    outlier_count = int(df[flag_col].sum())
    outlier_pct = (outlier_count / len(df)) * 100 if len(df) > 0 else 0.0
    logger.info(f"  {column}: {outlier_count:,} outliers ({outlier_pct:.2f}%) | bounds=[{bounds['lower_bound']:.2f}, {bounds['upper_bound']:.2f}]")

    return df, bounds, outlier_count, outlier_pct


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str] = None,
    multiplier: float = IQR_MULTIPLIER,
    output_dir: Path = TABLES_DIR,
) -> Dict:
    logger.info("=" * 60)
    logger.info("RUNNING IQR OUTLIER DETECTION")
    logger.info("=" * 60)

    if columns is None:
        columns = ["Quantity", "UnitPrice"]

    if "TotalAmount" in df.columns and "TotalAmount" not in columns:
        columns = columns + ["TotalAmount"]

    results = {}
    df_out = df.copy()
    summary_rows = []

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue
        consider_positive = col in ["Quantity", "UnitPrice", "TotalAmount"]
        df_out, bounds, count, pct = flag_iqr_outliers(
            df_out, col, multiplier=multiplier, consider_positive_only=consider_positive
        )
        summary_rows.append({
            "column": col,
            "Q1": bounds["Q1"],
            "Q3": bounds["Q3"],
            "IQR": bounds["IQR"],
            "lower_bound": bounds["lower_bound"],
            "upper_bound": bounds["upper_bound"],
            "outlier_count": count,
            "outlier_percentage": round(pct, 4),
            "column_min": bounds["min"],
            "column_max": bounds["max"],
            "column_mean": bounds["mean"],
            "column_median": bounds["median"],
        })
        results[f"{col}_bounds"] = bounds
        results[f"{col}_outlier_count"] = count
        results[f"{col}_outlier_pct"] = pct

    summary_df = pd.DataFrame(summary_rows)
    results["summary"] = summary_df
    results["dataset_flagged"] = df_out

    if len(summary_df) > 0 and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "outlier_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Outlier summary saved to: {summary_path}")

    logger.info("=" * 60)
    logger.info("OUTLIER DETECTION COMPLETE")
    logger.info("=" * 60)
    return results


def extreme_values(df: pd.DataFrame, column: str, n: int = 10, largest: bool = True) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()
    series = pd.to_numeric(df[column], errors="coerce")
    if largest:
        idx = series.nlargest(n).index
    else:
        idx = series.nsmallest(n).index
    cols = [c for c in ["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "CustomerID", "Country", column] if c in df.columns]
    return df.loc[idx, cols].copy()


def outlier_impact_analysis(
    df: pd.DataFrame,
    column: str,
    outlier_flag_col: str = None,
) -> pd.DataFrame:
    if outlier_flag_col is None:
        outlier_flag_col = f"{column}_IsOutlier"
    if outlier_flag_col not in df.columns:
        logger.warning(f"Outlier flag column {outlier_flag_col} not found")
        return pd.DataFrame()

    numeric = pd.to_numeric(df[column], errors="coerce")
    full = numeric.dropna()
    non_outliers = numeric[~df[outlier_flag_col].fillna(False)].dropna()

    rows = []
    for label, data in [("With outliers", full), ("Without outliers", non_outliers)]:
        rows.append({
            "subset": label,
            "count": int(len(data)),
            "mean": float(data.mean()),
            "median": float(data.median()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "sum": float(data.sum()),
        })
    impact = pd.DataFrame(rows)
    return impact
