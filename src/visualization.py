import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


def _save(fig: plt.Figure, filename: str, output_dir: Path = FIGURES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved figure to: {filepath}")
    return filepath


def plot_missing_values(df: pd.DataFrame, filename: str = "missing_values_bar.png") -> Path:
    logger.info("Plotting missing values by column")
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df)) * 100
    non_zero = missing[missing > 0]
    if non_zero.empty:
        logger.info("No missing values; skipping plot")
        return Path()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(non_zero.index, missing_pct[non_zero.index].values, color="#c44e52")
    ax.set_title("Percentage of Missing Values by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing (%)")
    ax.set_ylim(0, max(missing_pct[non_zero.index].values) * 1.15 if len(non_zero) else 1)
    for bar, val in zip(bars, missing_pct[non_zero.index].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=10)
    return _save(fig, filename)


def plot_distribution(series: pd.Series, title: str, xlabel: str, filename: str,
                      log_scale: bool = False, drop_zero_or_neg: bool = True) -> Path:
    logger.info(f"Plotting distribution: {title}")
    s = pd.to_numeric(series, errors="coerce").dropna()
    if drop_zero_or_neg or log_scale:
        s = s[s > 0]
    if log_scale and len(s) > 0:
        s = np.log1p(s)
        xlabel = f"{xlabel} (log1p scale)"

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(s, kde=True, color="#4c72b0", ax=ax, bins=50)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.axvline(s.mean(), color="red", linestyle="--", label=f"Mean: {s.mean():.2f}")
    ax.axvline(s.median(), color="green", linestyle="-.", label=f"Median: {s.median():.2f}")
    ax.legend()
    return _save(fig, filename)


def plot_boxplot(series: pd.Series, title: str, ylabel: str, filename: str,
                 drop_zero_or_neg: bool = True) -> Path:
    logger.info(f"Plotting boxplot: {title}")
    s = pd.to_numeric(series, errors="coerce").dropna()
    if drop_zero_or_neg:
        s = s[s > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(y=s, color="#55a868", ax=ax, flierprops={"marker": "o", "alpha": 0.3})
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return _save(fig, filename)


def plot_country_distribution(df: pd.DataFrame, filename: str = "country_distribution.png",
                              top_n: int = 10) -> Path:
    logger.info("Plotting country distribution")
    if "Country" not in df.columns:
        return Path()
    counts = df["Country"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(counts)), counts.values, color="#8172b3")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=10)
    ax.set_title(f"Top {top_n} Countries by Transaction Count")
    ax.set_ylabel("Transaction Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}",
                ha="center", va="bottom", fontsize=9)
    return _save(fig, filename)


def plot_monthly_transactions(df: pd.DataFrame, filename: str = "monthly_transactions.png") -> Path:
    logger.info("Plotting monthly transaction trend")
    temp = df.copy()
    temp["InvoiceDate"] = pd.to_datetime(temp["InvoiceDate"], errors="coerce")
    temp = temp.dropna(subset=["InvoiceDate"])
    if temp.empty:
        return Path()
    temp["YearMonth"] = temp["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = temp.groupby("YearMonth").size()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly.index, monthly.values, marker="o", color="#ccb974", linewidth=2)
    ax.fill_between(monthly.index, monthly.values, alpha=0.2, color="#ccb974")
    ax.set_title("Monthly Transaction Volume")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Transactions")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_total_amount_distribution(df: pd.DataFrame, filename: str = "total_amount_distribution.png") -> Path:
    if "TotalAmount" not in df.columns:
        return Path()
    return plot_distribution(
        df["TotalAmount"],
        title="Total Transaction Amount Distribution",
        xlabel="TotalAmount (Quantity x UnitPrice)",
        filename=filename,
        log_scale=True,
        drop_zero_or_neg=True,
    )


def plot_before_after_comparison(comparison_df: pd.DataFrame,
                                 filename: str = "before_after_comparison.png") -> Path:
    logger.info("Plotting before/after data quality comparison")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("Row Count", "Row Count", "#4c72b0"),
        ("Total Missing Values", "Total Missing Values", "#c44e52"),
        ("Negative Quantity Count", "Negative Qty Count", "#dd8452"),
    ]
    for ax, (key, title, color) in zip(axes, metrics):
        if key in comparison_df.columns:
            bars = ax.bar(comparison_df["Dataset"], comparison_df[key], color=color)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=30)
            for bar, val in zip(bars, comparison_df[key].values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{int(val):,}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Before / After Data Quality Comparison", fontsize=15)
    fig.tight_layout()
    return _save(fig, filename)


def plot_cancellation_analysis(df: pd.DataFrame, filename: str = "cancellation_analysis.png") -> Path:
    logger.info("Plotting cancellation analysis")
    if "IsCancellation" not in df.columns:
        invoices = df["InvoiceNo"].astype(str)
        is_cancel = invoices.str.startswith("C", na=False)
    else:
        is_cancel = df["IsCancellation"]
    counts = [int((~is_cancel).sum()), int(is_cancel.sum())]
    labels = ["Regular", "Cancellation"]
    colors = ["#55a868", "#c44e52"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].pie(counts, labels=labels, autopct="%1.2f%%", colors=colors, startangle=90,
                textprops={"fontsize": 11})
    axes[0].set_title("Cancellations vs Regular Transactions")

    if "IsCancellation" in df.columns:
        neg_by_cancel = df.groupby("IsCancellation")["IsNegativeQuantity"].sum()
        if len(neg_by_cancel) > 0:
            neg_by_cancel.plot(kind="bar", ax=axes[1], color=colors)
            axes[1].set_title("Negative Quantity Count by Cancellation Flag")
            axes[1].set_ylabel("Count")
            axes[1].set_xlabel("IsCancellation")
            axes[1].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    return _save(fig, filename)


def generate_all_visualizations(
    df_raw: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    df_positive_sales: pd.DataFrame,
    comparison_df: pd.DataFrame = None,
) -> Dict[str, Path]:
    logger.info("=" * 60)
    logger.info("GENERATING ALL VISUALIZATIONS")
    logger.info("=" * 60)

    paths: Dict[str, Path] = {}

    paths["missing_values"] = plot_missing_values(df_raw)

    for data_label, df_use in [("cleaned", df_cleaned), ("positive", df_positive_sales)]:
        paths[f"qty_dist_{data_label}"] = plot_distribution(
            df_use["Quantity"], f"Quantity Distribution ({data_label})",
            "Quantity", f"quantity_distribution_{data_label}.png", log_scale=True
        )
        paths[f"price_dist_{data_label}"] = plot_distribution(
            df_use["UnitPrice"], f"UnitPrice Distribution ({data_label})",
            "UnitPrice", f"unitprice_distribution_{data_label}.png", log_scale=True
        )
        paths[f"qty_box_{data_label}"] = plot_boxplot(
            df_use["Quantity"], f"Quantity Boxplot ({data_label})",
            "Quantity", f"quantity_boxplot_{data_label}.png"
        )
        paths[f"price_box_{data_label}"] = plot_boxplot(
            df_use["UnitPrice"], f"UnitPrice Boxplot ({data_label})",
            "UnitPrice", f"unitprice_boxplot_{data_label}.png"
        )
        paths[f"total_amount_{data_label}"] = plot_total_amount_distribution(
            df_use, f"total_amount_distribution_{data_label}.png"
        )

    paths["country_distribution"] = plot_country_distribution(df_cleaned)
    paths["monthly_transactions"] = plot_monthly_transactions(df_cleaned)
    paths["cancellation_analysis"] = plot_cancellation_analysis(df_cleaned)

    if comparison_df is not None:
        paths["before_after"] = plot_before_after_comparison(comparison_df)

    logger.info(f"Generated {len(paths)} figures")
    logger.info("=" * 60)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 60)
    return paths
