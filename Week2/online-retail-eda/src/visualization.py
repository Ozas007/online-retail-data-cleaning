import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, FIGURE_DPI

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
FIGSIZE_WIDE = (12, 6)
FIGSIZE_SQUARE = (10, 8)


def _save(fig: plt.Figure, filename: str, output_dir: Path = FIGURES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")
    return path


def plot_missing_values(missing_df: pd.DataFrame, filename: str = "01_missing_values.png") -> Path:
    df = missing_df.copy().sort_values("Missing Count", ascending=True)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.barh(df["Column"], df["Missing Count"], color="#c44e52")
    for bar, val, pct in zip(bars, df["Missing Count"], df["Missing Percentage (%)"]):
        if val > 0:
            ax.text(bar.get_width() + max(df["Missing Count"].max()*0.01, 10),
                    bar.get_y() + bar.get_height()/2,
                    f"{int(val):,} ({pct:.2f}%)", va="center", fontsize=10)
    ax.set_xlabel("Number of Missing Values")
    ax.set_ylabel("Column")
    ax.set_title("Missing Values by Column in UCI Online Retail Dataset")
    ax.set_xlim(0, max(df["Missing Count"].max() * 1.25, 100))
    return _save(fig, filename)


def _plot_hist_box(series: pd.Series, title: str, xlabel: str, filename_hist: str,
                   filename_box: str, use_log: bool = True) -> Tuple[Path, Path]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    s_pos = s[s > 0] if use_log else s
    s_plot = np.log1p(s_pos) if use_log else s_pos
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    label_suffix = " (log1p scale, positive only)" if use_log else ""
    sns.histplot(s_plot, kde=True, color="#4c72b0", bins=60, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(f"{xlabel}{label_suffix}")
    ax.set_ylabel("Frequency")
    mean_v = float(s_plot.mean())
    ax.axvline(mean_v, color="red", linestyle="--", label=f"Mean={mean_v:.2f}")
    ax.legend()
    hist_path = _save(fig, filename_hist)

    fig2, ax2 = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.boxplot(x=s_pos, color="#dd8452", ax=ax2, orient="h", flierprops={"marker": ".", "alpha": 0.3})
    ax2.set_title(f"{title} — Boxplot (positive values, raw scale)")
    ax2.set_xlabel(xlabel)
    box_path = _save(fig2, filename_box)
    return hist_path, box_path


def plot_quantity_distribution(df_positive: pd.DataFrame) -> Tuple[Path, Path]:
    return _plot_hist_box(
        df_positive["Quantity"], "Quantity Distribution (Positive Sales)",
        "Quantity per transaction line",
        "02_quantity_distribution.png", "02b_quantity_boxplot.png"
    )


def plot_unitprice_distribution(df_positive: pd.DataFrame) -> Tuple[Path, Path]:
    return _plot_hist_box(
        df_positive["UnitPrice"], "Unit Price Distribution (Positive Sales)",
        "Unit Price (£)",
        "03_unit_price_distribution.png", "03b_unit_price_boxplot.png"
    )


def plot_monthly_transactions(monthly_df: pd.DataFrame, filename: str = "04_monthly_transaction_volume.png") -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(monthly_df["YearMonth"], monthly_df["transaction_lines"], marker="o", linewidth=2, color="#4c72b0", label="Transaction Lines")
    ax2 = ax.twinx()
    ax2.plot(monthly_df["YearMonth"], monthly_df["unique_invoices"], marker="s", linewidth=2, color="#dd8452", label="Unique Invoices")
    ax.set_xlabel("Month")
    ax.set_ylabel("Transaction Lines", color="#4c72b0")
    ax2.set_ylabel("Unique Invoices", color="#dd8452")
    ax.set_title("Monthly Transaction Volume (Lines & Unique Invoices)")
    plt.xticks(rotation=45)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    return _save(fig, filename)


def plot_monthly_revenue(revenue_df: pd.DataFrame, filename: str = "05_monthly_revenue.png") -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(revenue_df["YearMonth"], revenue_df["revenue"], color="#55a868", alpha=0.7, label="Revenue (£)")
    ax2 = ax.twinx()
    ax2.plot(revenue_df["YearMonth"], revenue_df["avg_order_value"], marker="D", color="#c44e52", linewidth=2, label="Avg Order Value (£)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Revenue (£)", color="#55a868")
    ax2.set_ylabel("Avg TotalAmount per Line (£)", color="#c44e52")
    ax.set_title("Monthly Revenue with Average Transaction Value")
    plt.xticks(rotation=45)
    ax.ticklabel_format(axis="y", style="plain")
    ax2.ticklabel_format(axis="y", style="plain")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    return _save(fig, filename)


def plot_top_countries(country_df: pd.DataFrame, top_n: int = 10, filename: str = "06_top_countries.png") -> Path:
    df = country_df.head(top_n).copy().sort_values("lines", ascending=True)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.barh(df["Country"], df["lines"], color="#4c72b0")
    for bar, val in zip(bars, df["lines_pct"]):
        ax.text(bar.get_width() + max(df["lines"].max()*0.01, 1000),
                bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va="center", fontsize=10)
    ax.set_xlabel("Number of Transaction Lines")
    ax.set_ylabel("Country")
    ax.set_title(f"Top {top_n} Countries by Transaction-Line Count")
    ax.ticklabel_format(axis="x", style="plain")
    return _save(fig, filename)


def _top_product_chart(df: pd.DataFrame, value_col: str, title: str, filename: str) -> Path:
    df_use = df.head(10).copy()
    df_use["label"] = [f"{d[:45]}..." if len(str(d)) > 45 else str(d) for d in df_use["Description"]]
    df_use = df_use.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(df_use["label"], df_use[value_col], color="#8172b3")
    for bar, val in zip(bars, df_use[value_col]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                f"{val:,.0f}" if val >= 1000 else f"{val:.2f}",
                va="center", fontsize=9)
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_title(title)
    return _save(fig, filename)


def plot_top_products_quantity(top_qty: pd.DataFrame) -> Path:
    return _top_product_chart(top_qty, "total_quantity", "Top 10 Products by Quantity Sold", "07_top_products_quantity.png")


def plot_top_products_revenue(top_rev: pd.DataFrame) -> Path:
    df = top_rev.copy()
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    return _top_product_chart(df, "revenue", "Top 10 Products by Revenue Generated (£)", "08_top_products_revenue.png")


def plot_transactions_by_hour(hourly_df: pd.DataFrame, filename: str = "09_transactions_by_hour.png") -> Path:
    df = hourly_df.sort_values("Hour")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(df["Hour"].astype(int), df["lines"], marker="o", linewidth=2.5, color="#ccb974")
    ax.fill_between(df["Hour"].astype(int), df["lines"], alpha=0.2, color="#ccb974")
    ax.set_xlabel("Hour of Day (24h)")
    ax.set_ylabel("Number of Transaction Lines")
    ax.set_title("Transaction Activity by Hour of Day")
    ax.set_xticks(range(6, 22))
    ax.set_xlim(5.5, 20.5)
    return _save(fig, filename)


def plot_transactions_by_weekday(weekday_df: pd.DataFrame, filename: str = "10_transactions_by_weekday.png") -> Path:
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df = weekday_df.copy()
    df["DayOfWeekName"] = pd.Categorical(df["DayOfWeekName"], categories=weekday_order, ordered=True)
    df = df.sort_values("DayOfWeekName")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.bar(df["DayOfWeekName"], df["lines"], color="#64b5cd")
    for bar, val in zip(bars, df["lines"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df["lines"].max()*0.01, 1000),
                f"{int(val):,}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Number of Transaction Lines")
    ax.set_title("Transaction Volume by Day of Week")
    ax.ticklabel_format(axis="y", style="plain")
    return _save(fig, filename)


def plot_quantity_vs_totalamount(qta: Dict, filename: str = "11_quantity_vs_totalamount.png") -> Path:
    df_plot = qta["scatter_sample"].copy()
    s_full = qta["full"]
    qty_max = float(s_full["Quantity"].quantile(0.99))
    amt_max = float(s_full["TotalAmount"].quantile(0.99))
    df_plot = df_plot[(df_plot["Quantity"] <= qty_max) & (df_plot["TotalAmount"] <= amt_max)]
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.scatterplot(data=df_plot, x="Quantity", y="TotalAmount", alpha=0.3, s=8, color="#4c72b0", ax=ax)
    sns.regplot(data=df_plot, x="Quantity", y="TotalAmount", scatter=False, color="#c44e52", line_kws={"lw": 2}, ax=ax)
    corr_val = qta["correlation"]
    ax.set_title(f"Quantity vs TotalAmount (P99 Clipped; Pearson r = {corr_val:.3f})")
    ax.set_xlabel("Quantity (per line)")
    ax.set_ylabel("Total Amount (£)")
    return _save(fig, filename)


def plot_unitprice_vs_quantity(uvq: Dict, filename: str = "12_unit_price_vs_quantity.png") -> Path:
    df_plot = uvq["scatter_sample"].copy()
    s_full = uvq["full"]
    q_max = float(s_full["Quantity"].quantile(0.99))
    p_max = float(s_full["UnitPrice"].quantile(0.99))
    df_plot = df_plot[(df_plot["Quantity"] <= q_max) & (df_plot["UnitPrice"] <= p_max)]
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.scatterplot(data=df_plot, x="UnitPrice", y="Quantity", alpha=0.25, s=8, color="#dd8452", ax=ax)
    sns.regplot(data=df_plot, x="UnitPrice", y="Quantity", scatter=False, color="#55a868", line_kws={"lw": 2}, ax=ax)
    corr_val = uvq["correlation"]
    ax.set_title(f"Unit Price vs Quantity (P99 Clipped; Pearson r = {corr_val:.3f})")
    ax.set_xlabel("Unit Price (£)")
    ax.set_ylabel("Quantity (per line)")
    return _save(fig, filename)


def plot_correlation_heatmap(corr_df: pd.DataFrame, filename: str = "13_correlation_heatmap.png") -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="RdBu_r", vmin=-1, vmax=1,
                mask=mask, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Pearson Correlation Matrix: Numerical Variables")
    return _save(fig, filename)


def plot_totalamount_distribution(df_positive: pd.DataFrame) -> Tuple[Path, Path]:
    return _plot_hist_box(
        df_positive["TotalAmount"], "Total Transaction Amount Distribution (Positive Sales)",
        "TotalAmount = Quantity × UnitPrice (£)",
        "14_totalamount_distribution.png", "14b_totalamount_boxplot.png"
    )


def run_all_visualizations(analyses: Dict, df_full: pd.DataFrame, df_positive: pd.DataFrame,
                           missing_df: pd.DataFrame) -> Dict[str, Path]:
    logger.info("=" * 60)
    logger.info("GENERATING ALL FIGURES")
    logger.info("=" * 60)
    paths = {}
    paths["missing_values"] = plot_missing_values(missing_df)
    paths["qty_hist"], paths["qty_box"] = plot_quantity_distribution(df_positive)
    paths["price_hist"], paths["price_box"] = plot_unitprice_distribution(df_positive)
    paths["amt_hist"], paths["amt_box"] = plot_totalamount_distribution(df_positive)
    paths["monthly_vol"] = plot_monthly_transactions(analyses["temporal"]["monthly_volume"])
    paths["monthly_rev"] = plot_monthly_revenue(analyses["temporal"]["monthly_revenue"])
    paths["countries"] = plot_top_countries(analyses["geographic"]["country_transactions"])
    paths["products_qty"] = plot_top_products_quantity(analyses["product"]["top_quantity"])
    paths["products_rev"] = plot_top_products_revenue(analyses["product"]["top_revenue"])
    paths["hourly"] = plot_transactions_by_hour(analyses["temporal"]["hourly"])
    paths["weekday"] = plot_transactions_by_weekday(analyses["temporal"]["weekday"])
    paths["qty_vs_amt"] = plot_quantity_vs_totalamount(analyses["relationship"]["qty_vs_totalamount"])
    paths["price_vs_qty"] = plot_unitprice_vs_quantity(analyses["relationship"]["unitprice_vs_quantity"])
    paths["corr_heatmap"] = plot_correlation_heatmap(analyses["relationship"]["correlation_matrix"])
    logger.info(f"Generated {len(paths)} figures")
    logger.info("VISUALIZATION COMPLETE")
    return paths
