import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from src.config import FIGURES_DIR, FIGURE_DPI, RANDOM_SEED

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
FIGSIZE_WIDE = (12, 6)
FIGSIZE_SQUARE = (10, 8)


def _save(fig: plt.Figure, filename: str, output_dir: Path = FIGURES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")
    return path


def _plot_eda_monthly_revenue(monthly: pd.DataFrame, filename: str = "01_eda_monthly_revenue.png") -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    if monthly.empty or "YearMonth" not in monthly.columns or "revenue" not in monthly.columns:
        ax.text(0.5, 0.5, "No monthly revenue data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Monthly Revenue")
        return _save(fig, filename)
    df = monthly.sort_values("YearMonth").copy()
    ax.bar(df["YearMonth"].astype(str), df["revenue"].astype(float), color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Total Revenue")
    ax.set_title("EDA: Monthly Revenue Over Time")
    plt.xticks(rotation=45, ha="right")
    ax.ticklabel_format(axis="y", style="plain")
    return _save(fig, filename)


def _plot_eda_top10_countries(countries: pd.DataFrame, filename: str = "02_eda_top10_countries.png") -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    if countries.empty or "Country" not in countries.columns or "transactions" not in countries.columns:
        ax.text(0.5, 0.5, "No country data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Top 10 Countries by Transactions")
        return _save(fig, filename)
    df = countries.head(10).copy()
    ax.barh(df["Country"].astype(str)[::-1], df["transactions"].astype(float)[::-1], color="#55a868", edgecolor="white")
    ax.set_xlabel("Number of Transactions")
    ax.set_title("EDA: Top 10 Countries by Transactions")
    ax.ticklabel_format(axis="x", style="plain")
    return _save(fig, filename)


def _plot_cluster_sizes(profiles: pd.DataFrame, filename: str = "03_cluster_sizes.png") -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    if profiles.empty or "cluster" not in profiles.columns or "customer_count" not in profiles.columns:
        ax.text(0.5, 0.5, "No cluster data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Cluster Sizes")
        return _save(fig, filename)
    df = profiles.copy()
    df["cluster_label"] = "Cluster " + df["cluster"].astype(int).astype(str)
    ax.bar(df["cluster_label"], df["customer_count"].astype(int), color="#dd8452", edgecolor="white")
    pcts = df.get("pct_of_customers")
    for i, (_, row) in enumerate(df.iterrows()):
        label = f"{int(row['customer_count']):,}"
        if pcts is not None and not df.empty:
            label += f"\n({float(row['pct_of_customers']):.1f}%)"
        ax.text(i, int(row["customer_count"]), label, ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Customer Count")
    ax.set_title("Unsupervised: Customer Count per Cluster")
    ax.ticklabel_format(axis="y", style="plain")
    return _save(fig, filename)


def _plot_cluster_recency_vs_monetary(
    labels_array: np.ndarray, rfm_df: pd.DataFrame, filename: str = "04_cluster_recency_vs_monetary.png"
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    if rfm_df.empty or "Recency" not in rfm_df.columns or "Monetary" not in rfm_df.columns:
        ax.text(0.5, 0.5, "No RFM data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Recency vs Monetary Colored by Cluster")
        return _save(fig, filename)
    df = rfm_df.copy()
    df["Cluster"] = labels_array
    palette = sns.color_palette("tab10", n_colors=max(2, len(np.unique(labels_array))))
    for cid in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == cid]
        ax.scatter(
            subset["Recency"], subset["Monetary"],
            label=f"Cluster {int(cid)}", alpha=0.6, s=30, edgecolors="none",
            color=palette[int(cid) % len(palette)],
        )
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Monetary (total revenue)")
    ax.set_title("Unsupervised: Recency vs Monetary Colored by Cluster")
    ax.legend(fontsize=9)
    ax.ticklabel_format(axis="y", style="plain")
    return _save(fig, filename)


def _plot_roc_curve(
    models_results: Dict[str, Dict[str, Any]], filename: str = "05_roc_curve.png"
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    if not models_results:
        ax.text(0.5, 0.5, "No model results available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("ROC Curve")
        return _save(fig, filename)
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    ci = 0
    for name, res in models_results.items():
        yt = res.get("y_true")
        yp = res.get("y_proba")
        if yt is None or yp is None:
            continue
        y_true_arr = np.asarray(yt).astype(int)
        proba = np.asarray(yp)
        proba_use = proba[:, 1] if proba.ndim == 2 else proba
        try:
            fpr, tpr, _ = roc_curve(y_true_arr, proba_use)
            roc_auc = float(auc(fpr, tpr))
            ax.plot(fpr, tpr, lw=2, color=colors[ci % len(colors)], label=f"{name} (AUC={roc_auc:.3f})")
            ci += 1
        except Exception as e:
            logger.warning(f"ROC skipped for {name}: {e}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Supervised: ROC Curve (Random Forest vs Logistic Regression)")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, filename)


def _plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15, filename: str = "06_feature_importance.png") -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    if importance_df is None or importance_df.empty or "feature" not in importance_df.columns or "importance" not in importance_df.columns:
        ax.text(0.5, 0.5, "No feature importance data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Top 15 Feature Importance (Random Forest)")
        return _save(fig, filename)
    df = importance_df.head(top_n).copy()
    df = df.sort_values("importance", ascending=True)
    ax.barh(df["feature"].astype(str), df["importance"].astype(float), color="#c44e52", edgecolor="white")
    ax.set_xlabel("Importance (MDI)")
    ax.set_title(f"Supervised: Top {len(df)} Feature Importance (Random Forest)")
    return _save(fig, filename)


def _plot_confusion_matrix_heatmap(cm_df: pd.DataFrame, filename: str = "07_confusion_matrix.png") -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    if cm_df is None or cm_df.empty:
        ax.text(0.5, 0.5, "No confusion matrix available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Confusion Matrix Heatmap (Random Forest)")
        return _save(fig, filename)
    try:
        values = cm_df.values.astype(int)
        n = values.shape[0]
        true_labels = [f"Actual={i}" for i in range(n)]
        pred_labels = [f"Pred={i}" for i in range(n)]
        sns.heatmap(
            values, annot=True, fmt="d", cmap="Blues",
            xticklabels=pred_labels, yticklabels=true_labels, ax=ax,
            cbar_kws={"label": "Count"},
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Supervised: Confusion Matrix Heatmap (Random Forest)")
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not render confusion matrix: {e}", ha="center", va="center", transform=ax.transAxes)
    return _save(fig, filename)


def _plot_cluster_profile_heatmap(scaled_profiles_df: pd.DataFrame, filename: str = "08_cluster_profile_heatmap.png") -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    if scaled_profiles_df is None or scaled_profiles_df.empty:
        ax.text(0.5, 0.5, "No scaled profiles available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Cluster Profile Heatmap (Z-Scored)")
        return _save(fig, filename)
    try:
        sp = scaled_profiles_df.copy()
        if "cluster" in sp.columns:
            sp = sp.set_index("cluster")
        sp.index = [f"Cluster {int(c)}" if str(c).isdigit() else f"Cluster {c}" for c in sp.index]
        sns.heatmap(
            sp, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            cbar_kws={"label": "Z-Score vs Overall Mean"},
        )
        ax.set_title("Unsupervised: Cluster Profile Heatmap (Z-Scored R/F/M)")
        ax.set_ylabel("")
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not render profile heatmap: {e}", ha="center", va="center", transform=ax.transAxes)
    return _save(fig, filename)


def run_all_visualizations(
    eda_monthly: Optional[pd.DataFrame] = None,
    eda_top_countries: Optional[pd.DataFrame] = None,
    cluster_profiles: Optional[pd.DataFrame] = None,
    unsupervised_labels: Optional[np.ndarray] = None,
    rfm_df: Optional[pd.DataFrame] = None,
    models_results_for_viz: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_importance_df: Optional[pd.DataFrame] = None,
    confusion_matrix_df: Optional[pd.DataFrame] = None,
    scaled_profiles_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    logger.info("Generating all 8 visualizations")
    np.random.seed(RANDOM_SEED)

    paths: Dict[str, Path] = {}

    paths["01_eda_monthly_revenue"] = _plot_eda_monthly_revenue(eda_monthly if eda_monthly is not None else pd.DataFrame())
    paths["02_eda_top10_countries"] = _plot_eda_top10_countries(eda_top_countries if eda_top_countries is not None else pd.DataFrame())
    paths["03_cluster_sizes"] = _plot_cluster_sizes(cluster_profiles if cluster_profiles is not None else pd.DataFrame())
    paths["04_cluster_recency_vs_monetary"] = _plot_cluster_recency_vs_monetary(
        unsupervised_labels if unsupervised_labels is not None else np.array([]),
        rfm_df if rfm_df is not None else pd.DataFrame(),
    )
    paths["05_roc_curve"] = _plot_roc_curve(models_results_for_viz if models_results_for_viz is not None else {})
    paths["06_feature_importance"] = _plot_feature_importance(feature_importance_df, top_n=15)
    paths["07_confusion_matrix"] = _plot_confusion_matrix_heatmap(confusion_matrix_df)
    paths["08_cluster_profile_heatmap"] = _plot_cluster_profile_heatmap(scaled_profiles_df)

    return paths
