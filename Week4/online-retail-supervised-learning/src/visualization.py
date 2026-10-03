import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)

from src.config import FIGURES_DIR, FIGURE_DPI

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


def plot_target_distribution(
    target_df: pd.DataFrame,
    filename: str = "01_target_distribution.png",
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    y_col = "FuturePurchased" if "FuturePurchased" in target_df.columns else target_df.columns[-1]
    id_col = "CustomerID" if "CustomerID" in target_df.columns else target_df.columns[0]

    counts = target_df[y_col].astype(int).value_counts().sort_index()
    labels_map = {0: "No Future Purchase", 1: "Future Purchase"}
    xtick_labels = [labels_map.get(int(k), str(k)) for k in counts.index]
    values = counts.values
    pcts = values / values.sum() * 100.0

    colors = ["#dd8452", "#55a868"]
    bars = axes[0].bar(
        range(len(values)), values,
        color=colors[: len(values)], edgecolor="white", linewidth=0.8,
    )
    for i, (b, v, p) in enumerate(zip(bars, values, pcts)):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + max(values) * 0.01,
                     f"{v:,}\n({p:.1f}%)", ha="center", va="bottom", fontsize=10)
    axes[0].set_xticks(range(len(xtick_labels)))
    axes[0].set_xticklabels(xtick_labels)
    axes[0].set_ylabel("Number of Customers")
    axes[0].set_title("Class Distribution (Customer Count)")
    axes[0].ticklabel_format(axis="y", style="plain")

    axes[1].pie(
        pcts, labels=xtick_labels, colors=colors[: len(values)],
        autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white"},
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Class Balance")

    fig.suptitle("Target Distribution — FuturePurchased", fontsize=14, y=1.02)
    return _save(fig, filename)


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    title: str = "Confusion Matrix",
    filename: str = "02_confusion_matrix.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))

    cm_values = cm_df.values.astype(int)
    row_labels = [f"Act {int(lbl)}" for _, lbl in cm_df.index.tolist()]
    col_labels = [f"Pred {int(lbl)}" for _, lbl in cm_df.columns.tolist()]

    sns.heatmap(
        cm_values,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=col_labels, yticklabels=row_labels,
        ax=ax, cbar_kws={"label": "Count"}, linewidths=0.5,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    return _save(fig, filename)


def plot_roc_curves(
    models_results_dict: Dict[str, Dict[str, Any]],
    filename: str = "03_roc_curves.png",
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6, label="Random baseline (AUC=0.5)")

    palette = sns.color_palette("tab10", max(len(models_results_dict), 1))
    for idx, (model_name, results) in enumerate(models_results_dict.items()):
        y_true = results.get("y_true")
        y_proba = results.get("y_proba")
        if y_true is None or y_proba is None:
            continue
        y_true_arr = np.asarray(y_true).astype(int)
        proba = np.asarray(y_proba)
        if proba.ndim == 2:
            proba = proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]
        fpr, tpr, _ = roc_curve(y_true_arr, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=palette[idx], linewidth=2,
                label=f"{model_name} (AUC = {roc_auc:.3f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves by Model")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_pr_curves(
    models_results_dict: Dict[str, Dict[str, Any]],
    filename: str = "04_pr_curves.png",
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    palette = sns.color_palette("tab10", max(len(models_results_dict), 1))
    for idx, (model_name, results) in enumerate(models_results_dict.items()):
        y_true = results.get("y_true")
        y_proba = results.get("y_proba")
        if y_true is None or y_proba is None:
            continue
        y_true_arr = np.asarray(y_true).astype(int)
        proba = np.asarray(y_proba)
        if proba.ndim == 2:
            proba = proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]
        prec, rec, _ = precision_recall_curve(y_true_arr, proba)
        pr_auc = auc(rec, prec)
        baseline = (y_true_arr == 1).sum() / len(y_true_arr) if len(y_true_arr) else 0.0
        ax.plot(rec, prec, color=palette[idx], linewidth=2,
                label=f"{model_name} (AUC = {pr_auc:.3f})")

    if len(models_results_dict) > 0:
        first_vals = next(iter(models_results_dict.values()))
        yt = np.asarray(first_vals.get("y_true", [])).astype(int)
        if len(yt):
            bline = float((yt == 1).sum() / len(yt))
            ax.axhline(bline, color="gray", linestyle=":", linewidth=1.2, alpha=0.6,
                       label=f"Baseline (P = {bline:.3f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (True Positive Rate)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves by Model")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_feature_importance(
    importances_df: pd.DataFrame,
    top_n: int = 20,
    filename: str = "05_feature_importance.png",
) -> Path:
    df = importances_df.copy()
    if "feature" not in df.columns and df.shape[1] >= 2:
        df.columns = ["feature", "importance"]
    if "importance" not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df = df.rename(columns={num_cols[0]: "importance"})
            df["feature"] = df.index.astype(str)
        else:
            raise ValueError("Cannot find 'importance' column in importances_df")

    df_sorted = df.sort_values("importance", ascending=True).tail(top_n).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.55), max(6, len(df_sorted) * 0.4)))
    bars = ax.barh(df_sorted["feature"], df_sorted["importance"],
                   color=sns.color_palette("viridis", len(df_sorted)),
                   edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, df_sorted["importance"]):
        ax.text(val + max(df_sorted["importance"]) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", ha="left", va="center", fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance (Top {min(top_n, len(df_sorted))})")
    ax.grid(True, axis="x", alpha=0.3)
    return _save(fig, filename)


def plot_lr_coefficients(
    coef_df: pd.DataFrame,
    top_n: int = 20,
    filename: str = "06_lr_coefficients.png",
) -> Path:
    df = coef_df.copy()
    if "feature" not in df.columns and df.shape[1] >= 2:
        df.columns = ["feature", "coefficient"]
    if "coefficient" not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df = df.rename(columns={num_cols[0]: "coefficient"})
            df["feature"] = df.index.astype(str)
        else:
            raise ValueError("Cannot find 'coefficient' column in coef_df")

    df["abs_coef"] = df["coefficient"].abs()
    df_sorted = df.sort_values("abs_coef", ascending=True).tail(top_n).reset_index(drop=True)

    color_list = ["#c44e52" if v < 0 else "#4c72b0" for v in df_sorted["coefficient"]]

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.55), max(6, len(df_sorted) * 0.4)))
    bars = ax.barh(df_sorted["feature"], df_sorted["coefficient"],
                   color=color_list, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient (log-odds direction)")
    ax.set_title(f"Logistic Regression Coefficients (Top {min(top_n, len(df_sorted))} by Magnitude)")
    legend_patches = [
        plt.matplotlib.patches.Patch(color="#4c72b0", label="Positive (↑ Future Purchase Odds)"),
        plt.matplotlib.patches.Patch(color="#c44e52", label="Negative (↓ Future Purchase Odds)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    return _save(fig, filename)


def run_all_visualizations(
    *,
    target_df: Optional[pd.DataFrame] = None,
    confusion_matrices: Optional[Dict[str, pd.DataFrame]] = None,
    models_results: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_importance_df: Optional[pd.DataFrame] = None,
    lr_coefficients_df: Optional[pd.DataFrame] = None,
    top_n_importance: int = 20,
) -> Dict[str, Path]:
    logger.info("=" * 60)
    logger.info("GENERATING ALL SUPERVISED LEARNING FIGURES")
    logger.info("=" * 60)
    paths: Dict[str, Path] = {}

    if target_df is not None:
        paths["target_distribution"] = plot_target_distribution(target_df)

    if confusion_matrices:
        for name, cm_df in confusion_matrices.items():
            safe_name = str(name).replace("/", "_").replace(" ", "_")
            paths[f"confusion_{safe_name}"] = plot_confusion_matrix(
                cm_df,
                title=f"Confusion Matrix — {name}",
                filename=f"02_confusion_{safe_name}.png",
            )

    if models_results:
        paths["roc_curves"] = plot_roc_curves(models_results)
        paths["pr_curves"] = plot_pr_curves(models_results)

    if feature_importance_df is not None and not feature_importance_df.empty:
        paths["feature_importance"] = plot_feature_importance(
            feature_importance_df, top_n=top_n_importance,
        )

    if lr_coefficients_df is not None and not lr_coefficients_df.empty:
        paths["lr_coefficients"] = plot_lr_coefficients(
            lr_coefficients_df, top_n=top_n_importance,
        )

    logger.info(f"Generated {len(paths)} figures")
    logger.info("VISUALIZATION COMPLETE")
    return paths
