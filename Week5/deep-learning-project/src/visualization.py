import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import FIGURES_DIR, FIGURE_DPI

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
FIGSIZE_WIDE = (12, 5)
FIGSIZE_SQUARE = (10, 8)


def _save(fig: plt.Figure, filename: str, output_dir: Path = FIGURES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")
    return path


def _history_to_df(history: Any) -> pd.DataFrame:
    if isinstance(history, pd.DataFrame):
        return history.copy()
    if hasattr(history, "history"):
        history_dict = history.history
    elif isinstance(history, dict):
        history_dict = history
    else:
        history_dict = dict(history)
    return pd.DataFrame(history_dict)


def plot_training_curves(
    history: Any,
    filename: str = "01_training_curves.png",
    output_dir: Path = FIGURES_DIR,
) -> Path:
    df = _history_to_df(history)
    epochs = np.arange(1, len(df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    ax = axes[0]
    ax.plot(epochs, df["loss"], label="Train Loss", color="#4c72b0", linewidth=2, marker="o", markersize=4)
    if "val_loss" in df.columns:
        ax.plot(epochs, df["val_loss"], label="Val Loss", color="#c44e52", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Categorical Cross-Entropy Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, df["accuracy"], label="Train Acc", color="#4c72b0", linewidth=2, marker="o", markersize=4)
    if "val_accuracy" in df.columns:
        ax.plot(epochs, df["val_accuracy"], label="Val Acc", color="#c44e52", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training / Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("MLP Training Curves on MNIST", fontsize=13, y=1.02)
    return _save(fig, filename, output_dir)


def plot_confusion_matrix_heatmap(
    cm_df: pd.DataFrame,
    title: str = "MNIST Test Set Confusion Matrix",
    filename: str = "02_confusion_matrix.png",
    output_dir: Path = FIGURES_DIR,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    cm_vals = cm_df.values.astype(int)
    n_classes = cm_vals.shape[0]
    digits = list(range(n_classes))

    row_sums = cm_vals.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm_vals / row_sums_safe

    sns.heatmap(
        cm_norm,
        annot=cm_vals,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Row-Normalized Fraction"},
        xticklabels=digits,
        yticklabels=digits,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Digit")
    ax.set_ylabel("True Digit")
    ax.set_title(title)
    return _save(fig, filename, output_dir)


def _draw_fallback_architecture_diagram(
    filename: str,
    output_dir: Path,
) -> Path:
    layers_spec = [
        ("Input (784 features)", "#a1caf1"),
        ("Dense 128\nReLU", "#ffdfba"),
        ("Dropout 0.2", "#fff2cc"),
        ("Dense 64\nReLU", "#ffdfba"),
        ("Dropout 0.2", "#fff2cc"),
        ("Dense 10\nSoftmax Output", "#b7e1cd"),
    ]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers_spec) * 1.5 + 1)
    ax.axis("off")

    box_w, box_h = 5.5, 1.0
    x_center = 5.0
    y_step = 1.35
    y_start = len(layers_spec) * y_step

    for i, (label, color) in enumerate(layers_spec):
        y = y_start - i * y_step
        x0 = x_center - box_w / 2
        y0 = y - box_h / 2
        patch = FancyBboxPatch(
            (x0, y0), box_w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=color, edgecolor="#333333",
            linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(
            x_center, y, label,
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="#222222",
        )

        if i < len(layers_spec) - 1:
            y_arrow_from = y0
            y_arrow_to = y_arrow_from - y_step + box_h
            ax.annotate(
                "",
                xy=(x_center, y_arrow_to + 0.02),
                xytext=(x_center, y_arrow_from - 0.02),
                arrowprops=dict(
                    arrowstyle=ArrowStyle.Simple(tail_width=0.4, head_width=0.7, head_length=0.5),
                    color="#555555",
                    lw=1,
                ),
            )

    ax.set_title("Neural Network Architecture", fontsize=15, fontweight="bold", y=0.98)
    return _save(fig, filename, output_dir)


def plot_architecture_diagram(
    model: tf.keras.Model,
    filename: str = "03_architecture_diagram.png",
    output_dir: Path = FIGURES_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    try:
        import tensorflow as tf
        temp_path = output_dir / "_temp_model_diagram.png"
        tf.keras.utils.plot_model(
            model,
            to_file=str(temp_path),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=FIGURE_DPI,
        )
        if temp_path.exists() and temp_path.stat().st_size > 0:
            if path.exists():
                path.unlink()
            temp_path.rename(path)
            logger.info(f"Saved keras plot_model architecture diagram: {path}")
            return path
        logger.warning("keras plot_model produced empty file; using fallback matplotlib diagram")
    except Exception as exc:
        logger.warning(f"keras plot_model failed ({exc!r}); using fallback matplotlib diagram")

    return _draw_fallback_architecture_diagram(filename, output_dir)


def plot_sample_predictions(
    X_test_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str = "04_sample_predictions.png",
    output_dir: Path = FIGURES_DIR,
    n_samples: int = 16,
) -> Path:
    rng = np.random.RandomState(42)
    idxs = rng.choice(len(X_test_raw), size=min(n_samples, len(X_test_raw)), replace=False)

    ncols = 4
    nrows = int(np.ceil(len(idxs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, idxs):
        img = np.squeeze(X_test_raw[idx])
        true_d = int(y_true[idx])
        pred_d = int(y_pred[idx])
        color = "green" if true_d == pred_d else "red"
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True={true_d}  Pred={pred_d}", fontsize=9, color=color, fontweight="bold")
        ax.axis("off")

    for ax in axes[len(idxs):]:
        ax.axis("off")

    fig.suptitle("Sample Test Images — True vs Predicted Digits", fontsize=13, y=1.01)
    return _save(fig, filename, output_dir)


def run_all_visualizations(
    history: Any,
    confusion_matrix_df: pd.DataFrame,
    model: tf.keras.Model,
    X_test_raw: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    output_dir: Path = FIGURES_DIR,
) -> Dict[str, Path]:
    logger.info("=" * 60)
    logger.info("GENERATING ALL DEEP LEARNING FIGURES")
    logger.info("=" * 60)
    paths: Dict[str, Path] = {}
    paths["training_curves"] = plot_training_curves(history, output_dir=output_dir)
    paths["confusion_matrix"] = plot_confusion_matrix_heatmap(confusion_matrix_df, output_dir=output_dir)
    paths["architecture_diagram"] = plot_architecture_diagram(model, output_dir=output_dir)

    if X_test_raw is not None and y_true is not None and y_pred is not None:
        try:
            paths["sample_predictions"] = plot_sample_predictions(
                X_test_raw, y_true, y_pred, output_dir=output_dir
            )
        except Exception as exc:
            logger.warning(f"plot_sample_predictions failed: {exc!r}")

    logger.info(f"Generated {len(paths)} figures in {output_dir}")
    logger.info("VISUALIZATION COMPLETE")
    return paths
