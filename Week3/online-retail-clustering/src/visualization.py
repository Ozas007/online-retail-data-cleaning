import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import seaborn as sns

from src.config import FIGURES_DIR, FIGURE_DPI

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
FIGSIZE_WIDE = (12, 6)
FIGSIZE_SQUARE = (10, 8)


def _resolve_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns {candidates} found in DataFrame columns: {list(df.columns)}")


def _save(fig: plt.Figure, filename: str, output_dir: Path = FIGURES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")
    return path


def plot_rfm_distributions(rfm_df: pd.DataFrame, filename_prefix: str = "01_rfm_distribution") -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    r_col = _resolve_col(rfm_df, ["Recency", "recency"])
    f_col = _resolve_col(rfm_df, ["Frequency", "frequency"])
    m_col = _resolve_col(rfm_df, ["Monetary", "monetary"])
    cols = [r_col, f_col, m_col]
    titles = [
        "Recency Distribution (Days Since Last Purchase)",
        "Frequency Distribution (Number of Purchases)",
        "Monetary Distribution (Total Spend £)",
    ]
    colors = ["#4c72b0", "#55a868", "#dd8452"]
    xlabels = ["Recency (days)", "Frequency (count)", "Monetary (£)"]

    for ax, col, title, color, xlab in zip(axes, cols, titles, colors, xlabels):
        s = pd.to_numeric(rfm_df[col], errors="coerce").dropna()
        sns.histplot(s, kde=True, color=color, bins=50, ax=ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Count")
        mean_v = float(s.mean())
        ax.axvline(mean_v, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mean_v:.1f}")
        ax.legend(fontsize=9)

    fig.suptitle("RFM Distributions — Raw Scale", fontsize=14, y=1.02)
    return _save(fig, f"{filename_prefix}.png")


def plot_rfm_distributions_log(rfm_df_log: pd.DataFrame, filename_prefix: str = "02_rfm_log_distribution") -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cols = []
    raw_cols = []
    for base_cap, base_low in [("Recency", "recency"), ("Frequency", "frequency"), ("Monetary", "monetary")]:
        log_cap = f"Log_{base_cap}"
        log_low = f"{base_low}_log"
        if log_cap in rfm_df_log.columns:
            cols.append(log_cap)
            raw_cols.append(base_cap if base_cap in rfm_df_log.columns else base_low)
        elif log_low in rfm_df_log.columns:
            cols.append(log_low)
            raw_cols.append(base_low if base_low in rfm_df_log.columns else base_cap)
        else:
            cols.append(log_low)
            raw_cols.append(base_low if base_low in rfm_df_log.columns else base_cap)
    titles = [
        "Recency — Log1p Transformed",
        "Frequency — Log1p Transformed",
        "Monetary — Log1p Transformed",
    ]
    colors = ["#4c72b0", "#55a868", "#dd8452"]
    xlabels = [
        "log1p(Recency)",
        "log1p(Frequency)",
        "log1p(Monetary)",
    ]

    for ax, col, raw, title, color, xlab in zip(axes, cols, raw_cols, titles, colors, xlabels):
        if col in rfm_df_log.columns:
            s = pd.to_numeric(rfm_df_log[col], errors="coerce").dropna()
        else:
            s = np.log1p(pd.to_numeric(rfm_df_log[raw], errors="coerce").dropna())
        sns.histplot(s, kde=True, color=color, bins=50, ax=ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Count")
        mean_v = float(s.mean())
        ax.axvline(mean_v, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mean_v:.2f}")
        ax.legend(fontsize=9)

    fig.suptitle("RFM Distributions — Log1p Transformed Scale", fontsize=14, y=1.02)
    return _save(fig, f"{filename_prefix}.png")


def plot_elbow(k_metrics_df: pd.DataFrame, filename: str = "03_elbow_plot.png") -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    df = k_metrics_df.copy()
    ax.plot(df["k"], df["inertia"], marker="o", linewidth=2.5, color="#4c72b0", markersize=8)
    ax.fill_between(df["k"], df["inertia"], alpha=0.15, color="#4c72b0")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_title("Elbow Method — Inertia vs Number of Clusters")
    ax.set_xticks(df["k"].astype(int))
    ax.ticklabel_format(axis="y", style="plain")
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_silhouette(k_metrics_df: pd.DataFrame, filename: str = "04_silhouette_plot.png") -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    df = k_metrics_df.copy()
    ax.plot(df["k"], df["silhouette_score"], marker="s", linewidth=2.5, color="#c44e52", markersize=8)
    ax.fill_between(df["k"], df["silhouette_score"], alpha=0.15, color="#c44e52")
    best_idx = df["silhouette_score"].idxmax()
    best_k = df.loc[best_idx, "k"]
    best_s = df.loc[best_idx, "silhouette_score"]
    ax.axvline(best_k, color="green", linestyle="--", linewidth=1.5, label=f"Best k={int(best_k)} (silhouette={best_s:.3f})")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Analysis — Silhouette Score vs Number of Clusters")
    ax.set_xticks(df["k"].astype(int))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_cluster_sizes(labels, filename: str = "05_cluster_sizes.png") -> Path:
    if isinstance(labels, pd.DataFrame):
        if "Cluster" in labels.columns:
            label_series = labels["Cluster"]
        else:
            label_series = labels.iloc[:, -1]
    else:
        label_series = pd.Series(labels, name="Cluster")

    counts = label_series.value_counts().sort_index()
    cluster_names = [f"Cluster {int(c)}" for c in counts.index]
    sizes = counts.values

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.bar(cluster_names, sizes, color=sns.color_palette("viridis", len(sizes)), edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, sizes):
        pct = val / sum(sizes) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sizes) * 0.01,
                f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Distribution by Cluster Size")
    ax.ticklabel_format(axis="y", style="plain")
    return _save(fig, filename)


def plot_pca_clusters(pca_df: pd.DataFrame, labels, filename: str = "06_pca_clusters.png") -> Path:
    df = pca_df.copy()

    if isinstance(labels, pd.DataFrame):
        if "Cluster" in labels.columns:
            label_series = labels["Cluster"].reset_index(drop=True)
        else:
            label_series = labels.iloc[:, -1].reset_index(drop=True)
    else:
        label_series = pd.Series(labels, name="Cluster").reset_index(drop=True)

    df = df.reset_index(drop=True)
    df["Cluster"] = label_series.astype(int).astype(str)

    pc1_col = "PC1" if "PC1" in df.columns else df.columns[0]
    pc2_col = "PC2" if "PC2" in df.columns else df.columns[1]

    n_clusters = df["Cluster"].nunique()
    palette = sns.color_palette("viridis", n_clusters)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.scatterplot(
        data=df, x=pc1_col, y=pc2_col, hue="Cluster",
        palette=palette, s=40, alpha=0.7, edgecolor="white", linewidth=0.5, ax=ax
    )
    centroids = df.groupby("Cluster")[[pc1_col, pc2_col]].mean()
    for i, (cname, row) in enumerate(centroids.iterrows()):
        ax.scatter(row[pc1_col], row[pc2_col], color=palette[i], s=300, marker="X",
                   edgecolors="black", linewidths=1.5, zorder=10, label=f"Centroid {cname}")
        ax.text(row[pc1_col], row[pc2_col], f"C{cname}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="black", zorder=11)

    ax.set_xlabel(f"{pc1_col} (Principal Component 1)")
    ax.set_ylabel(f"{pc2_col} (Principal Component 2)")
    ax.set_title("Customer Clusters Visualized with PCA (PC1 vs PC2)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_cluster_profile_heatmap(cluster_profiles_scaled: pd.DataFrame, filename: str = "07_cluster_profile_heatmap.png") -> Path:
    df = cluster_profiles_scaled.copy()
    if "Cluster" in df.columns:
        df = df.set_index("Cluster")

    rfm_cols = [c for c in df.columns if c in ["recency", "frequency", "monetary",
                                                "recency_z", "frequency_z", "monetary_z",
                                                "Recency", "Frequency", "Monetary"]]
    if not rfm_cols:
        rfm_cols = list(df.select_dtypes(include=[np.number]).columns)
    df_plot = df[rfm_cols].astype(float)

    if df_plot.index.dtype != object and df_plot.index.dtype != str:
        df_plot.index = [f"Cluster {int(i)}" for i in df_plot.index]

    fig, ax = plt.subplots(figsize=(max(8, len(rfm_cols) * 1.8), max(5, len(df_plot) * 0.9)))
    sns.heatmap(
        df_plot, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
        linewidths=0.5, cbar_kws={"label": "Standardized Value (Z-score)"}, ax=ax
    )
    ax.set_title("Standardized Cluster Profiles — Heatmap of RFM Means")
    ax.set_xlabel("RFM Feature")
    ax.set_ylabel("Cluster")
    return _save(fig, filename)


def _radar_factory(num_vars, frame="circle"):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                x, y = line.get_data()
                if x[0] != x[-1]:
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                spine = Spine(axes=self, spine_type="circle", path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {"polar": spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def plot_cluster_profile_radar(cluster_profiles: pd.DataFrame, filename: str = "08_cluster_profile_radar.png") -> Path:
    df = cluster_profiles.copy()

    rfm_cols = [c for c in ["recency", "frequency", "monetary"] if c in df.columns]
    if not rfm_cols:
        rfm_cols = [c for c in ["Recency", "Frequency", "Monetary"] if c in df.columns]
    if not rfm_cols:
        rfm_cols = list(df.select_dtypes(include=[np.number]).columns)[:3]

    rfm_labels = [c.title() for c in rfm_cols]

    data_raw = df[rfm_cols].astype(float).values
    data_min = data_raw.min(axis=0)
    data_max = data_raw.max(axis=0)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0
    data_norm = (data_raw - data_min) / data_range

    if "Cluster" in df.columns:
        cluster_ids = df["Cluster"].astype(int).tolist()
    else:
        cluster_ids = list(df.index)
        if not all(isinstance(c, (int, np.integer)) for c in cluster_ids):
            cluster_ids = list(range(len(df)))

    n_clusters = len(cluster_ids)
    ncols = 3 if n_clusters > 4 else (2 if n_clusters > 2 else 1)
    nrows = int(np.ceil(n_clusters / ncols))

    theta = _radar_factory(len(rfm_cols), frame="polygon")
    colors = sns.color_palette("viridis", n_clusters)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5.5 * nrows),
        subplot_kw=dict(projection="radar")
    )
    if n_clusters == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for idx, (ax, cid, ccolor) in enumerate(zip(axes, cluster_ids, colors)):
        spoke_data = data_norm[idx]
        ax.plot(theta, spoke_data, color=ccolor, linewidth=2)
        ax.fill(theta, spoke_data, facecolor=ccolor, alpha=0.25)
        ax.set_varlabels(rfm_labels)
        ax.set_title(f"Cluster {int(cid)}", weight="bold", size="medium", pad=15)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    for j in range(n_clusters, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Radar Chart — Normalized RFM Profiles per Cluster", fontsize=14, y=1.02)
    return _save(fig, filename)


def plot_cluster_bar_comparison(cluster_profiles: pd.DataFrame, filename: str = "09_cluster_monetary_bar.png") -> Path:
    df = cluster_profiles.copy()

    monetary_col = "monetary" if "monetary" in df.columns else ("Monetary" if "Monetary" in df.columns else None)
    if monetary_col is None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        monetary_col = num_cols[-1] if len(num_cols) > 0 else None

    if "Cluster" in df.columns:
        clusters = df["Cluster"].astype(int)
    else:
        clusters = pd.Series(df.index, name="Cluster").astype(int)

    values = pd.to_numeric(df[monetary_col], errors="coerce").fillna(0).values
    cluster_labels = [f"Cluster {int(c)}" for c in clusters]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.bar(cluster_labels, values, color=sns.color_palette("viridis", len(values)),
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"£{val:,.2f}" if val >= 1000 else f"£{val:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Cluster")
    ax.set_ylabel(f"Average Total Monetary Value (£)")
    ax.set_title(f"Monetary Comparison — Average Total Spend per Customer Cluster")
    ax.ticklabel_format(axis="y", style="plain")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, filename)


def run_all_visualizations(
    rfm_df: pd.DataFrame,
    rfm_df_log: pd.DataFrame,
    k_metrics_df: pd.DataFrame,
    labels,
    pca_df: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    cluster_profiles_scaled: pd.DataFrame,
) -> dict:
    logger.info("=" * 60)
    logger.info("GENERATING ALL CLUSTERING FIGURES")
    logger.info("=" * 60)
    paths = {}
    paths["rfm_distribution"] = plot_rfm_distributions(rfm_df)
    paths["rfm_dist"] = paths["rfm_distribution"]
    paths["rfm_log_distribution"] = plot_rfm_distributions_log(rfm_df_log)
    paths["rfm_log_dist"] = paths["rfm_log_distribution"]
    paths["elbow"] = plot_elbow(k_metrics_df)
    paths["silhouette"] = plot_silhouette(k_metrics_df)
    paths["cluster_sizes"] = plot_cluster_sizes(labels)
    paths["pca_clusters"] = plot_pca_clusters(pca_df, labels)
    paths["profile_heatmap"] = plot_cluster_profile_heatmap(cluster_profiles_scaled)
    paths["profile_radar"] = plot_cluster_profile_radar(cluster_profiles)
    paths["monetary_bar"] = plot_cluster_bar_comparison(cluster_profiles)
    logger.info(f"Generated {len(paths)} figures")
    logger.info("VISUALIZATION COMPLETE")
    return paths
