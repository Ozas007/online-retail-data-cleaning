import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def compute_clustering_metrics(
    scaled_features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute three common internal clustering evaluation metrics.

    Parameters
    ----------
    scaled_features : np.ndarray
        2D array of standardized features, shape (n_samples, n_features).
    labels : np.ndarray
        1D array of cluster assignments, length n_samples.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - silhouette_score : Mean silhouette coefficient (-1 to 1, higher is better).
        - calinski_harabasz_score : Variance ratio criterion (higher is better).
        - davies_bouldin_score : Average similarity between clusters (lower is better).
    """
    logger.info("Computing clustering evaluation metrics")
    np.random.seed(RANDOM_SEED)

    if scaled_features.ndim != 2:
        raise ValueError(
            f"scaled_features must be 2D, got shape {scaled_features.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if len(scaled_features) != len(labels):
        raise ValueError(
            f"Length mismatch: scaled_features has {len(scaled_features)} samples, "
            f"labels has {len(labels)}"
        )
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        raise ValueError(
            f"Need at least 2 clusters to compute metrics, got {n_clusters}"
        )

    sil = float(silhouette_score(scaled_features, labels))
    ch = float(calinski_harabasz_score(scaled_features, labels))
    db = float(davies_bouldin_score(scaled_features, labels))

    metrics = {
        "silhouette_score": sil,
        "calinski_harabasz_score": ch,
        "davies_bouldin_score": db,
    }

    logger.info(
        f"Metrics: silhouette={sil:.4f}, calinski_harabasz={ch:.2f}, "
        f"davies_bouldin={db:.4f}"
    )
    return metrics


def rank_clusters_by_profit(
    cluster_profiles: pd.DataFrame,
    by_total: bool = True,
) -> pd.DataFrame:
    """
    Sort cluster profiles by monetary value to identify the most profitable segments.

    Parameters
    ----------
    cluster_profiles : pd.DataFrame
        DataFrame as returned by compute_cluster_profiles(). Must contain
        total_monetary and mean_monetary columns.
    by_total : bool, optional
        If True, sort by total_monetary descending. If False, sort by
        mean_monetary descending. Defaults to True.

    Returns
    -------
    pd.DataFrame
        Sorted copy of cluster_profiles with the most profitable cluster first.
    """
    logger.info(
        f"Ranking clusters by {'total' if by_total else 'mean'} monetary desc"
    )
    sort_col = "total_monetary" if by_total else "mean_monetary"
    if sort_col not in cluster_profiles.columns:
        raise ValueError(
            f"Missing required column '{sort_col}' in cluster_profiles"
        )

    ranked = cluster_profiles.sort_values(sort_col, ascending=False).reset_index(
        drop=True
    )
    logger.info(
        f"Top cluster: #{ranked.iloc[0]['cluster']} with "
        f"{sort_col}={ranked.iloc[0][sort_col]:.2f}"
    )
    return ranked


def describe_cluster(
    cluster_row: pd.Series,
    overall_rfm_stats: Dict[str, float],
) -> str:
    """
    Build a natural-language, evidence-based description of a single cluster
    by comparing its R/F/M metrics against the overall (dataset-level) means.

    Parameters
    ----------
    cluster_row : pd.Series
        Row from the cluster_profiles DataFrame. Must contain mean_recency,
        mean_frequency, mean_monetary, cluster, customer_count,
        pct_of_customers, total_monetary, pct_of_total_revenue.
    overall_rfm_stats : Dict[str, float]
        Dict with keys "Recency", "Frequency", "Monetary" mapping to the
        overall dataset-wide mean of each metric. Used as the comparison
        baseline to decide "high" vs "low".

    Returns
    -------
    str
        A human-readable description of the cluster.
    """
    logger.info(
        f"Generating description for cluster {cluster_row.get('cluster', '?')}"
    )
    needed_keys = ["Recency", "Frequency", "Monetary"]
    missing = [k for k in needed_keys if k not in overall_rfm_stats]
    if missing:
        raise ValueError(
            f"overall_rfm_stats missing required keys: {missing}"
        )

    cluster_id = int(cluster_row["cluster"])
    cust_count = int(cluster_row["customer_count"])
    pct_cust = float(cluster_row["pct_of_customers"])
    total_rev = float(cluster_row["total_monetary"])
    pct_rev = float(cluster_row["pct_of_total_revenue"])

    mean_recency = float(cluster_row["mean_recency"])
    mean_frequency = float(cluster_row["mean_frequency"])
    mean_monetary = float(cluster_row["mean_monetary"])

    overall_rec = overall_rfm_stats["Recency"]
    overall_freq = overall_rfm_stats["Frequency"]
    overall_mon = overall_rfm_stats["Monetary"]

    recency_ratio = mean_recency / overall_rec if overall_rec != 0 else 1.0
    frequency_ratio = mean_frequency / overall_freq if overall_freq != 0 else 1.0
    monetary_ratio = mean_monetary / overall_mon if overall_mon != 0 else 1.0

    HIGH = 1.2
    LOW = 0.8

    if recency_ratio <= LOW:
        recency_label = "recent"
    elif recency_ratio >= HIGH:
        recency_label = "stale"
    else:
        recency_label = "moderately recent"

    if frequency_ratio >= HIGH:
        freq_label = "high-frequency"
    elif frequency_ratio <= LOW:
        freq_label = "low-frequency"
    else:
        freq_label = "moderate-frequency"

    if monetary_ratio >= HIGH:
        mon_label = "high-monetary"
    elif monetary_ratio <= LOW:
        mon_label = "low-monetary"
    else:
        mon_label = "moderate-monetary"

    traits = [freq_label, mon_label]
    if recency_label == "recent":
        traits.append("with relatively recent purchases")
    elif recency_label == "stale":
        traits.append("with relatively stale (old) purchases")
    else:
        traits.append("with average recency")

    traits_str = ", ".join(traits)
    description = (
        f"Cluster {cluster_id}: {traits_str.capitalize()}. "
        f"Contains {cust_count} customers ({pct_cust:.1f}% of all customers), "
        f"generating total revenue of {total_rev:.2f} ({pct_rev:.1f}% of total revenue). "
        f"Mean R/F/M: Recency={mean_recency:.1f}d (vs overall {overall_rec:.1f}d), "
        f"Frequency={mean_frequency:.1f} orders (vs overall {overall_freq:.1f}), "
        f"Monetary={mean_monetary:.2f} (vs overall {overall_mon:.2f})."
    )

    logger.debug(f"Cluster {cluster_id} description: {description}")
    return description
