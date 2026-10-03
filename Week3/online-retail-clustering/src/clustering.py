import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config import (
    K_RANGE_END,
    K_RANGE_START,
    PCA_N_COMPONENTS,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def find_optimal_k(
    scaled_features: np.ndarray,
    k_range: range = None,
) -> pd.DataFrame:
    """
    Iterate over a range of k values, fit KMeans for each, and record
    inertia and silhouette score to help select an optimal cluster count.

    Parameters
    ----------
    scaled_features : np.ndarray
        2D array of standardized features, shape (n_samples, n_features).
    k_range : range, optional
        Iterable of k values to test. Defaults to range(K_RANGE_START, K_RANGE_END + 1)
        as defined in config.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: k, inertia, silhouette_score. One row per tested k.
    """
    logger.info("Finding optimal k via inertia and silhouette analysis")
    if k_range is None:
        k_range = range(K_RANGE_START, K_RANGE_END + 1)
        logger.info(f"Using default k range: {list(k_range)}")

    if scaled_features.ndim != 2:
        raise ValueError(
            f"scaled_features must be 2D, got shape {scaled_features.shape}"
        )

    records: List[dict] = []
    for k in k_range:
        logger.info(f"Fitting KMeans with k={k}")
        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=10,
        )
        labels = kmeans.fit_predict(scaled_features)
        inertia = kmeans.inertia_
        sil = silhouette_score(scaled_features, labels)
        logger.info(
            f"k={k}: inertia={inertia:.2f}, silhouette_score={sil:.4f}"
        )
        records.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": sil,
        })

    result = pd.DataFrame(records)
    return result


def run_kmeans(
    scaled_features: np.ndarray,
    k: int,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, KMeans]:
    """
    Fit KMeans with a specified number of clusters and return labels + model.

    Parameters
    ----------
    scaled_features : np.ndarray
        2D array of standardized features.
    k : int
        Number of clusters.
    random_state : int, optional
        Random seed for reproducibility. Defaults to RANDOM_SEED from config.

    Returns
    -------
    Tuple[np.ndarray, KMeans]
        - labels : 1D array of cluster assignments, length n_samples
        - kmeans_model : Fitted sklearn KMeans instance
    """
    logger.info(f"Running KMeans with k={k}, random_state={random_state}")
    if scaled_features.ndim != 2:
        raise ValueError(
            f"scaled_features must be 2D, got shape {scaled_features.shape}"
        )
    if not isinstance(k, int) or k < 2:
        raise ValueError(f"k must be an integer >= 2, got {k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(scaled_features)

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        logger.info(f"Cluster {int(u)} size: {c} ({c / len(labels) * 100:.1f}%)")

    return labels, kmeans


def apply_pca_for_visualization(
    scaled_features: np.ndarray,
    n_components: int = PCA_N_COMPONENTS,
) -> Tuple[pd.DataFrame, List[float], PCA]:
    """
    Project scaled features into 2D (or n_components-D) space using PCA for visualization.

    Parameters
    ----------
    scaled_features : np.ndarray
        2D array of standardized features.
    n_components : int, optional
        Number of principal components to keep. Defaults to PCA_N_COMPONENTS from config.

    Returns
    -------
    Tuple[pd.DataFrame, List[float], PCA]
        - pca_df : DataFrame with columns PC1, PC2 (, PC3, ...) for each sample.
        - explained_variance_ratio_list : List of explained variance ratios per component.
        - pca_model : Fitted sklearn PCA instance.
    """
    logger.info(
        f"Applying PCA for visualization with n_components={n_components}"
    )
    if scaled_features.ndim != 2:
        raise ValueError(
            f"scaled_features must be 2D, got shape {scaled_features.shape}"
        )

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca_array = pca.fit_transform(scaled_features)

    col_names = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_array, columns=col_names)

    evr = list(pca.explained_variance_ratio_)
    logger.info(f"PCA explained variance ratios: {[f'{r:.4f}' for r in evr]}")
    logger.info(f"Total explained variance: {sum(evr):.4f}")

    return pca_df, evr, pca


def compute_cluster_profiles(
    rfm_df_with_labels: pd.DataFrame,
    cluster_col: str = "Cluster",
) -> pd.DataFrame:
    """
    Compute aggregate profile statistics per cluster.

    Parameters
    ----------
    rfm_df_with_labels : pd.DataFrame
        DataFrame containing Recency, Frequency, Monetary plus a cluster label column.
    cluster_col : str, optional
        Name of the cluster label column. Defaults to "Cluster".

    Returns
    -------
    pd.DataFrame
        Profile DataFrame sorted by cluster number, with columns:
        cluster, customer_count, pct_of_customers,
        mean_recency, median_recency,
        mean_frequency, median_frequency,
        mean_monetary, median_monetary,
        total_monetary, pct_of_total_revenue.
    """
    logger.info(f"Computing cluster profiles grouped by '{cluster_col}'")
    required = ["Recency", "Frequency", "Monetary", cluster_col]
    missing = [c for c in required if c not in rfm_df_with_labels.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = rfm_df_with_labels.copy()
    total_customers = len(df)
    total_revenue = df["Monetary"].sum()

    profiles = (
        df.groupby(cluster_col)
        .agg(
            customer_count=("CustomerID", "nunique")
            if "CustomerID" in df.columns
            else (cluster_col, "size"),
            mean_recency=("Recency", "mean"),
            median_recency=("Recency", "median"),
            mean_frequency=("Frequency", "mean"),
            median_frequency=("Frequency", "median"),
            mean_monetary=("Monetary", "mean"),
            median_monetary=("Monetary", "median"),
            total_monetary=("Monetary", "sum"),
        )
        .reset_index()
        .rename(columns={cluster_col: "cluster"})
    )

    profiles["customer_count"] = profiles["customer_count"].astype(int)
    profiles["pct_of_customers"] = (
        profiles["customer_count"] / total_customers * 100
    )
    profiles["pct_of_total_revenue"] = (
        profiles["total_monetary"] / total_revenue * 100
    )

    ordered_cols = [
        "cluster",
        "customer_count",
        "pct_of_customers",
        "mean_recency",
        "median_recency",
        "mean_frequency",
        "median_frequency",
        "mean_monetary",
        "median_monetary",
        "total_monetary",
        "pct_of_total_revenue",
    ]
    profiles = profiles[ordered_cols].sort_values("cluster").reset_index(drop=True)

    logger.info(f"Built profile for {len(profiles)} clusters")
    logger.debug(f"Cluster profiles:\n{profiles}")
    return profiles
