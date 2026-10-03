import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from src.config import (
    K_RANGE_END,
    K_RANGE_START,
    PCA_N_COMPONENTS,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def _log1p_scale_rfm(rfm_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    cols = ["Recency", "Frequency", "Monetary"]
    missing = [c for c in cols if c not in rfm_df.columns]
    if missing:
        raise ValueError(f"Missing RFM columns for scaling: {missing}")
    log_df = pd.DataFrame()
    for c in cols:
        log_df[c] = np.log1p(rfm_df[c])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_df.values)
    return scaled, scaler


def run_unsupervised_pipeline(rfm_df: pd.DataFrame) -> Dict:
    logger.info("Starting unsupervised learning pipeline (KMeans on RFM)")
    np.random.seed(RANDOM_SEED)

    required = ["CustomerID", "Recency", "Frequency", "Monetary"]
    missing = [c for c in required if c not in rfm_df.columns]
    if missing:
        raise ValueError(f"Missing required RFM columns: {missing}")

    scaled_features, _ = _log1p_scale_rfm(rfm_df)

    records: List[dict] = []
    k_range = range(K_RANGE_START, K_RANGE_END + 1)
    best_sil = -1.0
    best_k = K_RANGE_START
    best_labels: np.ndarray | None = None
    best_model: KMeans | None = None

    for k in k_range:
        logger.info(f"Fitting KMeans with k={k}")
        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=10,
        )
        labels = kmeans.fit_predict(scaled_features)
        inertia = kmeans.inertia_
        sil = float(silhouette_score(scaled_features, labels))
        records.append({"k": k, "inertia": inertia, "silhouette_score": sil})
        logger.info(f"k={k}: inertia={inertia:.2f}, silhouette_score={sil:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels.copy()
            best_model = kmeans

    k_metrics_df = pd.DataFrame(records)
    logger.info(f"Best k selected: {best_k} (silhouette={best_sil:.4f})")

    assert best_labels is not None and best_model is not None
    labels_array = best_labels.astype(int)

    rfm_labelled = rfm_df.copy()
    rfm_labelled["Cluster"] = labels_array

    sil_final = float(silhouette_score(scaled_features, labels_array))
    ch = float(calinski_harabasz_score(scaled_features, labels_array))
    db = float(davies_bouldin_score(scaled_features, labels_array))
    clustering_metrics_dict = {
        "silhouette_score": sil_final,
        "calinski_harabasz_score": ch,
        "davies_bouldin_score": db,
    }
    logger.info(f"Final metrics: silhouette={sil_final:.4f}, CH={ch:.2f}, DB={db:.4f}")

    total_customers = len(rfm_labelled)
    total_revenue = float(rfm_labelled["Monetary"].sum())
    profiles = (
        rfm_labelled.groupby("Cluster")
        .agg(
            customer_count=("CustomerID", "nunique"),
            mean_recency=("Recency", "mean"),
            median_recency=("Recency", "median"),
            mean_frequency=("Frequency", "mean"),
            median_frequency=("Frequency", "median"),
            mean_monetary=("Monetary", "mean"),
            median_monetary=("Monetary", "median"),
            total_monetary=("Monetary", "sum"),
        )
        .reset_index()
        .rename(columns={"Cluster": "cluster"})
    )
    profiles["customer_count"] = profiles["customer_count"].astype(int)
    profiles["pct_of_customers"] = profiles["customer_count"] / total_customers * 100
    profiles["pct_of_total_revenue"] = profiles["total_monetary"] / total_revenue * 100 if total_revenue > 0 else 0.0
    ordered_cols = [
        "cluster", "customer_count", "pct_of_customers",
        "mean_recency", "median_recency",
        "mean_frequency", "median_frequency",
        "mean_monetary", "median_monetary",
        "total_monetary", "pct_of_total_revenue",
    ]
    profiles = profiles[[c for c in ordered_cols if c in profiles.columns]].sort_values("cluster").reset_index(drop=True)
    cluster_profiles_df = profiles

    rfm_cols = ["Recency", "Frequency", "Monetary"]
    mean_ = rfm_labelled[rfm_cols].mean()
    std_ = rfm_labelled[rfm_cols].std(ddof=0).replace(0, np.nan)
    scaled_profiles = pd.DataFrame()
    scaled_profiles["cluster"] = cluster_profiles_df["cluster"]
    for col in rfm_cols:
        mean_col = f"mean_{col.lower()}"
        if mean_col in cluster_profiles_df.columns:
            vals = (cluster_profiles_df[mean_col] - mean_[col]) / std_[col]
            scaled_profiles[f"{col}_z"] = vals
    scaled_profiles_df = scaled_profiles

    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)
    pca_arr = pca.fit_transform(scaled_features)
    pca_col_names = [f"PC{i + 1}" for i in range(PCA_N_COMPONENTS)]
    pca_df = pd.DataFrame(pca_arr, columns=pca_col_names)
    pca_df["Cluster"] = labels_array
    pca_evr = list(pca.explained_variance_ratio_)
    logger.info(f"PCA EVR: {[f'{r:.4f}' for r in pca_evr]}  total={sum(pca_evr):.4f}")

    return {
        "best_k": int(best_k),
        "labels_array": labels_array,
        "k_metrics_df": k_metrics_df,
        "cluster_profiles_df": cluster_profiles_df,
        "clustering_metrics_dict": clustering_metrics_dict,
        "pca_df": pca_df,
        "pca_evr": pca_evr,
        "scaled_profiles_df": scaled_profiles_df,
    }
