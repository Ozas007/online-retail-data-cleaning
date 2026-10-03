import numpy as np
import pandas as pd
import pytest

from src.unsupervised import run_unsupervised_pipeline
from src.preprocessing import clean_dataset
from src.feature_engineering import build_rfm_features


def _make_synthetic_rfm(n_cust: int = 80):
    rng = np.random.default_rng(42)
    cluster0 = pd.DataFrame({
        "CustomerID": np.arange(10000, 10000 + n_cust // 2),
        "Recency": rng.integers(60, 300, size=n_cust // 2),
        "Frequency": rng.integers(1, 4, size=n_cust // 2),
        "Monetary": rng.uniform(10, 200, size=n_cust // 2),
    })
    cluster1 = pd.DataFrame({
        "CustomerID": np.arange(20000, 20000 + n_cust // 2),
        "Recency": rng.integers(1, 45, size=n_cust // 2),
        "Frequency": rng.integers(5, 20, size=n_cust // 2),
        "Monetary": rng.uniform(500, 5000, size=n_cust // 2),
    })
    return pd.concat([cluster0, cluster1], ignore_index=True)


def test_run_unsupervised_returns_structure():
    rfm = _make_synthetic_rfm()
    result = run_unsupervised_pipeline(rfm)
    assert isinstance(result, dict)
    assert "best_k" in result
    assert "labels_array" in result
    assert "k_metrics_df" in result
    assert "cluster_profiles_df" in result
    assert "clustering_metrics_dict" in result
    assert "pca_df" in result


def test_run_unsupervised_labels_and_clustering_metrics():
    rfm = _make_synthetic_rfm()
    result = run_unsupervised_pipeline(rfm)
    labels = result["labels_array"]
    assert len(labels) == len(rfm)
    best_k = int(result["best_k"])
    unique_labels = np.unique(labels)
    assert len(unique_labels) == best_k or len(unique_labels) >= 2
    prof = result["cluster_profiles_df"]
    assert "cluster" in prof.columns
    assert len(prof) >= 2
    metrics = result["clustering_metrics_dict"]
    assert "silhouette_score" in metrics
    sil = float(metrics["silhouette_score"])
    assert -1.0 <= sil <= 1.0
