import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import (
    find_optimal_k,
    run_kmeans,
    apply_pca_for_visualization,
    compute_cluster_profiles,
)


def _three_blobs(n_per_blob: int = 15) -> np.ndarray:
    np.random.seed(42)
    centers = np.array([-3.0, 0.0, 3.0])
    blobs = []
    for c in centers:
        samples = np.random.normal(loc=c, scale=0.3, size=(n_per_blob, 3))
        blobs.append(samples)
    data = np.vstack(blobs)
    np.random.seed(42)
    perm = np.random.permutation(len(data))
    return data[perm]


class TestFindOptimalK:
    def test_returns_dataframe_with_metrics(self):
        data = _three_blobs()
        result = find_optimal_k(data, k_range=range(2, 7))
        assert isinstance(result, pd.DataFrame)
        for col in ["k", "inertia", "silhouette_score"]:
            assert col in result.columns
        assert list(result["k"]) == list(range(2, 7))
        inertias = result["inertia"].values
        assert np.all(np.diff(inertias) <= 0)
        sil_scores = result["silhouette_score"].values
        assert np.all(sil_scores >= -1) and np.all(sil_scores <= 1)


class TestRunKmeans:
    def test_returns_labels_and_model(self):
        data = _three_blobs()
        labels, model = run_kmeans(data, k=3)
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (len(data),)
        assert isinstance(model, KMeans)
        assert model.cluster_centers_.shape == (3, data.shape[1])
        unique_labels = np.unique(labels)
        assert set(unique_labels).issubset({0, 1, 2})
        assert len(unique_labels) == 3


class TestApplyPcaForVisualization:
    def test_returns_df_evr_and_pca_model(self):
        data = _three_blobs()
        pca_df, evr, pca_model = apply_pca_for_visualization(data, n_components=2)
        assert isinstance(pca_df, pd.DataFrame)
        assert list(pca_df.columns) == ["PC1", "PC2"]
        assert len(pca_df) == len(data)
        assert isinstance(evr, list)
        assert len(evr) == 2
        assert all(isinstance(v, float) for v in evr)
        assert sum(evr) <= 1.0 + 1e-9
        assert isinstance(pca_model, PCA)


class TestComputeClusterProfiles:
    def test_output_has_expected_columns_and_one_row_per_cluster(self):
        np.random.seed(42)
        n = 30
        clusters = np.repeat([0, 1, 2], 10)
        rfm_df = pd.DataFrame({
            "CustomerID": np.arange(n),
            "Recency": np.concatenate([
                np.random.normal(10, 2, 10),
                np.random.normal(100, 10, 10),
                np.random.normal(300, 20, 10),
            ]),
            "Frequency": np.concatenate([
                np.random.randint(20, 40, 10),
                np.random.randint(5, 15, 10),
                np.random.randint(1, 5, 10),
            ]),
            "Monetary": np.concatenate([
                np.random.uniform(2000, 5000, 10),
                np.random.uniform(500, 1500, 10),
                np.random.uniform(50, 300, 10),
            ]),
            "Cluster": clusters,
        })
        profiles = compute_cluster_profiles(rfm_df)
        expected_cols = [
            "cluster", "customer_count", "pct_of_customers",
            "mean_recency", "median_recency",
            "mean_frequency", "median_frequency",
            "mean_monetary", "median_monetary",
            "total_monetary", "pct_of_total_revenue",
        ]
        assert list(profiles.columns) == expected_cols
        assert len(profiles) == 3
        assert profiles["cluster"].tolist() == [0, 1, 2]
        assert profiles["customer_count"].sum() == n
        np.testing.assert_almost_equal(profiles["pct_of_customers"].sum(), 100.0, decimal=5)
        np.testing.assert_almost_equal(profiles["pct_of_total_revenue"].sum(), 100.0, decimal=5)
