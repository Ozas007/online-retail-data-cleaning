import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    compute_clustering_metrics,
    rank_clusters_by_profit,
    describe_cluster,
)


def _three_blobs_with_labels(n_per_blob: int = 15):
    np.random.seed(42)
    centers = np.array([-3.0, 0.0, 3.0])
    blobs = []
    labels_list = []
    for i, c in enumerate(centers):
        samples = np.random.normal(loc=c, scale=0.3, size=(n_per_blob, 3))
        blobs.append(samples)
        labels_list.append(np.full(n_per_blob, i))
    data = np.vstack(blobs)
    labels = np.concatenate(labels_list)
    np.random.seed(42)
    perm = np.random.permutation(len(data))
    return data[perm], labels[perm]


def _cluster_profiles_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "cluster": 0,
            "customer_count": 10,
            "pct_of_customers": 10.0,
            "mean_recency": 10.0,
            "median_recency": 9.0,
            "mean_frequency": 30.0,
            "median_frequency": 28.0,
            "mean_monetary": 350.0,
            "median_monetary": 340.0,
            "total_monetary": 3500.0,
            "pct_of_total_revenue": 70.0,
        },
        {
            "cluster": 1,
            "customer_count": 50,
            "pct_of_customers": 50.0,
            "mean_recency": 100.0,
            "median_recency": 95.0,
            "mean_frequency": 10.0,
            "median_frequency": 9.0,
            "mean_monetary": 100.0,
            "median_monetary": 95.0,
            "total_monetary": 1000.0,
            "pct_of_total_revenue": 20.0,
        },
        {
            "cluster": 2,
            "customer_count": 40,
            "pct_of_customers": 40.0,
            "mean_recency": 300.0,
            "median_recency": 290.0,
            "mean_frequency": 3.0,
            "median_frequency": 2.0,
            "mean_monetary": 12.5,
            "median_monetary": 11.0,
            "total_monetary": 500.0,
            "pct_of_total_revenue": 10.0,
        },
    ])


class TestComputeClusteringMetrics:
    def test_returns_all_metrics_finite_and_in_range(self):
        data, labels = _three_blobs_with_labels()
        metrics = compute_clustering_metrics(data, labels)
        assert isinstance(metrics, dict)
        for key in ["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]:
            assert key in metrics
            assert np.isfinite(metrics[key])
            assert isinstance(metrics[key], float)
        assert -1.0 <= metrics["silhouette_score"] <= 1.0
        assert metrics["calinski_harabasz_score"] > 0
        assert metrics["davies_bouldin_score"] > 0


class TestRankClustersByProfit:
    def test_sorts_by_total_monetary_descending(self):
        profiles = _cluster_profiles_df()
        ranked = rank_clusters_by_profit(profiles, by_total=True)
        totals = ranked["total_monetary"].values
        assert np.all(np.diff(totals) <= 0)
        assert ranked.iloc[0]["total_monetary"] == profiles["total_monetary"].max()
        assert ranked.iloc[0]["cluster"] == 0


class TestDescribeCluster:
    def test_returns_non_empty_string_with_keywords(self):
        profiles = _cluster_profiles_df()
        overall_stats = {
            "Recency": 170.0,
            "Frequency": 10.0,
            "Monetary": 100.0,
        }
        row = profiles.iloc[0]
        desc = describe_cluster(row, overall_stats)
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "Cluster" in desc
        assert "customers" in desc
