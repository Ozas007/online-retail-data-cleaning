from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    build_recommendations,
    summarize_supervised,
    summarize_unsupervised,
)


def _sample_unsupervised_results() -> Dict[str, Any]:
    profiles = pd.DataFrame({
        "cluster": [0, 1],
        "customer_count": [120, 80],
        "pct_of_customers": [60.0, 40.0],
        "pct_of_total_revenue": [15.0, 85.0],
        "mean_recency": [135.0, 25.0],
        "mean_frequency": [1.5, 9.2],
        "mean_monetary": [180.0, 3500.0],
    })
    metrics = {
        "silhouette_score": 0.4328,
        "calinski_harabasz_score": 2100.5,
        "davies_bouldin_score": 0.82,
    }
    return {"profiles": profiles, "metrics": metrics}


def _sample_supervised_results() -> Dict[str, Any]:
    metrics = {
        "accuracy": 0.704,
        "precision": 0.702,
        "recall": 0.699,
        "f1": 0.700,
        "roc_auc": 0.755,
    }
    importances = pd.DataFrame({
        "feature": ["Frequency", "Recency", "Monetary", "PurchaseFrequency"],
        "importance": [0.154, 0.120, 0.110, 0.145],
    })
    return {"metrics": metrics, "importances": importances}


def test_summarize_unsupervised_returns_text():
    u = _sample_unsupervised_results()
    text = summarize_unsupervised(u["metrics"], u["profiles"])
    assert isinstance(text, str)
    assert len(text) > 100
    assert "Silhouette" in text
    assert "Cluster" in text


def test_summarize_supervised_returns_text():
    s = _sample_supervised_results()
    text = summarize_supervised(s["metrics"], s["importances"])
    assert isinstance(text, str)
    assert len(text) > 50
    assert "ROC-AUC" in text
    assert "Frequency" in text


def test_build_recommendations_returns_at_least_5():
    u = _sample_unsupervised_results()
    s = _sample_supervised_results()
    combined: Dict[str, Any] = {
        "unsupervised": {
            "cluster_profiles_df": u["profiles"],
            "clustering_metrics_dict": u["metrics"],
            "best_k": 2,
        },
        "supervised": {
            "test_metrics_dict": s["metrics"],
            "feature_importance_df": s["importances"],
            "best_model_name": "Random Forest",
            "temporal_dates_dict": {
                "days_historical": 314,
                "days_future": 59,
            },
        },
        "eda": {
            "top10_countries_by_transactions": pd.DataFrame({
                "Country": ["United Kingdom", "Germany", "France"],
                "transactions": [16000, 450, 360],
                "revenue": [7_200_000.0, 180_000.0, 150_000.0],
            }),
            "top10_products_by_revenue": pd.DataFrame({
                "StockCode": ["S001", "S002", "S003"],
                "Description": ["Widget A", "Gadget B", "Tool C"],
                "revenue": [60000.0, 45000.0, 30000.0],
            }),
            "monthly_volume": pd.DataFrame({
                "YearMonth": ["2011-01", "2011-09", "2011-11"],
                "revenue": [500_000.0, 900_000.0, 1_200_000.0],
            }),
        },
        "n_feature_customers": 200,
        "deep_learning": {
            "mlp_test_metrics": {"roc_auc": 0.748, "f1": 0.680},
        },
    }
    recs = build_recommendations(combined)
    assert isinstance(recs, list)
    assert len(recs) >= 5
    for r in recs:
        assert isinstance(r, str)
        assert len(r) > 20
