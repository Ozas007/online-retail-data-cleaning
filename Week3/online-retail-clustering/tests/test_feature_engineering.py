import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (
    compute_rfm,
    apply_log1p_transformation,
    scale_rfm_features,
    inspect_rfm_distributions,
)


def _positive_sales_df() -> pd.DataFrame:
    return pd.DataFrame({
        "CustomerID": [1001.0, 1001.0, 1001.0, 1002.0, 1002.0, 1003.0, np.nan],
        "InvoiceNo": ["A001", "A002", "A003", "B001", "B002", "C001", "D001"],
        "InvoiceDate": pd.to_datetime([
            "2011-12-01",
            "2011-12-05",
            "2011-12-09",
            "2011-11-20",
            "2011-12-01",
            "2011-10-15",
            "2011-12-10",
        ]),
        "Quantity": [2, 3, 1, 5, 2, 10, 1],
        "UnitPrice": [10.0, 20.0, 5.0, 4.0, 15.0, 3.0, 50.0],
        "TotalAmount": [20.0, 60.0, 5.0, 20.0, 30.0, 30.0, 50.0],
    })


class TestComputeRfm:
    def test_recency_frequency_monetary_values(self):
        df = _positive_sales_df()
        rfm = compute_rfm(df)
        max_date = pd.Timestamp("2011-12-09")
        expected_ref = max_date + timedelta(days=1)
        rfm_sorted = rfm.set_index("CustomerID").sort_index()
        assert rfm_sorted.loc[1001.0, "Recency"] == (expected_ref - pd.Timestamp("2011-12-09")).days
        assert rfm_sorted.loc[1002.0, "Recency"] == (expected_ref - pd.Timestamp("2011-12-01")).days
        assert rfm_sorted.loc[1003.0, "Recency"] == (expected_ref - pd.Timestamp("2011-10-15")).days
        assert rfm_sorted.loc[1001.0, "Frequency"] == 3
        assert rfm_sorted.loc[1002.0, "Frequency"] == 2
        assert rfm_sorted.loc[1003.0, "Frequency"] == 1
        assert rfm_sorted.loc[1001.0, "Monetary"] == pytest.approx(85.0)
        assert rfm_sorted.loc[1002.0, "Monetary"] == pytest.approx(50.0)
        assert rfm_sorted.loc[1003.0, "Monetary"] == pytest.approx(30.0)

    def test_excludes_null_customer_id(self):
        df = _positive_sales_df()
        rfm = compute_rfm(df)
        assert 1001.0 in rfm["CustomerID"].values
        assert 1002.0 in rfm["CustomerID"].values
        assert 1003.0 in rfm["CustomerID"].values
        assert len(rfm) == 3

    def test_default_reference_date_is_max_plus_one(self):
        df = _positive_sales_df()
        df_clean = df[df["CustomerID"].notna()].copy()
        max_d = df_clean["InvoiceDate"].max()
        rfm = compute_rfm(df)
        c1001_max = df_clean[df_clean["CustomerID"] == 1001.0]["InvoiceDate"].max()
        expected_ref = max_d + timedelta(days=1)
        expected_recency = (expected_ref - c1001_max).days
        actual = rfm.loc[rfm["CustomerID"] == 1001.0, "Recency"].iloc[0]
        assert actual == expected_recency

    def test_custom_reference_date(self):
        df = _positive_sales_df()
        ref = datetime(2011, 12, 15)
        rfm = compute_rfm(df, reference_date=ref)
        r1001 = rfm.loc[rfm["CustomerID"] == 1001.0, "Recency"].iloc[0]
        assert r1001 == (ref - pd.Timestamp("2011-12-09")).days


class TestApplyLog1pTransformation:
    def test_produces_log_columns(self):
        df = pd.DataFrame({
            "CustomerID": [1, 2, 3],
            "Recency": [10, 100, 500],
            "Frequency": [1, 5, 20],
            "Monetary": [100.0, 500.0, 5000.0],
        })
        result = apply_log1p_transformation(df)
        for col in ["Recency", "Frequency", "Monetary"]:
            log_col = f"Log_{col}"
            assert log_col in result.columns
            expected = np.log1p(df[col].values)
            np.testing.assert_array_almost_equal(result[log_col].values, expected)


class TestScaleRfmFeatures:
    def test_returns_scaled_array_and_fitted_scaler(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "CustomerID": np.arange(10),
            "Recency": np.random.randint(1, 365, 10),
            "Frequency": np.random.randint(1, 50, 10),
            "Monetary": np.random.uniform(10, 5000, 10),
        })
        scaled, scaler = scale_rfm_features(df)
        n_customers = len(df)
        assert scaled.shape == (n_customers, 3)
        assert isinstance(scaler, StandardScaler)
        assert np.all(np.isfinite(scaler.mean_))


class TestInspectRfmDistributions:
    def test_returns_dict_with_rfm_keys(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "CustomerID": np.arange(10),
            "Recency": np.random.randint(1, 365, 10),
            "Frequency": np.random.randint(1, 50, 10),
            "Monetary": np.random.uniform(10, 5000, 10),
        })
        result = inspect_rfm_distributions(df)
        assert isinstance(result, dict)
        for key in ["Recency", "Frequency", "Monetary"]:
            assert key in result
            assert isinstance(result[key], pd.DataFrame)
