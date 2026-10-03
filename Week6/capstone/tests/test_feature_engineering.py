import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_rfm_features, build_supervised_features
from src.preprocessing import clean_dataset


def _make_synthetic_txns(n_cust: int = 40, n_txns: int = 600):
    rng = np.random.default_rng(42)
    customer_ids = np.arange(12346, 12346 + n_cust)
    cids = rng.choice(customer_ids, size=n_txns)
    return pd.DataFrame({
        "InvoiceNo": [f"T{i:06d}" for i in range(n_txns)],
        "StockCode": [f"STK{i%30:03d}" for i in range(n_txns)],
        "Description": [f"P{i%30}" for i in range(n_txns)],
        "Quantity": rng.integers(1, 15, size=n_txns),
        "InvoiceDate": pd.date_range("2011-02-01", periods=n_txns, freq="30min"),
        "UnitPrice": np.round(rng.uniform(1.0, 20.0, size=n_txns), 2),
        "CustomerID": cids.astype(np.float64),
        "Country": rng.choice(["United Kingdom", "Germany", "France"], size=n_txns),
    })


def test_build_rfm_returns_expected_columns():
    df_raw = _make_synthetic_txns()
    _, df_pos = clean_dataset(df_raw)
    rfm = build_rfm_features(df_pos)
    assert isinstance(rfm, pd.DataFrame)
    assert "CustomerID" in rfm.columns
    for col in ("Recency", "Frequency", "Monetary"):
        assert col in rfm.columns
    assert (rfm["Frequency"] >= 1).all()
    assert (rfm["Monetary"] >= 0).all()
    assert len(rfm) <= df_pos["CustomerID"].nunique()


def test_build_rfm_recency_is_non_negative():
    df_raw = _make_synthetic_txns()
    _, df_pos = clean_dataset(df_raw)
    rfm = build_rfm_features(df_pos)
    assert (rfm["Recency"] >= 0).all()


def test_build_supervised_features_returns_all_numeric():
    df_raw = _make_synthetic_txns()
    _, df_pos = clean_dataset(df_raw)
    start = df_pos["InvoiceDate"].min()
    end = df_pos["InvoiceDate"].max()
    features = build_supervised_features(df_pos, historical_start=start, historical_end=end)
    assert isinstance(features, pd.DataFrame)
    assert "CustomerID" in features.columns
    for col in ["Recency", "Frequency", "Monetary", "AOV",
                "UniqueProducts", "AvgQuantity", "TotalQuantity", "PurchaseFrequency"]:
        assert col in features.columns, f"Missing column {col}"
    numeric_cols = features.drop(columns=["CustomerID", "Country"], errors="ignore")
    assert numeric_cols.isna().sum().sum() == 0


def test_build_supervised_features_respects_cutoff():
    rng = np.random.default_rng(0)
    cid_kept = 12346
    cutoff = pd.Timestamp("2011-03-15 00:00:00")
    before = pd.DataFrame({
        "InvoiceNo": ["T001", "T002", "T003"],
        "StockCode": ["A", "B", "C"],
        "Description": ["P1", "P2", "P3"],
        "Quantity": [5, 3, 10],
        "InvoiceDate": pd.to_datetime(["2011-02-01", "2011-02-15", "2011-03-10"]),
        "UnitPrice": [10.0, 20.0, 5.0],
        "CustomerID": [cid_kept, cid_kept, 99999],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom"],
    })
    after = pd.DataFrame({
        "InvoiceNo": ["T100", "T101"],
        "StockCode": ["Z", "Y"],
        "Description": ["PZ", "PY"],
        "Quantity": [1000, 1000],
        "InvoiceDate": pd.to_datetime(["2011-03-20", "2011-04-01"]),
        "UnitPrice": [100.0, 100.0],
        "CustomerID": [cid_kept, 99999],
        "Country": ["United Kingdom", "United Kingdom"],
    })
    combined = pd.concat([before, after], ignore_index=True)
    features = build_supervised_features(
        combined,
        historical_start=pd.Timestamp("2011-01-01"),
        historical_end=cutoff,
    )
    row = features[features["CustomerID"].astype(int) == cid_kept].iloc[0]
    assert int(row["TotalQuantity"]) == 8, (
        f"Post-cutoff rows must be excluded from feature computation. Got TotalQuantity={row['TotalQuantity']}"
    )
