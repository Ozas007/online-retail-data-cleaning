import numpy as np
import pandas as pd
import pytest

from src.preprocessing import clean_dataset


def _make_synthetic_raw(n_rows: int = 400):
    rng = np.random.default_rng(42)
    n_cust = 25
    idx = np.arange(n_rows)
    cancellations = int(n_rows * 0.05)
    invoices = [f"{100000+i}" for i in range(n_rows - cancellations)]
    invoices.extend([f"C{900000+i}" for i in range(cancellations)])
    rng.shuffle(invoices)
    return pd.DataFrame({
        "InvoiceNo": invoices,
        "StockCode": [f"STK{i%40:04d}" for i in idx],
        "Description": [f"Product {i%40}" for i in idx],
        "Quantity": np.where([s.startswith("C") for s in invoices],
                             rng.integers(-10, -1, size=n_rows),
                             rng.integers(1, 20, size=n_rows)),
        "InvoiceDate": pd.date_range("2011-02-01", periods=n_rows, freq="H"),
        "UnitPrice": np.round(rng.uniform(0.5, 30.0, size=n_rows), 2),
        "CustomerID": np.where(
            rng.random(n_rows) < 0.9,
            rng.integers(12346, 12346 + n_cust, size=n_rows).astype(np.float64),
            np.nan,
        ),
        "Country": rng.choice(["United Kingdom", "Germany", "France"], size=n_rows),
    })


def test_clean_dataset_returns_tuple():
    df_raw = _make_synthetic_raw()
    df_clean, df_pos = clean_dataset(df_raw)
    assert isinstance(df_clean, pd.DataFrame)
    assert isinstance(df_pos, pd.DataFrame)
    assert "TotalAmount" in df_clean.columns
    assert len(df_pos) <= len(df_clean)


def test_clean_dataset_drops_missing_customer_id():
    df_raw = _make_synthetic_raw()
    n_before = len(df_raw)
    _, df_pos = clean_dataset(df_raw)
    assert df_pos["CustomerID"].isna().sum() == 0
    assert len(df_pos) <= n_before


def test_clean_dataset_removes_duplicates():
    df_raw = _make_synthetic_raw()
    dup = df_raw.iloc[[0]].copy()
    df_raw_dup = pd.concat([df_raw, dup], ignore_index=True)
    df_clean, _ = clean_dataset(df_raw_dup)
    assert len(df_clean) <= len(df_raw_dup) - 1


def test_clean_dataset_positive_filter_excludes_neg():
    df_raw = _make_synthetic_raw()
    _, df_pos = clean_dataset(df_raw)
    assert (df_pos["Quantity"] > 0).all()
    assert (df_pos["UnitPrice"] >= 0).all()
    assert not df_pos["InvoiceNo"].str.startswith("C").any()
