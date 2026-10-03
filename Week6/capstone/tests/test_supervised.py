import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_supervised_features
from src.preprocessing import clean_dataset
from src.supervised import build_target, derive_dates_from_df, run_supervised_pipeline


def _make_synthetic_balanced(n_cust: int = 200, n_txns_per_cust_hist: int = 20, n_txns_per_cust_fut: int = 10):
    rng = np.random.default_rng(42)
    all_cids = np.arange(12346, 12346 + n_cust)
    n_pos = n_cust // 2
    pos_cids = set(all_cids[:n_pos])

    hist_dates = pd.date_range("2010-12-01", "2011-04-30", freq="6h")
    fut_dates = pd.date_range("2011-07-01", "2011-12-09", freq="12h")

    rows = []
    inv_counter = 0
    for cid in all_cids:
        dates_for_hist = rng.choice(hist_dates, size=n_txns_per_cust_hist, replace=True)
        for d in dates_for_hist:
            rows.append({
                "InvoiceNo": f"INV{inv_counter:06d}",
                "StockCode": f"S{inv_counter%30:03d}",
                "Description": f"P{inv_counter%30}",
                "Quantity": int(rng.integers(1, 12)),
                "InvoiceDate": d,
                "UnitPrice": float(np.round(rng.uniform(1.0, 25.0), 2)),
                "CustomerID": float(cid),
                "Country": str(rng.choice(["United Kingdom", "Germany", "France"])),
            })
            inv_counter += 1
        if cid in pos_cids:
            dates_for_fut = rng.choice(fut_dates, size=n_txns_per_cust_fut, replace=True)
            for d in dates_for_fut:
                rows.append({
                    "InvoiceNo": f"INV{inv_counter:06d}",
                    "StockCode": f"S{inv_counter%30:03d}",
                    "Description": f"P{inv_counter%30}",
                    "Quantity": int(rng.integers(1, 12)),
                    "InvoiceDate": d,
                    "UnitPrice": float(np.round(rng.uniform(1.0, 25.0), 2)),
                    "CustomerID": float(cid),
                    "Country": str(rng.choice(["United Kingdom", "Germany", "France"])),
                })
                inv_counter += 1
    return pd.DataFrame(rows)


def test_derive_dates_from_df_returns_bounds():
    df = _make_synthetic_balanced()
    out = derive_dates_from_df(df)
    assert isinstance(out, dict)
    for k in ("HISTORICAL_START", "HISTORICAL_END",
              "FUTURE_START", "FUTURE_END",
              "days_historical", "days_future"):
        assert k in out, f"Missing key {k}"
    assert out["days_historical"] > 0
    assert out["days_future"] > 0


def test_build_target_returns_customers_and_binary_label():
    df = _make_synthetic_balanced()
    dates = derive_dates_from_df(df)
    target_df = build_target(
        df,
        historical_end=dates["HISTORICAL_END"],
        future_start=dates["FUTURE_START"],
        future_end=dates["FUTURE_END"],
    )
    assert isinstance(target_df, pd.DataFrame)
    assert list(target_df.columns) == ["CustomerID", "FuturePurchased"]
    assert target_df["FuturePurchased"].nunique() <= 2
    assert target_df["FuturePurchased"].isin([0, 1]).all()


def test_build_target_excludes_non_historical_customers():
    future_only_cid = 999999
    base = _make_synthetic_balanced()
    fut = pd.DataFrame({
        "InvoiceNo": ["F001", "F002"],
        "StockCode": ["S00", "S01"],
        "Description": ["PF0", "PF1"],
        "Quantity": [2, 3],
        "InvoiceDate": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "UnitPrice": [1.0, 1.0],
        "CustomerID": [future_only_cid, future_only_cid],
        "Country": ["United Kingdom"] * 2,
    })
    combined = pd.concat([base, fut], ignore_index=True)
    dates = derive_dates_from_df(base)
    target = build_target(
        combined,
        historical_end=dates["HISTORICAL_END"],
        future_start=dates["HISTORICAL_END"],
        future_end=dates["FUTURE_END"],
    )
    assert (target["CustomerID"].astype(int) != future_only_cid).all()


def test_run_supervised_pipeline_synthetic(tmp_path):
    df_raw = _make_synthetic_balanced()
    _, df_pos = clean_dataset(df_raw)
    result = run_supervised_pipeline(df_pos)
    assert isinstance(result, dict)
    assert "best_model_name" in result
    assert result["best_model_name"] in ("Random Forest", "Logistic Regression")
    assert "test_metrics_dict" in result
    m = result["test_metrics_dict"]
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert k in m
        assert 0.0 <= float(m[k]) <= 1.0 or np.isnan(float(m[k]))
    assert "n_train" in result and "n_test" in result
    assert result["n_train"] > 0 and result["n_test"] > 0
