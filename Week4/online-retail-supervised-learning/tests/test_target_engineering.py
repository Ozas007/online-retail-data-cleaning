import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.target_engineering import (
    build_target,
    derive_dates_from_df,
)


def _synthetic_transactions_df() -> pd.DataFrame:
    dates_hist = pd.date_range("2011-01-01", "2011-08-31", freq="5D")
    dates_future = pd.date_range("2011-09-09", "2011-12-09", freq="12D")

    rows = []
    cust_purchases_future = {
        1001: True,
        1002: True,
        1003: False,
        1004: False,
    }

    for cid, will_return in cust_purchases_future.items():
        for i, d in enumerate(dates_hist[: min(len(dates_hist), 4 + (cid % 2) * 2)]):
            rows.append({
                "CustomerID": float(cid),
                "InvoiceNo": f"H{cid}-{i}",
                "InvoiceDate": d,
                "StockCode": f"S{i}",
                "Description": f"Product {i}",
                "Quantity": 1 + i,
                "UnitPrice": 10.0 + i,
                "Country": "United Kingdom",
            })
        if will_return:
            for i, d in enumerate(dates_future[:3]):
                rows.append({
                    "CustomerID": float(cid),
                    "InvoiceNo": f"F{cid}-{i}",
                    "InvoiceDate": d,
                    "StockCode": f"SF{i}",
                    "Description": f"Future {i}",
                    "Quantity": 1 + i,
                    "UnitPrice": 10.0 + i,
                    "Country": "United Kingdom",
                })

    rows.append({
        "CustomerID": np.nan, "InvoiceNo": "ANON1",
        "InvoiceDate": dates_hist[0], "StockCode": "S1",
        "Description": "x", "Quantity": 1, "UnitPrice": 5.0,
        "Country": "United Kingdom",
    })

    return pd.DataFrame(rows)


class TestDeriveDatesFromDf:
    def test_temporal_dates_range(self):
        df = _synthetic_transactions_df()
        dates = derive_dates_from_df(df, split_quantile=0.75)
        assert "HISTORICAL_START" in dates
        assert "HISTORICAL_END" in dates
        assert "FUTURE_START" in dates
        assert "FUTURE_END" in dates
        assert dates["HISTORICAL_START"] < dates["HISTORICAL_END"]
        assert dates["FUTURE_START"] == dates["HISTORICAL_END"]
        assert dates["FUTURE_START"] <= dates["FUTURE_END"]
        assert dates["days_historical"] > 0
        assert dates["days_future"] > 0

    def test_dates_actual_vs_hardcoded(self):
        df = _synthetic_transactions_df()
        dates = derive_dates_from_df(df, split_quantile=0.75)
        dt = pd.to_datetime(df["InvoiceDate"])
        assert dates["HISTORICAL_START"] == pd.Timestamp(dt.min())
        assert dates["FUTURE_END"] == pd.Timestamp(dt.max())


class TestBuildTarget:
    def test_target_binary(self):
        df = _synthetic_transactions_df()
        dates = derive_dates_from_df(df, split_quantile=0.75)
        tgt = build_target(
            df,
            historical_end=dates["HISTORICAL_END"],
            future_start=dates["FUTURE_START"],
            future_end=dates["FUTURE_END"],
        )
        assert set(tgt["FuturePurchased"].unique()) <= {0, 1}
        assert tgt["FuturePurchased"].dtype == int

    def test_build_target_no_leakage(self):
        df = _synthetic_transactions_df()
        dates = derive_dates_from_df(df, split_quantile=0.75)
        cutoff = pd.Timestamp(dates["HISTORICAL_END"])
        tgt = build_target(
            df,
            historical_end=dates["HISTORICAL_END"],
            future_start=dates["FUTURE_START"],
            future_end=dates["FUTURE_END"],
        )

        feature_customer_ids = set(tgt["CustomerID"])
        df_dated = df.copy()
        df_dated["_dt"] = pd.to_datetime(df_dated["InvoiceDate"], errors="coerce")
        df_dated["CustomerID"] = pd.to_numeric(df_dated["CustomerID"], errors="coerce")

        for cid in feature_customer_ids:
            c_rows = df_dated[(df_dated["CustomerID"] == cid) & (df_dated["_dt"].notna())]
            feature_rows = c_rows[c_rows["_dt"] < cutoff]
            assert len(feature_rows) >= 1, (
                f"Customer {cid} in target must have at least one transaction "
                f"strictly before cutoff {cutoff}"
            )
            max_feature_date = feature_rows["_dt"].max()
            assert max_feature_date < cutoff

        future_used = df_dated[
            (df_dated["CustomerID"].isin(feature_customer_ids))
            & (df_dated["_dt"] >= dates["FUTURE_START"])
            & (df_dated["_dt"] <= dates["FUTURE_END"])
        ]
        if len(future_used) > 0:
            assert future_used["_dt"].min() >= dates["FUTURE_START"]

    def test_target_customer_set_equals_historical_customers(self):
        df = _synthetic_transactions_df()
        dates = derive_dates_from_df(df, split_quantile=0.75)
        tgt = build_target(
            df,
            historical_end=dates["HISTORICAL_END"],
            future_start=dates["FUTURE_START"],
            future_end=dates["FUTURE_END"],
        )
        dt = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        cids_hist = set(
            pd.to_numeric(df.loc[(dt < dates["HISTORICAL_END"]) & (df["CustomerID"].notna()), "CustomerID"])
        )
        assert set(tgt["CustomerID"]) == cids_hist

    def test_excludes_anonymous_customers(self):
        df = _synthetic_transactions_df()
        dates = derive_dates_from_df(df, split_quantile=0.75)
        tgt = build_target(
            df,
            historical_end=dates["HISTORICAL_END"],
            future_start=dates["FUTURE_START"],
            future_end=dates["FUTURE_END"],
        )
        assert not tgt["CustomerID"].isna().any()
