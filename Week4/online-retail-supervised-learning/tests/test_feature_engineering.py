import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLUMNS, ID_COLUMN
from src.feature_engineering import build_features


def _synthetic_historical_df():
    rows = []
    for cid in [1001.0, 1002.0, 1003.0]:
        for i, iso in enumerate(["2011-03-01", "2011-05-15", "2011-07-20"]):
            rows.append({
                ID_COLUMN: cid,
                "InvoiceNo": f"INV{int(cid)}-{i}",
                "InvoiceDate": pd.Timestamp(iso),
                "StockCode": f"P{i}-{int(cid)}",
                "Quantity": 2 + i,
                "UnitPrice": 10.0 + i,
                "Country": "United Kingdom",
            })
    return pd.DataFrame(rows)


def _synthetic_with_post_cutoff_leaks():
    df = _synthetic_historical_df()
    extra = pd.DataFrame([
        {
            ID_COLUMN: 1001.0,
            "InvoiceNo": "LEAK-1",
            "InvoiceDate": pd.Timestamp("2011-09-15"),
            "StockCode": "LEAK",
            "Quantity": 99,
            "UnitPrice": 999.0,
            "Country": "United Kingdom",
        },
        {
            ID_COLUMN: 1002.0,
            "InvoiceNo": "LEAK-2",
            "InvoiceDate": pd.Timestamp("2011-09-01"),
            "StockCode": "CUTOFF-EDGE",
            "Quantity": 50,
            "UnitPrice": 50.0,
            "Country": "France",
        },
    ])
    return pd.concat([df, extra], ignore_index=True)


class TestBuildFeatureShapes:
    def test_feature_shapes_one_row_per_customer(self):
        df = _synthetic_historical_df()
        hist_end = pd.Timestamp("2011-09-09")
        features = build_features(df, historical_end=hist_end)
        expected_ids = set(df[ID_COLUMN].dropna().unique())
        assert len(features) == len(expected_ids)
        assert ID_COLUMN in features.columns
        for col in FEATURE_COLUMNS:
            assert col in features.columns

    def test_no_null_features(self):
        df = _synthetic_historical_df()
        hist_end = pd.Timestamp("2011-09-09")
        features = build_features(df, historical_end=hist_end)
        numeric_cols = [c for c in FEATURE_COLUMNS if c != "Country"]
        for col in numeric_cols:
            assert features[col].notna().all(), f"Null values found in feature column {col}"


class TestNoFutureLeakage:
    def test_no_future_leakage_on_synthetic(self):
        df_leaky = _synthetic_with_post_cutoff_leaks()
        hist_end = pd.Timestamp("2011-09-09")

        features_leaked_ignored_suppressed = build_features(df_leaky, historical_end=hist_end)

        df_hist_only = df_leaky.copy()
        df_hist_only["_dt"] = pd.to_datetime(df_hist_only["InvoiceDate"], errors="coerce")
        df_hist_only = df_hist_only[df_hist_only["_dt"] < hist_end].drop(columns="_dt")
        features_hist_only = build_features(df_hist_only, historical_end=hist_end)

        features_leaked_ignored_suppressed = features_leaked_ignored_suppressed.sort_values(ID_COLUMN).reset_index(drop=True)
        features_hist_only = features_hist_only.sort_values(ID_COLUMN).reset_index(drop=True)

        numeric_cols = [c for c in FEATURE_COLUMNS if c != "Country"]
        pd.testing.assert_frame_equal(
            features_leaked_ignored_suppressed[[ID_COLUMN] + numeric_cols],
            features_hist_only[[ID_COLUMN] + numeric_cols],
            check_names=True,
            rtol=1e-9,
        )

        c1001 = features_leaked_ignored_suppressed[features_leaked_ignored_suppressed[ID_COLUMN] == 1001.0].iloc[0]
        assert c1001["TotalQuantity"] < 50, (
            "Recency should be derived strictly pre-cutoff; post-cutoff 99-Qty row must be excluded"
        )

    def test_recency_strictly_geq_zero(self):
        df = _synthetic_historical_df()
        hist_end = pd.Timestamp("2011-09-09")
        features = build_features(df, historical_end=hist_end)
        assert (features["Recency"] >= 0).all()
        last_purchase = pd.Timestamp("2011-07-20")
        expected_recency_max = (hist_end - last_purchase).days
        assert features["Recency"].max() <= expected_recency_max
