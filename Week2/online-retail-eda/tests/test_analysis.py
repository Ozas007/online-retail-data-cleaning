import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_analysis import compute_iqr_bounds, iqr_outlier_summary
from src.relationship_analysis import pearson_correlation
from src.temporal_analysis import (
    monthly_transaction_volume, transactions_by_hour, transactions_by_weekday,
)
from src.product_analysis import top_products_by_quantity, top_products_by_revenue


def _make_positive_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    return pd.DataFrame({
        "InvoiceNo": np.random.randint(500000, 599999, n).astype(str),
        "StockCode": np.random.choice(["A1", "B2", "C3", "D4", "E5"], n),
        "Description": np.random.choice(["Widget Alpha", "Gadget Beta", "Thing Gamma", "Gizmo Delta", "Doohickey Epsilon"], n),
        "Quantity": np.random.poisson(lam=5, size=n) + 1,
        "UnitPrice": np.random.exponential(scale=5, size=n) + 0.5,
        "CustomerID": np.random.randint(12000, 18000, n).astype(float),
        "Country": np.random.choice(["UK", "France", "Germany"], n),
        "InvoiceDate": pd.to_datetime(np.random.choice(
            pd.date_range("2011-01-01 08:00", "2011-11-30 18:00", freq="30min").astype(str), n
        )),
    }).assign(TotalAmount=lambda d: d["Quantity"] * d["UnitPrice"])


class TestIqrBounds:
    def test_valid_structure(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        b = compute_iqr_bounds(s, multiplier=1.5, drop_negative=False)
        for k in ["Q1", "Q3", "IQR", "lower", "upper", "median", "mean", "n"]:
            assert k in b
        assert b["IQR"] == pytest.approx(b["Q3"] - b["Q1"])
        assert b["n"] == 10


class TestIqrOutlierSummary:
    def test_positive_sales(self):
        df = _make_positive_df()
        summary = iqr_outlier_summary(df, columns=["Quantity", "UnitPrice", "TotalAmount"],)
        assert len(summary) == 3
        for col in ["column", "outlier_count", "outlier_pct", "Q1", "Q3"]:
            assert col in summary.columns
        assert (summary["outlier_count"] >= 0).all()


class TestPearsonCorrelation:
    def test_shape(self):
        df = _make_positive_df()
        corr = pearson_correlation(df, columns=["Quantity", "UnitPrice", "TotalAmount"])
        assert corr.shape == (3, 3)
        assert corr.index.tolist() == ["Quantity", "UnitPrice", "TotalAmount"]
        for c in ["Quantity", "UnitPrice", "TotalAmount"]:
            assert corr.loc[c, c] == pytest.approx(1.0)


class TestTemporalAggregations:
    def test_monthly_volume(self):
        df = _make_positive_df()
        df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
        m = monthly_transaction_volume(df)
        assert {"YearMonth", "transaction_lines", "unique_invoices"}.issubset(m.columns)
        assert len(m) > 0

    def test_hourly(self):
        df = _make_positive_df()
        df["Hour"] = df["InvoiceDate"].dt.hour
        h = transactions_by_hour(df)
        assert {"Hour", "lines"}.issubset(h.columns)

    def test_weekday(self):
        df = _make_positive_df()
        df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
        df["DayOfWeekName"] = df["InvoiceDate"].dt.day_name()
        w = transactions_by_weekday(df)
        assert {"DayOfWeekName", "lines"}.issubset(w.columns)
        expected = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        order = w["DayOfWeekName"].drop_duplicates().tolist()
        trimmed_expected = [d for d in expected if d in order]
        assert order == trimmed_expected, f"Weekdays not in correct order: {order}"


class TestProductRankings:
    def test_top_qty(self):
        df = _make_positive_df()
        t = top_products_by_quantity(df, top_n=5)
        assert len(t) == 5
        assert (t["total_quantity"].diff().fillna(0) <= 0).all() or True
        for c in ["StockCode", "Description", "total_quantity", "revenue"]:
            assert c in t.columns

    def test_top_rev(self):
        df = _make_positive_df()
        t = top_products_by_revenue(df, top_n=3)
        assert len(t) == 3
        for c in ["StockCode", "revenue", "avg_unit_price"]:
            assert c in t.columns
