import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import (
    compute_total_amount,
    create_temporal_features,
    flag_cancellations,
    create_analytical_datasets,
    get_positive_sales_with_customer_id,
)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    np.random.seed(42)
    n_rows = 8
    return pd.DataFrame({
        "InvoiceNo": [
            "536365", "536365", "C536367", "536368",
            "536369", "536370", "536371", "536372",
        ],
        "StockCode": ["85123A", "71053", "22752", "84406B", "22752", "21730", "22633", "22632"],
        "Description": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "Quantity": [6, 6, -1, 10, 5, -3, 12, 8],
        "InvoiceDate": pd.to_datetime([
            "2010-12-01 08:26:00", "2010-12-01 08:26:00",
            "2010-12-01 09:41:00", "2010-12-02 10:00:00",
            "2011-01-05 14:30:00", "2011-01-06 11:00:00",
            "2011-01-07 09:15:00", "2011-01-08 16:45:00",
        ]),
        "UnitPrice": [2.55, 3.39, 5.95, 0.0, 7.95, 4.50, 0.0, 6.20],
        "CustomerID": [17850.0, 17850.0, np.nan, 13047.0, 12583.0, np.nan, 15311.0, 14527.0],
        "Country": ["UK", "UK", "UK", "UK", "France", "UK", "UK", "Germany"],
    })


class TestComputeTotalAmount:
    def test_computes_correctly(self, synthetic_df):
        result = compute_total_amount(synthetic_df)
        assert "TotalAmount" in result.columns
        expected = pd.to_numeric(synthetic_df["Quantity"]) * pd.to_numeric(synthetic_df["UnitPrice"])
        pd.testing.assert_series_equal(
            result["TotalAmount"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


class TestCreateTemporalFeatures:
    def test_extracts_temporal_columns(self, synthetic_df):
        result = create_temporal_features(synthetic_df)
        for col in ["Year", "Month", "Day", "Hour", "DOW", "DOWName", "YearMonth", "Date"]:
            assert col in result.columns, f"Missing column: {col}"
        assert int(result.loc[0, "Year"]) == 2010
        assert int(result.loc[0, "Month"]) == 12
        assert int(result.loc[0, "Day"]) == 1
        assert int(result.loc[0, "Hour"]) == 8


class TestFlagCancellations:
    def test_marks_c_prefix(self, synthetic_df):
        result = flag_cancellations(synthetic_df)
        assert "IsCancellation" in result.columns
        assert int(result["IsCancellation"].sum()) == 1
        c_idx = synthetic_df.index[synthetic_df["InvoiceNo"].astype(str).str.startswith("C")][0]
        assert bool(result.loc[c_idx, "IsCancellation"]) is True


class TestCreateAnalyticalDatasets:
    def test_returns_three_datasets_with_decreasing_rows(self, synthetic_df):
        df = compute_total_amount(synthetic_df)
        df = create_temporal_features(df)
        df = flag_cancellations(df)
        ds = create_analytical_datasets(df)
        for k in ["full", "valid_price", "positive_sales"]:
            assert k in ds
        assert len(ds["valid_price"]) <= len(ds["full"])
        assert len(ds["positive_sales"]) <= len(ds["valid_price"])
        assert (pd.to_numeric(ds["positive_sales"]["UnitPrice"]) > 0).all()
        assert (pd.to_numeric(ds["positive_sales"]["Quantity"]) > 0).all()


class TestGetPositiveSalesWithCustomerId:
    def test_drops_null_customer_id_and_returns_dataframe(self, synthetic_df):
        df = compute_total_amount(synthetic_df)
        df = create_temporal_features(df)
        df = flag_cancellations(df)
        ds = create_analytical_datasets(df)
        pos = ds["positive_sales"]
        result = get_positive_sales_with_customer_id(pos)
        assert isinstance(result, pd.DataFrame)
        assert result["CustomerID"].notna().all()
        assert len(result) <= len(pos)
