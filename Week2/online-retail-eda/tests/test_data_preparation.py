import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import (
    compute_total_amount, create_temporal_features,
    flag_cancellations, create_analytical_datasets, prepare_all,
)


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "InvoiceNo": ["536365", "536365", "C536367", "536368", "536369"],
        "StockCode": ["85123A", "71053", "22752", "84406B", "22752"],
        "Description": ["A", "B", "C", "D", "E"],
        "Quantity": [6, 6, -1, 10, 5],
        "InvoiceDate": pd.to_datetime([
            "2010-12-01 08:26:00", "2010-12-01 08:26:00",
            "2010-12-01 09:41:00", "2010-12-02 10:00:00",
            "2011-01-05 14:30:00",
        ]),
        "UnitPrice": [2.55, 3.39, 5.95, 0.0, 7.95],
        "CustomerID": [17850.0, 17850.0, np.nan, 13047.0, 12583.0],
        "Country": ["UK", "UK", "UK", "UK", "France"],
    })


class TestComputeTotalAmount:
    def test_calculation(self):
        df = _df()
        result = compute_total_amount(df)
        assert "TotalAmount" in result.columns
        expected = pd.to_numeric(df["Quantity"]) * pd.to_numeric(df["UnitPrice"])
        pd.testing.assert_series_equal(
            result["TotalAmount"].reset_index(drop=True),
            expected.reset_index(drop=True), check_names=False,
        )


class TestCreateTemporalFeatures:
    def test_extracts_fields(self):
        df = _df()
        result = create_temporal_features(df)
        for col in ["Year", "Month", "Day", "Hour", "DayOfWeek", "DayOfWeekName", "YearMonth", "Date"]:
            assert col in result.columns, f"Missing: {col}"
        assert int(result.loc[0, "Year"]) == 2010
        assert int(result.loc[0, "Month"]) == 12
        assert int(result.loc[0, "Day"]) == 1
        assert int(result.loc[0, "Hour"]) == 8


class TestFlagCancellations:
    def test_flag_column(self):
        df = _df()
        result = flag_cancellations(df)
        assert "IsCancellation" in result.columns
        assert int(result["IsCancellation"].sum()) == 1


class TestCreateAnalyticalDatasets:
    def test_three_datasets_returned(self):
        df = flag_cancellations(create_temporal_features(compute_total_amount(_df())))
        ds = create_analytical_datasets(df)
        for k in ["full", "valid_price", "positive_sales"]:
            assert k in ds
        assert len(ds["valid_price"]) <= len(ds["full"])
        assert len(ds["positive_sales"]) <= len(ds["valid_price"])
        assert (pd.to_numeric(ds["positive_sales"]["Quantity"]) > 0).all()
        assert (pd.to_numeric(ds["positive_sales"]["UnitPrice"]) > 0).all()


class TestPrepareAll:
    def test_integration(self):
        result = prepare_all(_df())
        for k in ["full", "valid_price", "positive_sales", "engineered"]:
            assert k in result
        assert "TotalAmount" in result["engineered"].columns
        assert "YearMonth" in result["engineered"].columns
