import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import (
    compute_total_amount,
    create_positive_sales_dataset,
    extract_datetime_features,
    flag_cancellation_invoices,
    flag_negative_quantities,
    flag_zero_negative_prices,
    remove_exact_duplicates,
    remove_invalid_prices,
)


def _make_synthetic_df() -> pd.DataFrame:
    return pd.DataFrame({
        "InvoiceNo": ["536365", "536365", "C536367", "536368", "536369", "536370"],
        "StockCode": ["85123A", "85123A", "22752", "21730", "22633", "22632"],
        "Description": ["A", "A", "B", "", None, "D"],
        "Quantity": [6, 6, -1, 10, 0, 5],
        "InvoiceDate": [
            "2010-12-01 08:26:00",
            "2010-12-01 08:26:00",
            "2010-12-01 09:41:00",
            "2010-12-01 10:00:00",
            "2010-12-02 11:00:00",
            "2010-12-03 12:30:00",
        ],
        "UnitPrice": [2.55, 2.55, 5.95, 0.0, -1.0, 7.95],
        "CustomerID": [17850.0, 17850.0, np.nan, 13047.0, 13047.0, 12583.0],
        "Country": ["UK", "UK", "UK", "UK", "France", "UK"],
    })


class TestRemoveExactDuplicates:
    def test_removes_duplicates(self):
        df = _make_synthetic_df()
        deduped, removed = remove_exact_duplicates(df)
        assert removed == 1
        assert len(deduped) == len(df) - 1

    def test_no_duplicates_unchanged(self):
        df = _make_synthetic_df().drop_duplicates()
        deduped, removed = remove_exact_duplicates(df)
        assert removed == 0
        assert len(deduped) == len(df)


class TestFlagCancellationInvoices:
    def test_flag_column_added(self):
        df = _make_synthetic_df()
        result = flag_cancellation_invoices(df)
        assert "IsCancellation" in result.columns
        assert int(result["IsCancellation"].sum()) == 1

    def test_original_not_modified(self):
        df = _make_synthetic_df()
        original_cols = list(df.columns)
        flag_cancellation_invoices(df)
        assert list(df.columns) == original_cols


class TestFlagNegativeQuantities:
    def test_flags(self):
        df = _make_synthetic_df()
        result = flag_negative_quantities(df)
        assert "IsNegativeQuantity" in result.columns
        assert "IsZeroQuantity" in result.columns
        assert int(result["IsNegativeQuantity"].sum()) == 1
        assert int(result["IsZeroQuantity"].sum()) == 1


class TestFlagZeroNegativePrices:
    def test_flags(self):
        df = _make_synthetic_df()
        result = flag_zero_negative_prices(df)
        assert "IsNegativePrice" in result.columns
        assert "IsZeroPrice" in result.columns
        assert int(result["IsNegativePrice"].sum()) == 1
        assert int(result["IsZeroPrice"].sum()) == 1


class TestRemoveInvalidPrices:
    def test_removes_non_positive(self):
        df = _make_synthetic_df()
        cleaned, removed = remove_invalid_prices(df)
        assert removed == 2
        assert len(cleaned) == len(df) - 2
        assert (pd.to_numeric(cleaned["UnitPrice"]) > 0).all()


class TestComputeTotalAmount:
    def test_calculation(self):
        df = _make_synthetic_df()
        result = compute_total_amount(df)
        assert "TotalAmount" in result.columns
        expected = pd.to_numeric(df["Quantity"]) * pd.to_numeric(df["UnitPrice"])
        pd.testing.assert_series_equal(
            result["TotalAmount"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


class TestExtractDatetimeFeatures:
    def test_extracts_fields(self):
        df = _make_synthetic_df()
        result = extract_datetime_features(df)
        for col in ["Year", "Month", "Day", "Hour", "DayOfWeek", "YearMonth"]:
            assert col in result.columns
        assert int(result.loc[0, "Year"]) == 2010
        assert int(result.loc[0, "Month"]) == 12
        assert int(result.loc[0, "Day"]) == 1


class TestCreatePositiveSalesDataset:
    def test_excludes_cancellations_and_non_positive_qty(self):
        df = _make_synthetic_df()
        df_flagged = flag_cancellation_invoices(df)
        pos, removed = create_positive_sales_dataset(df_flagged)
        assert (pos["Quantity"] > 0).all()
        assert not pos["IsCancellation"].any()
        assert len(pos) == 4
