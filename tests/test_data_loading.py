import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import (
    MissingColumnsError,
    validate_required_columns,
    validate_file_exists,
)


class TestValidateRequiredColumns:
    def test_all_columns_present(self):
        expected = ["InvoiceNo", "StockCode", "Description", "Quantity"]
        df = pd.DataFrame(columns=expected)
        ok, extra = validate_required_columns(df, expected_columns=expected)
        assert ok is True
        assert extra == []

    def test_missing_columns_raises(self):
        expected = ["InvoiceNo", "StockCode", "Description", "Quantity"]
        df = pd.DataFrame(columns=["InvoiceNo", "StockCode"])
        with pytest.raises(MissingColumnsError):
            validate_required_columns(df, expected_columns=expected)

    def test_extra_columns_reported(self):
        expected = ["InvoiceNo", "StockCode"]
        df = pd.DataFrame(columns=["InvoiceNo", "StockCode", "Description"])
        ok, extra = validate_required_columns(df, expected_columns=expected)
        assert ok is True
        assert "Description" in extra


class TestValidateFileExists:
    def test_file_not_found_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.xlsx"
        with pytest.raises(Exception):
            validate_file_exists(missing)

    def test_existing_file_passes(self, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"")
        result = validate_file_exists(f)
        assert result is True
