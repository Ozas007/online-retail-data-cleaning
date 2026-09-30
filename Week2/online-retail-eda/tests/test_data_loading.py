import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import (
    DatasetNotFoundError, MissingColumnsError,
    validate_file_exists, validate_required_columns,
)


class TestValidateFileExists:
    def test_file_not_found_raises(self, tmp_path):
        missing = tmp_path / "nope.xlsx"
        with pytest.raises(DatasetNotFoundError):
            validate_file_exists(missing)

    def test_existing_file_passes(self, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"")
        assert validate_file_exists(f) is True


class TestValidateRequiredColumns:
    def test_all_columns_present(self):
        expected = ["InvoiceNo", "StockCode", "Quantity"]
        df = pd.DataFrame(columns=expected)
        ok, extra = validate_required_columns(df, expected_columns=expected)
        assert ok is True
        assert extra == []

    def test_missing_columns_raises(self):
        expected = ["InvoiceNo", "StockCode"]
        df = pd.DataFrame(columns=["InvoiceNo"])
        with pytest.raises(MissingColumnsError):
            validate_required_columns(df, expected_columns=expected)

    def test_extra_columns_reported(self):
        expected = ["InvoiceNo", "StockCode"]
        df = pd.DataFrame(columns=["InvoiceNo", "StockCode", "Description"])
        ok, extra = validate_required_columns(df, expected_columns=expected)
        assert ok is True
        assert "Description" in extra
