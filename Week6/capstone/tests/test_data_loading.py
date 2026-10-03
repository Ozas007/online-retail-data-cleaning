from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import EXPECTED_COLUMNS
from src.data_loading import (
    get_dataset_shape,
    load_excel_dataset,
    validate_file_exists,
    validate_required_columns,
)


def _make_synthetic_df(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    countries = ["United Kingdom", "Germany", "France", "Spain"]
    return pd.DataFrame({
        "InvoiceNo": [f"{i+100000}" for i in range(n_rows)],
        "StockCode": [f"STK{i%50:04d}" for i in range(n_rows)],
        "Description": [f"Product {i%50}" for i in range(n_rows)],
        "Quantity": rng.integers(1, 25, size=n_rows),
        "InvoiceDate": pd.date_range("2011-01-01", periods=n_rows, freq="30min"),
        "UnitPrice": np.round(rng.uniform(0.5, 25.0, size=n_rows), 2),
        "CustomerID": rng.integers(12346, 12400, size=n_rows).astype(np.float64),
        "Country": rng.choice(countries, size=n_rows),
    })


def test_validate_file_exists(tmp_path: Path):
    from src.data_loading import DatasetNotFoundError
    p = tmp_path / "missing.xlsx"
    with pytest.raises(DatasetNotFoundError):
        validate_file_exists(p)
    p.write_bytes(b"PK\x03\x04fake_xlsx")
    assert validate_file_exists(p) is True


def test_validate_required_columns_ok():
    df = _make_synthetic_df()
    ok, extra = validate_required_columns(df, EXPECTED_COLUMNS)
    assert ok is True
    assert isinstance(extra, list)


def test_validate_required_columns_missing():
    from src.data_loading import MissingColumnsError
    df = _make_synthetic_df().drop(columns=["CustomerID"])
    with pytest.raises(MissingColumnsError):
        validate_required_columns(df, EXPECTED_COLUMNS)


def test_get_dataset_shape():
    df = _make_synthetic_df(n_rows=120)
    rows, cols = get_dataset_shape(df)
    assert rows == 120
    assert cols == len(EXPECTED_COLUMNS)


def test_load_excel_dataset_real_file():
    from src.config import RAW_DATA_PATH
    if RAW_DATA_PATH.exists():
        df = load_excel_dataset(RAW_DATA_PATH)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 1000
        ok, _ = validate_required_columns(df, EXPECTED_COLUMNS)
        assert ok is True
    else:
        pytest.skip("Real dataset not available")
