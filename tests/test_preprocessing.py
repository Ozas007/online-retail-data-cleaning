import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.outlier_detection import (
    compute_iqr_bounds,
    detect_outliers_iqr,
    flag_iqr_outliers,
)
from src.preprocessing import (
    ensure_datetime,
    ensure_numeric,
    ensure_string,
    scale_numeric,
)


class TestComputeIqrBounds:
    def test_correct_values(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bounds = compute_iqr_bounds(s, multiplier=1.5, drop_negative=False)
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        assert bounds["Q1"] == q1
        assert bounds["Q3"] == q3
        assert bounds["IQR"] == iqr
        assert bounds["lower_bound"] == q1 - 1.5 * iqr
        assert bounds["upper_bound"] == q3 + 1.5 * iqr


class TestFlagIqrOutliers:
    def test_outlier_flagged(self):
        np.random.seed(42)
        s = np.random.normal(0, 1, 1000).tolist() + [100.0, -999.0]
        df = pd.DataFrame({"x": s})
        df_out, bounds, count, pct = flag_iqr_outliers(
            df, "x", multiplier=1.5, consider_positive_only=False
        )
        assert "x_IsOutlier" in df_out.columns
        assert count >= 1
        assert bool(df_out.loc[1000, "x_IsOutlier"]) is True


class TestDetectOutliersIqr:
    def test_runs_for_multiple_columns(self, tmp_path):
        df = pd.DataFrame({
            "Quantity": [1, 2, 3, 4, 5, 100],
            "UnitPrice": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0],
        })
        result = detect_outliers_iqr(df, columns=["Quantity", "UnitPrice"], output_dir=tmp_path)
        assert "summary" in result
        assert len(result["summary"]) == 2
        assert "Quantity_IsOutlier" in result["dataset_flagged"].columns
        assert "UnitPrice_IsOutlier" in result["dataset_flagged"].columns
        assert (tmp_path / "outlier_summary.csv").exists()

    def test_no_output_when_output_dir_none(self):
        df = pd.DataFrame({
            "Quantity": [1, 2, 3, 4, 5, 100],
            "UnitPrice": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0],
        })
        result = detect_outliers_iqr(df, columns=["Quantity", "UnitPrice"], output_dir=None)
        assert "summary" in result
        assert len(result["summary"]) == 2


class TestEnsureNumeric:
    def test_coercion(self):
        df = pd.DataFrame({"a": ["1", "2", "x", None], "b": [10, 20, 30, 40]})
        result = ensure_numeric(df, ["a", "b"])
        assert pd.api.types.is_numeric_dtype(result["a"])
        assert result["a"].loc[2] is np.nan or pd.isna(result["a"].loc[2])


class TestEnsureDatetime:
    def test_parsing(self):
        df = pd.DataFrame({"dt": ["2010-12-01 08:26:00", "bad-date"]})
        result = ensure_datetime(df, ["dt"])
        assert pd.api.types.is_datetime64_any_dtype(result["dt"])
        assert pd.isna(result["dt"].loc[1])


class TestEnsureString:
    def test_stripping(self):
        df = pd.DataFrame({"s": ["  hello ", None, "world  "]})
        result = ensure_string(df, ["s"])
        assert result["s"].loc[0] == "hello"
        assert result["s"].loc[2] == "world"


class TestScaleNumeric:
    def test_standard_scaling(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = scale_numeric(df, ["x"], method="standard", exclude_ids=False)
        scaled = result["dataframe"]["x_Scaled"]
        assert abs(float(scaled.mean())) < 1e-9
        assert abs(float(scaled.std(ddof=0)) - 1.0) < 1e-3
