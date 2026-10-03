import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import build_full_pipeline, get_preprocessor
from src.models import create_logistic_regression


def _synthetic_feature_df(n_rows: int = 50):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Recency": rng.integers(1, 300, n_rows).astype(float),
        "Frequency": rng.integers(1, 20, n_rows).astype(float),
        "Monetary": rng.uniform(10, 5000, n_rows),
        "AOV": rng.uniform(5, 500, n_rows),
        "UniqueProducts": rng.integers(1, 100, n_rows).astype(float),
        "AvgQuantity": rng.uniform(1, 30, n_rows),
        "TotalQuantity": rng.integers(1, 1000, n_rows).astype(float),
        "PurchaseFrequency": rng.uniform(0.001, 0.1, n_rows),
        "Country": rng.choice(["United Kingdom", "France", "Germany"], n_rows),
    })


class TestGetPreprocessor:
    def test_preprocessor_output_shape(self):
        numeric = ["Recency", "Frequency", "Monetary", "AOV",
                   "UniqueProducts", "AvgQuantity", "TotalQuantity",
                   "PurchaseFrequency"]
        categorical = ["Country"]
        pre = get_preprocessor(numeric_features=numeric, categorical_features=categorical)
        assert isinstance(pre, ColumnTransformer)

        df = _synthetic_feature_df(n_rows=20)
        X_out = pre.fit_transform(df)
        assert X_out.shape[0] == 20
        n_countries = df["Country"].nunique()
        assert X_out.shape[1] == len(numeric) + n_countries
        assert np.isfinite(X_out).all()

    def test_preprocessor_without_categorical(self):
        numeric = ["Recency", "Frequency", "Monetary"]
        pre = get_preprocessor(numeric_features=numeric, categorical_features=[])
        df = _synthetic_feature_df(n_rows=10)
        X_out = pre.fit_transform(df)
        assert X_out.shape == (10, 3)


class TestBuildFullPipeline:
    def test_pipeline_fit_predict_shape(self):
        numeric = ["Recency", "Frequency", "Monetary", "AOV",
                   "UniqueProducts", "AvgQuantity", "TotalQuantity",
                   "PurchaseFrequency"]
        categorical = ["Country"]
        pre = get_preprocessor(numeric_features=numeric, categorical_features=categorical)
        clf = create_logistic_regression(random_state=42)
        pipe = build_full_pipeline(clf, pre)

        df_train = _synthetic_feature_df(n_rows=40)
        y_train = np.random.default_rng(0).choice([0, 1], 40, p=[0.5, 0.5])
        df_test = _synthetic_feature_df(n_rows=15)

        pipe.fit(df_train, y_train)
        y_pred = pipe.predict(df_test)
        y_proba = pipe.predict_proba(df_test)
        assert y_pred.shape == (15,)
        assert set(y_pred) <= {0, 1}
        assert y_proba.shape == (15, 2)
        np.testing.assert_allclose(y_proba.sum(axis=1), 1.0, rtol=1e-6)
