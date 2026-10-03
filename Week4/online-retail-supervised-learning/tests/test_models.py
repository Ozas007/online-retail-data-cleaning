import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    create_decision_tree,
    create_logistic_regression,
    create_random_forest,
)
from src.preprocessing import build_full_pipeline, get_preprocessor


def _tiny_labeled_df(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Recency": rng.integers(1, 300, n).astype(float),
        "Frequency": rng.integers(1, 15, n).astype(float),
        "Monetary": rng.uniform(10, 5000, n),
        "Country": rng.choice(["UK", "FR"], n),
    })
    y = ((X["Frequency"] > 7) & (X["Recency"] < 150)).astype(int).values
    return X, y


class TestCreateAllModels:
    def test_create_all_models_fit_predict_on_tiny_df(self):
        X, y = _tiny_labeled_df(n=60)
        X_train, y_train = X.iloc[:40].copy(), y[:40]
        X_test, y_test = X.iloc[40:].copy(), y[40:]

        numeric = ["Recency", "Frequency", "Monetary"]
        categorical = ["Country"]

        factories = [
            ("LR", lambda: create_logistic_regression(random_state=42)),
            ("DT", lambda: create_decision_tree(random_state=42)),
            ("RF", lambda: create_random_forest(random_state=42, n_estimators=10)),
        ]

        for name, factory in factories:
            pre = get_preprocessor(numeric_features=numeric, categorical_features=categorical)
            pipe = build_full_pipeline(factory(), pre)
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)
            assert preds.shape == (len(y_test),), f"{name}: predict shape mismatch"
            assert proba.shape == (len(y_test), 2), f"{name}: proba shape mismatch"
            assert set(preds) <= {0, 1}, f"{name}: preds contain non-binary values"
            acc = (preds == y_test).mean()
            assert acc >= 0.3, f"{name}: accuracy impossibly low — likely a logic error"

    def test_random_state_reproducibility(self):
        X, y = _tiny_labeled_df(n=80, seed=1)
        numeric = ["Recency", "Frequency", "Monetary"]
        categorical = ["Country"]

        def _train_and_predict(seed: int):
            results = {}
            for name, factory in [
                ("LR", lambda s=seed: create_logistic_regression(random_state=s)),
                ("DT", lambda s=seed: create_decision_tree(random_state=s)),
                ("RF", lambda s=seed: create_random_forest(random_state=s, n_estimators=20)),
            ]:
                pre = get_preprocessor(numeric_features=numeric, categorical_features=categorical)
                pipe = build_full_pipeline(factory(), pre)
                pipe.fit(X, y)
                results[name] = (pipe.predict(X), pipe.predict_proba(X))
            return results

        r1 = _train_and_predict(42)
        r2 = _train_and_predict(42)
        for name in r1:
            np.testing.assert_array_equal(r1[name][0], r2[name][0],
                err_msg=f"{name}: predictions differ with same random_state=42")
            np.testing.assert_allclose(r1[name][1], r2[name][1], rtol=1e-10,
                err_msg=f"{name}: probabilities differ with same random_state=42")
