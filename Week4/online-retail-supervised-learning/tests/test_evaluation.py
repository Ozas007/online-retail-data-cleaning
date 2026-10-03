import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    build_confusion_matrix_df,
    evaluate_classifier,
    run_cross_validation,
)
from src.models import create_logistic_regression
from src.preprocessing import build_full_pipeline, get_preprocessor


class TestEvaluateClassifierKeys:
    def test_evaluate_classifier_keys(self):
        rng = np.random.default_rng(0)
        y_true = rng.choice([0, 1], 100, p=[0.6, 0.4])
        y_pred = y_true.copy()
        idx = rng.choice(len(y_true), 15, replace=False)
        y_pred[idx] = 1 - y_pred[idx]
        y_proba = np.where(y_pred == 1, 0.85, 0.15)

        metrics = evaluate_classifier(y_true, y_pred, y_proba)
        required = ["accuracy", "precision", "recall", "f1",
                    "roc_auc", "support_positive", "support_negative"]
        for key in required:
            assert key in metrics, f"Missing key: {key}"
            if key == "roc_auc":
                assert isinstance(metrics[key], float)
            elif key.startswith("support"):
                assert isinstance(metrics[key], (int, np.integer))
            else:
                assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} outside [0,1]"

    def test_roc_auc_sanity_perfect(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = y_true.copy()
        y_proba = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
        m = evaluate_classifier(y_true, y_pred, y_proba)
        assert m["accuracy"] == 1.0
        assert m["roc_auc"] == pytest.approx(1.0)


class TestConfusionMatrix:
    def test_confusion_matrix_labels(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        cm_df = build_confusion_matrix_df(y_true, y_pred)
        assert cm_df.shape == (2, 2)


class TestRunCrossValidation:
    def test_cv_output_shape(self):
        rng = np.random.default_rng(42)
        n = 80
        X = pd.DataFrame({
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.integers(0, 5, n).astype(float),
        })
        y = ((X["f1"] + X["f2"]) > 0).astype(int).values

        pre = get_preprocessor(numeric_features=["f1", "f2", "f3"], categorical_features=[])
        pipe = build_full_pipeline(create_logistic_regression(random_state=42), pre)
        cv_df = run_cross_validation(X, y, pipe, cv=5, random_state=42)

        assert len(cv_df) == 5 + 1
        assert "fold" in cv_df.columns
        fold_rows = cv_df[cv_df["fold"] != "overall"]
        overall_row = cv_df[cv_df["fold"] == "overall"]
        assert len(fold_rows) == 5
        assert len(overall_row) == 1

        metric_bases = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        ov = overall_row.iloc[0]
        for base in metric_bases:
            assert f"{base}_mean" in ov.index
            assert f"{base}_std" in ov.index
            assert np.isfinite(float(ov[f"{base}_mean"]))
