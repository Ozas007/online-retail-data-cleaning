import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def build_dl_section_summary(
    week5_info: Optional[Dict[str, Any]] = None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[np.ndarray] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    logger.info("Building honest Deep Learning section summary")
    np.random.seed(RANDOM_SEED)

    summary: Dict[str, Any] = {
        "week5_dataset": "MNIST",
        "week5_dataset_description": (
            "Week 5 used the MNIST handwritten-digit dataset (LeCun et al.), a separate "
            "public dataset used strictly for neural-network technique demonstration. "
            "MNIST contains 28x28 grayscale images of digits 0-9 and is unrelated to "
            "UCI online retail transactions. MNIST results are not reported as retail results."
        ),
        "retail_tabular_approach": (
            "For the UCI retail tabular data, we additionally train a lightweight "
            "sklearn MLPClassifier as a deep-learning-style baseline on the supervised "
            "tabular features: 2 hidden layers of size (64, 32), relu activation, "
            "early_stopping=True, max_iter=200, random_state=42. This provides an "
            "honest comparison point versus Random Forest without needing TensorFlow."
        ),
        "mlp_trained": False,
        "mlp_train_metrics": {},
        "mlp_test_metrics": {},
    }

    can_train = (
        X_train is not None and y_train is not None
        and X_test is not None and y_test is not None
    )

    if can_train:
        try:
            from sklearn.neural_network import MLPClassifier
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            logger.info("Training tabular MLP baseline for honest comparison")

            numeric_in = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
            categorical_in = [c for c in X_train.columns if c not in numeric_in]

            transformers = []
            if numeric_in:
                transformers.append(("num", StandardScaler(), numeric_in))
            if categorical_in:
                transformers.append((
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_in,
                ))
            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop") if transformers else None

            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                early_stopping=True,
                max_iter=200,
                random_state=RANDOM_SEED,
            )

            if preprocessor is not None:
                pipe = Pipeline([("pre", preprocessor), ("mlp", mlp)])
            else:
                pipe = Pipeline([("mlp", mlp)])

            pipe.fit(X_train, y_train)

            y_train_pred = pipe.predict(X_train)
            y_train_proba = pipe.predict_proba(X_train)
            y_test_pred = pipe.predict(X_test)
            y_test_proba = pipe.predict_proba(X_test)

            def _metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
                yt = np.asarray(y_true).astype(int)
                yp = np.asarray(y_pred).astype(int)
                proba_use = y_proba[:, 1] if np.asarray(y_proba).ndim == 2 else np.asarray(y_proba)
                out = {
                    "accuracy": float(accuracy_score(yt, yp)),
                    "precision": float(precision_score(yt, yp, average="macro", zero_division=0)),
                    "recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
                    "f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
                }
                try:
                    out["roc_auc"] = float(roc_auc_score(yt, proba_use))
                except Exception:
                    out["roc_auc"] = float("nan")
                return out

            summary["mlp_train_metrics"] = _metrics(y_train, y_train_pred, y_train_proba)
            summary["mlp_test_metrics"] = _metrics(y_test, y_test_pred, y_test_proba)
            summary["mlp_trained"] = True
            summary["mlp_hidden_layers"] = (64, 32)
            summary["mlp_params"] = {
                "activation": "relu",
                "early_stopping": True,
                "max_iter": 200,
                "random_state": RANDOM_SEED,
            }
            logger.info(
                f"  MLP test metrics: acc={summary['mlp_test_metrics'].get('accuracy', float('nan')):.3f}  "
                f"f1={summary['mlp_test_metrics'].get('f1', float('nan')):.3f}  "
                f"roc_auc={summary['mlp_test_metrics'].get('roc_auc', float('nan')):.3f}"
            )

        except Exception as e:
            logger.warning(f"Tabular MLP baseline could not be trained: {e}")
            summary["mlp_train_error"] = str(e)

    if week5_info is not None:
        summary["week5_extra"] = week5_info

    return summary
