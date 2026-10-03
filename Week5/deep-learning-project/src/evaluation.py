import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf

from src.config import NUM_CLASSES

logger = logging.getLogger(__name__)


def evaluate_trained_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_cat: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict:
    logger.info("Evaluating trained model on test set")

    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    logger.info(f"Keras evaluate — test_loss={test_loss:.6f}, test_accuracy={test_accuracy:.6f}")

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.asarray(y_test).astype(int)

    precision_weighted = float(
        precision_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    recall_weighted = float(
        recall_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    f1_weighted = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    per_class_f1: List[float] = f1_score(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    ).tolist()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in range(num_classes)],
        columns=[f"pred_{i}" for i in range(num_classes)],
    )

    results: Dict = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "per_class_f1": per_class_f1,
        "confusion_matrix_df": cm_df,
        "n_samples": int(X_test.shape[0]),
        "num_classes": num_classes,
    }

    logger.info(
        f"Test metrics: acc={test_accuracy:.4f}, P_w={precision_weighted:.4f}, "
        f"R_w={recall_weighted:.4f}, F1_w={f1_weighted:.4f}"
    )
    return results
