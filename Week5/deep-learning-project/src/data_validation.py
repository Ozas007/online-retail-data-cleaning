import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.config import INPUT_SHAPE, NUM_CLASSES

logger = logging.getLogger(__name__)


def validate_data_shapes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_shape: tuple = INPUT_SHAPE,
) -> bool:
    logger.info("Validating data shapes, types, and label ranges")

    expected_H, expected_W = input_shape

    assert isinstance(X_train, np.ndarray), f"X_train must be np.ndarray, got {type(X_train)}"
    assert isinstance(y_train, np.ndarray), f"y_train must be np.ndarray, got {type(y_train)}"
    assert isinstance(X_test, np.ndarray), f"X_test must be np.ndarray, got {type(X_test)}"
    assert isinstance(y_test, np.ndarray), f"y_test must be np.ndarray, got {type(y_test)}"

    assert X_train.ndim == 3, f"X_train must be 3D (N, H, W), got {X_train.ndim}D shape {X_train.shape}"
    assert X_train.shape[1] == expected_H and X_train.shape[2] == expected_W, (
        f"X_train spatial dims must be ({expected_H}, {expected_W}), got {X_train.shape[1:]}"
    )

    assert X_test.ndim == 3, f"X_test must be 3D (N, H, W), got {X_test.ndim}D shape {X_test.shape}"
    assert X_test.shape[1] == expected_H and X_test.shape[2] == expected_W, (
        f"X_test spatial dims must be ({expected_H}, {expected_W}), got {X_test.shape[1:]}"
    )

    assert y_train.ndim == 1, f"y_train must be 1D (N,), got {y_train.ndim}D shape {y_train.shape}"
    assert y_test.ndim == 1, f"y_test must be 1D (N,), got {y_test.ndim}D shape {y_test.shape}"

    assert X_train.shape[0] == y_train.shape[0], (
        f"X_train/y_train size mismatch: X has {X_train.shape[0]} samples, y has {y_train.shape[0]}"
    )
    assert X_test.shape[0] == y_test.shape[0], (
        f"X_test/y_test size mismatch: X has {X_test.shape[0]} samples, y has {y_test.shape[0]}"
    )

    label_min = int(min(y_train.min(), y_test.min()))
    label_max = int(max(y_train.max(), y_test.max()))
    assert label_min >= 0 and label_max <= (NUM_CLASSES - 1), (
        f"Labels out of expected range [0, {NUM_CLASSES - 1}], got [{label_min}, {label_max}]"
    )

    logger.info(
        f"Shape OK: X_train {X_train.shape} (N, {expected_H}, {expected_W}), y_train {y_train.shape} | "
        f"X_test {X_test.shape}, y_test {y_test.shape} | labels in [{label_min}, {label_max}]"
    )
    return True


def check_class_balance(y: np.ndarray, num_classes: int = NUM_CLASSES) -> pd.DataFrame:
    logger.info(f"Checking class balance across {num_classes} classes")
    counts = np.bincount(y.astype(int), minlength=num_classes)
    total = int(counts.sum())
    df = pd.DataFrame({
        "label": np.arange(num_classes),
        "count": counts,
        "percentage": (counts / total * 100.0).round(3) if total > 0 else 0.0,
    })
    logger.info(f"Class distribution:\n{df.to_string(index=False)}")
    return df
