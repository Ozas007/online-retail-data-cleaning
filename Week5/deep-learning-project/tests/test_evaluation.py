import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.model import build_mlp_model
from src.preprocessing import preprocess_labels_categorical
from src.evaluation import evaluate_trained_model


REQUIRED_KEYS = {
    "test_loss",
    "test_accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "per_class_f1",
    "confusion_matrix_df",
    "n_samples",
    "num_classes",
}


def test_evaluate_returns_all_required_keys(tmp_path):
    tf.random.set_seed(42)
    np.random.seed(42)

    n_samples = 120
    num_classes = 10
    X = np.random.rand(n_samples, 784).astype(np.float32)
    y = np.random.randint(0, num_classes, size=n_samples).astype(np.int32)
    y_cat = preprocess_labels_categorical(y, num_classes=num_classes)

    model = build_mlp_model(
        input_dim=784, num_classes=num_classes,
        dropout_rate=0.0, dense1_units=16, dense2_units=8, learning_rate=1e-3,
    )
    model.fit(X, y_cat, epochs=1, batch_size=32, verbose=0)

    result = evaluate_trained_model(model, X, y, y_cat, num_classes=num_classes)

    missing = REQUIRED_KEYS - set(result.keys())
    assert not missing, f"Missing required keys: {missing}"

    assert isinstance(result["confusion_matrix_df"], pd.DataFrame)
    assert result["confusion_matrix_df"].shape == (num_classes, num_classes)
    assert len(result["per_class_f1"]) == num_classes
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert 0.0 <= result["f1_weighted"] <= 1.0
    assert result["n_samples"] == n_samples
    assert result["num_classes"] == num_classes
