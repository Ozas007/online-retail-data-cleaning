import numpy as np
import pytest

from src.preprocessing import preprocess_images, preprocess_labels_categorical


def test_preprocess_images_range():
    X_raw = np.random.randint(0, 256, size=(20, 28, 28), dtype=np.uint8)
    X_raw[0, 0, 0] = 0
    X_raw[0, 0, 1] = 255
    X_proc = preprocess_images(X_raw)
    assert X_proc.ndim == 2
    assert X_proc.shape == (20, 784)
    assert float(X_proc.min()) >= 0.0
    assert float(X_proc.max()) <= 1.0 + 1e-6
    assert X_proc.dtype == np.float32


def test_preprocess_images_normalizes():
    X_raw = np.full((3, 2, 2), 200, dtype=np.uint8)
    X_proc = preprocess_images(X_raw)
    assert X_proc.shape == (3, 4)
    assert np.allclose(X_proc, 200 / 255.0, atol=1e-5)


def test_label_onehot_shape():
    y = np.array([0, 3, 9, 2, 7, 1], dtype=np.int32)
    y_cat = preprocess_labels_categorical(y, num_classes=10)
    assert y_cat.shape == (6, 10)
    row_sums = y_cat.sum(axis=1)
    assert np.allclose(row_sums, 1.0)
    assert np.argmax(y_cat, axis=1).tolist() == y.tolist()
    assert y_cat.dtype == np.float32
