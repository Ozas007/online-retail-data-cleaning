import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def preprocess_images(X: np.ndarray) -> np.ndarray:
    logger.info(f"Preprocessing images: input shape = {X.shape}, dtype = {X.dtype}")

    X_flat = X.reshape(X.shape[0], -1)
    logger.info(f"Flattened shape = {X_flat.shape}")

    X_flat = X_flat.astype(np.float32)
    X_flat = X_flat / 255.0
    logger.info("Normalized pixel values to [0,1] (divided by 255)")

    X_min = float(X_flat.min())
    X_max = float(X_flat.max())
    logger.info(f"After preprocess: min={X_min:.4f}, max={X_max:.4f}, dtype={X_flat.dtype}")
    return X_flat


def preprocess_labels_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
    logger.info(f"One-hot encoding labels (num_classes={num_classes})")
    y_int = y.astype(np.int32)
    y_cat = tf.keras.utils.to_categorical(y_int, num_classes=num_classes).astype(np.float32)
    logger.info(f"Categorical labels shape = {y_cat.shape}, dtype = {y_cat.dtype}")
    return y_cat
