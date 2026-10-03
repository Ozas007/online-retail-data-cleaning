import logging
from typing import Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def load_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    logger.info("Loading MNIST dataset via tf.keras.datasets.mnist.load_data()")
    try:
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    except Exception as e:
        logger.error(f"Failed to load MNIST dataset: {str(e)}")
        raise RuntimeError(f"Failed to load MNIST dataset: {str(e)}") from e

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    logger.info(
        f"Loaded MNIST: train X shape = {X_train.shape}, train y shape = {y_train.shape} | "
        f"test X shape = {X_test.shape}, test y shape = {y_test.shape}"
    )
    logger.info(
        f"X train dtype = {X_train.dtype}, y train dtype = {y_train.dtype} | "
        f"pixel range train: [{int(X_train.min())}, {int(X_train.max())}]"
    )
    return (X_train, y_train), (X_test, y_test)
