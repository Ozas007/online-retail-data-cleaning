import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from src.config import (
    RANDOM_SEED,
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


def _set_all_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Set random seeds (random/numpy/tf) = {seed}")


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train_cat: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
    patience: int = EARLY_STOPPING_PATIENCE,
    seed: int = RANDOM_SEED,
    checkpoint_path: Optional[Path] = None,
    verbose: int = 1,
) -> tf.keras.callbacks.History:
    _set_all_seeds(seed)

    callbacks = []
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks.append(early_stop)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        )
        callbacks.append(checkpoint)
        logger.info(f"ModelCheckpoint enabled at: {checkpoint_path}")

    logger.info(
        f"Starting training: epochs={epochs}, batch_size={batch_size}, "
        f"val_split={validation_split}, patience={patience} | "
        f"train X shape = {X_train.shape}, y_cat shape = {y_train_cat.shape}"
    )

    history = model.fit(
        X_train,
        y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose,
    )

    if hasattr(early_stop, "stopped_epoch") and early_stop.stopped_epoch > 0:
        logger.info(f"Early stopping triggered at epoch {early_stop.stopped_epoch + 1}")
    else:
        logger.info(f"Training completed all {epochs} epochs (no early stopping)")

    has_val = "val_loss" in history.history and "val_accuracy" in history.history
    if has_val:
        logger.info(
            f"Final training metrics: loss={history.history['loss'][-1]:.4f}, "
            f"acc={history.history['accuracy'][-1]:.4f} | "
            f"val_loss={history.history['val_loss'][-1]:.4f}, "
            f"val_acc={history.history['val_accuracy'][-1]:.4f}"
        )
    else:
        logger.info(
            f"Final training metrics: loss={history.history['loss'][-1]:.4f}, "
            f"acc={history.history['accuracy'][-1]:.4f} (no validation split)"
        )
    return history
