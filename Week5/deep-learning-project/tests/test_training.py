import numpy as np
import pytest
import tensorflow as tf

from src.model import build_mlp_model
from src.training import train_model


def test_one_batch_overfit(tmp_path):
    tf.random.set_seed(0)
    np.random.seed(0)
    import random as _random
    _random.seed(0)

    n_batch = 16
    input_dim = 784
    num_classes = 10

    X = np.random.rand(n_batch, input_dim).astype(np.float32)
    y_int = np.random.randint(0, num_classes, size=n_batch).astype(np.int32)
    y = tf.keras.utils.to_categorical(y_int, num_classes=num_classes).astype(np.float32)

    tiny_model = build_mlp_model(
        input_dim=input_dim,
        num_classes=num_classes,
        dropout_rate=0.0,
        dense1_units=64,
        dense2_units=32,
        learning_rate=1e-3,
    )

    history = train_model(
        model=tiny_model,
        X_train=X,
        y_train_cat=y,
        epochs=20,
        batch_size=n_batch,
        validation_split=0.0,
        patience=20,
        seed=0,
        checkpoint_path=None,
        verbose=0,
    )

    losses = history.history["loss"]
    first_loss = float(losses[0])
    last_loss = float(losses[-1])
    assert first_loss > last_loss, (
        f"Expected training loss to decrease on 1-batch overfit, "
        f"but first_loss={first_loss:.4f} and last_loss={last_loss:.4f}"
    )

    final_acc = float(history.history["accuracy"][-1])
    assert final_acc >= 0.5, (
        f"Expected at least 50% accuracy on a single memorized batch after 20 epochs, got {final_acc:.4f}"
    )
