import numpy as np
import pytest
import tensorflow as tf

from src.model import build_mlp_model, get_model_summary_string


def test_build_model_output_shape():
    model = build_mlp_model(input_dim=784, num_classes=10, dropout_rate=0.2, dense1_units=128, dense2_units=64, learning_rate=0.001)
    fake_batch = np.random.rand(5, 784).astype(np.float32)
    output = model(fake_batch, training=False)
    assert output.shape == (5, 10), f"Expected (5, 10), got {output.shape}"
    row_sums = tf.reduce_sum(output, axis=1).numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Softmax row sums should equal 1"


def test_model_compiled_metrics():
    model = build_mlp_model()
    assert model.loss is not None, "Model must have a compiled loss"
    assert model.optimizer is not None, "Model must have a compiled optimizer"
    metric_names = [m.name for m in model.metrics] if hasattr(model, "metrics") else []
    assert any("accuracy" in m.lower() for m in metric_names), "Metrics should contain accuracy"
    assert model.optimizer.learning_rate is not None


def test_get_model_summary_string_returns_text():
    model = build_mlp_model()
    text = get_model_summary_string(model)
    assert isinstance(text, str) and len(text) > 100
    assert "dense1" in text or "Dense" in text or "dense2" in text
    assert "output" in text
