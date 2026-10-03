# Dataset: MNIST Handwritten Digits

## Overview

This project uses the **MNIST** (Modified National Institute of Standards and Technology) dataset of handwritten digits, loaded directly from TensorFlow/Keras built-in utilities. No local data files are required; the dataset is downloaded and cached automatically on first run via `tf.keras.datasets.mnist.load_data()`.

## Reference

**Citation:** LeCun, Y., Cortes, C., & Burges, C. J. C. *The MNIST Database of Handwritten Digits*.  
Available at: http://yann.lecun.com/exdb/mnist/

## Specifications

| Property | Value |
|---|---|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image dimensions | 28 x 28 pixels |
| Color space | Grayscale (single channel) |
| Pixel value range | 0–255 (uint8) |
| Number of classes | 10 (digits 0–9) |
| Class balance | Approximately balanced (≈ 6,000 images per digit in train) |

## Usage

```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

After loading:
- `X_train.shape == (60000, 28, 28)`
- `X_test.shape  == (10000, 28, 28)`
- `y_train.shape == (60000,)`   (integers in range `[0, 9]`)
- `y_test.shape  == (10000,)`   (integers in range `[0, 9]`)
