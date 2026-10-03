# Week 5 — Deep Learning: MLP on MNIST (TensorFlow/Keras)

A reproducible, modular deep learning pipeline that trains and evaluates a Multi-Layer Perceptron (MLP) classifier on the MNIST handwritten digits dataset using **TensorFlow / Keras only** (no PyTorch). The project follows the same clean folder structure as Week 3 (clustering): `config + src modules + tests + visualization + report + main orchestration`.

## Architecture Summary (3-Layer MLP)

```
Input (784 flattened pixels)
    └─► Dense(128, ReLU)
        └─► Dropout(0.2)
            └─► Dense(64, ReLU)
                └─► Dropout(0.2)
                    └─► Dense(10, Softmax)  →  predicted digit class
```

Compiled with:
- Optimizer: **Adam** (learning rate = 0.001)
- Loss: **CategoricalCrossentropy** (one-hot labels)
- Metric: **Accuracy**
- Regularizer: **EarlyStopping** (patience = 3, restore best weights on `val_loss`)

## Project Structure

```
Week5/deep-learning-project/
├── data/                      Built-in MNIST dataset docs
├── notebooks/                 Jupyter experiment notebook
├── outputs/
│   ├── figures/               Training curves, CM, architecture diagram PNGs
│   ├── tables/                CSV exports (history, eval metrics, CM, etc.)
│   └── models/                Saved Keras model (*.keras)
├── report/                    Week_5_Deep_Learning_Report.docx
├── src/
│   ├── config.py              Paths + hyperparams + RANDOM_SEED = 42
│   ├── data_loading.py        `load_mnist()`  → (X_train, y_train), (X_test, y_test)
│   ├── data_validation.py     Assert shapes, label range; check class balance
│   ├── preprocessing.py       Flatten + normalize [0,1]; one-hot labels
│   ├── model.py               `build_mlp_model(...)`, summary helper
│   ├── training.py            Train with EarlyStopping; seed TF + NumPy
│   ├── evaluation.py          Test loss/acc + P/R/F1 + confusion matrix
│   ├── visualization.py       Training curves, CM heatmap, architecture diagram
│   ├── report_generation.py   Build 26-section DOCX report (TOC + page numbers)
│   └── main.py                End-to-end pipeline orchestration
├── tests/                     pytest modules (data/model/training/evaluation)
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```powershell
# 1. (Optional) Create a virtual env
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

> **Only** TensorFlow/Keras is used as the deep learning framework. No PyTorch is allowed or installed.

## Running the Pipeline

```powershell
# From the project root (Week5/deep-learning-project):
python -m src.main
```

On success, the script will:

1. Load + validate MNIST (60k train / 10k test).
2. Preprocess: flatten to 784-dim vectors, rescale pixel values to `[0, 1]`, one-hot encode labels.
3. Build + compile the MLP model and print a layer summary.
4. Train for up to 15 epochs with `EarlyStopping` and 10% validation split.
5. Evaluate on the held-out test set: loss, accuracy, precision/recall/F1 (weighted), per-class F1, confusion matrix.
6. Save:
   - `outputs/models/mnist_mlp.keras`  (trained Keras model)
   - `outputs/figures/*.png`  (4 plots: architecture diagram, training curves, CM heatmap, etc.)
   - `outputs/tables/*.csv`   (history, CM, class balance, eval metrics)
7. Generate the full 26-section Word report at `report/Week_5_Deep_Learning_Report.docx`.
8. Print a final summary including the **overfitting gap** (`final_train_acc − final_val_acc`).

## Running Tests

```powershell
pytest tests/ -v
```

All tests are written to use `tmp_path` and do **not** overwrite production outputs or saved models.

## Expected Outputs

| Output | Location | Description |
|---|---|---|
| Saved model | `outputs/models/mnist_mlp.keras` | Keras v3 native format |
| Architecture diagram | `outputs/figures/03_architecture_diagram.png` | Block diagram (uses `keras.utils.plot_model` if graphviz installed; otherwise a matplotlib fallback that always works) |
| Training curves | `outputs/figures/01_training_curves.png` | Loss + accuracy, train vs val |
| Confusion matrix | `outputs/figures/02_confusion_matrix.png` | Heatmap of true vs predicted digit |
| Eval metrics | `outputs/tables/05_evaluation_metrics.csv` | loss, accuracy, P/R/F1 |
| DOCX report | `report/Week_5_Deep_Learning_Report.docx` | Full 26-section report with embedded figures |

All runs are deterministic — random seeds for `random`, `numpy`, and `tensorflow` are fixed to **42** before training.
