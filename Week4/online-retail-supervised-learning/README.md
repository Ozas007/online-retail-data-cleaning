# Week 4: Supervised Learning — Customer Purchase Prediction

Predicting future customer purchases (binary classification) using RFM-style behavioural features engineered from historical transactions, with a rigorous temporal train/test split to eliminate leakage.

## Overview

This project builds a supervised binary classifier to predict which customers will make a purchase in a 3-month future window, using features computed strictly from an earlier historical window. Three models are compared: Logistic Regression, Decision Tree, and Random Forest. Evaluation is performed on a held-out test set and via 5-fold stratified cross-validation, with ROC / PR curves, confusion matrices, and feature importance analysis.

## Dataset

The UCI Online Retail dataset (Excel) is expected at one of the following locations (automatic fallback):

1. `Week4/online-retail-supervised-learning/data/Online+Retail.xlsx`
2. `E:/project/online-retail-data-cleaning/data/Online+Retail.xlsx` (parent repo, preferred)

See `data/README.md` for the official UCI download link and filename. The raw Excel file is not committed to version control.

## Installation

```bash
cd Week4/online-retail-supervised-learning
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Run

```bash
# Full pipeline: load -> clean -> temporal split -> target -> features -> train 3 models -> evaluate -> visualize -> DOCX report
python -m src.main
```

Outputs:
- `outputs/tables/*.csv` — feature table, predictions, CV results, importances, etc.
- `outputs/figures/*.png` — target distribution, confusion matrices, ROC/PR curves, feature importance, LR coefficients
- `report/Week_4_Supervised_Learning_Report.docx` — 26-section professional Word report with embedded figures and tables

## Tests

```bash
pytest tests/ -v
```

Tests use `tmp_path`; they never overwrite production outputs.

## Project Structure

```
Week4/online-retail-supervised-learning/
├── data/README.md                     Dataset source documentation
├── notebooks/supervised_learning.ipynb 4-cell empty notebook skeleton
├── outputs/figures/                   Generated PNG plots
├── outputs/tables/                    Generated CSV tables
├── report/                            Week_4_Supervised_Learning_Report.docx
├── src/
│   ├── __init__.py
│   ├── config.py                      Paths, seed, prediction window, feature lists
│   ├── data_loading.py                Excel loader + column validation
│   ├── data_validation.py             Missing/dup/dtypes/unique/invalid/date
│   ├── target_engineering.py          Temporal split + FuturePurchased target
│   ├── feature_engineering.py         RFM + AOV + product features (historical only)
│   ├── preprocessing.py               ColumnTransformer + Pipeline builders
│   ├── models.py                      LR / DT / RF factory functions
│   ├── evaluation.py                  Metrics, confusion DF, stratified CV
│   ├── visualization.py               matplotlib/ seaborn plots (Agg backend)
│   ├── report_generation.py           26-section python-docx report builder
│   └── main.py                        End-to-end orchestration
└── tests/
    ├── __init__.py
    ├── test_target_engineering.py     Leakage, binary, dates
    ├── test_feature_engineering.py    Shapes, nulls, future leakage
    ├── test_preprocessing.py          Preprocessor / pipeline shapes
    ├── test_models.py                 Fit/predict, reproducibility
    └── test_evaluation.py             Metric keys, CV output shape
```

## Random Seed / Reproducibility

- `RANDOM_SEED = 42` everywhere (numpy, sklearn, python-docx)
- Temporal split cutoffs derived deterministically from `InvoiceDate` quantiles
- All feature engineering restricted to the historical window
