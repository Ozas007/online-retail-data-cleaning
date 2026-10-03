# Week 6 Integrative Capstone: End-to-End Online Retail ML Pipeline

Combines all validated components from Weeks 1–4 into a single coherent
customer-analytics project:

1. **Raw data loading + validation** (UCI Online Retail Excel)
2. **Data cleaning / preprocessing**: cancellation detection, de-duplication,
   positive-sales filter, CustomerID filter, TotalAmount derivation
3. **Exploratory Data Analysis**: monthly revenue, top-10 countries by
   transactions/revenue, top-10 products by revenue
4. **Feature engineering**:
   - RFM customer-level table (Recency / Frequency / Monetary)
   - 9-column supervised behavioural table + Country
5. **Unsupervised learning**: K-Means on log1p + StandardScaler RFM,
   k ∈ 2..10, optimal k by silhouette score, PCA 2D projection, cluster
   z-score profiles
6. **Supervised learning**: Temporal-split binary classification
   (Historical vs Future windows, no-data-leakage design) with
   Random Forest and Logistic Regression inside a
   `Pipeline(ColumnTransformer, Classifier)`, plus stratified 5-fold CV.
7. **Deep Learning (honest baseline)**: small tabular MLP trained on the
   same preprocessed design matrix — compared fairly against the
   best tree/linear model.
8. **Concrete, number-backed recommendations** (>= 5 items from real
   metrics).
9. **Report generation**: 24-section Word DOCX with embedded PNG figures
   and CSV tables.

## Running the pipeline

```bash
cd Week6/capstone
python -m src.main
```

Outputs are written to:
- `outputs/figures/` — 8 PNG visualisations
- `outputs/tables/`  — ~19 CSV tables
- `outputs/models/`  — any persisted scikit-learn / Keras artefacts
- `report/Week_6_Integrative_Capstone_Report.docx` — 24-section DOCX
- `logs/pipeline.log` — full run transcript

## Running the tests

```bash
cd Week6/capstone
python -m pytest tests/ -v
```

Tests use synthetic fixtures and `tmp_path` so they never overwrite
production outputs.

## Requirements

See `requirements.txt` in this directory for the full Python environment
spec (pandas, numpy, scikit-learn, openpyxl, python-docx, matplotlib,
seaborn, tensorflow/keras and pytest).
