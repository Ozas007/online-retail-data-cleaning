# UCI Online Retail — Data Cleaning & Preprocessing Pipeline

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![pytest](https://img.shields.io/badge/tests-22%20passed-success)](#testing)
[![pipeline](https://img.shields.io/badge/pipeline-reproducible-8A2BE2)](#running-the-pipeline)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **complete, production-quality academic data-cleaning and preprocessing project built on the
[UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail).
Every statistic, table and figure in this repository is computed from the actual dataset
— never fabricated.

---

## Overview

This project demonstrates a full, auditable data-cleaning workflow:

1. **Acquisition** → reliable public dataset from the UCI repository.
2. **Exploration** → structured EDA with `head/tail/info/describe + unique/missing/duplicate reports.
3. **Validation** → columns, dtypes, missingness, duplicates, negative quantities, invalid prices, cancellation invoices, blank strings.
4. **Cleaning** → deduplication, flags for cancellations/negatives, removal of invalid UnitPrice, feature engineering.
5. **Outlier detection** → IQR method on Quantity, UnitPrice and TotalAmount, documented before/after impact on mean/median/std.
6. **Preprocessing** → StandardScaler scaling, safe categorical encoding, datetime decomposition, ID-safe preprocessing note.
7. **Outputs** → 5 cleaned datasets, 14 summary CSV tables, 15 publication-style PNG figures.
8. **Tests** → 22 unit tests with pytest.
9. **Notebook** → Jupyter walkthrough with markdown explanations.
10. **GitHub readiness** → `.gitignore`, `LICENSE`, relative paths, documented `pathlib everywhere.

---

## Objectives

The assignment-style deliverables required by the academic brief:

* Demonstrate acquisition from a reliable public source
* Identify and treat missing values (count + % + reasoning
* Identify and treat exact duplicates
* Investigate erroneous / inconsistent values (negative qty, zero price, cancellation invoices)
* Detect statistical outliers (IQR), document decisions
* Transform and preprocess (TotalAmount, datetime decomposition, numerical scaling)
* Document every major decision with technical reasoning and downstream impact
* Produce a reproducible, professional GitHub-ready project

---

## Dataset

### Dataset Source

* **Name:** Online Retail
* **Host:** UCI Machine Learning Repository
* **Direct link:** <https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx>
* **Citation:** Chen, D., Sain, S. L., & Guo, K. (2012). *Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining.* Journal of Database Marketing & Customer Strategy Management, 19(3), 197–208.

### Dataset Characteristics

Computed from the actual raw `Online+Retail.xlsx` source file that this pipeline loads:

| Metric | Value |
|---|---|
| Rows | **541,909** |
| Columns | **8** |
| In-memory size | **126.18 MB** |
| InvoiceNo levels | 25,900 |
| StockCode levels | 4,070 |
| CustomerID levels | 4,372 |
| Country levels | 38 |
| Date range | 2010-12-01 08:26 → 2011-12-09 12:50 |

Columns (required by the loader):

* `InvoiceNo` — invoice number. Prefix `C` = cancellation.
* `StockCode` — product catalog code.
* `Description` — human-readable product name.
* `Quantity` — units of the product on the line item.
* `InvoiceDate` — date + time of the invoice.
* `UnitPrice` — sterling unit price.
* `CustomerID` — unique customer identifier (float; 24.93% missing).
* `Country` — shipping country of the customer.

### Data Quality Issues

All figures below come from actual computation on the raw dataset:

| Issue | Count | Percentage |
|---|---|---|
| Missing CustomerID | 135,080 | 24.93% |
| Missing Description | 1,454 | 0.27% |
| Exact duplicate rows | 5,268 | 0.97% |
| Negative Quantity rows | 10,624 | 1.96% |
| UnitPrice ≤ 0 rows | 2,517 | 0.46% |
| Cancellation invoices (prefix `"C"` on InvoiceNo) | 9,288 | 1.71% |
| Negative UnitPrice rows | 2 | <0.01% |
| Zero UnitPrice rows | 2,515 | 0.46% |

### Cleaning Approach

| Step | Action | Rows removed / Reason |
|---|---|---|
| Raw dataset | N/A | 541,909 rows |
| Exact duplicates | Drop (keep first) | 5,268 removed |
| Cancellation invoices | `IsCancellation` boolean flag | 0 rows removed |
| Negative / zero quantities | `IsNegativeQuantity`, `IsZeroQuantity` flags | 0 rows removed |
| Zero / negative prices | `IsNegativePrice`, `IsZeroPrice` flags + drop in cleaned set | 2,512 rows removed from cleaned dataset |
| Missing Description | Retained as-is; no arbitrary text imputation | 0 rows removed; product-level analysis unaffected in financial aggregations |
| Feature engineering | TotalAmount, datetime features | 0 rows removed |
| Positive-sales analytical set | Cancellations + non-positive qty excluded | 9,251 rows removed from analytical set |

Final sizes:
`cleaned` = **534,129 rows** | `positive_sales` = **524,878 rows** (flagged` still contains all flags).

### Outlier Methodology

* **Method:** IQR with multiplier 1.5, on strictly positive values only (so cancellation returns cannot skew the bounds artificially).
* **Applied to:** `Quantity`, `UnitPrice`, `TotalAmount`.

Outlier summary (computed on cleaned dataset 534,129 rows):

| Column | Q1 | Q3 | IQR | Lower | Upper | Count | % |
|---|---|---|---|---|---|---|---|
| Quantity | 1.0 | 11.0 | 10.0 | −14.0 | 26.0 | 27,111 | 5.08% |
| UnitPrice | 1.25 | 4.13 | 2.88 | −3.07 | 8.45 | 39,448 | 7.39% |
| TotalAmount | 3.90 | 17.70 | 13.80 | −16.80 | 38.40 | 42,624 | 7.98% |

Outliers are **flagged, not deleted. Bulk orders (up to Quantity = 80,995, TotalAmount = 80,995 × the relevant price) and legitimate high-end merchandise are real business activity and should not be discarded merely because they are statistically extreme. Impact analysis on mean vs median confirms the expected skewed behaviour see `outputs/tables/outlier_impact_*.csv.

### Preprocessing

The preprocessing module prepares the positive-sales analytical set:

* **Numerical features:** `Quantity`, `UnitPrice`, `TotalAmount`, `Year`, `Month`, `Day`, `Hour`, `DayOfWeek` → `StandardScaler` scaling with saved in `online_retail_scaled.csv`.
* **Categorical features:** `Country` is demonstrated with LabelEncoder (one-hot available on demand via parameter).
* **Identifier columns (`InvoiceNo`, `StockCode`, `CustomerID`) are explicitly excluded from ordinary categorical encoding because each ID refers to a unique entity. Blind one-hot encoding of these columns would inject spurious ordinal relationships, explode the feature space and overfit downstream models.

---

## Project Structure

```
online-retail-data-cleaning/
├── data/
│   ├── Online+Retail.xlsx      (user-provided, not in git)
│   └── README.md                 dataset README with download URL)
├── notebooks/
│   └── data_cleaning_walkthrough.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loading.py
│   ├── data_validation.py
│   ├── data_exploration.py
│   ├── data_cleaning.py
│   ├── outlier_detection.py
│   ├── preprocessing.py
│   ├── visualization.py
│   └── main.py                   one-step pipeline entry point)
├── tests/
│   ├── __init__.py
│   ├── test_data_loading.py
│   ├── test_data_cleaning.py
│   └── test_preprocessing.py
├── outputs/
│   ├── figures/                  15 charts, .png)
│   ├── tables/                   14 summary CSV tables)
│   └── cleaned_data/
│       ├── online_retail_flagged.csv           all rows + quality flags)
│       ├── online_retail_cleaned.csv       quality-cleaned: UnitPrice > 0)
│       ├── online_retail_positive_sales.csv   no cancels, qty > 0)
│       ├── online_retail_preprocessed.csv     base analytical set]
│       └── online_retail_scaled.csv         scaled analytical set]
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
└── pipeline.log                 (runtime)
```

---

## Installation

### Prerequisites

* Python **3.11+**
* A working internet connection for the first dataset download (see [`data/README.md`](data/README.md)).

### Install dependencies

```bash
cd online-retail-data-cleaning
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Obtain the dataset

```bash
cd data/
python -c "import urllib.request; urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx', 'Online+Retail.xlsx')"
cd ..
```

Expected file: `data/Online+Retail.xlsx` (~22.6 MB).

---

## Usage

### Running the pipeline

```bash
# From the project root:
python -m src.main
```

The entry point is `src/main.py`. It:

1. Loads & validates the raw Excel dataset.
2. Runs full data validation + exploratory analysis.
3. Saves 14 summary tables into `outputs/tables/`.
4. Runs the full cleaning pipeline and saves 5 cleaned datasets into `outputs/cleaned_data/`.
5. Runs IQR outlier detection with impact analysis.
6. Performs preprocessing StandardScaler on the analytical set.
7. Generates 15 figures into `outputs/figures/`.
8. Appends a before/after comparison.

### Notebook walkthrough

Open `notebooks/data_cleaning_walkthrough.ipynb` with Jupyter:

```bash
jupyter notebook notebooks/data_cleaning_walkthrough.ipynb
```

The notebook mirrors the pipeline but adds narrative markdown commentary inline interpretations of each output.

### Running individual modules

Every `src/*.py` modules are independently importable, allowing notebook or external analysis:

```python
from src.data_loading import load_excel_dataset
from src.data_cleaning import run_full_cleaning

df_raw = load_excel_dataset()
clean = run_full_cleaning(df_raw, save_outputs=False)
```

---

## Testing

```bash
cd online-retail-data-cleaning
python -m pytest tests/ -v
```

Expected output on Python 3.13 environment this was tested on:

```
collected 22 items
tests/test_data_loading.py ... 5 passed
tests/test_data_cleaning.py ... 10 passed
tests/test_preprocessing.py ... 7 passed
============================== 22 passed in 3.92s ==============================
```

Tests use small synthetic DataFrames instead of loading the full 541k-row dataset, so the suite is fast (<5 s) and CI-friendly.

Covered:

* Column validation.
* Duplicate removal behaviour.
* Cancellation + negative-qty + invalid-price flags.
* `TotalAmount = Quantity × UnitPrice` correctness.
* Datetime feature extraction.
* Positive-sales dataset filter.
* IQR bounds / outlier flagging.
* `ensure_numeric` / `ensure_datetime` / `ensure_string`.
* `StandardScaler` output (zero mean, unit std on scaled).

---

## Results

All results come from actual runs against the real dataset (2026-09-27.

### Dataset statistics

| Dataset | Rows | Columns | Missing CustomerID | Missing Description | Negative Qty | UnitPrice ≤ 0 | Cancels |
|---|---|---|---|---|---|---|---|
| Raw | 541,909 | 8 | 135,080 | 1,454 | 10,624 | 2,517 | 9,288 |
| Cleaned (quality) | 534,129 | 20 | 132,565 | 0 | 9,251 | 0 | 9,251 |
| Positive-sales analytical | 524,878 | 20 | 132,186 | 0 | 0 | 0 | 0 |

### TotalAmount summary (positive-sales analytical set)

| Stat | Value |
|---|---|
| Mean | £ 20.28 |
| Median | £ 9.92 |
| Std | £ 271.69 |
| Sum | £ 10,642,110.80 |

### Visualizations output

Saved into `outputs/figures/`:

* `missing_values_bar.png`
* `quantity_distribution_cleaned.png`, `quantity_distribution_positive.png`
* `unitprice_distribution_cleaned.png`, `unitprice_distribution_positive.png`
* `quantity_boxplot_cleaned.png`, `quantity_boxplot_positive.png`
* `unitprice_boxplot_cleaned.png`, `unitprice_boxplot_positive.png`
* `total_amount_distribution_cleaned.png`, `total_amount_distribution_positive.png`
* `country_distribution.png`
* `monthly_transactions.png`
* `cancellation_analysis.png`
* `before_after_comparison.png`

---

## Limitations

1. **CustomerID missingness (24.93%). Any RFM / customer segmentation must drop those rows.
2. **Description missing (0.27%):** Retained as-is in the pipeline; no arbitrary text imputation is applied. Product-level studies can optionally build a StockCode→Description lookup from the 99.73% populated rows, but the cleaning pipeline itself leaves Description untouched. Revenue/quantity aggregations are unaffected.
3. **Outlier methodology.** The IQR method assumes unimodal, roughly symmetric distributions. The Online Retail data strongly skews heavily and a domain expert judgement for a true generative model would catch more nuanced outlier handling (e.g., MCD / robust Mahalanobis on multivariate). I explicitly flag only.
4. **Cancellation semantics.** Not every negative-quantity row has a `"C"` prefixed invoice; some may represent manual reversals, not data issues are flagged but not resolved.
5. **Temporal coverage.** Dataset terminates 2011-12-09; December is incomplete.

---

## Reproducibility

* Random seeds: `config.RANDOM_SEED = 42` in config.py` (used only in sklearn preprocessors when needed deterministic).
* `pathlib` relative paths everywhere; nowhere hardcoded.
* All modules copy inputs via `.copy()`. The original raw DataFrame remains unmodified in memory; the source Excel file is never touched.
* Cleaning pipeline emits a deterministic log per output paths all tables are sorted.
* Every function with type hints, docstrings and clear function boundaries.
* pytest runs without network access after dataset is downloaded.

---

## References

1. Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197–208. https://doi.org/10.1057/dbm.2012.17
2. Dua, D., & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences. https://archive.ics.uci.edu/ml
3. McKinney, W. et al. pandas development team. (2024). pandas-dev/pandas: Pandas. Zenodo. https://pandas.pydata.org
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html
5. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.

---

## License

MIT License — see [LICENSE](LICENSE).

The UCI Online Retail dataset is © its original authors and UCI; see data/README.md for download instructions and attribution.
