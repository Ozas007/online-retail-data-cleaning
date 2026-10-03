# Week 3: Unsupervised Learning and Customer Clustering with RFM Analysis

Academic unsupervised-learning project for the **UCI Online Retail Dataset**,
built to satisfy a Week 3 university assignment requiring customer
segmentation via **RFM feature engineering + K-Means clustering**.

The goal of the project is to discover natural, data-driven customer groups
within the retailer's transaction history. By encoding each customer as an
RFM (Recency / Frequency / Monetary) vector and then applying K-Means with a
rigorously justified choice of *k* (Elbow + Silhouette), the pipeline
produces interpretable customer segments whose profiles can be used to
inform retention, upselling, and marketing decisions.  All numerical
outputs, cluster sizes, silhouette scores, and RFM values are computed from
the actual loaded Excel dataset.

---

## Objective

Customer segmentation is a core unsupervised-learning application in
marketing analytics. Given a raw transaction log, the objective is to group
customers based on their purchasing behaviour so that each group is
internally similar and meaningfully different from the others. The project
implements a standard RFM (Recency – Frequency – Monetary) framework on the
UCI Online Retail dataset: Recency measures how recently each customer
purchased, Frequency counts distinct orders, and Monetary aggregates total
spend. After a log1p + StandardScaler transform, K-Means is swept across
k = 2..10, and the best k is selected by combining the Elbow heuristic with
the highest Silhouette score. The final clusters are then profiled, ranked
by revenue, visualized in PCA space, and documented with evidence-based
natural-language descriptions. The output is a complete, reproducible
clustering pipeline whose results populate a Word-format course report.

---

## Dataset

- **Name:** UCI Online Retail
- **Expected file in this project:** `data/Online+Retail.xlsx`
- **Source URL:** <https://archive.ics.uci.edu/dataset/352/online+retail>
- **Rows:** 541,909 transaction lines
- **Columns:** 8 (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country)
- **CustomerID missing rate:** ~25% (rows with null CustomerID are **excluded** from segmentation; clustering is performed only on customers with valid IDs)
- **Date range:** 2010-12-01 to 2011-12-09
- **Retailer context:** UK-based, non-store online gift retailer

---

## Project Structure

```
online-retail-clustering/
│
├── data/
│   ├── Online+Retail.xlsx          ← Raw UCI Excel dataset (user must supply)
│   └── README.md                   ← Dataset documentation
│
├── notebooks/
│   └── customer_segmentation.ipynb ← Complete Jupyter walkthrough
│
├── outputs/
│   ├── figures/                    ← 9+ PNG figures (200 DPI)
│   └── tables/                     ← RFM, k-metrics, cluster-profiles CSVs & JSON
│
├── report/
│   └── Week_3_Unsupervised_Learning_and_Clustering_Report.docx
│
├── src/
│   ├── __init__.py
│   ├── config.py                   ← Paths, constants, k-range, random seed
│   ├── data_loading.py             ← Excel loader, file/column validation
│   ├── data_preparation.py         ← TotalAmount, temporal features, positive-sales filter
│   ├── feature_engineering.py      ← RFM computation, log1p, StandardScaler
│   ├── clustering.py               ← k-sweep, K-Means, PCA, cluster profiles
│   ├── evaluation.py               ← Silhouette/CH/DB metrics, descriptions
│   ├── visualization.py            ← 9 plots: RFM, elbow, silhouette, PCA, heatmap, radar, bar
│   └── main.py                     ← Full pipeline orchestrator
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loading.py
│   ├── test_data_preparation.py
│   └── test_clustering.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Requirements

Key libraries declared in `requirements.txt`:

| Library | Purpose |
|---|---|
| `pandas` | DataFrames, data manipulation, RFM aggregation |
| `numpy` | Numeric utilities, log1p, array operations |
| `scikit-learn` | KMeans, StandardScaler, PCA, Silhouette/CH/DB metrics |
| `matplotlib` | Static plotting (9+ figures, 200 DPI PNGs) |
| `seaborn` | Statistical visualization styling, heatmaps, scatterplots |
| `scipy` | Statistical distributions / supporting numerics |
| `openpyxl` | Excel reader for `Online+Retail.xlsx` |
| `python-docx` | DOCX report builder for `report/` output |
| `jupyter` | Notebook runtime for `notebooks/customer_segmentation.ipynb` |
| `pytest` | Unit / integration test runner |

---

## How to Run

Before executing, ensure the dataset file `Online+Retail.xlsx` has been
placed inside the `data/` directory. The pipeline will raise a clear
`DatasetNotFoundError` with the expected absolute path if the file is
missing.

From the **project root** (`online-retail-clustering/`):

```bash
python -m src.main
```

Running this command executes the complete pipeline:
1. Load & validate the Excel dataset
2. Engineer TotalAmount + temporal features, build positive-sales dataset
3. Compute per-customer RFM (Recency/Frequency/Monetary)
4. Apply log1p transform + StandardScaler
5. Sweep k = 2..10 with Elbow + Silhouette to select optimal k
6. Run final K-Means (random_state=42, n_init=10)
7. Compute internal cluster evaluation metrics (Silhouette, CH, DB)
8. Run PCA (2 components, visualization only)
9. Build cluster profiles and evidence-based descriptions
10. Generate 9+ figures → `outputs/figures/`
11. Export 10+ tables → `outputs/tables/`
12. Build the final DOCX report → `report/Week_3_Unsupervised_Learning_and_Clustering_Report.docx`

---

## How to Run Tests

From the project root:

```bash
pytest tests/ -v
# or, equivalently:
python -m pytest tests/ -v
```

The test suite validates file loading, column presence, RFM computation
shape/columns, feature transforms, k-sweep DataFrame, K-Means label shape,
and profile aggregation.

---

## Outputs

### `outputs/figures/` — 9+ publication-quality PNG plots (200 DPI)

| # | Figure | Filename |
|---|--------|----------|
| 1 | RFM distributions on the raw scale (Recency / Frequency / Monetary) | `01_rfm_distribution.png` |
| 2 | RFM distributions after log1p transformation | `02_rfm_log_distribution.png` |
| 3 | Elbow method — Inertia vs k | `03_elbow_plot.png` |
| 4 | Silhouette analysis — Silhouette score vs k (best k highlighted) | `04_silhouette_plot.png` |
| 5 | Cluster size bar chart (counts + percentages) | `05_cluster_sizes.png` |
| 6 | PCA 2D scatter plot with cluster colours + centroids | `06_pca_clusters.png` |
| 7 | Standardized cluster-profile heatmap (RFM means, z-score) | `07_cluster_profile_heatmap.png` |
| 8 | Normalized radar chart of RFM profiles — one per cluster | `08_cluster_profile_radar.png` |
| 9 | Average monetary spend bar comparison across clusters | `09_cluster_monetary_bar.png` |

### `outputs/tables/` — all intermediate and final tables

| File | Description |
|---|---|
| `rfm_raw.csv` | Raw per-customer Recency / Frequency / Monetary (CustomerID keyed) |
| `rfm_distribution_stats.csv` | Describe + skew + kurtosis for each RFM column |
| `rfm_log_transformed.csv` | RFM with added Log_Recency / Log_Frequency / Log_Monetary columns |
| `k_metrics.csv` | Inertia + Silhouette score for each k in 2..10 |
| `cluster_labels.csv` | Per-customer CustomerID + assigned Cluster |
| `clustering_metrics.json` | Silhouette / Calinski-Harabasz / Davies-Bouldin scores |
| `cluster_profiles.csv` | Aggregate profile per cluster (count, mean/median RFM, revenue share) |
| `cluster_descriptions.json` | Evidence-based natural-language description per cluster |
| `pca_coordinates.csv` | Per-customer PC1 / PC2 coordinates |
| `pca_explained_variance.csv` | Explained variance ratio for each principal component |

### `report/`

- `Week_3_Unsupervised_Learning_and_Clustering_Report.docx` — full
  submission report with figures, tables, methodology, results,
  limitations, and references.

---

## Methodology

The full clustering pipeline follows these reproducible steps, in order:

1. **Reference date derivation.** Set `reference_date = max(InvoiceDate) + 1 day`
   directly from the loaded dataset so Recency is computed relative to a
   snapshot just after the final transaction.
2. **Per-customer RFM aggregation.** On the positive-sales,
   non-null-CustomerID subset:
   - *Recency* = `(reference_date − max InvoiceDate).days`
   - *Frequency* = `nunique(InvoiceNo)` per customer
   - *Monetary* = `sum(TotalAmount)` per customer (positive sales only)
3. **Transform & scale.** Apply `np.log1p` to each of Recency, Frequency,
   Monetary to correct strong positive skew; then z-score with
   `sklearn.preprocessing.StandardScaler` (zero mean, unit variance).
4. **Optimal k sweep.** Fit KMeans for each `k ∈ {2, 3, …, 10}`
   (random_state=42, n_init=10) and record Inertia (WCSS) plus the
   Silhouette score. Choose the k with the *highest Silhouette score*; the
   Elbow plot is produced as secondary corroborating evidence.
5. **Final K-Means fit.** Re-fit KMeans with the selected k
   (random_state=42, n_init=10) and assign cluster labels to every
   customer.
6. **Internal cluster evaluation.** Report Silhouette coefficient,
   Calinski-Harabasz (variance-ratio) score, and Davies-Bouldin index on
   the scaled RFM feature space.
7. **PCA for visualization only.** Project scaled RFM into two principal
   components (random_state=42) purely for 2D plotting; PCA is **not** used
   as a feature for the actual clustering.
8. **Cluster profiles & descriptions.** Aggregate each cluster's RFM means,
   customer counts, and revenue share; then generate evidence-based
   natural-language descriptions by comparing cluster-level R/F/M means
   against the dataset-wide means using ±20% thresholds for "high" / "low"
   / "moderate".

---

## Key Results

*(Placeholder style — actual numeric values are produced by running
`python -m src.main` and reading the generated outputs/ and report/)*

- **Number of clusters selected:** k = X (chosen by highest silhouette score in k ∈ 2..10, with the Elbow plot produced as corroborating evidence).
- **Silhouette score (final clustering):** Y (reported together with Calinski-Harabasz = Z and Davies-Bouldin = W in `outputs/tables/clustering_metrics.json`).
- **Cluster sizes:** See `outputs/tables/cluster_profiles.csv` and the final DOCX report for per-cluster customer counts, percentage of base, mean/median RFM, total revenue, and revenue-share breakdowns.
- **PCA variance explained:** See `outputs/tables/pca_explained_variance.csv` for the individual and cumulative explained variance of PC1 and PC2 used to draw the 2D cluster scatter.
- **Natural-language descriptions:** One evidence-based paragraph per cluster, keyed by cluster id, is written to `outputs/tables/cluster_descriptions.json`.

---

## Reproducibility

The pipeline is designed to produce byte-for-byte equivalent results across
runs:

- **Fixed random seeds:** `random_state=42` is used for every stochastic
  step — KMeans (all k in the sweep and the final fit) and PCA (2D
  projection for visualization).
- **Deterministic reference date:** `reference_date = max(InvoiceDate) + 1 day`
  is derived from the actual loaded data rather than hardcoded, so any
  dataset of the same file yields the same snapshot date.
- **Relative paths:** All I/O (dataset, outputs, report, figures, tables)
  is resolved relative to `PROJECT_ROOT = Path(__file__).parent.parent` in
  `src/config.py` so the project is portable to any working directory.
- **Pin requirements:** Version numbers are pinned in `requirements.txt`
  (`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, etc.) so
  `pip install -r requirements.txt` installs the exact same dependency
  versions used during development.
- **Stable sort order:** Cluster profiles, k-metrics tables, and CSV
  exports are sorted by cluster id or by explicit keys before being saved.

---

## Dataset Source & Citations

- **UCI Machine Learning Repository dataset page:**
  <https://archive.ics.uci.edu/dataset/352/online+retail>
- **Primary RFM segmentation paper reference:**
  Chen, D. Sain, S. L. & Guo, K. (2012). *Data mining for the online retail
  industry: A case study of RFM model-based customer segmentation using
  data mining.* Journal of Database Marketing & Customer Strategy
  Management, 19(3), 197–208.
- **UCI citation:**
  Dua, D. & Graff, C. (2019). *UCI Machine Learning Repository.*
  <https://archive.ics.uci.edu/ml>. Irvine, CA: University of California,
  School of Information and Computer Science.

---

## Limitations

1. **Missing CustomerID rows excluded.** Roughly 25% of the raw UCI
   transaction lines carry a null CustomerID and are dropped before
   segmentation; cluster insights therefore apply only to the identifiable
   customer subset, not to anonymous or guest checkout activity.
2. **Single UK non-store retailer.** The dataset represents one niche
   retailer (predominantly UK customers, gifts & household). Clusters
   learned on this data should not be extrapolated to other geographies,
   categories, or business models.
3. **K-Means spherical assumption.** K-Means implicitly assumes clusters
   are roughly spherical, equally sized, and with similar intra-cluster
   variance — real customer segments in RFM space often violate this
   (e.g., a tiny but valuable "VIP" cluster is intrinsically smaller and
   denser than a large "dormant" cluster).
4. **Snapshot-in-time segmentation.** RFM is computed on the fixed window
   2010-12-01 → 2011-12-09. The segments represent behaviour *up to that
   exact snapshot* and do not reflect customer lifetime trajectories,
   churn timing, or post-period behaviour.
5. **No causal claims.** Cluster profiles describe *statistical
   associations* between RFM traits and group membership, not causal
   effects. The report uses hedged language ("is consistent with", "may
   reflect") and never attributes a customer's value *to* cluster
   membership.
