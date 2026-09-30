# Week 2 — Exploratory Data Analysis & Visualization

Academic EDA project for the **UCI Online Retail Dataset**, built to satisfy a
Week 2 university assignment requiring Exploratory Data Analysis and
Visualization using **Pandas, NumPy, Matplotlib, and Seaborn**.

This project is deliberately distinct from Week 1's data-cleaning assignment:
Week 1 focused on data quality, cleansing decisions, and preprocessing outputs.
Week 2 focuses on **exploration, transformation, analysis, visualization,
interpretation, and insight-generation** — the EDA workflow proper.

All numerical results (percentages, correlations, rankings, counts, anomaly
rates, etc.) are computed from the actual UCI dataset loaded from Excel.
**Nothing is fabricated or approximated.**

---

## Dataset

- **Name:** UCI Online Retail
- **Source:** <https://archive.ics.uci.edu/dataset/352/online+retail>
- **Rows:** 541,909 transaction lines (8 columns)
- **Period:** 01 Dec 2010 → 09 Dec 2011
- **Retailer:** UK-based, non-store online retailer
- **File location in this project:** `data/Online+Retail.xlsx` (22.6 MB)

---

## Project Structure

```
online-retail-eda/
│
├── data/
│   ├── Online+Retail.xlsx          ← Raw UCI dataset (22.6 MB)
│   └── README.md                   ← Dataset docs, column dictionary
│
├── src/
│   ├── __init__.py                 ← Package metadata (v2.0.0)
│   ├── config.py                   ← All paths, column lists, constants
│   ├── data_loading.py             ← Excel load, column/file validation
│   ├── data_validation.py          ← Missing / types / duplicates / uniques
│   ├── data_preparation.py         ← TotalAmount, temporal features, 3 analytical datasets
│   ├── descriptive_analysis.py     ← Overview, categorical frequencies, stats tables
│   ├── temporal_analysis.py        ← Monthly volume/revenue, hourly, weekday
│   ├── product_analysis.py         ← Top products qty/rev, ranking overlap, Pareto
│   ├── geographic_analysis.py      ← Top countries by lines & revenue, ranking diff
│   ├── relationship_analysis.py   ← Correlation matrix + two scatter-plot studies
│   ├── anomaly_analysis.py         ← IQR summary, concrete-dataframe examples, tail percentiles
│   ├── visualization.py            ← 13+ publication-quality figures (Matplotlib/Seaborn, 200 DPI)
│   ├── report_generation.py        ← 12-section DOCX builder with embedded figures
│   └── main.py                     ← Full pipeline orchestrator
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb  ← Complete Jupyter walkthrough
│
├── outputs/
│   ├── figures/                   ← All PNG figures (200 DPI)
│   ├── tables/                    ← All CSV tables (≥21 descriptive tables)
│   └── analysis/                  ← Any serialized analytical outputs
│
├── report/
│   └── Week_2_EDA_and_Visualization_Report.docx   ← Final submission DOCX
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loading.py       ← File + column validation
│   ├── test_data_preparation.py   ← Feature engineering, analytical datasets
│   └── test_analysis.py           ← IQR, correlation, temporal, product aggregates
│
├── .gitignore
├── requirements.txt
├── LICENSE                        ← MIT
└── README.md
```

---

## 12 Required Visualizations (plus extras)

| # | Figure | Filename |
|---|--------|----------|
| 1 | Missing Values by Column | `01_missing_values.png` |
| 2 | Quantity Distribution (log1p) + Boxplot | `02_quantity_distribution.png`, `02b_quantity_boxplot.png` |
| 3 | Unit Price Distribution (log1p) + Boxplot | `03_unit_price_distribution.png`, `03b_unit_price_boxplot.png` |
| 4 | Monthly Transaction Volume (twin axis) | `04_monthly_transaction_volume.png` |
| 5 | Monthly Revenue + Avg Order Value | `05_monthly_revenue.png` |
| 6 | Top 10 Countries (horizontal bars) | `06_top_countries.png` |
| 7 | Top 10 Products by Quantity | `07_top_products_quantity.png` |
| 8 | Top 10 Products by Revenue | `08_top_products_revenue.png` |
| 9 | Transactions by Hour of Day | `09_transactions_by_hour.png` |
| 10 | Transactions by Day of Week (ordered Mon–Sun) | `10_transactions_by_weekday.png` |
| 11 | Quantity vs TotalAmount (P99 clipped, w/ regression) | `11_quantity_vs_totalamount.png` |
| 12 | Unit Price vs Quantity (P99 clipped, w/ regression) | `12_unit_price_vs_quantity.png` |
| 13 | Pearson Correlation Heatmap (annotated) | `13_correlation_heatmap.png` |
| + | TotalAmount Distribution + Boxplot | `14_totalamount_distribution.png`, `14b_totalamount_boxplot.png` |

---

## Running the Project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete EDA pipeline
python -m src.main
# This produces:
#   - All figures in outputs/figures/ (200 DPI PNGs)
#   - All CSV tables in outputs/tables/
#   - Week_2_EDA_and_Visualization_Report.docx in report/

# 3. Run the test suite
pytest tests/ -v

# 4. Open the walkthrough notebook
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

---

## Analyses Performed (each producing tables + interpretations in the DOCX)

1. **Descriptive** — rows, cols, memory, uniques, missing, duplicates, stats, categorical frequencies.
2. **Temporal** — monthly volume & revenue, hourly pattern, weekday pattern.
3. **Product** — top-10 quantity vs top-10 revenue, ranking overlap, Pareto concentration.
4. **Geographic** — top countries by lines & revenue, ranking comparison.
5. **Relationship / Correlation** — 3×3 Pearson matrix with heatmap; two scatter studies with clipped scales, sample sizes, and actual correlation values reported.
6. **Anomaly / Outlier** — IQR multipliers on positive-sales data; concrete *actual rows* from the dataset for largest-qty, highest-amount, cancellation, and zero-price examples; heavy-tail P99/P50 diagnostics.
7. **Interpretation** — every major figure is followed in the report by: **Observation / Evidence / Interpretation / Caution** — as required.

---

## DOCX Report Contents (`report/Week_2_EDA_and_Visualization_Report.docx`)

1. **Title Page**
2. §1 Introduction & Dataset Selection (with data-loading code snippet)
3. §2 Initial Exploratory Analysis & Basic Statistics (overview table, data types, missing values table + figure)
4. §3 Data Transformations & Feature Engineering (code snippets; 3 analytical datasets clearly defined)
5. §4 Numerical Distributions (Quantity / UnitPrice / TotalAmount stats + histograms + boxplots)
6. §5 Temporal Analysis (monthly volume, monthly revenue, hourly, weekday — all plotted and interpreted)
7. §6 Geographic Distribution (top-10 countries chart with UK-concentration interpretation)
8. §7 Product Analysis — Quantity vs Revenue rankings (two top-10 charts + overlap/comparison table + Pareto table)
9. §8 Correlations & Scatter Plots (correlation heatmap, Qty vs Amt, Price vs Qty — each with actual Pearson r)
10. §9 Anomaly Analysis (IQR table, tail diagnostics, 4 categories of **actual concrete rows** from the dataset)
11. §10 Limitations & Challenges (7 explicit limitations: single window, missing CustomerIDs, skewed data, Dec-2011 truncation, no causal ID, single retailer, description nulls)
12. §11 Key Findings & Conclusions (numbered list — every bullet cites a real computed number)
13. §12 References (6 academic / canonical references)

---

## Validation & Honesty Rules Strictly Enforced

The entire pipeline, DOCX builder, and notebook follow these rules from the
assignment specification:

- ✅ **Do not invent dataset values.** Every count, percentage, correlation,
  ranking, and anomaly count is computed from the actual loaded dataset.
- ✅ **Do not fabricate observations.** Text interpretations are written only
  after numerical evidence is computed.
- ✅ **Do not silently filter.** Every row exclusion (UnitPrice>0, positive
  sales, P99 clipping **for display only**, log1p scaling) is explicitly
  documented in code and in the report text.
- ✅ **Separate analytical datasets.** `full`, `valid_price`, `positive_sales`
  — never mixed without explanation.
- ✅ **No causation claims.** All causal language uses hedged wording
  ("may indicate", "could reflect", "is consistent with").
- ✅ **No misleading charts.** No deceptive axis limits; no invented data
  points; day-of-week charts use logical Mon→Sun order, not alphabetical.
- ✅ **Concrete real examples.** All "example rows" in §9 are actual rows
  sampled from the data, not synthetic.

---

## Tests Passed

```
tests/test_data_loading.py       – 5/5
tests/test_data_preparation.py   – 5/5
tests/test_analysis.py           – 7/7
                              ————
                              17/17  (pytest, no warnings)
```
