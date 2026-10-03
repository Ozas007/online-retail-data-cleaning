# Data Directory — Week 4 Supervised Learning

## Expected Dataset File

Place the raw UCI Excel file here (or rely on the parent-repo fallback):

- Expected filename: `Online+Retail.xlsx`
- Preferred fallback location (auto-used): `E:/project/online-retail-data-cleaning/data/Online+Retail.xlsx`

## Official UCI Source

- Dataset name: **Online Retail Data Set**
- UCI ML Repository page: https://archive.ics.uci.edu/ml/datasets/Online+Retail
- Direct download (XLSX): https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
- Hosted by: Dr. Daqing Chen, Brunel University (UK)
- License / citation: Chen, D., Sain, S. L., & Guo, K. (2012). *Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining*. Journal of Database Marketing & Customer Strategy Management, 19(3), 197-208. Dua, D., & Graff, C. (2019). UCI Machine Learning Repository.

## NOT Committed to Version Control

The raw Excel file is **not** committed to this repository (see `.gitignore` → `data/*.xlsx`). To reproduce the full pipeline:

1. Download from the UCI link above.
2. Save as exactly `Online+Retail.xlsx` into this directory, or into the parent repo `E:/project/online-retail-data-cleaning/data/`.
3. Run `python -m src.main` from the project root.

The `src/config.py` module automatically falls back to the parent repo's `data/` directory if the local copy is absent.
