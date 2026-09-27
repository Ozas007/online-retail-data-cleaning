# Raw Dataset Directory

This folder is intended to contain the raw source dataset for the project.

## Expected File

| File | Source | Notes |
|---|---|---|
| `Online+Retail.xlsx` | UCI Machine Learning Repository | Original, unmodified Excel file (~22.6 MB) |

## Dataset Location

**Download URL:** https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx

**UCI Dataset Page:** https://archive.ics.uci.edu/ml/datasets/online+retail

**How to obtain the file:**

```bash
cd data/
# Option 1 — curl
curl -L -o "Online+Retail.xlsx" "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

# Option 2 — wget
wget -O "Online+Retail.xlsx" "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

# Option 3 — Python
python -c "import urllib.request; urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx', 'Online+Retail.xlsx')"
```

## Why is the Excel file not in Git?

The dataset is ~22.6 MB. Hosting it on GitHub would:

1. Exceed the GitHub repository file-size recommendation for long-term maintainability.
2. Potentially conflict with UCI's redistribution preferences (the dataset is hosted by UCI for direct download).

**The repository is designed to work after cloning:** a user only has to place the `Online+Retail.xlsx` file in this directory before running the pipeline. The loading module (`src/data_loading.py`) validates the file existence and fails loudly with a clear error message if it is missing.

## Dataset Description

* **Domain:** E-commerce transactions from a UK-based registered non-store online gift retailer.
* **Format:** Microsoft Excel (.xlsx), single sheet.
* **Expected rows:** 541,909
* **Expected columns:** 8 — `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.
* **Date range:** 2010-12-01 to 2011-12-09 (inclusive).
* **Conventions:** Invoice numbers starting with the letter `C` denote **cancellations / returns**. `CustomerID` is a nullable float because the Excel source stores missing values as empty cells.

## References

Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197-208. https://doi.org/10.1057/dbm.2012.17

Dua, D., & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences. https://archive.ics.uci.edu/ml
