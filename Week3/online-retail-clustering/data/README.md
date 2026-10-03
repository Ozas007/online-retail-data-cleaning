# Raw Dataset Directory

Place the UCI Online Retail Excel dataset in this directory.

Expected filename: `Online+Retail.xlsx`

Official source:
- https://archive.ics.uci.edu/dataset/352/online+retail

Alternative (official mirror):
- https://uci-ics-mlr-prod.aws.uci.edu/dataset/352/online%2Bretail

Dataset citation:
> Dua, D. and Graff, C. (2019). UCI Machine Learning Repository.
> Irvine, CA: University of California, School of Information and Computer Science.

---

## Why this file is not committed to Git

The raw dataset file (`Online+Retail.xlsx`) is approximately 23 MB uncompressed and
contains proprietary transaction data from the UCI ML Repository. Per repository
best practices, large binary files are excluded via `.gitignore` to keep clones
fast and repository size manageable. You must download the file once and place it
in this folder before running the pipeline.

---

## Expected dataset shape

- **Rows:** 541,909 transaction line-items (± rounding; exact count may vary by file revision)
- **Columns:** 8
- **InvoiceDate range:** 2010-12-01 to 2011-12-09 (inclusive)
- **Unique customers (non-null):** ~4,372
- **Cancellation convention:** `InvoiceNo` values **starting with `'C'`** represent cancellation/refund transactions and carry negative `Quantity`.

The data spreadsheet contains the following 8 columns (per UCI documentation):

| Column        | Type    | Description |
| ------------- | ------- | ----------- |
| InvoiceNo     | Nominal | 6-digit integral number uniquely assigned to each transaction. Prefix `C` = cancellation. |
| StockCode     | Nominal | 5-digit integral number uniquely assigned to each distinct product. |
| Description   | Nominal | Product (item) name. |
| Quantity      | Numeric | The quantities of each product per transaction line. Cancels are negative. |
| InvoiceDate   | Numeric | Invoice date and time (day + time when each transaction was generated). |
| UnitPrice     | Numeric | Product price per unit in sterling (£). |
| CustomerID    | Nominal | 5-digit integral number uniquely assigned to each customer. **Nullable — ~25% missing.** |
| Country       | Nominal | Name of the country where each customer resides. |

---

## Quick download

From a shell in the project root (requires `curl`):

```bash
curl -L "https://archive.ics.uci.edu/static/public/352/online+retail.zip" \
  -o data/online+retail.zip && \
  unzip -j data/online+retail.zip -d data/
```

Then verify the expected file is present:

```
data/Online+Retail.xlsx
```
