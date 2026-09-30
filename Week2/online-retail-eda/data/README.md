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

The data spreadsheet contains the following 8 columns (per UCI documentation):

| Column        | Type    | Description |
| ------------- | ------- | ----------- |
| InvoiceNo     | Nominal | 6-digit integral number uniquely assigned to each transaction. Prefix 'C' = cancellation. |
| StockCode     | Nominal | 5-digit integral number uniquely assigned to each distinct product. |
| Description   | Nominal | Product (item) name. |
| Quantity      | Numeric | The quantities of each product per transaction. |
| InvoiceDate   | Numeric | Invoice date and time (day + time when each transaction was generated). |
| UnitPrice     | Numeric | Product price per unit in sterling (£). |
| CustomerID    | Nominal | 5-digit integral number uniquely assigned to each customer. |
| Country       | Nominal | Name of the country where each customer resides. |
