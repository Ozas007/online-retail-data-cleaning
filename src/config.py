from pathlib import Path
from typing import List

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "Online+Retail.xlsx"
DATA_README: Path = DATA_DIR / "README.md"

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
TABLES_DIR: Path = OUTPUTS_DIR / "tables"
CLEANED_DATA_DIR: Path = OUTPUTS_DIR / "cleaned_data"

REPORT_DIR: Path = PROJECT_ROOT / "report"
REPORT_DOCX: Path = REPORT_DIR / "Data_Cleaning_and_Preprocessing_Report.docx"

CLEANED_DATA_CSV: Path = CLEANED_DATA_DIR / "online_retail_cleaned.csv"

EXPECTED_COLUMNS: List[str] = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

NUMERIC_COLUMNS: List[str] = ["Quantity", "UnitPrice", "CustomerID"]
CATEGORICAL_COLUMNS: List[str] = ["InvoiceNo", "StockCode", "Description", "Country"]
DATETIME_COLUMNS: List[str] = ["InvoiceDate"]

RANDOM_SEED: int = 42

CANCELLATION_PREFIX: str = "C"

IQR_MULTIPLIER: float = 1.5

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
