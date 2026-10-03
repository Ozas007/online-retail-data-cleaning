from pathlib import Path
from typing import List

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "Online+Retail.xlsx"

_PARENT_REPO_ROOT: Path = PROJECT_ROOT.parent.parent
_PARENT_DATA_DIR: Path = _PARENT_REPO_ROOT / "data"
_PARENT_RAW_DATA_PATH: Path = _PARENT_DATA_DIR / "Online+Retail.xlsx"

if not RAW_DATA_PATH.exists() and _PARENT_RAW_DATA_PATH.exists():
    RAW_DATA_PATH = _PARENT_RAW_DATA_PATH
    DATA_DIR = _PARENT_DATA_DIR

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
TABLES_DIR: Path = OUTPUTS_DIR / "tables"

REPORT_DIR: Path = PROJECT_ROOT / "report"
REPORT_DOCX: Path = REPORT_DIR / "Week_4_Supervised_Learning_Report.docx"

EXPECTED_COLUMNS: List[str] = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]

NUMERIC_COLUMNS: List[str] = ["Quantity", "UnitPrice", "CustomerID"]
CATEGORICAL_COLUMNS: List[str] = ["InvoiceNo", "StockCode", "Description", "Country"]
DATETIME_COLUMNS: List[str] = ["InvoiceDate"]

RANDOM_SEED: int = 42
CANCELLATION_PREFIX: str = "C"
FIGURE_DPI: int = 200

PREDICTION_WINDOW_MONTHS: int = 3
TEMPORAL_SPLIT_QUANTILE: float = 0.75

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

NUMERIC_FEATURES: List[str] = [
    "Recency", "Frequency", "Monetary", "AOV",
    "UniqueProducts", "AvgQuantity", "TotalQuantity", "PurchaseFrequency",
]

CATEGORICAL_FEATURES: List[str] = ["Country"]

FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN: str = "FuturePurchased"
ID_COLUMN: str = "CustomerID"
