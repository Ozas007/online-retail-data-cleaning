import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.config import EXPECTED_COLUMNS, RAW_DATA_PATH

logger = logging.getLogger(__name__)


class DatasetNotFoundError(FileNotFoundError):
    pass


class MissingColumnsError(ValueError):
    pass


def validate_file_exists(file_path: Path) -> bool:
    if not file_path.exists():
        raise DatasetNotFoundError(
            f"Dataset file not found at: {file_path.resolve()}. "
            "Please place the Online_Retail.xlsx file in the data/ directory "
            "or in the parent repo's data/ directory."
        )
    if not file_path.is_file():
        raise DatasetNotFoundError(f"Path exists but is not a file: {file_path.resolve()}")
    logger.info(f"Dataset file found at: {file_path.resolve()}")
    return True


def validate_required_columns(df: pd.DataFrame, expected_columns: list = EXPECTED_COLUMNS) -> Tuple[bool, list]:
    actual_columns = list(df.columns)
    missing_cols = [col for col in expected_columns if col not in actual_columns]
    extra_cols = [col for col in actual_columns if col not in expected_columns]

    if missing_cols:
        raise MissingColumnsError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Found columns: {actual_columns}"
        )
    if extra_cols:
        logger.warning(f"Dataset contains extra columns not in expected list: {extra_cols}")

    logger.info(f"All {len(expected_columns)} required columns are present")
    return True, extra_cols


def load_excel_dataset(
    file_path: Path = RAW_DATA_PATH,
    sheet_name: str | int = 0,
    validate: bool = True,
) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {file_path}")
    validate_file_exists(file_path)

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
        logger.info(f"Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Failed to load Excel file: {str(e)}")
        raise RuntimeError(f"Failed to load Excel file: {str(e)}") from e

    if validate:
        validate_required_columns(df)

    return df.copy()


def get_dataset_shape(df: pd.DataFrame) -> Tuple[int, int]:
    rows, cols = df.shape
    logger.info(f"Dataset shape: {rows} rows x {cols} columns")
    return rows, cols
