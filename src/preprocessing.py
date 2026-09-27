import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

logger = logging.getLogger(__name__)


ID_COLUMNS = {"InvoiceNo", "StockCode", "CustomerID"}


def ensure_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    logger.info(f"Ensuring numeric columns: {columns}")
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_datetime(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    logger.info(f"Ensuring datetime columns: {columns}")
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def ensure_string(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    logger.info(f"Ensuring string columns with consistent formatting: {columns}")
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def scale_numeric(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
    exclude_ids: bool = True,
) -> Dict[str, pd.DataFrame | object]:
    if exclude_ids:
        columns = [c for c in columns if c not in ID_COLUMNS]
    columns = [c for c in columns if c in df.columns]

    logger.info(f"Scaling {len(columns)} columns using {method} scaling")
    df = df.copy()
    scaled_df = df.copy()

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}. Use 'standard' or 'minmax'.")

    numeric = ensure_numeric(df, columns)
    scaled_values = scaler.fit_transform(numeric[columns].fillna(0))

    for i, col in enumerate(columns):
        scaled_df[f"{col}_Scaled"] = scaled_values[:, i]

    return {
        "dataframe": scaled_df,
        "scaler": scaler,
        "scaled_columns": columns,
    }


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "label",
    exclude_ids: bool = True,
) -> Dict[str, pd.DataFrame | Dict]:
    if exclude_ids:
        columns = [c for c in columns if c not in ID_COLUMNS]
    columns = [c for c in columns if c in df.columns]

    logger.info(f"Encoding {len(columns)} categorical columns using {method} encoding")
    df = df.copy()
    encoded_df = df.copy()
    encoders: Dict = {}

    for col in columns:
        if method == "label":
            le = LabelEncoder()
            encoded_df[f"{col}_Encoded"] = le.fit_transform(df[col].astype(str).fillna("MISSING"))
            encoders[col] = le
        elif method == "onehot":
            dummies = pd.get_dummies(df[col].astype(str).fillna("MISSING"), prefix=col, drop_first=False)
            encoded_df = pd.concat([encoded_df, dummies], axis=1)
            encoders[col] = list(dummies.columns)
        else:
            raise ValueError(f"Unknown encoding method: {method}. Use 'label' or 'onehot'.")

    return {
        "dataframe": encoded_df,
        "encoders": encoders,
        "encoded_columns": columns,
    }


def create_analytical_dataset(
    df_positive_sales: pd.DataFrame,
    perform_scaling: bool = True,
    perform_encoding: bool = False,
) -> Dict:
    logger.info("Building preprocessed analytical dataset")
    df = df_positive_sales.copy()

    df = ensure_numeric(df, ["Quantity", "UnitPrice", "TotalAmount", "CustomerID"])
    df = ensure_datetime(df, ["InvoiceDate"])
    df = ensure_string(df, ["InvoiceNo", "StockCode", "Description", "Country"])

    flag_cols_notes = (
        "Identifier columns (InvoiceNo, StockCode, CustomerID) are treated as keys and NOT encoded "
        "as ordinary categorical predictors, because each ID refers to a unique entity and encoding "
        "them would introduce spurious ordinal relationships."
    )
    logger.info(flag_cols_notes)

    result = {"base_dataset": df, "identifier_treatment_note": flag_cols_notes}

    if perform_scaling:
        scale_result = scale_numeric(
            df,
            columns=["Quantity", "UnitPrice", "TotalAmount", "Year", "Month", "Day", "Hour", "DayOfWeek"],
            method="standard",
        )
        result["scaled_dataset"] = scale_result["dataframe"]
        result["scaler"] = scale_result["scaler"]

    if perform_encoding:
        enc_result = encode_categorical(
            df,
            columns=["Country"],
            method="label",
        )
        result["encoded_dataset"] = enc_result["dataframe"]
        result["encoders"] = enc_result["encoders"]

    logger.info("Preprocessing complete")
    return result


def before_after_comparison(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    df_positive: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Generating before/after comparison table")

    def _metrics(name, df):
        qty = pd.to_numeric(df.get("Quantity", pd.Series(dtype=float)), errors="coerce")
        price = pd.to_numeric(df.get("UnitPrice", pd.Series(dtype=float)), errors="coerce")
        total = df.get("TotalAmount", qty * price)
        total = pd.to_numeric(total, errors="coerce")
        return {
            "Dataset": name,
            "Row Count": len(df),
            "Column Count": len(df.columns),
            "Total Missing Values": int(df.isna().sum().sum()),
            "Missing CustomerID": int(df.get("CustomerID", pd.Series(dtype=float)).isna().sum()),
            "Missing Description": int(df.get("Description", pd.Series(dtype=float)).isna().sum()),
            "Exact Duplicate Rows": int(df.duplicated().sum()),
            "Negative Quantity Count": int((qty < 0).sum()),
            "Zero Quantity Count": int((qty == 0).sum()),
            "UnitPrice <= 0 Count": int((price <= 0).sum()),
            "Cancellation Count": int(df.get("IsCancellation", df["InvoiceNo"].astype(str).str.startswith("C", na=False)).sum()) if "InvoiceNo" in df.columns else 0,
            "TotalAmount Mean": float(total.mean()) if total.notna().any() else np.nan,
            "TotalAmount Median": float(total.median()) if total.notna().any() else np.nan,
            "TotalAmount Std": float(total.std()) if total.notna().any() else np.nan,
            "TotalAmount Sum": float(total.sum()) if total.notna().any() else np.nan,
        }

    rows = [
        _metrics("Raw dataset", df_raw),
        _metrics("Cleaned (quality)", df_clean),
        _metrics("Positive-sales analytical", df_positive),
    ]
    comparison = pd.DataFrame(rows)
    return comparison
