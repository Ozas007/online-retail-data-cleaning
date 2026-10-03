import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CANCELLATION_PREFIX,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEMPORAL_SPLIT_QUANTILE,
)
from src.feature_engineering import build_supervised_features

logger = logging.getLogger(__name__)


def _ensure_datetime(df: pd.DataFrame, col: str = "InvoiceDate") -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def derive_dates_from_df(
    df: pd.DataFrame,
    datetime_col: str = "InvoiceDate",
    split_quantile: float = TEMPORAL_SPLIT_QUANTILE,
) -> Dict[str, Any]:
    logger.info("Deriving temporal split dates from InvoiceDate distribution")
    np.random.seed(RANDOM_SEED)

    dt = _ensure_datetime(df, datetime_col)
    valid_mask = dt.notna()
    if valid_mask.sum() == 0:
        raise ValueError(f"No valid dates found in column '{datetime_col}'")

    dt_valid = dt[valid_mask]
    min_date = pd.Timestamp(dt_valid.min())
    max_date = pd.Timestamp(dt_valid.max())

    unique_sorted = np.sort(dt_valid.drop_duplicates().values.astype("datetime64[ns]"))
    if len(unique_sorted) < 2:
        raise ValueError("Not enough unique dates to compute a temporal split")

    q_idx = int(np.clip(np.floor(split_quantile * len(unique_sorted)), 0, len(unique_sorted) - 1))
    historical_end = pd.Timestamp(unique_sorted[q_idx])

    if historical_end <= min_date:
        historical_end = min_date + (max_date - min_date) * split_quantile
        historical_end = pd.Timestamp(historical_end)

    future_start = historical_end
    future_end = max_date

    dates = {
        "HISTORICAL_START": min_date,
        "HISTORICAL_END": historical_end,
        "FUTURE_START": future_start,
        "FUTURE_END": future_end,
        "days_historical": (historical_end - min_date).days,
        "days_future": (future_end - future_start).days,
    }
    logger.info(
        f"  Historical: {min_date.date()} -> {historical_end.date()} ({dates['days_historical']} d)  "
        f"Future: {future_start.date()} -> {future_end.date()} ({dates['days_future']} d)"
    )
    return dates


def build_target(
    df_positive_sales: pd.DataFrame,
    historical_end: pd.Timestamp,
    future_start: pd.Timestamp,
    future_end: pd.Timestamp,
    datetime_col: str = "InvoiceDate",
) -> pd.DataFrame:
    logger.info("Building customer-level FuturePurchased target")
    np.random.seed(RANDOM_SEED)

    historical_end = pd.Timestamp(historical_end)
    future_start = pd.Timestamp(future_start)
    future_end = pd.Timestamp(future_end)

    df = df_positive_sales.copy()
    df["_InvoiceNo_str"] = df["InvoiceNo"].astype(str)
    df["_IsCancellation"] = df["_InvoiceNo_str"].str.startswith(CANCELLATION_PREFIX, na=False)
    df["_Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["_UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    mask = (
        (~df["_IsCancellation"])
        & (df["_Quantity"] > 0)
        & (df["_UnitPrice"] > 0)
        & (df[ID_COLUMN].notna())
    )
    df = df.loc[mask].copy()
    df.drop(columns=["_InvoiceNo_str", "_IsCancellation", "_Quantity", "_UnitPrice"], inplace=True)

    df["_dt"] = _ensure_datetime(df, datetime_col)
    df = df[df["_dt"].notna()].copy()

    historical_mask = df["_dt"] < historical_end
    future_mask = (df["_dt"] >= future_start) & (df["_dt"] <= future_end)

    hist_customers = df.loc[historical_mask, ID_COLUMN].dropna().astype(float)
    future_customers = df.loc[future_mask, ID_COLUMN].dropna().astype(float)

    hist_unique = pd.Series(hist_customers.unique(), name=ID_COLUMN)
    future_unique_set = set(future_customers.unique())

    if hist_unique.empty:
        raise ValueError("No customers found in historical window — cannot build target")

    target_df = pd.DataFrame({ID_COLUMN: hist_unique.values})
    target_df[TARGET_COLUMN] = target_df[ID_COLUMN].isin(future_unique_set).astype(int)

    n_total = len(target_df)
    n_pos = int(target_df[TARGET_COLUMN].sum())
    pct_pos = n_pos / n_total * 100.0 if n_total else 0.0
    logger.info(f"  Target customers: {n_total:,}  |  Positive: {n_pos:,} ({pct_pos:.1f}%)")

    target_df = target_df.sort_values(ID_COLUMN).reset_index(drop=True)
    return target_df


def _get_preprocessor(numeric_in: List[str], categorical_in: List[str]) -> ColumnTransformer:
    transformers = []
    if numeric_in:
        transformers.append(("num", StandardScaler(), numeric_in))
    if categorical_in:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_in,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_pipeline(clf, preprocessor) -> Pipeline:
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def _create_rf():
    np.random.seed(RANDOM_SEED)
    return RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def _create_lr():
    np.random.seed(RANDOM_SEED)
    return LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear",
        class_weight="balanced", random_state=RANDOM_SEED, max_iter=2000,
    )


def _evaluate(y_true, y_pred, y_proba) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
    }
    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        if proba.shape[1] >= 2:
            proba_use = proba[:, 1]
        else:
            proba_use = proba[:, 0]
    else:
        proba_use = proba
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, proba_use))
    except ValueError as e:
        logger.warning(f"ROC-AUC issue: {e}")
        metrics["roc_auc"] = float("nan")
    return metrics


def _cm_df(y_true, y_pred) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    labels = sorted(np.unique(np.concatenate([y_true_arr, y_pred_arr])).tolist())
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    row_index = pd.MultiIndex.from_tuples(
        [("Actual", int(lbl)) for lbl in labels], names=["Condition", "Label"],
    )
    col_index = pd.MultiIndex.from_tuples(
        [("Predicted", int(lbl)) for lbl in labels], names=["Prediction", "Label"],
    )
    return pd.DataFrame(cm, index=row_index, columns=col_index)


def _run_cv(X, y, pipeline, cv=5) -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    y_arr = np.asarray(y).astype(int)
    results = cross_validate(pipeline, X, y_arr, cv=skf, scoring=scoring, return_train_score=False, error_score="raise")
    fold_df = pd.DataFrame()
    for key in scoring:
        short = key.replace("_macro", "")
        fold_df[short] = results[f"test_{key}"]
    fold_df.insert(0, "fold", np.arange(1, len(fold_df) + 1))
    summary = {"fold": "overall"}
    for col in [c for c in fold_df.columns if c != "fold"]:
        vals = fold_df[col].values
        summary[f"{col}_mean"] = float(np.mean(vals))
        summary[f"{col}_std"] = float(np.std(vals, ddof=0))
    summary_df = pd.DataFrame([summary])
    return pd.concat([fold_df, summary_df], ignore_index=True)


def _extract_feature_names(pipeline, X_sample) -> List[str]:
    try:
        numeric_in = [c for c in NUMERIC_FEATURES if c in X_sample.columns]
        categorical_in = [c for c in CATEGORICAL_FEATURES if c in X_sample.columns]
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            ohe = preprocessor.named_transformers_["cat"]
            if hasattr(ohe, "get_feature_names_out"):
                cat_names = list(ohe.get_feature_names_out(categorical_in))
            else:
                cat_names = []
        except Exception:
            cat_names = []
        names = numeric_in + cat_names
        if len(names) == 0:
            names = [f"f{i}" for i in range(len(X_sample.columns))]
        return names
    except Exception:
        return [f"f{i}" for i in range(len(X_sample.columns))]


def run_supervised_pipeline(df_positive_sales: pd.DataFrame) -> Dict:
    logger.info("Starting supervised learning pipeline")
    np.random.seed(RANDOM_SEED)

    temporal_dates = derive_dates_from_df(df_positive_sales)
    HISTORICAL_START = temporal_dates["HISTORICAL_START"]
    HISTORICAL_END = temporal_dates["HISTORICAL_END"]
    FUTURE_START = temporal_dates["FUTURE_START"]
    FUTURE_END = temporal_dates["FUTURE_END"]

    target_df = build_target(df_positive_sales, HISTORICAL_END, FUTURE_START, FUTURE_END)

    historical_mask = pd.to_datetime(df_positive_sales["InvoiceDate"], errors="coerce") < HISTORICAL_END
    df_hist = df_positive_sales.loc[historical_mask].copy()
    logger.info(f"Historical-only transactions (pre-filtered): {len(df_hist):,}")
    features_df = build_supervised_features(df_hist, HISTORICAL_END, HISTORICAL_START)

    merged = features_df.merge(target_df, on=ID_COLUMN, how="inner")
    logger.info(f"Merged design matrix: {len(merged):,} rows")

    feature_cols_used = [c for c in FEATURE_COLUMNS if c in merged.columns]
    X = merged[[ID_COLUMN] + feature_cols_used].copy()
    y = merged[TARGET_COLUMN].astype(int).values
    X_model = X[feature_cols_used].copy()
    X_id = X[[ID_COLUMN]].copy()

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X_model, y, X_id[ID_COLUMN].values,
        test_size=0.30, random_state=RANDOM_SEED, stratify=y,
    )
    n_train, n_test = len(y_train), len(y_test)
    logger.info(f"Train: {n_train:,}  |  Test: {n_test:,}  |  Train pos rate: {y_train.mean():.3f}  |  Test pos rate: {y_test.mean():.3f}")

    numeric_in = [c for c in NUMERIC_FEATURES if c in X_model.columns]
    categorical_in = [c for c in CATEGORICAL_FEATURES if c in X_model.columns]
    preprocessor = _get_preprocessor(numeric_in, categorical_in)

    models = [
        ("Random Forest", _create_rf, "rf"),
        ("Logistic Regression", _create_lr, "lr"),
    ]

    fitted: Dict[str, Pipeline] = {}
    test_metrics_per_model: Dict[str, Dict[str, float]] = {}
    cm_per_model: Dict[str, pd.DataFrame] = {}
    models_results_for_viz: Dict[str, Dict[str, Any]] = {}
    feature_names_per_model: Dict[str, List[str]] = {}
    all_y_proba_test: Dict[str, np.ndarray] = {}

    for model_name, factory, _short in models:
        logger.info(f"Training {model_name}")
        clf = factory()
        pipe = _build_pipeline(clf, preprocessor)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        metrics = _evaluate(y_test, y_pred, y_proba)
        cm = _cm_df(y_test, y_pred)
        fitted[model_name] = pipe
        test_metrics_per_model[model_name] = metrics
        cm_per_model[model_name] = cm
        feature_names_per_model[model_name] = _extract_feature_names(pipe, X_train)
        models_results_for_viz[model_name] = {"y_true": y_test, "y_pred": y_pred, "y_proba": y_proba}
        yp = np.asarray(y_proba)
        if yp.ndim == 2:
            if yp.shape[1] >= 2:
                all_y_proba_test[model_name] = yp[:, 1]
            else:
                all_y_proba_test[model_name] = yp[:, 0]
        else:
            all_y_proba_test[model_name] = yp
        logger.info(f"  {model_name}: acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

    rf_name = "Random Forest"
    lr_name = "Logistic Regression"
    best_model_name = rf_name if test_metrics_per_model[rf_name]["roc_auc"] >= test_metrics_per_model[lr_name]["roc_auc"] else lr_name
    logger.info(f"Best model by ROC-AUC: {best_model_name}")

    cv_results_list: List[pd.DataFrame] = []
    for model_name, factory, short in models:
        cv_pre = _get_preprocessor(numeric_in, categorical_in)
        cv_pipe = _build_pipeline(factory(), cv_pre)
        cv_df = _run_cv(X_model, y, cv_pipe, cv=5)
        cv_df["model"] = model_name
        cv_results_list.append(cv_df)
    cv_results_df = pd.concat(cv_results_list, ignore_index=True)

    feature_importance_df = pd.DataFrame()
    if rf_name in fitted:
        try:
            rf_clf = fitted[rf_name].named_steps["classifier"]
            names = feature_names_per_model.get(rf_name, [f"f{i}" for i in range(len(rf_clf.feature_importances_))])
            fi = sorted(list(zip(names, rf_clf.feature_importances_.tolist())), key=lambda t: -t[1])
            feature_importance_df = pd.DataFrame(fi, columns=["feature", "importance"])
        except Exception as e:
            logger.warning(f"Could not extract RF feature importance: {e}")

    confusion_matrix_df = cm_per_model.get(best_model_name, cm_per_model.get(rf_name, pd.DataFrame()))

    y_proba_test = all_y_proba_test.get(best_model_name, np.zeros_like(y_test, dtype=float))

    test_metrics_dict = test_metrics_per_model.get(best_model_name, {})

    return {
        "best_model_name": best_model_name,
        "test_metrics_dict": test_metrics_dict,
        "cv_results_df": cv_results_df,
        "feature_importance_df": feature_importance_df,
        "confusion_matrix_df": confusion_matrix_df,
        "y_test": y_test,
        "y_proba_test": y_proba_test,
        "temporal_dates_dict": temporal_dates,
        "target_df": target_df,
        "features_df": features_df,
        "merged_df": merged,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "id_test": id_test,
        "fitted_pipelines": fitted,
        "test_metrics_per_model": test_metrics_per_model,
        "cm_per_model": cm_per_model,
        "models_results_for_viz": models_results_for_viz,
        "feature_names_per_model": feature_names_per_model,
        "numeric_in": numeric_in,
        "categorical_in": categorical_in,
        "n_train": n_train,
        "n_test": n_test,
    }
