import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CANCELLATION_PREFIX,
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    ID_COLUMN,
    LOG_FORMAT,
    NUMERIC_FEATURES,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DATA_PATH,
    REPORT_DOCX,
    REPORT_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
)
from src.data_loading import get_dataset_shape, load_excel_dataset
from src.data_validation import run_full_validation
from src.evaluation import (
    build_confusion_matrix_df,
    evaluate_classifier,
    run_cross_validation,
)
from src.feature_engineering import build_features
from src.models import (
    create_decision_tree,
    create_logistic_regression,
    create_random_forest,
)
from src.preprocessing import build_full_pipeline, get_preprocessor
from src.report_generation import build_report
from src.target_engineering import build_target, derive_dates_from_df
from src.visualization import run_all_visualizations

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "pipeline.log", mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )


def _ensure_directories() -> None:
    for d in (FIGURES_DIR, TABLES_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output directories exist under {PROJECT_ROOT}")


def _prepare_positive_sales(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replicate Week3 data_preparation logic inline (TotalAmount, temporal features,
    cancellation flags, dedup, positive-sales filter, non-null CustomerID filter).
    Returns (df_engineered_with_all_cols, positive_sales_with_customer_id).
    """
    logger.info("Preparing positive-sales analytical dataset")
    df = df_raw.copy()

    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["TotalAmount"] = qty * price

    dt = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Hour"] = dt.dt.hour
    df["DOW"] = dt.dt.dayofweek
    df["DOWName"] = dt.dt.day_name()
    df["YearMonth"] = dt.dt.to_period("M").astype(str)
    df["Date"] = dt.dt.date

    invoices = df["InvoiceNo"].astype(str)
    df["IsCancellation"] = invoices.str.startswith(CANCELLATION_PREFIX, na=False)

    before = len(df)
    df = df.drop_duplicates(keep="first").copy()
    logger.info(f"  Deduplicated rows: before={before:,}  after={len(df):,}")

    valid_price = pd.to_numeric(df["UnitPrice"], errors="coerce") > 0
    qty_pos = pd.to_numeric(df["Quantity"], errors="coerce") > 0
    not_cancel = ~df["IsCancellation"]
    positive_sales = df.loc[valid_price & not_cancel & qty_pos].copy()
    logger.info(f"  Positive sales (non-cancel, Qty>0, Price>0): {len(positive_sales):,}")

    positive_sales_cid = positive_sales.dropna(subset=[ID_COLUMN]).copy()
    logger.info(
        f"  With non-null CustomerID: {len(positive_sales_cid):,}  "
        f"(dropped {len(positive_sales) - len(positive_sales_cid):,} anonymous rows)"
    )
    return df, positive_sales_cid


def _merge_features_and_target(
    features_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Merging features and target on CustomerID")
    merged = features_df.merge(target_df, on=ID_COLUMN, how="inner")
    n_diff = len(features_df) - len(merged)
    if n_diff > 0:
        logger.warning(
            f"{n_diff} customers in features but not in target (should be 0 by design) — inner join dropped them."
        )
    logger.info(f"  Merged design matrix: {len(merged):,} rows × {len(merged.columns)} cols")
    return merged


def _extract_feature_names_from_pipeline(
    pipeline,
    X_fit_sample: pd.DataFrame,
) -> list:
    """
    Recover post-preprocessing feature names so that RF importances and LR coefficients
    can be labelled with the actual encoded column names.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        numeric_features = list(NUMERIC_FEATURES)
        cat_features = list(CATEGORICAL_FEATURES)
        try:
            ohe = preprocessor.named_transformers_["cat"]
            if hasattr(ohe, "get_feature_names_out"):
                cat_names = list(ohe.get_feature_names_out(cat_features))
            else:
                cat_names = []
        except Exception:
            cat_names = []
        names = numeric_features + cat_names
        if len(names) == 0:
            names = [f"f{i}" for i in range(len(X_fit_sample.columns))]
        return names
    except Exception:
        return [f"f{i}" for i in range(len(X_fit_sample.columns))]


def _train_and_evaluate(
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    preprocessor,
    model_factory,
    model_name: str,
) -> Tuple[Any, Dict[str, float], pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    logger.info(f"Training and evaluating {model_name}")
    clf = model_factory()
    pipeline = build_full_pipeline(clf, preprocessor)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    eval_metrics = evaluate_classifier(y_test, y_pred, y_proba)
    cm_df = build_confusion_matrix_df(y_test, y_pred)
    return pipeline, eval_metrics, cm_df, y_pred, y_proba, _extract_feature_names_from_pipeline(pipeline, X_train)


def _save_tables(
    *,
    validation: Dict[str, Any],
    target_df: pd.DataFrame,
    features_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    test_eval: Dict[str, Dict[str, float]],
    cv_results: Dict[str, pd.DataFrame],
    predictions_df: pd.DataFrame,
    feature_importance_df: Optional[pd.DataFrame],
    lr_coefficients_df: Optional[pd.DataFrame],
) -> Dict[str, Path]:
    saved: Dict[str, Path] = {}

    def _save(df: pd.DataFrame, fname: str) -> Path:
        p = TABLES_DIR / fname
        df.to_csv(p, index=False)
        saved[fname] = p
        return p

    if "missing_values" in validation:
        _save(validation["missing_values"], "00_validation_missing_values.csv")
    if "data_types" in validation:
        _save(validation["data_types"], "00_validation_data_types.csv")

    _save(target_df, "01_target_customer_level.csv")
    _save(features_df, "02_features_customer_level.csv")
    _save(merged_df, "03_merged_design_matrix_X_y.csv")

    eval_rows = []
    for model_name, md in test_eval.items():
        row = {"model": model_name}
        row.update(md)
        eval_rows.append(row)
    _save(pd.DataFrame(eval_rows), "04_holdout_metrics_all_models.csv")

    for model_name, cv_df in cv_results.items():
        safe = model_name.replace("/", "_").replace(" ", "_")
        _save(cv_df, f"05_cv_{safe}.csv")

    _save(predictions_df, "06_test_set_predictions_all_models.csv")

    if feature_importance_df is not None and not feature_importance_df.empty:
        _save(feature_importance_df, "07_feature_importance_random_forest.csv")
    if lr_coefficients_df is not None and not lr_coefficients_df.empty:
        _save(lr_coefficients_df, "08_logistic_regression_coefficients.csv")

    logger.info(f"Saved {len(saved)} CSV tables to {TABLES_DIR}")
    return saved


def main() -> Path:
    _setup_logging()
    _ensure_directories()

    logger.info("=" * 70)
    logger.info("WEEK 4 — SUPERVISED LEARNING: CUSTOMER FUTURE-PURCHASE PREDICTION")
    logger.info("=" * 70)
    logger.info(f"PROJECT_ROOT = {PROJECT_ROOT}")
    logger.info(f"RAW_DATA_PATH = {RAW_DATA_PATH.resolve()}")
    logger.info(f"Random seed = {RANDOM_SEED}")

    np.random.seed(RANDOM_SEED)

    # Step 1+2: load + validate raw dataset
    logger.info("STEP 1-2: Load and validate raw dataset")
    df_raw = load_excel_dataset(file_path=RAW_DATA_PATH)
    n_raw_rows, n_raw_cols = get_dataset_shape(df_raw)
    validation = run_full_validation(df_raw)

    # Step 3: prepare positive-sales dataset
    logger.info("STEP 3: Prepare analytical dataset (TotalAmount, cancellation, positive-sales, CID filter)")
    df_engineered, positive_sales_cid = _prepare_positive_sales(df_raw)
    n_positive_sales_rows = len(positive_sales_cid)
    logger.info(
        f"Raw rows: {n_raw_rows:,}  |  Positive sales with CustomerID: {n_positive_sales_rows:,}  "
        f"({n_positive_sales_rows / n_raw_rows * 100:.1f}%)"
    )

    # Step 4: derive temporal dates
    logger.info("STEP 4: Derive temporal split dates from InvoiceDate distribution")
    temporal_dates = derive_dates_from_df(positive_sales_cid)
    HISTORICAL_START = temporal_dates["HISTORICAL_START"]
    HISTORICAL_END = temporal_dates["HISTORICAL_END"]
    FUTURE_START = temporal_dates["FUTURE_START"]
    FUTURE_END = temporal_dates["FUTURE_END"]

    # Step 5: build target
    logger.info("STEP 5: Build FuturePurchased target")
    target_df = build_target(
        positive_sales_cid,
        historical_end=HISTORICAL_END,
        future_start=FUTURE_START,
        future_end=FUTURE_END,
    )

    # Step 6: build features — historical transactions only
    logger.info("STEP 6: Build customer-level features (strictly historical window only)")
    historical_mask = pd.to_datetime(positive_sales_cid["InvoiceDate"], errors="coerce") < HISTORICAL_END
    df_hist_transactions = positive_sales_cid.loc[historical_mask].copy()
    logger.info(f"  Historical-only transactions (pre-filtered): {len(df_hist_transactions):,}")
    features_df = build_features(
        df_hist_transactions,
        historical_end=HISTORICAL_END,
        historical_start=HISTORICAL_START,
    )
    n_feature_customers = len(features_df)

    # Step 7: merge X + y
    logger.info("STEP 7: Inner-merge features and target on CustomerID")
    merged_df = _merge_features_and_target(features_df, target_df)

    X = merged_df[[c for c in merged_df.columns if c != TARGET_COLUMN]].copy()
    y = merged_df[TARGET_COLUMN].astype(int).values
    feature_cols_used = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in X.columns]
    X_model = X[feature_cols_used].copy()
    X_id = X[[ID_COLUMN]].copy()
    logger.info(f"  Final design matrix: X = {X_model.shape[0]:,} × {X_model.shape[1]}  |  y={y.shape}")

    # Step 8: train/test split (70/30, post-temporal-split so no leakage possible)
    logger.info("STEP 8: 70/30 train/test split on already-temporally-safe rows")
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X_model, y, X_id[ID_COLUMN].values,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    n_train, n_test = len(y_train), len(y_test)
    logger.info(f"  Train: {n_train:,}  |  Test: {n_test:,}  |  Train pos rate: {y_train.mean():.3f}  |  Test pos rate: {y_test.mean():.3f}")

    # Step 9: create shared preprocessor + 3 pipelines
    logger.info("STEP 9: Create preprocessor + 3 model pipelines")
    numeric_in = [c for c in NUMERIC_FEATURES if c in X_model.columns]
    categorical_in = [c for c in CATEGORICAL_FEATURES if c in X_model.columns]
    preprocessor = get_preprocessor(numeric_features=numeric_in, categorical_features=categorical_in)

    # Step 10+11+12: fit each pipeline on train, predict on test, evaluate
    logger.info("STEP 10-12: Fit 3 models, predict test, evaluate")
    model_factories = [
        ("Logistic Regression", create_logistic_regression),
        ("Decision Tree", create_decision_tree),
        ("Random Forest", create_random_forest),
    ]

    fitted_pipelines: Dict[str, Any] = {}
    test_eval: Dict[str, Dict[str, float]] = {}
    confusion_matrices: Dict[str, pd.DataFrame] = {}
    models_results_for_viz: Dict[str, Dict[str, Any]] = {}
    predictions_df = pd.DataFrame({ID_COLUMN: id_test, "y_true": y_test})
    feature_names_by_model: Dict[str, list] = {}

    for model_name, factory in model_factories:
        pipeline, metrics, cm_df, y_pred, y_proba, feature_names = _train_and_evaluate(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            preprocessor=preprocessor,
            model_factory=factory,
            model_name=model_name,
        )
        fitted_pipelines[model_name] = pipeline
        test_eval[model_name] = metrics
        confusion_matrices[model_name] = cm_df
        feature_names_by_model[model_name] = feature_names
        models_results_for_viz[model_name] = {"y_true": y_test, "y_pred": y_pred, "y_proba": y_proba}

        safe = model_name.replace("/", "_").replace(" ", "_")
        predictions_df[f"y_pred_{safe}"] = y_pred
        if y_proba.ndim == 2:
            predictions_df[f"y_proba_{safe}_class1"] = y_proba[:, 1]
        else:
            predictions_df[f"y_proba_{safe}_class1"] = y_proba

    # Step 13: 5-fold stratified CV for each model
    logger.info("STEP 13: Stratified 5-fold CV (refit preprocessor per fold)")
    cv_results: Dict[str, pd.DataFrame] = {}
    for model_name, factory in model_factories:
        clf_cv = factory()
        cv_preprocessor = get_preprocessor(numeric_features=numeric_in, categorical_features=categorical_in)
        cv_pipeline = build_full_pipeline(clf_cv, cv_preprocessor)
        cv_df = run_cross_validation(X_model, y, cv_pipeline, cv=5, random_state=RANDOM_SEED)
        cv_results[model_name] = cv_df

    # Step 14: extract feature importance (Random Forest) and LR coefficients
    logger.info("STEP 14: Extract feature importance and LR coefficients")
    feature_importance_df = None
    lr_coefficients_df = None

    rf_name = "Random Forest"
    if rf_name in fitted_pipelines:
        try:
            rf_clf = fitted_pipelines[rf_name].named_steps["classifier"]
            rf_names = feature_names_by_model.get(rf_name, [f"f{i}" for i in range(len(rf_clf.feature_importances_))])
            fi_rows = sorted(
                list(zip(rf_names, rf_clf.feature_importances_.tolist())),
                key=lambda t: -t[1],
            )
            feature_importance_df = pd.DataFrame(fi_rows, columns=["feature", "importance"])
            logger.info(f"  RF importance top-3: {fi_rows[:3]}")
        except Exception as e:
            logger.warning(f"Could not extract RF feature importance: {e}")

    lr_name = "Logistic Regression"
    if lr_name in fitted_pipelines:
        try:
            lr_clf = fitted_pipelines[lr_name].named_steps["classifier"]
            lr_names = feature_names_by_model.get(lr_name, [f"f{i}" for i in range(lr_clf.coef_.shape[1])])
            coef_vec = np.asarray(lr_clf.coef_).ravel()
            coef_rows = sorted(
                list(zip(lr_names, coef_vec.tolist())),
                key=lambda t: -abs(t[1]),
            )
            lr_coefficients_df = pd.DataFrame(coef_rows, columns=["feature", "coefficient"])
            logger.info(f"  LR coefficient top-3 (by magnitude): {coef_rows[:3]}")
        except Exception as e:
            logger.warning(f"Could not extract LR coefficients: {e}")

    # Step 15: generate all visualizations
    logger.info("STEP 15: Generate all figures")
    paths_figs = run_all_visualizations(
        target_df=target_df,
        confusion_matrices=confusion_matrices,
        models_results=models_results_for_viz,
        feature_importance_df=feature_importance_df,
        lr_coefficients_df=lr_coefficients_df,
        top_n_importance=20,
    )

    # Step 16: save CSV tables
    logger.info("STEP 16: Save CSV tables to outputs/tables/")
    paths_tables = _save_tables(
        validation=validation,
        target_df=target_df,
        features_df=features_df,
        merged_df=merged_df,
        test_eval=test_eval,
        cv_results=cv_results,
        predictions_df=predictions_df,
        feature_importance_df=feature_importance_df,
        lr_coefficients_df=lr_coefficients_df,
    )

    # Step 17: build DOCX report
    logger.info("STEP 17: Build 26-section DOCX report")
    report_path = build_report(
        target_df=target_df,
        features_df=features_df,
        temporal_dates=temporal_dates,
        test_eval=test_eval,
        cv_results=cv_results,
        feature_importance_df=feature_importance_df,
        lr_coefficients_df=lr_coefficients_df,
        confusion_matrices=confusion_matrices,
        paths_figs=paths_figs,
        output_path=REPORT_DOCX,
        n_raw_rows=n_raw_rows,
        n_positive_sales_rows=n_positive_sales_rows,
        n_feature_customers=n_feature_customers,
        n_train=n_train,
        n_test=n_test,
        validation=validation,
        paths_tables=paths_tables,
    )

    # Step 18: summary
    logger.info("=" * 70)
    logger.info("WEEK 4 PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Raw rows:                       {n_raw_rows:,}")
    logger.info(f"Positive sales (w/ CustomerID): {n_positive_sales_rows:,}")
    logger.info(f"Labelled customers (X ∩ y):     {n_feature_customers:,}")
    logger.info(f"Train / Test:                   {n_train:,} / {n_test:,}")
    logger.info(f"Temporal historical window:     {HISTORICAL_START.date()} → {HISTORICAL_END.date()}  "
                f"({temporal_dates.get('days_historical', 0)} d)")
    logger.info(f"Temporal future window:         {FUTURE_START.date()} → {FUTURE_END.date()}  "
                f"({temporal_dates.get('days_future', 0)} d)")
    for mname, md in test_eval.items():
        logger.info(
            f"  {mname:<22} | acc={md.get('accuracy', float('nan')):.3f}  "
            f"f1={md.get('f1', float('nan')):.3f}  "
            f"roc_auc={md.get('roc_auc', float('nan')):.3f}"
        )
    logger.info(f"Figures directory:              {FIGURES_DIR}  ({len(list(FIGURES_DIR.glob('*.png')))} files)")
    logger.info(f"Tables directory:               {TABLES_DIR}  ({len(list(TABLES_DIR.glob('*.csv')))} files)")
    logger.info(f"Report saved at:                {report_path}")
    logger.info(f"Report file size:               {report_path.stat().st_size / 1024:.1f} KB")

    return report_path


if __name__ == "__main__":
    main()
