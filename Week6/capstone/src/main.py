import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    LOG_FORMAT,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DATA_PATH,
    REPORT_DOCX,
    REPORT_DIR,
    TABLES_DIR,
)
from src.data_loading import get_dataset_shape, load_excel_dataset
from src.deep_learning import build_dl_section_summary
from src.eda import run_eda
from src.evaluation import (
    build_recommendations,
    summarize_supervised,
    summarize_unsupervised,
)
from src.feature_engineering import build_rfm_features, build_supervised_features
from src.preprocessing import clean_dataset
from src.report_generation import build_report
from src.supervised import run_supervised_pipeline
from src.unsupervised import run_unsupervised_pipeline
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


def _save_all_tables(
    eda_summaries: Dict[str, pd.DataFrame],
    unsupervised_results: Dict[str, Any],
    supervised_results: Dict[str, Any],
    rfm_df: pd.DataFrame,
) -> Dict[str, Path]:
    saved: Dict[str, Path] = {}

    def _save(df: pd.DataFrame, fname: str) -> Path:
        p = TABLES_DIR / fname
        df.to_csv(p, index=False)
        saved[fname] = p
        return p

    for key, df_val in eda_summaries.items():
        _save(df_val, f"01_eda_{key}.csv")

    if not rfm_df.empty:
        _save(rfm_df, "02_rfm_customer_level.csv")
    k_metrics = unsupervised_results.get("k_metrics_df", pd.DataFrame())
    if not k_metrics.empty:
        _save(k_metrics, "03_unsupervised_k_metrics.csv")
    profiles = unsupervised_results.get("cluster_profiles_df", pd.DataFrame())
    if not profiles.empty:
        _save(profiles, "04_cluster_profiles.csv")
    scaled_profiles = unsupervised_results.get("scaled_profiles_df", pd.DataFrame())
    if not scaled_profiles.empty:
        _save(scaled_profiles, "05_cluster_profiles_zscored.csv")
    pca_df = unsupervised_results.get("pca_df", pd.DataFrame())
    if not pca_df.empty:
        _save(pca_df, "06_pca_projection.csv")

    target_df = supervised_results.get("target_df", pd.DataFrame())
    if not target_df.empty:
        _save(target_df, "07_target_customer_level.csv")
    features_df = supervised_results.get("features_df", pd.DataFrame())
    if not features_df.empty:
        _save(features_df, "08_features_customer_level.csv")
    merged_df = supervised_results.get("merged_df", pd.DataFrame())
    if not merged_df.empty:
        _save(merged_df, "09_merged_design_matrix_X_y.csv")
    cv_df = supervised_results.get("cv_results_df", pd.DataFrame())
    if not cv_df.empty:
        _save(cv_df, "10_supervised_cv_results.csv")
    fi_df = supervised_results.get("feature_importance_df", pd.DataFrame())
    if not fi_df.empty:
        _save(fi_df, "11_supervised_feature_importance.csv")
    cm_df = supervised_results.get("confusion_matrix_df", pd.DataFrame())
    if not cm_df.empty:
        cm_df.to_csv(TABLES_DIR / "12_supervised_confusion_matrix_best.csv")
        saved["12_supervised_confusion_matrix_best.csv"] = TABLES_DIR / "12_supervised_confusion_matrix_best.csv"

    per_model_rows = []
    for mname, md in supervised_results.get("test_metrics_per_model", {}).items():
        row = {"model": mname}
        row.update({k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in md.items()})
        per_model_rows.append(row)
    if per_model_rows:
        _save(pd.DataFrame(per_model_rows), "13_supervised_holdout_metrics_all_models.csv")

    logger.info(f"Saved {len(saved)} CSV tables to {TABLES_DIR}")
    return saved


def main() -> Path:
    _setup_logging()
    _ensure_directories()

    logger.info("=" * 70)
    logger.info("WEEK 6 — INTEGRATIVE CAPSTONE: END-TO-END ONLINE RETAIL PIPELINE")
    logger.info("=" * 70)
    logger.info(f"PROJECT_ROOT = {PROJECT_ROOT}")
    logger.info(f"RAW_DATA_PATH = {RAW_DATA_PATH.resolve()}")
    logger.info(f"Random seed = {RANDOM_SEED}")

    np.random.seed(RANDOM_SEED)

    logger.info("STEP 1-2: Load raw dataset")
    df_raw = load_excel_dataset(file_path=RAW_DATA_PATH)
    n_raw_rows, n_raw_cols = get_dataset_shape(df_raw)

    logger.info("STEP 3: Clean and preprocess dataset (TotalAmount, cancellation, dedup, positive-sales, CID filter)")
    df_clean, df_positive = clean_dataset(df_raw)
    n_positive_sales_rows = len(df_positive)
    logger.info(
        f"Raw rows: {n_raw_rows:,}  |  Positive sales w/ CustomerID: {n_positive_sales_rows:,}  "
        f"({n_positive_sales_rows / n_raw_rows * 100:.1f}%)"
    )

    logger.info("STEP 4: Exploratory Data Analysis — save CSVs + return summaries dict")
    eda_summaries = run_eda(df_clean, df_positive)
    logger.info(f"EDA summary keys produced: {list(eda_summaries.keys())}")

    logger.info("STEP 5: Feature engineering — RFM for clustering + supervised behavioural features")
    rfm_df = build_rfm_features(df_positive)
    n_feature_customers = len(rfm_df)
    logger.info(f"RFM rows = {n_feature_customers:,} customers")

    logger.info("STEP 6: Unsupervised learning pipeline (log1p + std + KMeans k=2..10, best by silhouette)")
    unsupervised_results = run_unsupervised_pipeline(rfm_df)
    best_k = int(unsupervised_results["best_k"])
    labels_array = unsupervised_results["labels_array"]
    clust_metrics = unsupervised_results["clustering_metrics_dict"]
    silhouette = float(clust_metrics.get("silhouette_score", float("nan")))
    profiles = unsupervised_results["cluster_profiles_df"]
    logger.info(f"Best k={best_k}, Silhouette={silhouette:.4f}, CH={float(clust_metrics.get('calinski_harabasz_score', float('nan'))):.2f}, DB={float(clust_metrics.get('davies_bouldin_score', float('nan'))):.4f}")

    logger.info("STEP 7: Supervised learning pipeline (temporal split + RF + LR, 70/30 stratify)")
    supervised_results = run_supervised_pipeline(df_positive)
    best_model_name = supervised_results["best_model_name"]
    sup_metrics = supervised_results["test_metrics_dict"]
    roc_auc = float(sup_metrics.get("roc_auc", float("nan")))
    accuracy = float(sup_metrics.get("accuracy", float("nan")))
    f1 = float(sup_metrics.get("f1", float("nan")))
    importances = supervised_results["feature_importance_df"]
    temporal_dates = supervised_results["temporal_dates_dict"]
    n_train = int(supervised_results.get("n_train", 0))
    n_test = int(supervised_results.get("n_test", 0))
    logger.info(
        f"Best model: {best_model_name}  |  Acc={accuracy:.3f}  F1={f1:.3f}  ROC-AUC={roc_auc:.3f}  |  "
        f"Train={n_train:,}  Test={n_test:,}"
    )

    logger.info("STEP 8: Deep Learning honesty section + optional tabular MLP baseline")
    X_train_mlp = supervised_results.get("X_train")
    y_train_mlp = supervised_results.get("y_train")
    X_test_mlp = supervised_results.get("X_test")
    y_test_mlp = supervised_results.get("y_test")
    dl_summary = build_dl_section_summary(
        X_train=X_train_mlp,
        y_train=y_train_mlp,
        X_test=X_test_mlp,
        y_test=y_test_mlp,
    )
    if dl_summary.get("mlp_trained"):
        logger.info(
            f"  MLP test: roc_auc={float(dl_summary['mlp_test_metrics'].get('roc_auc', float('nan'))):.3f}  "
            f"f1={float(dl_summary['mlp_test_metrics'].get('f1', float('nan'))):.3f}"
        )
    else:
        logger.info("  MLP baseline skipped in this run; honesty text still included.")

    logger.info("STEP 9: Combine results and build number-backed recommendations list")
    combined_results: Dict[str, Any] = {
        "unsupervised": unsupervised_results,
        "supervised": supervised_results,
        "deep_learning": dl_summary,
        "eda": eda_summaries,
        "n_raw_rows": n_raw_rows,
        "n_positive_sales_rows": n_positive_sales_rows,
        "n_feature_customers": n_feature_customers,
        "rfm_df": rfm_df,
    }
    recommendations = build_recommendations(combined_results)
    logger.info(f"Built {len(recommendations)} recommendations")
    for i, r in enumerate(recommendations[:5], start=1):
        logger.info(f"  Rec[{i}]: {r[:140]}...")

    logger.info("STEP 10: Generate all 8 visualizations")
    paths_figs = run_all_visualizations(
        eda_monthly=eda_summaries.get("monthly_volume", pd.DataFrame()),
        eda_top_countries=eda_summaries.get("top10_countries_by_transactions", pd.DataFrame()),
        cluster_profiles=profiles,
        unsupervised_labels=labels_array,
        rfm_df=rfm_df,
        models_results_for_viz=supervised_results.get("models_results_for_viz", {}),
        feature_importance_df=importances,
        confusion_matrix_df=supervised_results.get("confusion_matrix_df", pd.DataFrame()),
        scaled_profiles_df=unsupervised_results.get("scaled_profiles_df", pd.DataFrame()),
    )

    logger.info("STEP 11: Save all CSV tables")
    paths_tables = _save_all_tables(eda_summaries, unsupervised_results, supervised_results, rfm_df)

    logger.info("STEP 12: Build 24-section DOCX report")
    report_path = build_report(
        output_path=REPORT_DOCX,
        eda_summaries=eda_summaries,
        unsupervised_results=unsupervised_results,
        supervised_results=supervised_results,
        deep_learning_summary=dl_summary,
        recommendations=recommendations,
        paths_figs=paths_figs,
        paths_tables=paths_tables,
        n_raw_rows=n_raw_rows,
        n_positive_sales_rows=n_positive_sales_rows,
        n_feature_customers=n_feature_customers,
        best_k=best_k,
        silhouette_score=silhouette,
        best_model_name=best_model_name,
        best_roc_auc=roc_auc,
        rfm_df=rfm_df,
    )

    logger.info("=" * 70)
    logger.info("WEEK 6 CAPSTONE PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Customers (RFM rows):              {n_feature_customers:,}")
    logger.info(f"Best KMeans k:                      {best_k}")
    logger.info(f"Silhouette score:                   {silhouette:.4f}")
    logger.info(f"Best supervised model:              {best_model_name}")
    logger.info(f"Best ROC-AUC:                       {roc_auc:.4f}")
    logger.info(f"Accuracy / F1:                      {accuracy:.4f} / {f1:.4f}")
    logger.info(f"Recommendations count:              {len(recommendations)}")
    logger.info("Top 5 recommendations:")
    for i, r in enumerate(recommendations[:5], start=1):
        logger.info(f"  [{i}] {r}")
    logger.info(f"Figures directory:                  {FIGURES_DIR}  ({len(list(FIGURES_DIR.glob('*.png')))} files)")
    logger.info(f"Tables directory:                   {TABLES_DIR}  ({len(list(TABLES_DIR.glob('*.csv')))} files)")
    logger.info(f"Report saved at:                    {report_path}")
    try:
        logger.info(f"Report file size:                   {report_path.stat().st_size / 1024:.1f} KB")
    except Exception:
        pass

    return report_path


if __name__ == "__main__":
    main()
