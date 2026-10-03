import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    TABLES_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DATA_PATH,
    REPORT_DOCX,
    REPORT_DIR,
    LOG_FORMAT,
    K_RANGE_START,
    K_RANGE_END,
)
from src.data_loading import load_excel_dataset, get_dataset_shape
from src.data_validation import run_full_validation
from src.data_preparation import prepare_all, get_positive_sales_with_customer_id
from src.feature_engineering import (
    compute_rfm,
    inspect_rfm_distributions,
    apply_log1p_transformation,
    scale_rfm_features,
)
from src.clustering import (
    find_optimal_k,
    run_kmeans,
    apply_pca_for_visualization,
    compute_cluster_profiles,
)
from src.evaluation import (
    compute_clustering_metrics,
    describe_cluster,
)
from src.visualization import run_all_visualizations
from src.report_generation import build_report

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


def _rfm_distributions_to_df(stats_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for col_name, desc_df in stats_dict.items():
        row = {"column": col_name}
        for metric in desc_df.columns:
            row[metric] = float(desc_df[metric].iloc[0])
        rows.append(row)
    result = pd.DataFrame(rows)
    return result


def _build_scaled_profiles(
    cluster_profiles: pd.DataFrame,
    rfm_df: pd.DataFrame,
) -> pd.DataFrame:
    rfm_cols = ["Recency", "Frequency", "Monetary"]
    mean = rfm_df[rfm_cols].mean()
    std = rfm_df[rfm_cols].std(ddof=0).replace(0, np.nan)
    scaled = pd.DataFrame()
    scaled["cluster"] = cluster_profiles["cluster"]
    for col in rfm_cols:
        mean_col = f"mean_{col.lower()}"
        if mean_col in cluster_profiles.columns:
            vals = (cluster_profiles[mean_col] - mean[col]) / std[col]
            scaled[f"{col}_z"] = vals
    return scaled


def _save_tables(
    *,
    validation: Dict[str, Any],
    rfm_df: pd.DataFrame,
    rfm_dist_stats_df: pd.DataFrame,
    k_metrics_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    cluster_profiles_scaled: pd.DataFrame,
    clustering_metrics: Dict[str, float],
    pca_explained_var_df: pd.DataFrame,
    overall_rfm_stats: Dict[str, float],
) -> None:
    rfm_df.to_csv(TABLES_DIR / "01_rfm_customer_level.csv", index=False)
    rfm_dist_stats_df.to_csv(TABLES_DIR / "02_rfm_distribution_statistics.csv", index=False)
    pd.DataFrame([overall_rfm_stats]).to_csv(TABLES_DIR / "03_rfm_overall_means.csv", index=False)
    k_metrics_df.to_csv(TABLES_DIR / "04_k_selection_metrics.csv", index=False)
    cluster_profiles.to_csv(TABLES_DIR / "05_cluster_profiles_original_scale.csv", index=False)
    cluster_profiles_scaled.to_csv(TABLES_DIR / "06_cluster_profiles_scaled_zscore.csv", index=False)
    labels_df.to_csv(TABLES_DIR / "07_customer_cluster_assignments.csv", index=False)
    pd.DataFrame([clustering_metrics]).to_csv(TABLES_DIR / "08_clustering_evaluation_metrics.csv", index=False)
    pca_explained_var_df.to_csv(TABLES_DIR / "09_pca_explained_variance.csv", index=False)
    if "missing_values" in validation:
        validation["missing_values"].to_csv(TABLES_DIR / "00_validation_missing_values.csv", index=False)
    if "data_types" in validation:
        validation["data_types"].to_csv(TABLES_DIR / "00_validation_data_types.csv", index=False)
    logger.info(f"Saved CSV tables to {TABLES_DIR}")


def main() -> Path:
    _setup_logging()
    _ensure_directories()

    logger.info("=" * 70)
    logger.info("WEEK 3 — RFM + K-MEANS CUSTOMER SEGMENTATION — REAL DATASET PIPELINE")
    logger.info("=" * 70)
    logger.info(f"PROJECT_ROOT = {PROJECT_ROOT}")
    logger.info(f"RAW_DATA_PATH = {RAW_DATA_PATH.resolve()}")
    logger.info(f"Random seed = {RANDOM_SEED}")

    np.random.seed(RANDOM_SEED)

    logger.info("STEP 1: Load and validate raw dataset")
    df_raw = load_excel_dataset(file_path=RAW_DATA_PATH)
    n_raw_rows, n_raw_cols = get_dataset_shape(df_raw)
    validation = run_full_validation(df_raw)

    logger.info("STEP 2: Prepare analytical datasets (TotalAmount, temporal features, cancellation flags, positive sales)")
    datasets = prepare_all(df_raw)
    positive_sales = datasets["positive_sales"]
    positive_sales_cid = get_positive_sales_with_customer_id(positive_sales)
    n_positive_sales_rows = len(positive_sales_cid)
    logger.info(f"Raw rows: {n_raw_rows:,}  |  Positive sales with CustomerID: {n_positive_sales_rows:,}")

    logger.info("STEP 3: Compute RFM features per customer")
    rfm_df = compute_rfm(positive_sales_cid)
    total_customers = len(rfm_df)
    logger.info(f"Total customers with RFM: {total_customers:,}")

    rfm_distributions_dict = inspect_rfm_distributions(rfm_df)
    rfm_dist_stats_df = _rfm_distributions_to_df(rfm_distributions_dict)
    overall_rfm_stats = {
        "Recency": float(rfm_df["Recency"].mean()),
        "Frequency": float(rfm_df["Frequency"].mean()),
        "Monetary": float(rfm_df["Monetary"].mean()),
    }
    logger.info(f"Overall RFM means: {overall_rfm_stats}")

    logger.info("STEP 4: Apply log1p transformation to stabilise right-skewed RFM")
    log_cols = ["Recency", "Frequency", "Monetary"]
    rfm_df_log = apply_log1p_transformation(rfm_df, columns=log_cols)
    use_log_feature_cols = [f"Log_{c}" for c in log_cols]

    logger.info("STEP 5: StandardScaler on log-transformed features")
    scaled_features, scaler = scale_rfm_features(rfm_df_log, feature_cols=use_log_feature_cols)
    logger.info(f"Scaled features shape: {scaled_features.shape}  |  means={scaler.mean_.round(3)}  stds={scaler.scale_.round(3)}")

    logger.info(f"STEP 6: Optimal K search (k = {K_RANGE_START}..{K_RANGE_END})")
    k_metrics_df = find_optimal_k(scaled_features)
    logger.info(f"K metrics:\n{k_metrics_df.to_string(index=False)}")

    best_k_row_idx = int(k_metrics_df["silhouette_score"].idxmax())
    best_k = int(k_metrics_df.loc[best_k_row_idx, "k"])
    logger.info(f"Selected best k = {best_k} (highest silhouette score = {k_metrics_df.loc[best_k_row_idx, 'silhouette_score']:.4f})")

    logger.info(f"STEP 7: Final K-Means fit with k = {best_k}")
    labels_array, kmeans_model = run_kmeans(scaled_features, k=best_k, random_state=RANDOM_SEED)

    labels_df = rfm_df[["CustomerID", "Recency", "Frequency", "Monetary"]].copy()
    labels_df["Cluster"] = labels_array

    logger.info("STEP 8: PCA projection for visualization")
    pca_df, pca_evr_list, pca_model = apply_pca_for_visualization(scaled_features)
    pca_explained_var_df = pd.DataFrame({
        "Component": [f"PC{i + 1}" for i in range(len(pca_evr_list))],
        "ExplainedVarianceRatio": pca_evr_list,
        "CumulativeExplainedVariance": np.cumsum(pca_evr_list),
    })
    logger.info(f"PCA explained variance ratio: {[f'{r:.4f}' for r in pca_evr_list]}  |  cumulative: {sum(pca_evr_list):.4f}")

    logger.info("STEP 9: Cluster profiles (original and scaled)")
    cluster_profiles = compute_cluster_profiles(labels_df, cluster_col="Cluster")
    cluster_profiles_scaled = _build_scaled_profiles(cluster_profiles, rfm_df)
    logger.info(f"Cluster profiles (original scale):\n{cluster_profiles.to_string(index=False)}")

    logger.info("STEP 10: Compute clustering evaluation metrics")
    clustering_metrics = compute_clustering_metrics(scaled_features, labels_array)
    logger.info(f"Clustering metrics: {clustering_metrics}")

    logger.info("STEP 11: Generate all visualization figures")
    viz_rfm = labels_df.rename(columns={
        "Recency": "Recency", "Frequency": "Frequency", "Monetary": "Monetary",
    }).copy()
    viz_rfm_log = rfm_df_log.copy()
    paths_figs = run_all_visualizations(
        rfm_df=viz_rfm,
        rfm_df_log=viz_rfm_log,
        k_metrics_df=k_metrics_df,
        labels=labels_array,
        pca_df=pca_df,
        cluster_profiles=cluster_profiles.rename(columns={"cluster": "Cluster"}),
        cluster_profiles_scaled=cluster_profiles_scaled.rename(columns={"cluster": "Cluster"}),
    )

    logger.info("STEP 12: Save all CSV tables")
    _save_tables(
        validation=validation,
        rfm_df=rfm_df,
        rfm_dist_stats_df=rfm_dist_stats_df,
        k_metrics_df=k_metrics_df,
        labels_df=labels_df,
        cluster_profiles=cluster_profiles,
        cluster_profiles_scaled=cluster_profiles_scaled,
        clustering_metrics=clustering_metrics,
        pca_explained_var_df=pca_explained_var_df,
        overall_rfm_stats=overall_rfm_stats,
    )

    logger.info("STEP 13: Build evidence-based cluster descriptions")
    cluster_descriptions: Dict[Any, str] = {}
    for _, row in cluster_profiles.iterrows():
        cid = int(row["cluster"])
        cluster_descriptions[cid] = describe_cluster(row, overall_rfm_stats)
        logger.info(f"Cluster {cid} description: {cluster_descriptions[cid][:160]}...")

    logger.info("STEP 14: Build DOCX report")
    report_path = build_report(
        rfm_df=rfm_df,
        rfm_distribution_stats=rfm_dist_stats_df,
        k_metrics_df=k_metrics_df,
        best_k=best_k,
        labels_df=labels_df,
        cluster_profiles=cluster_profiles,
        cluster_profiles_scaled=cluster_profiles_scaled,
        clustering_metrics=clustering_metrics,
        pca_explained_var_df=pca_explained_var_df,
        cluster_descriptions=cluster_descriptions,
        paths_figs=paths_figs,
        reference_date=(positive_sales_cid["InvoiceDate"].max() + pd.Timedelta(days=1)),
        use_log_columns=True,
        output_path=REPORT_DOCX,
        n_raw_rows=n_raw_rows,
        n_positive_sales_rows=n_positive_sales_rows,
        overall_rfm_stats=overall_rfm_stats,
    )

    logger.info("=" * 70)
    logger.info("WEEK 3 PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total customers:              {total_customers:,}")
    logger.info(f"Best K selected:              {best_k}")
    logger.info(f"Silhouette score:             {clustering_metrics['silhouette_score']:.4f}")
    logger.info(f"Calinski-Harabasz:            {clustering_metrics['calinski_harabasz_score']:,.2f}")
    logger.info(f"Davies-Bouldin:               {clustering_metrics['davies_bouldin_score']:.4f}")
    logger.info(f"Figures directory:            {FIGURES_DIR}  ({len(list(FIGURES_DIR.glob('*.png')))} files)")
    logger.info(f"Tables directory:             {TABLES_DIR}  ({len(list(TABLES_DIR.glob('*.csv')))} files)")
    logger.info(f"Report saved at:              {report_path}")
    logger.info(f"Report file size:             {report_path.stat().st_size / 1024:.1f} KB")
    return report_path


if __name__ == "__main__":
    main()
