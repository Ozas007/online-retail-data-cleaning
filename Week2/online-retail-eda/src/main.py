import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import (
    ANALYSIS_DIR, DATA_DIR, FIGURES_DIR, LOG_FORMAT, OUTPUTS_DIR,
    PROJECT_ROOT, RAW_DATA_PATH, REPORT_DOCX, REPORT_DIR, TABLES_DIR,
)
from src.data_loading import load_excel_dataset
from src.data_validation import count_missing_values, run_full_validation
from src.data_preparation import numeric_summary, prepare_all
from src.descriptive_analysis import run_descriptive_analysis, save_table
from src.temporal_analysis import run_temporal_analysis
from src.product_analysis import run_product_analysis
from src.geographic_analysis import run_geographic_analysis
from src.relationship_analysis import run_relationship_analysis
from src.anomaly_analysis import run_anomaly_analysis
from src.visualization import run_all_visualizations
from src.report_generation import build_report

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    for d in [DATA_DIR, FIGURES_DIR, TABLES_DIR, ANALYSIS_DIR, REPORT_DIR, OUTPUTS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(PROJECT_ROOT / "eda_pipeline.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def main() -> Dict:
    _setup_logging()
    logger.info("=" * 70)
    logger.info("WEEK 2 — EXPLORATORY DATA ANALYSIS & VISUALIZATION PIPELINE")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Dataset:       {RAW_DATA_PATH}")
    logger.info("=" * 70)

    # ============== PHASE 2: LOAD + VALIDATE ==============
    logger.info("\n[STEP 1] Loading & validating dataset")
    df_raw = load_excel_dataset()
    validation = run_full_validation(df_raw)
    missing_df = count_missing_values(df_raw)
    save_table(validation["data_types"], "00_data_types.csv")
    save_table(validation["unique_values"], "00b_unique_values.csv")

    # ============== PHASE 3: PREPARE ==============
    logger.info("\n[STEP 2] Preparing analytical datasets + features")
    datasets = prepare_all(df_raw)
    df_engineered = datasets["engineered"]
    df_full = datasets["full"]
    df_positive = datasets["positive_sales"]
    logger.info(f"Positive-sales rows: {len(df_positive):,}")

    # ============== PHASES 4–8: ALL ANALYSES ==============
    logger.info("\n[STEP 3] Running Descriptive Analysis")
    n_raw_cols = len(df_raw.columns)
    desc = run_descriptive_analysis(df_engineered, df_positive, raw_columns_count=n_raw_cols)

    logger.info("\n[STEP 4] Running Temporal Analysis")
    temporal = run_temporal_analysis(df_engineered, df_positive)

    logger.info("\n[STEP 5] Running Product Analysis")
    product = run_product_analysis(df_positive)

    logger.info("\n[STEP 6] Running Geographic Analysis")
    geographic = run_geographic_analysis(df_engineered, df_positive)

    logger.info("\n[STEP 7] Running Relationship / Correlation Analysis")
    relationship = run_relationship_analysis(df_positive)

    logger.info("\n[STEP 8] Running Anomaly / Outlier Analysis")
    anomaly = run_anomaly_analysis(df_engineered, df_positive)

    analyses = {
        "descriptive": desc,
        "temporal": temporal,
        "product": product,
        "geographic": geographic,
        "relationship": relationship,
        "anomaly": anomaly,
    }

    # Save numeric summary table (used by report)
    num_summary = numeric_summary(df_positive, columns=["Quantity", "UnitPrice", "TotalAmount"])
    save_table(num_summary, "04_numeric_summary.csv")

    # ============== PHASE 8: VISUALIZE ==============
    logger.info("\n[STEP 9] Generating all figures")
    figs = run_all_visualizations(analyses, df_engineered, df_positive, missing_df)

    # ============== PHASE 9: REPORT ==============
    logger.info("\n[STEP 10] Building DOCX report")
    report_path = build_report(
        analyses=analyses,
        df_overview=desc["overview"],
        paths_figs=figs,
        missing_df=missing_df,
        iqr_summary=anomaly["iqr_summary"],
        output_path=REPORT_DOCX,
    )

    # ============== SUMMARY ==============
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETED — FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Raw rows:       {len(df_raw):>10,}")
    logger.info(f"Positive sales: {len(df_positive):>10,}")
    logger.info(f"Figures:        {len(figs):>10}  -> {FIGURES_DIR}")
    logger.info(f"Tables:         {len(list(TABLES_DIR.glob('*.csv'))):>10}  -> {TABLES_DIR}")
    logger.info(f"Report:         {report_path.stat().st_size/1024:>9.1f} KB -> {report_path}")
    logger.info("=" * 70)
    return {
        "datasets": datasets,
        "analyses": analyses,
        "figures": figs,
        "report_path": report_path,
        "validation": validation,
    }


if __name__ == "__main__":
    main()
