import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLEANED_DATA_DIR,
    FIGURES_DIR,
    LOG_FORMAT,
    RAW_DATA_PATH,
    TABLES_DIR,
)
from src.data_cleaning import run_full_cleaning
from src.data_exploration import run_full_exploration, save_table
from src.data_loading import load_excel_dataset
from src.data_validation import run_full_validation
from src.outlier_detection import detect_outliers_iqr, outlier_impact_analysis
from src.preprocessing import before_after_comparison, create_analytical_dataset
from src.visualization import generate_all_visualizations

logger = logging.getLogger()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PROJECT_ROOT / "pipeline.log", mode="w", encoding="utf-8"),
        ],
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("seaborn").setLevel(logging.WARNING)


def main() -> None:
    setup_logging()

    logger.info("")
    logger.info("=" * 70)
    logger.info("ONLINE RETAIL DATA CLEANING & PREPROCESSING PIPELINE")
    logger.info("=" * 70)
    logger.info("")

    for directory in [FIGURES_DIR, TABLES_DIR, CLEANED_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Expected dataset: {RAW_DATA_PATH}")

    df_raw = load_excel_dataset(RAW_DATA_PATH, validate=True, preserve_original=True)

    validation_results = run_full_validation(df_raw)

    exploration_results = run_full_exploration(df_raw, save_outputs=True)

    quality_summary = {
        "Metric": [
            "Total rows",
            "Total columns",
            "Memory (MB)",
            "Exact duplicates",
            "Missing CustomerID count",
            "Missing CustomerID %",
            "Missing Description count",
            "Missing Description %",
            "Negative Quantity count",
            "UnitPrice <= 0 count",
            "Cancellation invoices (prefix 'C')",
        ],
        "Value": [
            validation_results["shape"][0],
            validation_results["shape"][1],
            exploration_results["basic_info"]["memory_usage_mb"],
            validation_results["duplicates"][0],
            validation_results["missing_values"].loc[
                validation_results["missing_values"]["Column"] == "CustomerID", "Missing Count"
            ].values[0] if (validation_results["missing_values"]["Column"] == "CustomerID").any() else 0,
            validation_results["missing_values"].loc[
                validation_results["missing_values"]["Column"] == "CustomerID", "Missing Percentage (%)"
            ].values[0] if (validation_results["missing_values"]["Column"] == "CustomerID").any() else 0.0,
            validation_results["missing_values"].loc[
                validation_results["missing_values"]["Column"] == "Description", "Missing Count"
            ].values[0] if (validation_results["missing_values"]["Column"] == "Description").any() else 0,
            validation_results["missing_values"].loc[
                validation_results["missing_values"]["Column"] == "Description", "Missing Percentage (%)"
            ].values[0] if (validation_results["missing_values"]["Column"] == "Description").any() else 0.0,
            validation_results["negative_quantities"][0],
            sum(validation_results["negative_prices"][:2]),
            validation_results["suspicious_invoices"]["cancellation_invoices"],
        ],
    }
    save_table(__import__("pandas").DataFrame(quality_summary), "data_quality_summary.csv")

    cleaning_results = run_full_cleaning(df_raw, save_outputs=True)
    df_flagged = cleaning_results["flagged_dataset"]
    df_cleaned = cleaning_results["cleaned_dataset"]
    df_positive_sales = cleaning_results["positive_sales_dataset"]

    outlier_results = detect_outliers_iqr(df_cleaned, columns=["Quantity", "UnitPrice", "TotalAmount"])
    df_outlier_flagged = outlier_results["dataset_flagged"]

    for col in ["Quantity", "UnitPrice", "TotalAmount"]:
        impact_df = outlier_impact_analysis(df_outlier_flagged, col)
        if not impact_df.empty:
            save_table(impact_df, f"outlier_impact_{col.lower()}.csv")

    comparison_df = before_after_comparison(df_raw, df_cleaned, df_positive_sales)
    save_table(comparison_df, "before_after_comparison.csv")

    preprocessing_results = create_analytical_dataset(
        df_positive_sales,
        perform_scaling=True,
        perform_encoding=False,
    )
    preprocessed_path = CLEANED_DATA_DIR / "online_retail_preprocessed.csv"
    preprocessing_results["base_dataset"].to_csv(preprocessed_path, index=False)
    logger.info(f"Preprocessed analytical dataset saved to: {preprocessed_path}")

    if "scaled_dataset" in preprocessing_results:
        scaled_path = CLEANED_DATA_DIR / "online_retail_scaled.csv"
        preprocessing_results["scaled_dataset"].to_csv(scaled_path, index=False)
        logger.info(f"Scaled analytical dataset saved to: {scaled_path}")

    figure_paths = generate_all_visualizations(
        df_raw, df_cleaned, df_positive_sales, comparison_df
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Raw dataset:            {len(df_raw):,} rows x {df_raw.shape[1]} cols")
    logger.info(f"After dedup:            {len(df_flagged):,} rows")
    logger.info(f"Quality-cleaned:        {len(df_cleaned):,} rows (UnitPrice > 0)")
    logger.info(f"Positive-sales dataset: {len(df_positive_sales):,} rows (no cancels, qty > 0)")
    logger.info(f"Figures generated:      {len(figure_paths)}")
    logger.info(f"Figures directory:      {FIGURES_DIR}")
    logger.info(f"Tables directory:       {TABLES_DIR}")
    logger.info(f"Cleaned data directory: {CLEANED_DATA_DIR}")
    logger.info("")
    logger.info("Cleaning log and decision tables saved to outputs/tables/")
    logger.info("Pipeline log saved to pipeline.log")
    logger.info("")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
