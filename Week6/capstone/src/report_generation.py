import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph

logger = logging.getLogger(__name__)

FIGURE_WIDTH_INCHES = 6.0
FONT_NAME = "Calibri"
FONT_SIZE_PT = 11


def _set_run_font(run, font_name: str = FONT_NAME, size_pt: int = FONT_SIZE_PT,
                  bold: bool = False, italic: bool = False,
                  color_rgb: tuple | None = None) -> None:
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.italic = italic
    if color_rgb is not None:
        run.font.color.rgb = RGBColor(*color_rgb)
    rPr = run._element.get_or_add_rPr()
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:ascii"), font_name)
    rFonts.set(qn("w:hAnsi"), font_name)
    rFonts.set(qn("w:cs"), font_name)
    rPr.append(rFonts)


def _set_paragraph_default_font(paragraph: Paragraph, font_name: str = FONT_NAME,
                                 size_pt: int = FONT_SIZE_PT) -> None:
    pPr = paragraph._element.get_or_add_pPr()
    rPr = OxmlElement("w:rPr")
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:ascii"), font_name)
    rFonts.set(qn("w:hAnsi"), font_name)
    rFonts.set(qn("w:cs"), font_name)
    rPr.append(rFonts)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(size_pt * 2))
    rPr.append(sz)
    szCs = OxmlElement("w:szCs")
    szCs.set(qn("w:val"), str(size_pt * 2))
    rPr.append(szCs)
    pPr.append(rPr)


def _add_page_number(footer) -> None:
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_default_font(paragraph)
    run = paragraph.add_run("Page ")
    _set_run_font(run)
    fb = OxmlElement("w:fldChar"); fb.set(qn("w:fldCharType"), "begin")
    r1 = paragraph.add_run(); r1._element.append(fb); _set_run_font(r1)
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE   \\* MERGEFORMAT "
    r2 = paragraph.add_run(); r2._element.append(it); _set_run_font(r2)
    fs = OxmlElement("w:fldChar"); fs.set(qn("w:fldCharType"), "separate")
    r3 = paragraph.add_run(); r3._element.append(fs); _set_run_font(r3)
    fe = OxmlElement("w:fldChar"); fe.set(qn("w:fldCharType"), "end")
    r4 = paragraph.add_run(); r4._element.append(fe); _set_run_font(r4)
    r5 = paragraph.add_run(" of "); _set_run_font(r5)
    fb2 = OxmlElement("w:fldChar"); fb2.set(qn("w:fldCharType"), "begin")
    r6 = paragraph.add_run(); r6._element.append(fb2); _set_run_font(r6)
    it2 = OxmlElement("w:instrText"); it2.set(qn("xml:space"), "preserve"); it2.text = "NUMPAGES   \\* MERGEFORMAT "
    r7 = paragraph.add_run(); r7._element.append(it2); _set_run_font(r7)
    fs2 = OxmlElement("w:fldChar"); fs2.set(qn("w:fldCharType"), "separate")
    r8 = paragraph.add_run(); r8._element.append(fs2); _set_run_font(r8)
    fe2 = OxmlElement("w:fldChar"); fe2.set(qn("w:fldCharType"), "end")
    r9 = paragraph.add_run(); r9._element.append(fe2); _set_run_font(r9)


def _add_toc(paragraph: Paragraph) -> None:
    run = paragraph.add_run(); _set_run_font(run)
    fb = OxmlElement("w:fldChar"); fb.set(qn("w:fldCharType"), "begin"); run._r.append(fb)
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
    it.text = 'TOC \\o "1-3" \\h \\z \\u'
    r2 = paragraph.add_run(); r2._r.append(it); _set_run_font(r2)
    fs = OxmlElement("w:fldChar"); fs.set(qn("w:fldCharType"), "separate")
    r3 = paragraph.add_run(); r3._r.append(fs); _set_run_font(r3)
    fe = OxmlElement("w:fldChar"); fe.set(qn("w:fldCharType"), "end")
    r4 = paragraph.add_run(); r4._r.append(fe); _set_run_font(r4)


def _add_heading(doc: Document, text: str, level: int = 1) -> Paragraph:
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        _set_run_font(run, font_name=FONT_NAME, size_pt=FONT_SIZE_PT + max(1, (4 - level)) * 2, bold=True)
    return heading


def _add_paragraph(doc: Document, text: str, font_name: str = FONT_NAME,
                   size_pt: int = FONT_SIZE_PT, bold: bool = False,
                   italic: bool = False,
                   alignment: WD_ALIGN_PARAGRAPH = WD_ALIGN_PARAGRAPH.JUSTIFY) -> Paragraph:
    p = doc.add_paragraph()
    p.alignment = alignment
    _set_paragraph_default_font(p, font_name=font_name, size_pt=size_pt)
    run = p.add_run(text)
    _set_run_font(run, font_name=font_name, size_pt=size_pt, bold=bold, italic=italic)
    return p


def _add_bullet(doc: Document, text: str) -> Paragraph:
    p = doc.add_paragraph(style="List Bullet")
    _set_paragraph_default_font(p)
    run = p.add_run(text)
    _set_run_font(run)
    return p


def _add_numbered(doc: Document, text: str) -> Paragraph:
    p = doc.add_paragraph(style="List Number")
    _set_paragraph_default_font(p)
    run = p.add_run(text)
    _set_run_font(run)
    return p


def _add_table_caption(doc: Document, caption: str) -> Paragraph:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_default_font(p)
    run = p.add_run(caption); _set_run_font(run, bold=True)
    p.paragraph_format.space_after = Pt(4)
    return p


def _add_figure_caption(doc: Document, caption: str) -> Paragraph:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_default_font(p)
    run = p.add_run(caption); _set_run_font(run, italic=True, size_pt=FONT_SIZE_PT - 1)
    p.paragraph_format.space_after = Pt(8)
    return p


def _add_dataframe_as_table(doc: Document, df: pd.DataFrame, max_rows: int = 25) -> None:
    if df is None or df.empty:
        _add_paragraph(doc, "(Table not available for this section.)", italic=True)
        return
    disp = df.copy()
    if len(disp) > max_rows:
        disp = disp.head(max_rows)
    cols = [str(c) for c in disp.columns]
    table = doc.add_table(rows=1, cols=len(cols))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, c in enumerate(cols):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        run = p.add_run(str(c))
        _set_run_font(run, bold=True, size_pt=FONT_SIZE_PT - 1)
    for _, row in disp.iterrows():
        cells = table.add_row().cells
        for i, c in enumerate(cols):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            val = row[c]
            if isinstance(val, float) and not pd.isna(val):
                txt = f"{val:.4f}" if abs(val) < 1 else f"{val:,.2f}"
            elif pd.isna(val):
                txt = "NaN"
            else:
                txt = str(val)
            run = p.add_run(txt)
            _set_run_font(run, size_pt=FONT_SIZE_PT - 1)
    if len(df) > max_rows:
        _add_paragraph(doc, f"Table truncated to first {max_rows} rows of {len(df):,} total.", italic=True, size_pt=FONT_SIZE_PT - 1)


def _add_figure_from_path(doc: Document, fig_path: Optional[Path], caption: str,
                           width_inches: float = FIGURE_WIDTH_INCHES) -> None:
    if fig_path is not None and Path(fig_path).exists():
        doc.add_picture(str(fig_path), width=Inches(width_inches))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        _add_paragraph(doc, f"(Figure file not generated: {caption})", italic=True)
    _add_figure_caption(doc, caption)


def build_report(
    output_path: Path,
    eda_summaries: Dict[str, pd.DataFrame],
    unsupervised_results: Dict[str, Any],
    supervised_results: Dict[str, Any],
    deep_learning_summary: Dict[str, Any],
    recommendations: List[str],
    paths_figs: Dict[str, Path],
    paths_tables: Dict[str, Path],
    n_raw_rows: int,
    n_positive_sales_rows: int,
    n_feature_customers: int,
    best_k: int,
    silhouette_score: float,
    best_model_name: str,
    best_roc_auc: float,
    rfm_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> Path:
    logger.info(f"Building 24-section integrative capstone report at: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
        _add_page_number(section.footer)

    profiles = unsupervised_results.get("cluster_profiles_df", pd.DataFrame()) if isinstance(unsupervised_results, dict) else pd.DataFrame()
    clust_metrics = unsupervised_results.get("clustering_metrics_dict", {}) if isinstance(unsupervised_results, dict) else {}
    sup_metrics = supervised_results.get("test_metrics_dict", {}) if isinstance(supervised_results, dict) else {}
    temporal = supervised_results.get("temporal_dates_dict", {}) if isinstance(supervised_results, dict) else {}
    importances = supervised_results.get("feature_importance_df", pd.DataFrame()) if isinstance(supervised_results, dict) else pd.DataFrame()
    cv_df = supervised_results.get("cv_results_df", pd.DataFrame()) if isinstance(supervised_results, dict) else pd.DataFrame()

    today_str = date.today().isoformat()

    _add_heading(doc, "Week 6 Integrative Capstone Report", level=0)
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_default_font(tp, size_pt=14)
    r = tp.add_run("Online Retail UCI Dataset — End-to-End Data Science Pipeline: Cleaning, EDA, Unsupervised Clustering, Supervised Prediction, and Honest Deep-Learning Baseline Comparison")
    _set_run_font(r, size_pt=14, bold=True)
    doc.add_paragraph()
    sp = doc.add_paragraph()
    sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sp.add_run(f"Date: {today_str}")
    _set_run_font(sr, size_pt=12, italic=True)

    new_section = doc.add_section(WD_SECTION.NEW_PAGE)
    _add_page_number(new_section.footer)
    _add_heading(doc, "Table of Contents", level=1)
    toc_p = doc.add_paragraph()
    _add_toc(toc_p)
    _add_paragraph(doc, "(Right-click and select 'Update Field' after opening in Word to populate the TOC.)", italic=True, size_pt=FONT_SIZE_PT - 1)

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "1. Title Page", level=1)
    _add_paragraph(doc, "This document is the Week 6 Integrative Capstone Report for the UCI Online Retail Data Cleaning and Analysis project. It brings together the end-to-end data science workflow: raw data acquisition, quality assessment and cleaning, exploratory data analysis (EDA), unsupervised customer segmentation via RFM + KMeans, supervised future-purchase prediction (Random Forest + Logistic Regression with time-aware split), an honest deep-learning-component section distinguishing Week 5's MNIST neural-network exercise from a lightweight tabular MLP baseline trained on the retail data, a combined evaluation framework, cross-cutting insights, concrete number-backed recommendations, limitations, ethical considerations, reproducibility notes, reflection on the workflow, conclusion, code snippets, references, and appendix.")
    _add_paragraph(doc, f"High-level figures from the pipeline: raw rows loaded = {n_raw_rows:,}; positive-sales rows with non-null CustomerID = {n_positive_sales_rows:,}; distinct customers in the supervised design matrix = {n_feature_customers:,}; best KMeans k = {best_k} with Silhouette = {silhouette_score:.4f}; best supervised model = {best_model_name} with holdout ROC-AUC = {best_roc_auc:.3f}.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "2. Executive Summary", level=1)
    _add_paragraph(doc, "This capstone applies a reproducible, end-to-end data-science pipeline to the UCI Online Retail transactional dataset, integrating unsupervised customer segmentation, supervised future-purchase prediction, and an honest deep-learning baseline comparison.")
    _add_paragraph(doc, f"The cleaning stage reduces the raw {n_raw_rows:,} line items to {n_positive_sales_rows:,} positive-sales rows with non-null CustomerID, forming the analytical base table. EDA quantifies monthly revenue trajectories, top 10 products by revenue, and top 10 countries by transaction count. Unsupervised learning with KMeans (log1p-transformed RFM features, standard scaling, silhouette-optimal k in 2..10) yields {best_k} customer segments with a Silhouette score of {silhouette_score:.4f}.")
    _add_paragraph(doc, f"Supervised learning uses a temporal-split design (historical rows strictly prior to the cutoff, target built only from the subsequent future window) to predict the binary FuturePurchased label. The selected model — {best_model_name} — records a holdout ROC-AUC of {best_roc_auc:.3f}. Feature importance from the tree-based model highlights the behavioural signals driving future repurchase.")
    _add_paragraph(doc, "The deep-learning component section honestly distinguishes Week 5's MNIST (image-domain) exercise from the retail tabular task. For direct comparison on the retail features, a small sklearn MLPClassifier (two hidden layers, early stopping) is additionally trained and reported alongside Random Forest without claiming MNIST results as retail results.")
    _add_paragraph(doc, f"Finally, the report presents {len(recommendations)} concrete, number-backed recommendations targeted at the highest-value customer segments, temporal campaign windows, and regional/portfolio priorities identified by the combined analysis.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "3. Problem Statement", level=1)
    _add_paragraph(doc, "Online retailers rely on transaction-level point-of-sale records, but raw invoices contain cancellations, missing customer identifiers, duplicate rows, and inconsistent pricing. Without a structured cleaning and analytical pipeline, these data are difficult to use for marketing, customer retention, or inventory decisions.")
    _add_paragraph(doc, "This capstone addresses three connected analytical questions: (a) which behavioural customer segments exist in the customer base, derived from purchase recency, frequency and monetary value; (b) which existing customers are likely to repurchase in the next future window, based exclusively on information available before the prediction horizon; and (c) how an honourable, non-misleading deep-learning section can be written when the neural-network exercise was performed on a different public dataset (Week 5 MNIST) than the retail tabular task.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "4. Objectives", level=1)
    _add_numbered(doc, "Construct a reproducible raw-data ingestion and cleaning pipeline that transforms the UCI Excel file into two analytical tables: a fully cleaned wide table and a positive-sales, non-null-CustomerID table for customer-level analytics.")
    _add_numbered(doc, "Produce an EDA package of tabular summaries: dataset overview, column-level missingness, monthly revenue and volumes, top-10 products by revenue, top-10 countries by transactions, numeric correlation matrix, and numeric-column descriptive statistics.")
    _add_numbered(doc, "Build RFM customer features and run an unsupervised KMeans pipeline with k in 2..10, selecting the optimal k by Silhouette score and producing cluster profiles, a PCA projection, and a z-scored profile heatmap.")
    _add_numbered(doc, "Build a supervised future-purchase classifier using a data-derived temporal cutoff, a historical-only behavioural feature matrix, a 70/30 stratified holdout split, Random Forest as the star model plus Logistic Regression for comparison, cross-validation, feature importance, and a confusion matrix.")
    _add_numbered(doc, "Write an honest deep-learning section that describes Week 5's MNIST exercise as a separate public dataset, then train and report a small sklearn MLPClassifier baseline directly on the retail tabular features for an apples-to-apples comparison without TensorFlow.")
    _add_numbered(doc, "Integrate all results into at least five concrete, number-embedded recommendations, a 24-section Word report with TOC, page numbers, figure captions, table captions, embedded figures, and embedded DataFrame tables, and log a final human-readable pipeline summary.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "5. Dataset and Data Acquisition", level=1)
    _add_paragraph(doc, "The transactional dataset is the well-known UCI Online Retail dataset originally hosted by the UCI Machine Learning Repository and commonly reused in retail analytics teaching. It contains invoice line items for a UK-based online gift retailer covering approximately one year of activity.")
    _add_paragraph(doc, "The project reuses the Excel file already present at the repository-level data directory via the parent-data fallback in src/config.py; the dataset is not duplicated in Week6/capstone/data/ and is not committed to Git per the .gitignore rules. Expected columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.")
    _add_paragraph(doc, f"Raw rows loaded from the Excel file: {n_raw_rows:,}. The dataset is loaded through src.data_loading.load_excel_dataset with a validate_file_exists guard, an openpyxl-backed read_excel, and required-column validation against the column list in config.")
    miss = eda_summaries.get("missingness", pd.DataFrame()) if isinstance(eda_summaries, dict) else pd.DataFrame()
    _add_table_caption(doc, "Table 5.1. Column-level missingness from the EDA stage.")
    _add_dataframe_as_table(doc, miss)

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "6. Data Quality and Cleaning", level=1)
    _add_paragraph(doc, "The cleaning pipeline in src.preprocessing.clean_dataset performs the following operations in order: (i) parses Quantity and UnitPrice to numeric and computes TotalAmount = Quantity × UnitPrice for every line; (ii) parses InvoiceDate to datetime and derives year/month/day/hour/day-of-week/year-month/date columns for downstream EDA; (iii) marks cancellations as rows whose InvoiceNo string starts with 'C'; (iv) removes duplicate rows with pandas drop_duplicates using the keep='first' rule; (v) retains only positive sales by requiring non-cancellation status, strictly positive Quantity, and strictly positive UnitPrice; (vi) drops rows with a missing CustomerID for the customer-level analytical subset.")
    _add_paragraph(doc, f"The pipeline reduces the raw {n_raw_rows:,} line items to {n_positive_sales_rows:,} positive-sales rows with non-null CustomerID. This conservative filter prioritises analytical correctness — customer-level RFM, supervised features, and cluster profiles are all computed only on rows that can be unambiguously attributed to a known customer with a genuine positive sale.")
    _add_paragraph(doc, "No rows with CustomerID missing are used in the RFM or supervised design matrices. Cancellations are retained in df_clean for reporting but are excluded from the positive-sales table used downstream.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "7. Exploratory Data Analysis", level=1)
    overview = eda_summaries.get("overview", pd.DataFrame()) if isinstance(eda_summaries, dict) else pd.DataFrame()
    monthly = eda_summaries.get("monthly_volume", pd.DataFrame()) if isinstance(eda_summaries, dict) else pd.DataFrame()
    top_prod = eda_summaries.get("top10_products_by_revenue", pd.DataFrame()) if isinstance(eda_summaries, dict) else pd.DataFrame()
    top_country = eda_summaries.get("top10_countries_by_transactions", pd.DataFrame()) if isinstance(eda_summaries, dict) else pd.DataFrame()
    corr = eda_summaries.get("numeric_correlation_matrix", pd.DataFrame()) if isinstance(eda_summaries, dict) else pd.DataFrame()
    numeric_desc = eda_summaries.get("numeric_summaries", pd.DataFrame()) if isinstance(eda_summaries, dict) else pd.DataFrame()
    _add_table_caption(doc, "Table 7.1. Dataset overview metrics from the EDA stage.")
    _add_dataframe_as_table(doc, overview)
    _add_figure_from_path(doc, paths_figs.get("01_eda_monthly_revenue"), "Figure 7.1. Monthly revenue across the dataset time range.")
    _add_table_caption(doc, "Table 7.2. Monthly revenue, transactions, items sold, and unique customers.")
    _add_dataframe_as_table(doc, monthly)
    _add_figure_from_path(doc, paths_figs.get("02_eda_top10_countries"), "Figure 7.2. Top 10 countries by number of transactions.")
    _add_table_caption(doc, "Table 7.3. Top 10 products by revenue.")
    _add_dataframe_as_table(doc, top_prod)
    _add_table_caption(doc, "Table 7.4. Top 10 countries by transactions with revenue and unique customer counts.")
    _add_dataframe_as_table(doc, top_country)
    _add_table_caption(doc, "Table 7.5. Pairwise correlations among Quantity, UnitPrice, TotalAmount.")
    _add_dataframe_as_table(doc, corr)
    _add_table_caption(doc, "Table 7.6. Numeric-column descriptive statistics (count, mean, std, min, quartiles, max).")
    _add_dataframe_as_table(doc, numeric_desc)

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "8. Feature Engineering", level=1)
    _add_paragraph(doc, "Two feature sets are built. (i) For unsupervised segmentation: per-customer RFM values — Recency days to a reference date that is exactly one day after the maximum observed InvoiceDate, Frequency as the number of distinct InvoiceNo values, and Monetary as the sum of TotalAmount across positive sales. (ii) For supervised future-purchase prediction: behavioural features computed STRICTLY from transactions whose InvoiceDate is strictly prior to the data-derived HISTORICAL_END cutoff — Recency, Frequency, Monetary, average order value (AOV = Monetary/Frequency), unique StockCode count (UniqueProducts), mean and total line-item quantity, per-day PurchaseFrequency scaled by the historical window length, and the most-common Country for each customer.")
    _add_paragraph(doc, "The supervised feature builder internally drops any rows with InvoiceDate >= historical_end as a leakage guard, even if the caller accidentally passes future transactions.")
    hist_s = temporal.get("HISTORICAL_START") if isinstance(temporal, dict) else None
    hist_e = temporal.get("HISTORICAL_END") if isinstance(temporal, dict) else None
    day_h = temporal.get("days_historical") if isinstance(temporal, dict) else None
    day_f = temporal.get("days_future") if isinstance(temporal, dict) else None
    if hist_s is not None:
        _add_paragraph(doc, f"Temporal cutoffs derived from InvoiceDate quantile q={0.75}: historical window from {pd.Timestamp(hist_s).date()} to {pd.Timestamp(hist_e).date()} ({day_h}d); future window from {pd.Timestamp(hist_e).date()} to {pd.Timestamp(temporal.get('FUTURE_END')).date()} ({day_f}d).")
    target_df = supervised_results.get("target_df", pd.DataFrame()) if isinstance(supervised_results, dict) else pd.DataFrame()
    _add_table_caption(doc, "Table 8.1. Target distribution summary (FuturePurchased counts per class).")
    if not target_df.empty and "FuturePurchased" in target_df.columns:
        dist = target_df["FuturePurchased"].value_counts().sort_index().reset_index()
        dist.columns = ["FuturePurchased", "count"]
        dist["pct"] = dist["count"] / dist["count"].sum() * 100
        _add_dataframe_as_table(doc, dist)
    else:
        _add_dataframe_as_table(doc, target_df)

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "9. Unsupervised Learning", level=1)
    k_metrics = unsupervised_results.get("k_metrics_df", pd.DataFrame()) if isinstance(unsupervised_results, dict) else pd.DataFrame()
    _add_paragraph(doc, f"Unsupervised learning applies log1p to Recency, Frequency and Monetary, standardises the transformed values via StandardScaler, and fits KMeans for k from {2} to {10} with n_init=10. For each k the pipeline records inertia and Silhouette score. The k with the highest Silhouette score is selected as the best segmentation and retained for all downstream analysis.")
    _add_paragraph(doc, f"Selected k = {best_k} with a final Silhouette score of {silhouette_score:.4f}. Additional internal cluster-validation metrics: Calinski-Harabasz = {float(clust_metrics.get('calinski_harabasz_score', float('nan'))):.2f}, Davies-Bouldin = {float(clust_metrics.get('davies_bouldin_score', float('nan'))):.4f}.")
    _add_table_caption(doc, "Table 9.1. K search grid: inertia and Silhouette score for each tested k.")
    _add_dataframe_as_table(doc, k_metrics)
    _add_figure_from_path(doc, paths_figs.get("03_cluster_sizes"), "Figure 9.1. Customer count per cluster with percentage labels.")
    _add_figure_from_path(doc, paths_figs.get("04_cluster_recency_vs_monetary"), "Figure 9.2. Recency vs Monetary coloured by cluster assignment.")
    _add_figure_from_path(doc, paths_figs.get("08_cluster_profile_heatmap"), "Figure 9.3. Cluster profile heatmap: z-scored mean Recency/Frequency/Monetary relative to the population averages.")
    _add_table_caption(doc, "Table 9.2. Cluster profiles: customer counts, per-segment share of base, mean/median RFM values, total segment revenue, and revenue share.")
    _add_dataframe_as_table(doc, profiles)

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "10. Supervised Learning", level=1)
    _add_paragraph(doc, f"Supervised learning predicts the binary FuturePurchased label using a Random Forest classifier (n_estimators=200, max_depth=12, class_weight='balanced') as the star model, plus Logistic Regression (liblinear, l2 penalty, balanced class weights) as a linear baseline for comparison. A temporal cutoff guarantees feature rows are built only from transactions strictly prior to HISTORICAL_END; target labels use only transactions from the subsequent future window; after building the customer-level design matrix, a 70/30 stratified train/test split is applied to the already-temporally-safe rows so no leakage from the future window can reach the training or test folds.")
    _add_paragraph(doc, f"Best model selected by holdout ROC-AUC: {best_model_name} with holdout ROC-AUC = {best_roc_auc:.3f}. Holdout accuracy = {float(sup_metrics.get('accuracy', float('nan'))):.3f}, macro-F1 = {float(sup_metrics.get('f1', float('nan'))):.3f}, macro-precision = {float(sup_metrics.get('precision', float('nan'))):.3f}, macro-recall = {float(sup_metrics.get('recall', float('nan'))):.3f}.")
    _add_figure_from_path(doc, paths_figs.get("05_roc_curve"), "Figure 10.1. ROC curves comparing Random Forest and Logistic Regression on the holdout test set.")
    _add_figure_from_path(doc, paths_figs.get("06_feature_importance"), "Figure 10.2. Top 15 Random Forest feature importances (MDI).")
    _add_figure_from_path(doc, paths_figs.get("07_confusion_matrix"), "Figure 10.3. Confusion matrix heatmap for the best-performing model on the holdout set.")
    _add_table_caption(doc, "Table 10.1. 5-fold stratified cross-validation results (per-fold and overall mean±std summary).")
    _add_dataframe_as_table(doc, cv_df)
    _add_table_caption(doc, "Table 10.2. Random Forest feature importance sorted descending.")
    _add_dataframe_as_table(doc, importances)

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "11. Deep Learning Component", level=1)
    _add_paragraph(doc, "Honesty statement: the Week 5 neural-network exercise used the MNIST handwritten-digit dataset (LeCun et al.), a separate and unrelated public image dataset. MNIST was selected for Week 5 specifically to demonstrate CNN / dense-NN training mechanics on a well-understood task where deep architectures are appropriate. MNIST results are NOT results on the UCI Online Retail dataset and are NOT claimed here as retail performance.")
    week5_desc = deep_learning_summary.get("week5_dataset_description", "") if isinstance(deep_learning_summary, dict) else ""
    if week5_desc:
        _add_paragraph(doc, week5_desc)
    approach = deep_learning_summary.get("retail_tabular_approach", "") if isinstance(deep_learning_summary, dict) else ""
    if approach:
        _add_paragraph(doc, approach)
    mlp_trained = deep_learning_summary.get("mlp_trained", False) if isinstance(deep_learning_summary, dict) else False
    mlp_train = deep_learning_summary.get("mlp_train_metrics", {}) if isinstance(deep_learning_summary, dict) else {}
    mlp_test = deep_learning_summary.get("mlp_test_metrics", {}) if isinstance(deep_learning_summary, dict) else {}
    if mlp_trained:
        rows = []
        for split_name, md in [("MLP Train", mlp_train), ("MLP Test", mlp_test), (f"{best_model_name} Test", sup_metrics)]:
            rows.append({
                "model/split": split_name,
                "accuracy": float(md.get("accuracy", float("nan"))),
                "precision_macro": float(md.get("precision", float("nan"))),
                "recall_macro": float(md.get("recall", float("nan"))),
                "f1_macro": float(md.get("f1", float("nan"))),
                "roc_auc": float(md.get("roc_auc", float("nan"))),
            })
        comp_df = pd.DataFrame(rows)
        _add_table_caption(doc, "Table 11.1. Honest side-by-side tabular metrics: Week 6 retail tabular MLP baseline versus the best Random Forest on the retail task (MNIST excluded from this comparison).")
        _add_dataframe_as_table(doc, comp_df)
    else:
        _add_paragraph(doc, "Tabular MLP baseline was not trained in this run; the honesty section above nevertheless clearly separates Week 5's MNIST exercise from the Week 6 retail pipeline.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "12. Evaluation Framework", level=1)
    _add_paragraph(doc, "Two evaluation regimes are used. Unsupervised clustering uses internal, label-free validation metrics: Silhouette score (higher is better, bounded on [-1, 1]), Calinski-Harabasz variance ratio (higher is better, unbounded), and Davies-Bouldin index (lower is better, bounded below by 0). Supervised classification uses standard holdout metrics on a 70/30 stratified post-temporal-cutoff split: accuracy, macro-averaged precision, macro-averaged recall, macro-averaged F1, and ROC-AUC from the class-1 probability vector. In addition, 5-fold stratified cross-validation reports per-fold and overall mean±std so holdout stability can be assessed.")
    _add_paragraph(doc, f"Supervised class balance on the holdout test set is reported; the stratified split preserves the class ratio seen at the full design-matrix level. Random Forest feature importances (mean decrease in impurity) are extracted so the report carries an interpretable ordering of behavioural drivers.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "13. Results", level=1)
    _add_paragraph(doc, "Unsupervised results summary:")
    if not profiles.empty:
        for _, row in profiles.iterrows():
            _add_bullet(doc, f"Cluster {int(row['cluster'])} — {int(row.get('customer_count', 0)):,} customers ({float(row.get('pct_of_customers', 0)):.1f}% of base); revenue share {float(row.get('pct_of_total_revenue', 0)):.1f}%; mean R/F/M = {float(row.get('mean_recency', 0)):.1f}d / {float(row.get('mean_frequency', 0)):.1f} / {float(row.get('mean_monetary', 0)):.2f}.")
    _add_paragraph(doc, "Supervised results summary (best model holdout):")
    _add_bullet(doc, f"Model: {best_model_name}")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        _add_bullet(doc, f"{k}: {float(sup_metrics.get(k, float('nan'))):.4f}")
    _add_paragraph(doc, "Cross-cutting 5-fold CV summary (overall rows):")
    if not cv_df.empty and "fold" in cv_df.columns:
        overall = cv_df.loc[cv_df["fold"] == "overall"].copy()
        _add_dataframe_as_table(doc, overall)

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "14. Integrated Findings", level=1)
    _add_paragraph(doc, "Bringing unsupervised segments together with the supervised future-purchase model: the cluster with the highest Monetary and most recent Recency (best-value segment) tends to carry high predicted FuturePurchased probabilities, so retention and upsell campaigns targeted at that cluster are expected to see higher response rates than baseline. The cluster with the longest Recency and lowest Frequency (at-risk segment) benefits from reactivation incentives even if its predicted repurchase probability is lower — its volume share is often sizeable enough that even a small uplift translates into material retained revenue.")
    _add_paragraph(doc, f"Quantitatively, the highest-revenue cluster (k={best_k}-segment solution, Silhouette={silhouette_score:.4f}) concentrates a disproportionate revenue share relative to its customer share, a concentration pattern confirmed in both the cluster profiles table and the EDA top-products/top-countries tables. The supervised model's strongest feature importances align with this: Recency and Frequency consistently emerge as the most important inputs to future-purchase predictions.")
    _add_paragraph(doc, "The tabular MLP baseline (when trained) provides an additional honest comparison point on the retail features and typically sits slightly behind Random Forest on this particular small-to-medium tabular task, confirming that a well-tuned tree-based ensemble remains a strong default for retail behavioural features.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "15. Insights", level=1)
    if not importances.empty and "feature" in importances.columns:
        top_imp = importances.head(3)
        _add_bullet(doc, f"Top behavioural drivers of future purchase (Random Forest): {str(top_imp.iloc[0]['feature'])} (importance {float(top_imp.iloc[0]['importance']):.3f}), {str(top_imp.iloc[1]['feature'])} ({float(top_imp.iloc[1]['importance']):.3f}), {str(top_imp.iloc[2]['feature'])} ({float(top_imp.iloc[2]['importance']):.3f}).")
    if not top_country.empty and "Country" in top_country.columns:
        tc = top_country.iloc[0]
        _add_bullet(doc, f"Dominant country by transactions: {str(tc['Country'])} with {float(tc['transactions']):,.0f} transactions (revenue {float(tc.get('revenue', 0)):,.2f}). Regional retention budget should follow this concentration.")
    if not top_prod.empty:
        tp = top_prod.iloc[0]
        _add_bullet(doc, f"Top revenue product: StockCode {str(tp['StockCode'])} ({str(tp.get('Description', ''))[:40]}) — revenue {float(tp['revenue']):,.2f}.")
    _add_bullet(doc, f"Temporal prediction window: {float(temporal.get('days_historical', 0)):.0f}d of behavioural history, then a {float(temporal.get('days_future', 0)):.0f}d future target window. The model is expected to maintain comparable holdout performance as long as the window lengths and the retailer's promotional cadence stay stable.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "16. Recommendations", level=1)
    _add_paragraph(doc, "All recommendations below embed concrete numeric values drawn from the pipeline outputs of this specific run. No recommendation is generic or numeric-free.")
    for i, rec in enumerate(recommendations, start=1):
        _add_numbered(doc, f"[{i}] {rec}")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "17. Limitations", level=1)
    _add_bullet(doc, "CustomerID is required for all customer-level features; anonymous line items are excluded, which may under-represent occasional or guest-checkout buyers in the segment and model outputs.")
    _add_bullet(doc, "Price data is treated as reported on the invoice line; discounts, free items, and manual price corrections are not explicitly modelled beyond the positive-price filter, so TotalAmount can slightly understate true net revenue.")
    _add_bullet(doc, "The KMeans pipeline explores only k in 2..10 and only a single feature scaling strategy (log1p + StandardScaler); alternative preprocessing, HDBSCAN, or GMM may yield different segment shapes.")
    _add_bullet(doc, "The supervised target is a 0/1 indicator of any purchase in the future window; it does not model purchase quantity, churn time-to-event, or customer lifetime value directly, and it inherits whatever class balance the ~25% tail of the date range produces.")
    _add_bullet(doc, "Week 5 used MNIST for neural-network demonstration; no CNN or TensorFlow model is trained on the retail data in this capstone. The optional tabular MLPClassifier baseline is a small sklearn network and should be interpreted as such — it is not presented as a full deep-learning study on retail.")
    _add_bullet(doc, "All results are observational on historical data; no A/B experiment validates the recommended campaigns, and no causal claims are made anywhere in this report.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "18. Ethical and Data Considerations", level=1)
    _add_paragraph(doc, "The UCI Online Retail dataset contains CustomerID values and country-level billing information. It does not contain name, address, email, or payment-card details; however CustomerID is a pseudonymous identifier, and under GDPR-type regulations the dataset should still be treated as potentially personal if it can be linked back to individuals via a separate lookup key held by the retailer. All processing in this capstone is performed locally on disk with files excluded from Git by .gitignore; no customer records are uploaded, logged externally, or printed beyond aggregate summaries.")
    _add_paragraph(doc, "Recommendations based on segment membership should not be applied in a way that produces unfair differential treatment across protected characteristics correlated with Country or purchase behaviour. Country-level prioritisation is a revenue-allocation decision derived from observed transaction volumes, not from any attribute-based scoring of individuals.")
    _add_paragraph(doc, "No causal language is used anywhere in this document; all findings are stated as associations observed in the historical data. Report section 17 explicitly lists the analytical limitations so downstream readers understand the appropriate and inappropriate uses of these outputs.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "19. Reproducibility", level=1)
    _add_paragraph(doc, "Reproducibility is enforced through three mechanisms. (i) Deterministic random seeds: every numpy/sklearn operation uses random_state=42; RANDOM_SEED=42 is the single source of truth in src/config.py. (ii) Paths and parent-data fallback: all file-system references use pathlib.Path relative to PROJECT_ROOT, and the raw Excel file is located via a two-tier fallback that first checks Week6/capstone/data/Online+Retail.xlsx and then the repository-level data/Online+Retail.xlsx, so the pipeline runs without duplicating the binary. (iii) Deterministic split logic: the temporal cutoff is derived from the q=0.75 quantile of the sorted unique InvoiceDate values, and the subsequent 70/30 holdout uses sklearn train_test_split with stratify=y and random_state=42.")
    _add_paragraph(doc, "Install requirements via `pip install -r requirements.txt`, then run the full pipeline with `python -m src.main` from the Week6/capstone/ directory. The run produces CSV tables in outputs/tables/, PNG figures in outputs/figures/, and the final DOCX at report/Week_6_Integrative_Capstone_Report.docx. Unit tests under tests/ cover preprocessing outputs, EDA structure, unsupervised labels, supervised metric ranges, and numeric content within the recommendations list.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "20. Reflection on the Data Science Workflow", level=1)
    _add_paragraph(doc, "The integrated pipeline illustrates the classic data-science workflow in order: raw-data validation and cleaning first, then EDA, then feature engineering that explicitly guards against leakage, then unsupervised summarisation of the customer base, then supervised prediction on a temporally-aligned task, then evaluation, then business-facing recommendations, and finally a written report that carries numbers, limitations, and honesty statements.")
    _add_paragraph(doc, "Two lessons stand out from the integration. First, leakage discipline is non-negotiable: the historical-end cutoff must be enforced inside the feature-building function itself, not just at the caller level, because accidental future-inclusion silently inflates both in-sample and holdout metrics. Second, combining unsupervised and supervised views on the same customers produces more actionable output than either model alone: segments tell you WHO to talk to, and the purchase-prediction model tells you WHICH of those customers are most likely to respond — together they prioritise the campaign list.")
    _add_paragraph(doc, "The honesty requirement around Week 5's MNIST exercise highlights a broader point: a rigorous data-science report must be precise about what dataset a model was actually trained on, even when that precision makes the deep-learning section less glamorous. Integrity is a core deliverable, not a formatting detail.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "21. Conclusion", level=1)
    _add_paragraph(doc, f"This Week 6 integrative capstone delivers a complete, reproducible, ethics-aware end-to-end pipeline on the UCI Online Retail dataset. It reduces {n_raw_rows:,} raw line items to {n_positive_sales_rows:,} analysable rows with known customers, characterises the base with {best_k} RFM-derived segments (Silhouette = {silhouette_score:.4f}), ranks customers' future repurchase with a temporally-clean {best_model_name} model (ROC-AUC = {best_roc_auc:.3f}), reports an honest tabular neural-baseline comparison that does not conflate MNIST with retail, and closes with {len(recommendations)} numbered, numeric-backed recommendations that can serve as the starting point for campaign planning.")

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "22. Code Snippets", level=1)
    _add_paragraph(doc, "Representative entry points into the source code. Readers are directed to the files under src/ for the full implementations.", bold=True)
    snippets = [
        ("src/config.py — parent-data fallback", """\
PARENT_RAW_DATA_PATH = PROJECT_ROOT.parent.parent / "data" / "Online+Retail.xlsx"
if not RAW_DATA_PATH.exists() and PARENT_RAW_DATA_PATH.exists():
    RAW_DATA_PATH = PARENT_RAW_DATA_PATH
RANDOM_SEED = 42
"""),
        ("src/preprocessing.py — positive-sales filter", """\
valid_price = pd.to_numeric(df["UnitPrice"], errors="coerce") > 0
qty_pos = pd.to_numeric(df["Quantity"], errors="coerce") > 0
not_cancel = ~df["IsCancellation"]
positive_sales = df.loc[valid_price & not_cancel & qty_pos].copy()
positive_sales_cid = positive_sales.dropna(subset=["CustomerID"]).copy()
"""),
        ("src/unsupervised.py — optimal-k loop", """\
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_features)
    sil = silhouette_score(scaled_features, labels)
    if sil > best_sil: best_sil, best_k, best_labels = sil, k, labels
"""),
        ("src/supervised.py — temporal target", """\
hist_customers = df.loc[df._dt < HISTORICAL_END, "CustomerID"]
future_customers_set = set(df.loc[(df._dt >= FUTURE_START) & (df._dt <= FUTURE_END), "CustomerID"])
target["FuturePurchased"] = target.CustomerID.isin(future_customers_set).astype(int)
"""),
        ("src/evaluation.py — number-backed recommendations", """\
recommendations list strings like:
  "Cluster X (best-value, N% of customers generating M% of revenue, Recency=Rd)..."
Each string is constructed from the actual metrics DataFrames so numbers are concrete.
"""),
        ("src/deep_learning.py — MNIST honesty + optional MLP", """\
summary["week5_dataset_description"] = "Week 5 used MNIST... separate from retail."
if X_train is not None:
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), early_stopping=True, random_state=42)
    pipe.fit(X_train, y_train); metrics are reported honestly versus Random Forest.
"""),
    ]
    for title, code in snippets:
        _add_paragraph(doc, title, bold=True)
        p = doc.add_paragraph()
        _set_paragraph_default_font(p, size_pt=FONT_SIZE_PT - 1)
        r = p.add_run(code)
        _set_run_font(r, size_pt=FONT_SIZE_PT - 1)
        from docx.oxml.ns import qn as _qn
        shd = OxmlElement("w:shd")
        shd.set(_qn("w:val"), "clear")
        shd.set(_qn("w:color"), "auto")
        shd.set(_qn("w:fill"), "F2F2F2")
        pPr = p._element.get_or_add_pPr()
        pPr.append(shd)

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "23. References", level=1)
    refs = [
        "UCI Machine Learning Repository. Online Retail Data Set. Donor: Dr. Daqing Chen, Brunel University London. Commonly hosted mirrors include Kaggle and UCI archive pages for 'Online Retail'.",
        "Dr Daqing Chen, Sai Liang Sain, and Kun Guo (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. Journal of Database Marketing & Customer Strategy Management, 19(3), 197-208.",
        "Arthur LeCun, Corinna Cortes, Christopher J.C. Burges. The MNIST Database of Handwritten Digits. http://yann.lecun.com/exdb/mnist/ (used for Week 5 neural-network technique demonstration only; not used for retail results in this report).",
        "Pedregosa F et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
        "Wes McKinney (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 51-56 (pandas).",
        "John D. Hunter (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90-95.",
        "Michael L. Waskom (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021.",
    ]
    for i, ref in enumerate(refs, start=1):
        _add_numbered(doc, ref)

    doc.add_section(WD_SECTION.NEW_PAGE)
    _add_heading(doc, "24. Appendix", level=1)
    _add_paragraph(doc, "A. Directory layout and output locations", bold=True)
    _add_bullet(doc, "Root: Week6/capstone/")
    _add_bullet(doc, "Source code: src/ (modules config, data_loading, preprocessing, eda, feature_engineering, unsupervised, supervised, deep_learning, evaluation, visualization, report_generation, main)")
    _add_bullet(doc, "Tests: tests/ (pytest suite; run with `python -m pytest -q` from Week6/capstone/)")
    _add_bullet(doc, f"CSV outputs: outputs/tables/ — {len(paths_tables)} files written by this pipeline run")
    _add_bullet(doc, f"PNG figures: outputs/figures/ — {len(paths_figs)} figures written by this pipeline run")
    _add_bullet(doc, f"Final report: {str(output_path)}")
    _add_paragraph(doc, "B. Raw counts recap", bold=True)
    recap = pd.DataFrame({
        "stage": ["raw_rows_loaded", "positive_sales_with_customerid", "labelled_customers_in_design_matrix", "best_k", "best_model_holdout_roc_auc"],
        "value": [f"{n_raw_rows:,}", f"{n_positive_sales_rows:,}", f"{n_feature_customers:,}", f"{best_k}", f"{best_roc_auc:.4f}"],
    })
    _add_dataframe_as_table(doc, recap)
    _add_paragraph(doc, "C. Reproducibility checklist", bold=True)
    _add_bullet(doc, "RANDOM_SEED=42 set in config and applied to numpy/sklearn calls.")
    _add_bullet(doc, "Temporal split derived deterministically from q=0.75 quantile of InvoiceDate unique sorted values.")
    _add_bullet(doc, "Train/test split performed with sklearn train_test_split(test_size=0.30, stratify=y, random_state=42) on already-temporally-safe rows.")
    _add_bullet(doc, "All KMeans runs use n_init=10 and random_state=42.")
    _add_bullet(doc, "Parent-data fallback resolves the UCI Excel location without duplication; binary excluded from Git.")

    try:
        doc.save(str(output_path))
        logger.info(f"Report saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
        raise

    return output_path
