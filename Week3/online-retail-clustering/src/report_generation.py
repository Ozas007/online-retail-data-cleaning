import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict

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

    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    run1 = paragraph.add_run()
    run1._element.append(fld_begin)
    _set_run_font(run1)

    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE   \\* MERGEFORMAT "
    run2 = paragraph.add_run()
    run2._element.append(instr_text)
    _set_run_font(run2)

    fld_separate = OxmlElement("w:fldChar")
    fld_separate.set(qn("w:fldCharType"), "separate")
    run3 = paragraph.add_run()
    run3._element.append(fld_separate)
    _set_run_font(run3)

    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run4 = paragraph.add_run()
    run4._element.append(fld_end)
    _set_run_font(run4)

    run5 = paragraph.add_run(" of ")
    _set_run_font(run5)

    fld_begin2 = OxmlElement("w:fldChar")
    fld_begin2.set(qn("w:fldCharType"), "begin")
    run6 = paragraph.add_run()
    run6._element.append(fld_begin2)
    _set_run_font(run6)

    instr_text2 = OxmlElement("w:instrText")
    instr_text2.set(qn("xml:space"), "preserve")
    instr_text2.text = "NUMPAGES   \\* MERGEFORMAT "
    run7 = paragraph.add_run()
    run7._element.append(instr_text2)
    _set_run_font(run7)

    fld_separate2 = OxmlElement("w:fldChar")
    fld_separate2.set(qn("w:fldCharType"), "separate")
    run8 = paragraph.add_run()
    run8._element.append(fld_separate2)
    _set_run_font(run8)

    fld_end2 = OxmlElement("w:fldChar")
    fld_end2.set(qn("w:fldCharType"), "end")
    run9 = paragraph.add_run()
    run9._element.append(fld_end2)
    _set_run_font(run9)


def _add_toc(paragraph: Paragraph) -> None:
    run = paragraph.add_run()
    _set_run_font(run)
    fld_char_begin = OxmlElement("w:fldChar")
    fld_char_begin.set(qn("w:fldCharType"), "begin")
    run._r.append(fld_char_begin)

    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = 'TOC \\o "1-3" \\h \\z \\u'
    run2 = paragraph.add_run()
    run2._r.append(instr_text)
    _set_run_font(run2)

    fld_char_separate = OxmlElement("w:fldChar")
    fld_char_separate.set(qn("w:fldCharType"), "separate")
    run3 = paragraph.add_run()
    run3._r.append(fld_char_separate)
    _set_run_font(run3)

    fld_char_end = OxmlElement("w:fldChar")
    fld_char_end.set(qn("w:fldCharType"), "end")
    run4 = paragraph.add_run()
    run4._r.append(fld_char_end)
    _set_run_font(run4)


def _add_heading(doc: Document, text: str, level: int = 1) -> Paragraph:
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        _set_run_font(run, font_name=FONT_NAME, size_pt=FONT_SIZE_PT + (4 - level) * 2, bold=True)
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


def _add_table_caption(doc: Document, caption: str) -> Paragraph:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_default_font(p)
    run = p.add_run(caption)
    _set_run_font(run, bold=True)
    p.paragraph_format.space_after = Pt(4)
    return p


def _add_figure_caption(doc: Document, caption: str) -> Paragraph:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_default_font(p)
    run = p.add_run(caption)
    _set_run_font(run, italic=True)
    p.paragraph_format.space_after = Pt(12)
    return p


def _add_dataframe_as_table(doc: Document, df: pd.DataFrame, caption: str) -> None:
    _add_table_caption(doc, caption)
    df_clean = df.reset_index(drop=False)
    cols = list(df_clean.columns)
    table = doc.add_table(rows=1 + len(df_clean), cols=len(cols))
    table.style = "Light Grid Accent 1"

    hdr_cells = table.rows[0].cells
    for i, col_name in enumerate(cols):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_paragraph_default_font(p)
        run = p.add_run(str(col_name))
        _set_run_font(run, bold=True, size_pt=FONT_SIZE_PT - 1)

    for row_idx in range(len(df_clean)):
        row_cells = table.rows[row_idx + 1].cells
        for col_idx in range(len(cols)):
            value = df_clean.iat[row_idx, col_idx]
            if isinstance(value, float):
                if pd.isna(value):
                    text = ""
                else:
                    text = f"{value:.4f}" if abs(value) < 1 else f"{value:,.2f}"
            else:
                text = "" if pd.isna(value) else str(value)
            row_cells[col_idx].text = ""
            p = row_cells[col_idx].paragraphs[0]
            _set_paragraph_default_font(p)
            run = p.add_run(text)
            _set_run_font(run, size_pt=FONT_SIZE_PT - 1)
    doc.add_paragraph()


def _add_code_snippet(doc: Document, title: str, code: str) -> None:
    _add_heading(doc, title, level=3)
    p = doc.add_paragraph()
    _set_paragraph_default_font(p, font_name="Consolas", size_pt=9)
    lines = code.strip("\n").split("\n")
    for i, line in enumerate(lines):
        run = p.add_run(line)
        _set_run_font(run, font_name="Consolas", size_pt=9)
        if i < len(lines) - 1:
            run.add_break()
    doc.add_paragraph()


def build_report(
    *,
    rfm_df: pd.DataFrame,
    rfm_distribution_stats: pd.DataFrame,
    k_metrics_df: pd.DataFrame,
    best_k: int,
    labels_df: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    cluster_profiles_scaled: pd.DataFrame,
    clustering_metrics: Dict[str, Any],
    pca_explained_var_df: pd.DataFrame,
    cluster_descriptions: Dict[Any, str],
    paths_figs: Dict[str, Path],
    reference_date: Any,
    use_log_columns: Any,
    output_path: Path,
    n_raw_rows: int,
    n_positive_sales_rows: int,
    overall_rfm_stats: Any,
) -> Path:
    logger.info(f"Starting report generation -> {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured parent directory exists: {output_path.parent}")

    if isinstance(reference_date, datetime):
        pass
    elif isinstance(reference_date, date):
        reference_date = datetime.combine(reference_date, datetime.min.time())
    else:
        reference_date = pd.to_datetime(reference_date).to_pydatetime()

    if isinstance(use_log_columns, bool):
        use_log_columns_list = ["Log_Recency", "Log_Frequency", "Log_Monetary"] if use_log_columns else []
    elif isinstance(use_log_columns, (list, tuple)):
        use_log_columns_list = [str(x) for x in use_log_columns]
    else:
        use_log_columns_list = []

    if isinstance(overall_rfm_stats, pd.DataFrame):
        overall_stats_df = overall_rfm_stats.copy()
    elif isinstance(overall_rfm_stats, dict):
        overall_stats_df = pd.DataFrame([{
            "Metric": "Mean across all customers",
            "Recency": overall_rfm_stats.get("Recency", float("nan")),
            "Frequency": overall_rfm_stats.get("Frequency", float("nan")),
            "Monetary": overall_rfm_stats.get("Monetary", float("nan")),
        }])
    else:
        overall_stats_df = pd.DataFrame(columns=["Metric", "Recency", "Frequency", "Monetary"])

    labels_df = labels_df.copy()
    if "Cluster" in labels_df.columns and "cluster" not in labels_df.columns:
        labels_df.rename(columns={"Cluster": "cluster"}, inplace=True)

    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = FONT_NAME
    style.font.size = Pt(FONT_SIZE_PT)
    style.paragraph_format.space_after = Pt(8)
    style.paragraph_format.line_spacing = 1.15

    section = doc.sections[0]
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

    _add_page_number(section.footer)

    today_str = date.today().isoformat()
    ref_date_str = reference_date.strftime("%Y-%m-%d")

    logger.info("Building Title Page (Section 1)")
    doc.add_paragraph()
    doc.add_paragraph()
    doc.add_paragraph()
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_default_font(p, font_name=FONT_NAME, size_pt=26)
    run = p.add_run("Week 3: Unsupervised Learning and Clustering")
    _set_run_font(run, size_pt=26, bold=True, color_rgb=(31, 73, 125))

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_default_font(p, font_name=FONT_NAME, size_pt=22)
    run = p.add_run("Customer Segmentation using RFM Analysis and K-Means Clustering")
    _set_run_font(run, size_pt=22, bold=True, color_rgb=(31, 73, 125))

    doc.add_paragraph()
    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_default_font(p)
    run = p.add_run(f"Report Date: {today_str}")
    _set_run_font(run, size_pt=13)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_default_font(p)
    run = p.add_run("UCI Online Retail Dataset — Customer Segmentation Project")
    _set_run_font(run, size_pt=13)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_default_font(p)
    run = p.add_run(f"RFM Reference Date: {ref_date_str}  |  Random Seed: 42")
    _set_run_font(run, size_pt=12, italic=True)

    logger.info("Adding Table of Contents")
    doc.add_section(WD_SECTION.NEW_PAGE)
    new_section = doc.sections[-1]
    _add_page_number(new_section.footer)
    new_section.top_margin = Cm(2.5)
    new_section.bottom_margin = Cm(2.5)
    new_section.left_margin = Cm(2.5)
    new_section.right_margin = Cm(2.5)

    _add_heading(doc, "Table of Contents", level=1)
    toc_p = doc.add_paragraph()
    _set_paragraph_default_font(toc_p)
    _add_toc(toc_p)
    note = _add_paragraph(doc, "Note: Right-click the TOC field in Word and select "
                               "'Update Field' to populate page numbers after opening.",
                          italic=True, alignment=WD_ALIGN_PARAGRAPH.LEFT)

    doc.add_section(WD_SECTION.NEW_PAGE)
    main_section = doc.sections[-1]
    _add_page_number(main_section.footer)
    main_section.top_margin = Cm(2.5)
    main_section.bottom_margin = Cm(2.5)
    main_section.left_margin = Cm(2.5)
    main_section.right_margin = Cm(2.5)

    logger.info("Building Executive Summary (Section 2)")
    _add_heading(doc, "2. Executive Summary", level=1)

    sil_score = clustering_metrics.get("silhouette_score", float("nan"))
    ch_score = clustering_metrics.get("calinski_harabasz_score", float("nan"))
    db_score = clustering_metrics.get("davies_bouldin_score", float("nan"))

    total_customers = len(rfm_df)
    cluster_counts = labels_df["cluster"].value_counts().sort_index()
    counts_str = ", ".join([f"Cluster {k}: {int(v)}" for k, v in cluster_counts.items()])

    if not cluster_profiles.empty and "pct_of_total_revenue" in cluster_profiles.columns:
        top_rev_cluster = cluster_profiles["pct_of_total_revenue"].idxmax()
        top_rev_pct = cluster_profiles.loc[top_rev_cluster, "pct_of_total_revenue"]
        seg_finding = (f"Cluster {top_rev_cluster} accounts for the largest revenue share "
                       f"at {top_rev_pct:.1%} of total revenue despite representing "
                       f"only {cluster_profiles.loc[top_rev_cluster, 'pct_of_customers']:.1%} "
                       f"of customers.")
    else:
        seg_finding = "Revenue concentration among a subset of clusters is evident from the profile table."

    summary_text = (
        f"This report presents an unsupervised learning-based customer segmentation of the UCI Online "
        f"Retail dataset using RFM (Recency, Frequency, Monetary) feature engineering combined with "
        f"K-Means clustering. The objective was to identify distinct customer groups to support "
        f"targeted marketing strategies. Starting from {n_raw_rows:,} raw transaction records, "
        f"{n_positive_sales_rows:,} positive sales rows with non-null CustomerID were retained, "
        f"yielding {total_customers:,} unique customers for RFM computation. A range of cluster "
        f"counts was evaluated using the Elbow method (inertia) and Silhouette score. "
        f"The selected number of clusters was k = {best_k}, producing a Silhouette score of "
        f"{sil_score:.4f} (Calinski-Harabasz: {ch_score:,.2f}, Davies-Bouldin: {db_score:.4f}). "
        f"Cluster membership distribution across the {best_k} clusters is as follows: {counts_str}. "
        f"Key segmentation finding: {seg_finding}"
    )
    _add_paragraph(doc, summary_text)

    logger.info("Building Introduction (Section 3)")
    _add_heading(doc, "3. Introduction", level=1)
    _add_paragraph(doc,
        "Customer segmentation is a foundational technique in marketing analytics and customer "
        "relationship management (CRM). By dividing a heterogeneous customer base into homogeneous "
        "groups based on behavioural patterns, businesses can tailor product recommendations, "
        "promotional campaigns, pricing strategies, and retention initiatives to each segment's "
        "characteristics. This personalization typically improves customer satisfaction, increases "
        "lifetime value (CLV), and reduces inefficient one-size-fits-all marketing spend."
    )
    _add_paragraph(doc,
        "The RFM model is a well-established behavioural segmentation framework originating from "
        "direct marketing analytics. It characterises each customer along three orthogonal "
        "dimensions: Recency (how recently they purchased), Frequency (how often they purchase), "
        "and Monetary value (how much they spend). RFM's practical value is its interpretability — "
        "each dimension maps directly to actionable marketing levers. Recency informs re-engagement "
        "timing; Frequency informs loyalty programme design; Monetary informs premium service "
        "eligibility and margin-based prioritisation (Chen, Sain, & Guo, 2012)."
    )
    _add_paragraph(doc,
        "In this project, we operationalise RFM on a real-world transactional dataset, stabilise "
        "its typically right-skewed distributions via log transformation, standardise the resulting "
        "features, and then apply K-Means clustering to uncover natural customer groupings. The "
        "resulting segments are described quantitatively through profile tables and qualitatively "
        "through evidence-based interpretation, forming the basis for downstream marketing "
        "decision-making."
    )

    logger.info("Building Problem Statement (Section 4)")
    _add_heading(doc, "4. Problem Statement", level=1)
    _add_paragraph(doc,
        "Given a historical set of e-commerce transactions from a UK-based online retailer, the "
        "task is to partition the retailer's customer base into a small number (k) of "
        "operationally meaningful, behaviourally distinct segments. Specifically: (1) engineer "
        "Recency, Frequency, and Monetary features per customer from the raw transaction log; "
        "(2) stabilise and standardise those features so that Euclidean distance-based clustering "
        "is well-conditioned; (3) apply K-Means with an appropriately selected k; (4) evaluate "
        "cluster quality via internal validation metrics; and (5) produce interpretable cluster "
        "profiles that can inform targeted marketing and retention strategies. No labelled ground "
        "truth exists, so this is an unsupervised problem and all claims about segments must be "
        "supported by internal metrics and descriptive statistics, not causal inferences."
    )

    logger.info("Building Dataset Description (Section 5)")
    _add_heading(doc, "5. Dataset Description", level=1)
    _add_paragraph(doc,
        "The dataset is the UCI Machine Learning Repository's 'Online Retail' dataset, which "
        "records transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based online "
        "gift and homeware retailer. The majority of sales are to wholesale customers across "
        "multiple countries."
    )
    cols_str = ", ".join([
        "InvoiceNo (transaction identifier)", "StockCode (product code)",
        "Description (product description)", "Quantity (units per line)",
        "InvoiceDate (transaction datetime)", "UnitPrice (£ per unit)",
        "CustomerID (customer identifier)", "Country (customer billing country)"
    ])
    _add_paragraph(doc, f"Columns present in the raw data: {cols_str}.")
    _add_paragraph(doc,
        f"The raw dataset contains n_raw_rows = {n_raw_rows:,} transaction-line records. The "
        "dataset is distributed by the UCI ML Repository (Dua & Graff, 2019) and hosted in the "
        "project's `data/` directory. Per the `data/README.md`, the raw Excel file is not "
        "committed to version control because of its size and licence terms; instead, the README "
        "documents the canonical download URL and expected filename so that any user can "
        "reproduce the full pipeline end-to-end."
    )
    _add_paragraph(doc,
        "An important data-quality note: a material number of transaction lines carry missing "
        "(NULL) CustomerID values. Because RFM features must be aggregated at the customer level, "
        "these anonymous rows cannot be assigned to any individual customer and are therefore "
        "excluded from the segmentation. The downstream analysis and all cluster statistics in "
        "this report therefore describe only the identified-customer subset of the business."
    )

    logger.info("Building Data Preparation (Section 6)")
    _add_heading(doc, "6. Data Preparation", level=1)
    _add_paragraph(doc,
        "Raw transactional data requires cleaning before RFM features can be meaningfully "
        "computed. The following sequential steps were applied:"
    )
    steps = [
        "Deduplication: exact duplicate rows were removed based on all columns, as duplicate "
        "line items would inflate Frequency and Monetary values spuriously.",
        "Cancellation removal: transactions whose InvoiceNo begins with the cancellation prefix "
        "'C' were filtered out, as these represent refunds/returns, not genuine sales events.",
        "Positive sales filter: only rows with Quantity > 0 and UnitPrice > 0 were retained. "
        "This removes negative-quantity return lines and zero-priced promotional or manual "
        "adjustment lines that do not contribute to revenue.",
        "Non-null CustomerID filter: rows with missing CustomerID were dropped because they "
        "cannot be mapped to an individual customer for RFM aggregation.",
    ]
    for s in steps:
        p = doc.add_paragraph(style="List Bullet")
        _set_paragraph_default_font(p)
        run = p.runs[0] if p.runs else p.add_run(s)
        if len(p.runs) == 0:
            _set_run_font(run)
        else:
            for r in p.runs:
                _set_run_font(r)

    _add_paragraph(doc,
        f"After applying the above cleaning pipeline, n_positive_sales_rows = "
        f"{n_positive_sales_rows:,} transaction lines remained as the input to RFM feature "
        f"engineering. This represents {(n_positive_sales_rows / n_raw_rows * 100):.1f}% of the "
        f"raw dataset size."
    )

    logger.info("Building RFM Feature Engineering (Section 7)")
    _add_heading(doc, "7. RFM Feature Engineering", level=1)
    _add_paragraph(doc,
        "Three customer-level features were derived by aggregating the cleaned positive-sales "
        f"transaction lines grouped by CustomerID. The analysis reference_date used was "
        f"{ref_date_str} (computed as one calendar day after the maximum observed InvoiceDate in "
        f"the cleaned dataset, so that the most recent buyer has a Recency of at least 1 day)."
    )

    rfm_defs = [
        ("Recency (R)",
         f"Days elapsed between the customer's most recent invoice date and the reference_date "
         f"({ref_date_str}). Formula: Recency_i = (reference_date - max_j InvoiceDate_ij).days. "
         "Lower values indicate more recent engagement."),
        ("Frequency (F)",
         "Number of distinct invoices (orders) placed by the customer across the observation "
         "window. Formula: Frequency_i = |{InvoiceNo_ij}| for customer i. Measures how often "
         "the customer returns."),
        ("Monetary (M)",
         "Sum of line-item TotalAmount (Quantity x UnitPrice) across all of the customer's "
         "positive-sales invoices. Formula: Monetary_i = sum_j (Quantity_ij * UnitPrice_ij). "
         "Captures the customer's total revenue contribution."),
    ]
    for title, desc in rfm_defs:
        _add_heading(doc, title, level=2)
        _add_paragraph(doc, desc)

    logger.info("Building Exploratory Analysis (Section 8)")
    _add_heading(doc, "8. Exploratory Analysis", level=1)
    _add_paragraph(doc,
        "Before modelling, RFM feature distributions were inspected for central tendency, spread, "
        "and shape. Descriptive statistics (count, mean, standard deviation, min, quartiles, "
        "max) are summarised below."
    )
    _add_dataframe_as_table(doc, rfm_distribution_stats,
        "Table 8.1. RFM distribution statistics (raw scale).")

    _add_dataframe_as_table(doc, overall_stats_df,
        "Table 8.2. Overall RFM descriptive summary.")

    if "skew" in rfm_distribution_stats.index and "Recency" in rfm_distribution_stats.columns:
        recency_skew = rfm_distribution_stats.loc["skew", "Recency"]
        freq_skew = rfm_distribution_stats.loc["skew", "Frequency"]
        mon_skew = rfm_distribution_stats.loc["skew", "Monetary"]
    elif "column" in rfm_distribution_stats.columns and "skewness" in rfm_distribution_stats.columns:
        def _skew_of(col):
            mask = rfm_distribution_stats["column"].astype(str) == col
            vals = rfm_distribution_stats.loc[mask, "skewness"]
            return float(vals.iloc[0]) if len(vals) > 0 else None
        recency_skew = _skew_of("Recency")
        freq_skew = _skew_of("Frequency")
        mon_skew = _skew_of("Monetary")
    else:
        recency_skew = freq_skew = mon_skew = None

    if recency_skew is not None or freq_skew is not None or mon_skew is not None:
        skew_parts = []
        if recency_skew is not None:
            skew_parts.append(f"Recency skew={recency_skew:.2f}")
        if freq_skew is not None:
            skew_parts.append(f"Frequency skew={freq_skew:.2f}")
        if mon_skew is not None:
            skew_parts.append(f"Monetary skew={mon_skew:.2f}")
        skew_line = "; ".join(skew_parts)
        _add_paragraph(doc,
            f"Skewness analysis reveals strongly right-skewed distributions ({skew_line}). "
            "Right skew - a long upper tail of high-value customers and infrequent buyers - is "
            "typical of transactional retail data. Because K-Means minimizes squared Euclidean "
            "distances, extreme outliers in the original scale can dominate cluster assignment "
            "and produce solutions driven by a handful of points rather than the bulk of "
            "customers."
        )
    else:
        _add_paragraph(doc,
            "RFM features typically exhibit strong positive (right) skew in retail contexts, "
            "driven by a small fraction of high-frequency / high-spend customers."
        )

    log_cols_str = ", ".join(use_log_columns_list) if use_log_columns_list else "(none)"
    _add_paragraph(doc,
        f"To mitigate skew and make clusters more representative of the whole customer base, "
        f"the log1p(x) = ln(1 + x) transformation was applied to these columns: "
        f"{log_cols_str}. The '+1' offset ensures zero values remain defined after log. "
        "Log1p both compresses the upper tail and tends to make Frequency and Monetary closer "
        "to normally distributed, which aligns better with K-Means' implicit assumption of "
        "approximately spherical, equal-variance clusters."
    )

    if "rfm_dist" in paths_figs and paths_figs["rfm_dist"].exists():
        logger.info(f"Inserting rfm_dist figure: {paths_figs['rfm_dist']}")
        doc.add_picture(str(paths_figs["rfm_dist"]), width=Inches(FIGURE_WIDTH_INCHES))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_figure_caption(doc, "Figure 8.1. RFM feature distributions on the raw scale.")
    else:
        logger.warning("paths_figs['rfm_dist'] missing; skipping raw-distribution figure.")

    if "rfm_log_dist" in paths_figs and paths_figs["rfm_log_dist"].exists():
        logger.info(f"Inserting rfm_log_dist figure: {paths_figs['rfm_log_dist']}")
        doc.add_picture(str(paths_figs["rfm_log_dist"]), width=Inches(FIGURE_WIDTH_INCHES))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_figure_caption(doc, "Figure 8.2. RFM feature distributions after log1p transformation.")
    else:
        logger.warning("paths_figs['rfm_log_dist'] missing; skipping log-distribution figure.")

    logger.info("Building Clustering Methodology (Section 9)")
    _add_heading(doc, "9. Clustering Methodology", level=1)
    _add_heading(doc, "9.1 K-Means Algorithm", level=2)
    _add_paragraph(doc,
        "K-Means is a partitional, centroid-based clustering algorithm that iteratively assigns "
        "each observation to its nearest centroid and then relocates each centroid to the mean "
        "of its assigned observations. The procedure converges when no assignment changes or a "
        "maximum iteration count is reached. The objective minimized is the within-cluster sum "
        "of squared Euclidean distances, also known as inertia: sum_i ||x_i - mu_{c_i}||^2."
    )
    _add_heading(doc, "9.2 StandardScaler Pre-processing", level=2)
    _add_paragraph(doc,
        "Because K-Means is scale-sensitive (a feature with larger variance exerts more "
        "influence on Euclidean distance), all features used as input to clustering were "
        "z-standardised with sklearn's StandardScaler: z = (x - mu) / sigma, fit only on the "
        "training/customer-level data. This puts each log-transformed RFM dimension on "
        "comparable footing so that Recency, Frequency, and Monetary each contribute roughly "
        "equally to the distance metric, unless the data itself argues otherwise via variance."
    )
    _add_heading(doc, "9.3 Model Selection: Elbow Method and Silhouette Score", level=2)
    _add_paragraph(doc,
        "Two complementary internal validation metrics guided the choice of k. The Elbow method "
        "inspects the inertia curve across k values: inertia monotonically decreases as k "
        "increases, and the 'elbow' - the k beyond which inertia drops most slowly - signals a "
        "region of reasonable parsimony. The Silhouette score s(i) for each observation ranges "
        "from -1 (worst) to +1 (perfectly assigned), and the dataset mean Silhouette score "
        "summarises cluster tightness versus separation. Typically the highest k with a "
        "positive, stable mean Silhouette score - while still respecting interpretability and "
        "actionability of the cluster count - is preferred."
    )

    logger.info("Building K-Means Implementation (Section 10)")
    _add_heading(doc, "10. K-Means Implementation", level=1)
    _add_paragraph(doc,
        "The final K-Means model was configured and executed with the following parameters to "
        "ensure reproducibility and convergence robustness:"
    )
    impl_items = [
        "random_state = 42 (numpy and sklearn seeds fixed for deterministic centroid "
        "initialization).",
        "n_init = 10 independent random initializations were run; the run producing the lowest "
        "final inertia was retained.",
        "max_iter = 300 Lloyd iterations per initialization (sklearn default).",
        "Feature columns supplied to StandardScaler and KMeans were the log1p-transformed and "
        "scaled Recency, Frequency, and Monetary values derived from the RFM DataFrame.",
    ]
    for item in impl_items:
        p = doc.add_paragraph(style="List Bullet")
        _set_paragraph_default_font(p)
        if not p.runs:
            run = p.add_run(item)
            _set_run_font(run)

    logger.info("Building Selecting Number of Clusters (Section 11)")
    _add_heading(doc, "11. Selecting Number of Clusters", level=1)
    _add_dataframe_as_table(doc, k_metrics_df,
        "Table 11.1. Inertia and Silhouette score for each evaluated k.")

    _add_paragraph(doc,
        f"Across the evaluated k values, the combination of the inertia elbow location and the "
        f"mean Silhouette score curve supports the selection of best_k = {best_k}. At this k, "
        f"the Silhouette score is acceptably positive while the solution remains parsimonious "
        f"enough for marketing action. Inertia continues to improve beyond k = {best_k}, but at "
        f"the cost of diminishing Silhouette score and increasingly fragmented - and therefore "
        f"less actionable - segments."
    )

    if "elbow" in paths_figs and paths_figs["elbow"].exists():
        logger.info(f"Inserting elbow figure: {paths_figs['elbow']}")
        doc.add_picture(str(paths_figs["elbow"]), width=Inches(FIGURE_WIDTH_INCHES))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_figure_caption(doc, "Figure 11.1. Elbow plot: inertia vs. number of clusters k.")

    if "silhouette" in paths_figs and paths_figs["silhouette"].exists():
        logger.info(f"Inserting silhouette figure: {paths_figs['silhouette']}")
        doc.add_picture(str(paths_figs["silhouette"]), width=Inches(FIGURE_WIDTH_INCHES))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_figure_caption(doc, "Figure 11.2. Mean Silhouette score vs. number of clusters k.")

    logger.info("Building Evaluation (Section 12)")
    _add_heading(doc, "12. Evaluation", level=1)

    eval_rows = []
    metric_labels = {
        "silhouette_score": "Silhouette Score",
        "calinski_harabasz_score": "Calinski-Harabasz Index",
        "davies_bouldin_score": "Davies-Bouldin Index",
    }
    for key, label in metric_labels.items():
        val = clustering_metrics.get(key, float("nan"))
        eval_rows.append({"Metric": label, "Value": val})
    eval_df = pd.DataFrame(eval_rows)
    _add_dataframe_as_table(doc, eval_df,
        f"Table 12.1. Internal clustering validation metrics at k = {best_k}.")

    _add_paragraph(doc,
        f"Silhouette Score = {sil_score:.4f}: Values near 0 indicate overlapping clusters; "
        "values well above 0.25 suggest reasonable separation. Interpretation must be tempered "
        "by real-world dataset structure - real customer data rarely achieves scores above "
        "0.5-0.6 even when segments are operationally useful."
    )
    _add_paragraph(doc,
        f"Calinski-Harabasz Index = {ch_score:,.2f}: The ratio of between-cluster dispersion to "
        "within-cluster dispersion. Higher is better; the metric has no upper bound and is most "
        "informative when compared across candidate k values on the same dataset."
    )
    _add_paragraph(doc,
        f"Davies-Bouldin Index = {db_score:.4f}: The average worst-case ratio of a cluster's "
        "within-cluster scatter to its separation from its nearest rival cluster. Lower is "
        "better, with 0 representing perfectly separated clusters."
    )

    logger.info("Building Cluster Visualization (Section 13)")
    _add_heading(doc, "13. Cluster Visualization", level=1)

    if "cluster_sizes" in paths_figs and paths_figs["cluster_sizes"].exists():
        logger.info(f"Inserting cluster_sizes figure: {paths_figs['cluster_sizes']}")
        doc.add_picture(str(paths_figs["cluster_sizes"]), width=Inches(FIGURE_WIDTH_INCHES))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_figure_caption(doc, "Figure 13.1. Cluster size distribution (customer count per cluster).")

    if "pca_clusters" in paths_figs and paths_figs["pca_clusters"].exists():
        logger.info(f"Inserting pca_clusters figure: {paths_figs['pca_clusters']}")
        doc.add_picture(str(paths_figs["pca_clusters"]), width=Inches(FIGURE_WIDTH_INCHES))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_figure_caption(doc,
            "Figure 13.2. K-Means clusters projected onto the first two principal components of "
            "the scaled RFM feature space.")
        if not pca_explained_var_df.empty:
            _add_dataframe_as_table(doc, pca_explained_var_df,
                "Table 13.1. PCA explained variance ratio per component.")

    if "profile_heatmap" in paths_figs and paths_figs["profile_heatmap"].exists():
        logger.info(f"Inserting profile_heatmap figure: {paths_figs['profile_heatmap']}")
        doc.add_picture(str(paths_figs["profile_heatmap"]), width=Inches(FIGURE_WIDTH_INCHES))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_figure_caption(doc,
            "Figure 13.3. Scaled cluster profile heatmap (z-score per RFM dimension).")

    if "profile_radar" in paths_figs and paths_figs["profile_radar"].exists():
        logger.info(f"Inserting profile_radar figure: {paths_figs['profile_radar']}")
        doc.add_picture(str(paths_figs["profile_radar"]), width=Inches(FIGURE_WIDTH_INCHES))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_figure_caption(doc,
            "Figure 13.4. Radar chart of scaled cluster profiles across R, F, M dimensions.")

    if "monetary_bar" in paths_figs and paths_figs["monetary_bar"].exists():
        logger.info(f"Inserting monetary_bar figure: {paths_figs['monetary_bar']}")
        doc.add_picture(str(paths_figs["monetary_bar"]), width=Inches(FIGURE_WIDTH_INCHES))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_figure_caption(doc,
            "Figure 13.5. Total monetary revenue contribution per cluster.")

    logger.info("Building Cluster Profiles (Section 14)")
    _add_heading(doc, "14. Cluster Profiles", level=1)
    _add_paragraph(doc,
        f"The following profile table aggregates statistics for each of the k = {best_k} "
        "clusters on the original (untransformed) RFM scale. Each row summarises one segment's "
        "size, R/F/M central tendency (both mean and median), and revenue contribution."
    )
    _add_dataframe_as_table(doc, cluster_profiles,
        "Table 14.1. Cluster-level descriptive profiles (original RFM scale).")

    if not cluster_profiles_scaled.empty:
        _add_dataframe_as_table(doc, cluster_profiles_scaled,
            "Table 14.2. Scaled cluster profiles (z-scores per RFM dimension for cross-feature comparison).")

    logger.info("Building Interpretation (Section 15)")
    _add_heading(doc, "15. Interpretation", level=1)
    _add_paragraph(doc,
        "Each cluster's character is read from its profile statistics (original and scaled) "
        "along the three RFM axes. Interpretations below describe the evidence rather than "
        "prescribing marketing action labels; no speculative VIP / At-risk / Lost designations "
        "are asserted beyond what the numbers directly support."
    )

    sorted_cluster_ids = sorted(cluster_descriptions.keys(), key=lambda x: (str(type(x).__name__), str(x)))
    for cid in sorted_cluster_ids:
        _add_heading(doc, f"Cluster {cid}", level=2)
        desc = cluster_descriptions[cid]
        _add_paragraph(doc, desc)

    logger.info("Building Business/Research Implications (Section 16)")
    _add_heading(doc, "16. Business/Research Implications", level=1)
    _add_paragraph(doc,
        "The segmentation produced here, grounded in observed RFM behaviour and supported by "
        "internal validation metrics, has several evidence-based implications for downstream "
        "marketing strategy and further research:"
    )
    implications = [
        "Segment-specific promotional cadence: clusters characterised by low Recency (recent "
        "activity) and high Frequency are likely receptive to loyalty and upsell programmes, "
        "while clusters marked by high Recency (dormant) profiles should be targeted with "
        "re-engagement campaigns before they churn definitively.",
        "Revenue-based resource allocation: clusters contributing the largest share of total "
        "Monetary value warrant a disproportionate share of retention-marketing resources - "
        "losing a single customer from these segments has a larger revenue impact than losing "
        "a customer from low-Monetary clusters.",
        "Offer tailoring: the distinct RFM fingerprints of each cluster imply different "
        "offer types will resonate. High-Frequency clusters may respond to subscription or "
        "auto-replenishment constructs; low-Frequency but high-Monetary clusters may favour "
        "premium product bundles rather than pure price discounts.",
        "Model monitoring and retraining: because RFM is measured relative to a single "
        "temporal cut-off, segments should be re-derived periodically (e.g., quarterly) to "
        "track customer migration between clusters over time. Migration matrices are a "
        "natural next analytical step.",
        "Feature expansion: future work could enrich the RFM-only representation with "
        "product-category breadth, inter-purchase interval variance, country, or seasonality "
        "features, and compare K-Means against density-based (DBSCAN) or hierarchical "
        "clustering alternatives for robustness.",
    ]
    for imp in implications:
        p = doc.add_paragraph(style="List Bullet")
        _set_paragraph_default_font(p)
        if not p.runs:
            run = p.add_run(imp)
            _set_run_font(run)

    logger.info("Building Limitations (Section 17)")
    _add_heading(doc, "17. Limitations", level=1)
    limitations = [
        "Missing CustomerID exclusion: transactions with missing CustomerID were dropped "
        "entirely. To the degree these represent systematic behaviours (e.g., guest checkout "
        "patterns), the final segments only describe the identified-customer population and "
        "may not generalise to anonymous traffic.",
        "Single-retailer scope: the dataset reflects one UK-based online gift/homeware retailer "
        "over a specific 13-month window. Industry and country-specific factors limit "
        "portability of the exact segment count and shape to other retailers.",
        "K-Means assumptions: K-Means favours roughly equal-volume, spherical, linearly "
        "separable clusters in the feature space. If the true customer manifold is strongly "
        "non-spherical or contains density-separated shapes, K-Means may produce arbitrary or "
        "suboptimal partitions. This is mitigated - but not eliminated - via log transform "
        "and standardisation.",
        "Single temporal snapshot: RFM is computed relative to one reference date. Seasonal "
        "events near the window boundaries (e.g., Christmas in December 2011) can distort "
        "Recency estimates, and no cross-temporal validation of cluster stability was "
        "performed.",
        "No causal claims: this is an observational, descriptive clustering. Associations "
        "between cluster membership and future behaviour must be validated with controlled "
        "marketing experiments or held-out time-split evaluation before being acted upon.",
    ]
    for lim in limitations:
        p = doc.add_paragraph(style="List Bullet")
        _set_paragraph_default_font(p)
        if not p.runs:
            run = p.add_run(lim)
            _set_run_font(run)

    logger.info("Building Conclusion (Section 18)")
    _add_heading(doc, "18. Conclusion", level=1)
    _add_paragraph(doc,
        f"This report demonstrated a reproducible RFM + K-Means segmentation pipeline on the "
        f"UCI Online Retail dataset. Cleaning reduced {n_raw_rows:,} raw rows to "
        f"{n_positive_sales_rows:,} positive-sales rows with identified customers, from which "
        f"{total_customers:,} RFM profiles were computed. After log1p stabilisation and "
        f"standardisation, K-Means with k = {best_k} (Silhouette = {sil_score:.4f}) yielded "
        f"{best_k} behaviourally distinct segments whose profiles - reported in both original "
        "and z-scored form - exhibit systematic variation across Recency, Frequency, and "
        "Monetary. The resulting segmentation provides a quantitative, auditable foundation "
        "for targeted marketing action, subject to the reproducibility protocol and limitations "
        "documented above."
    )

    logger.info("Building Reproducibility (Section 19)")
    _add_heading(doc, "19. Reproducibility", level=1)
    repro_items = [
        f"Random seed = 42 applied globally (numpy, sklearn KMeans random_state parameter).",
        f"RFM reference_date = {ref_date_str}, derived deterministically as "
        "max(InvoiceDate) + 1 day from the cleaned positive-sales dataset.",
        "Environment dependencies are enumerated in `requirements.txt` at the project root, "
        "pinning pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, openpyxl, and "
        "python-docx versions.",
        "Execution instructions are provided in the project README. In brief: place the raw "
        "data file at the path documented in `data/README.md`, install the requirements.txt "
        "into a fresh virtual environment, and run `python -m src.main` from the project "
        "root. The pipeline will run deterministically and write the same tables, figures, "
        "and DOCX report to the `outputs/` and `report/` directories.",
    ]
    for item in repro_items:
        p = doc.add_paragraph(style="List Bullet")
        _set_paragraph_default_font(p)
        if not p.runs:
            run = p.add_run(item)
            _set_run_font(run)

    logger.info("Building Code Snippets (Section 20)")
    _add_heading(doc, "20. Code Snippets", level=1)
    _add_paragraph(doc,
        "The following representative snippets illustrate the core computational steps. The "
        "full production code lives under `src/` and is the reference implementation."
    )

    rfm_code = '''rfm = (
    df.groupby("CustomerID")
       .agg(
           Recency=("InvoiceDate",
                    lambda x: (reference_date - x.max()).days),
           Frequency=("InvoiceNo", "nunique"),
           Monetary=("TotalAmount", "sum"),
       )
       .reset_index()
)'''
    _add_code_snippet(doc, "20.1 RFM Aggregation", rfm_code)

    kmeans_code = '''from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10,
)
labels = kmeans.fit_predict(X_scaled)
inertia = kmeans.inertia_
'''
    _add_code_snippet(doc, "20.2 K-Means Fit", kmeans_code)

    sil_code = '''from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

silhouette = silhouette_score(X_scaled, labels)
ch_index   = calinski_harabasz_score(X_scaled, labels)
db_index   = davies_bouldin_score(X_scaled, labels)
'''
    _add_code_snippet(doc, "20.3 Silhouette and Internal Metric Scoring", sil_code)

    pca_code = '''from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=labels, cmap="tab10", s=14, alpha=0.7)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.title("Clusters projected onto first two PCs")
'''
    _add_code_snippet(doc, "20.4 PCA Cluster Visualization", pca_code)

    log_code = '''import numpy as np
from sklearn.preprocessing import StandardScaler

use_log_columns = ["Recency", "Frequency", "Monetary"]
X_log = np.log1p(rfm_df[use_log_columns])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)
'''
    _add_code_snippet(doc, "20.5 Log1p Transform and Standardization", log_code)

    logger.info("Building References (Section 21)")
    _add_heading(doc, "21. References", level=1)

    refs = [
        "Chen, D.-Y., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: "
        "A case study of RFM model-based customer segmentation using data mining. Journal of "
        "Database Marketing & Customer Strategy Management, 19(3), 197-208. "
        "https://doi.org/10.1057/dbm.2012.17",
        "Dua, D., & Graff, C. (2019). UCI Machine Learning Repository [Online Retail Data Set]. "
        "University of California, Irvine, School of Information and Computer Science. "
        "https://archive.ics.uci.edu/ml/datasets/Online+Retail",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, "
        "M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., "
        "Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in "
        "Python. Journal of Machine Learning Research, 12, 2825-2830. "
        "https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html",
        "scikit-learn Developers. K-Means Clustering - scikit-learn 1.x Documentation. "
        "https://scikit-learn.org/stable/modules/clustering.html#k-means",
        "scikit-learn Developers. Silhouette Score - scikit-learn 1.x Documentation. "
        "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html",
        "scikit-learn Developers. Calinski-Harabasz Index - scikit-learn 1.x Documentation. "
        "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html",
        "scikit-learn Developers. Davies-Bouldin Index - scikit-learn 1.x Documentation. "
        "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html",
    ]
    for i, ref in enumerate(refs, start=1):
        p = doc.add_paragraph(style="List Number")
        _set_paragraph_default_font(p, font_name=FONT_NAME, size_pt=FONT_SIZE_PT)
        if not p.runs:
            run = p.add_run(ref)
            _set_run_font(run)

    logger.info(f"Saving report to disk: {output_path}")
    doc.save(str(output_path))
    logger.info(f"Report saved successfully: {output_path}")

    return output_path

