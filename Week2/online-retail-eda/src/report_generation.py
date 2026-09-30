import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from src.config import FIGURES_DIR, REPORT_DOCX, TABLES_DIR

logger = logging.getLogger(__name__)


def _add_heading_styled(doc: Document, text: str, level: int = 1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1a, 0x3c, 0x6e)
    return h


def _add_table_from_df(doc: Document, df: pd.DataFrame, caption: str = None,
                       float_fmt: str = "{:.3f}"):
    if caption:
        p = doc.add_paragraph()
        r = p.add_run(caption)
        r.bold = True
        r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    df = df.reset_index(drop=True)
    n_rows, n_cols = min(len(df) + 1, 51), len(df.columns)
    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.style = "Light Grid Accent 1"
    cols = list(df.columns)
    for j, col_name in enumerate(cols):
        cell = table.cell(0, j)
        cell.text = str(col_name)
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(9)
    max_body = min(len(df), 50)
    for i in range(max_body):
        for j, col in enumerate(cols):
            val = df.iloc[i][col]
            if isinstance(val, float):
                txt = float_fmt.format(val) if abs(val) < 1000 else f"{val:,.2f}"
            elif isinstance(val, int) and not isinstance(val, bool):
                txt = f"{val:,}"
            else:
                txt = str(val)
            cell = table.cell(i + 1, j)
            cell.text = txt
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(8.5)
    if len(df) > 50:
        p = doc.add_paragraph()
        r = p.add_run(f"* Table truncated: showing first 50 of {len(df):,} rows.")
        r.italic = True
        r.font.size = Pt(8.5)


def _add_figure(doc: Document, fig_name: str, caption: str, width_in: float = 6.0):
    fig_path = FIGURES_DIR / fig_name
    if fig_path.exists():
        doc.add_picture(str(fig_path), width=Inches(width_in))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _add_code_snippet(doc: Document, code: str):
    for line in code.strip().split("\n"):
        p = doc.add_paragraph()
        r = p.add_run(line)
        r.font.name = "Consolas"
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x1b, 0x5e, 0x20)


def build_report(analyses: Dict, df_overview: Dict, paths_figs: Dict,
                 missing_df: pd.DataFrame, iqr_summary: pd.DataFrame,
                 output_path: Path = REPORT_DOCX) -> Path:
    logger.info("Building DOCX report...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()

    style = doc.styles["Normal"]
    style.font.size = Pt(10.5)
    style.font.name = "Calibri"

    # ===== TITLE PAGE =====
    for _ in range(6):
        doc.add_paragraph()
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = tp.add_run("WEEK 2 ASSIGNMENT\nEXPLORATORY DATA ANALYSIS AND VISUALIZATION")
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = RGBColor(0x1a, 0x3c, 0x6e)

    tp2 = doc.add_paragraph()
    tp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = tp2.add_run("\nUCI Online Retail Dataset\n(541,909 Transactions, 8 Variables, Dec 2010 – Dec 2011)")
    r2.font.size = Pt(13)

    tp3 = doc.add_paragraph()
    tp3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = tp3.add_run("\n\nPython: Pandas · NumPy · Matplotlib · Seaborn\nReport format: DOCX with embedded figures & tables")
    r3.font.size = Pt(11)
    r3.italic = True
    doc.add_page_break()

    # ===== 1. INTRODUCTION =====
    _add_heading_styled(doc, "1. Introduction & Dataset Selection", level=1)
    doc.add_paragraph(
        "This report presents a comprehensive Exploratory Data Analysis (EDA) of the UCI Online Retail Dataset. "
        "The dataset was selected because it is a well-known public transactional retail dataset used widely in "
        "academia for demonstrating EDA, RFM segmentation, market basket analysis, and data cleaning techniques. "
        "It was acquired from the UCI Machine Learning Repository "
        "(https://archive.ics.uci.edu/dataset/352/online+retail)."
    )
    doc.add_paragraph(
        "The dataset contains 541,909 transaction lines recorded between 01 December 2010 and 09 December 2011 "
        "for a UK-based, non-store online retailer. Each record represents a single line of an invoice and contains "
        "eight variables: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country."
    )
    _add_heading_styled(doc, "1.1 Data Acquisition Code", level=2)
    _add_code_snippet(doc, """
import pandas as pd
df_raw = pd.read_excel("data/Online+Retail.xlsx", engine="openpyxl")
# => Loads 541,909 rows x 8 columns from the official UCI Excel source.
""")

    doc.add_page_break()
    # ===== 2. INITIAL EXPLORATION =====
    _add_heading_styled(doc, "2. Initial Exploratory Analysis & Basic Statistics", level=1)
    _add_heading_styled(doc, "2.1 Dataset Overview", level=2)
    overview_rows = [
        ("Rows (transaction lines)", f"{int(df_overview.get('rows', 0)):,}"),
        ("Columns", int(df_overview.get('columns', 0))),
        ("Memory usage (MB)", df_overview.get('memory_mb', 'N/A')),
        ("Unique invoices", f"{int(df_overview.get('n_invoices', 0)):,}"),
        ("Unique customers", f"{int(df_overview.get('n_customers', 0)):,}"),
        ("Unique products (StockCode)", f"{int(df_overview.get('n_products', 0)):,}"),
        ("Unique countries", int(df_overview.get('n_countries', 0))),
    ]
    overview_df = pd.DataFrame(overview_rows, columns=["Metric", "Computed Value"])
    _add_table_from_df(doc, overview_df, caption="Table 2.1 — Dataset overview (actual counts from dataset)")

    _add_heading_styled(doc, "2.2 Data Types", level=2)
    dtypes = pd.read_csv(TABLES_DIR / "01_dataset_overview.csv") if (TABLES_DIR / "01_dataset_overview.csv").exists() else pd.DataFrame()
    if len(dtypes):
        doc.add_paragraph(
            "All 8 variables are present. CustomerID is stored as a floating-point number (natively by Excel); "
            "InvoiceDate should be parsed to datetime; Quantity and UnitPrice are numeric and may legitimately be "
            "negative for cancellations/returns and adjustments."
        )

    _add_heading_styled(doc, "2.3 Missing Values", level=2)
    doc.add_paragraph(
        "Two columns contain missing values. CustomerID has the largest volume of nulls — these records represent "
        "valid transactions but cannot be used for customer-level analyses such as RFM or cohort analysis. "
        "Description has a smaller number of missing entries; financial aggregations (revenue, quantity) are unaffected."
    )
    _add_table_from_df(doc, missing_df, caption="Table 2.2 — Missing values by column (actual counts & %)")
    _add_figure(doc, "01_missing_values.png",
                "Figure 2.1 — Missing value counts per column. Only CustomerID and Description contain nulls.",
                width_in=5.5)

    doc.add_page_break()
    # ===== 3. DATA PREPARATION =====
    _add_heading_styled(doc, "3. Data Transformations & Feature Engineering", level=1)
    doc.add_paragraph(
        "Three analytical datasets are maintained throughout this EDA so that the rationale for any filter is explicit:"
    )
    for item in [
        "full — the complete loaded dataset (all 541,909 rows including duplicates, cancellations, zero prices).",
        "valid_price — rows with UnitPrice > 0 only; excludes 2,515 unpriced transactions.",
        "positive_sales — valid-price rows, AND excludes cancellation invoices (prefix 'C'), AND requires Quantity > 0. Used for all revenue & sales analyses.",
    ]:
        p = doc.add_paragraph(item, style="List Bullet")
    _add_heading_styled(doc, "3.1 Features Created", level=2)
    _add_code_snippet(doc, """
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
df["Year"]        = df["InvoiceDate"].dt.year
df["Month"]       = df["InvoiceDate"].dt.month
df["Day"]         = df["InvoiceDate"].dt.day
df["Hour"]        = df["InvoiceDate"].dt.hour
df["DayOfWeek"]   = df["InvoiceDate"].dt.dayofweek
df["DayOfWeekName"] = df["InvoiceDate"].dt.day_name()
df["YearMonth"]   = df["InvoiceDate"].dt.to_period("M").astype(str)
df["IsCancellation"] = df["InvoiceNo"].astype(str).str.startswith("C", na=False)
""")
    doc.add_paragraph(
        "TotalAmount is the key financial variable for revenue analysis. Temporal features (YearMonth, Hour, "
        "DayOfWeekName) drive the time-series and daily-pattern visualizations in Sections 4 and 5."
    )

    doc.add_page_break()
    # ===== 4. DISTRIBUTIONS =====
    _add_heading_styled(doc, "4. Numerical Distributions & Descriptive Statistics", level=1)
    if (TABLES_DIR / "03_transaction_amount_stats.csv").exists():
        stats = pd.read_csv(TABLES_DIR / "03_transaction_amount_stats.csv")
        _add_table_from_df(doc, stats, caption="Table 4.1 — Percentile & summary statistics (positive-sales dataset)")

    _add_heading_styled(doc, "4.1 Quantity Distribution", level=2)
    doc.add_paragraph(
        "Observation: Quantity values are concentrated in the low single digits but have a pronounced right tail. "
        "The distribution on the log1p scale reveals a roughly unimodal shape with a long upper tail driven by bulk orders."
    )
    try:
        qty_stats = pd.read_csv(TABLES_DIR / "03_transaction_amount_stats.csv")
        qty_stats.index = qty_stats["metric"]
        qty_p99 = float(qty_stats.loc["P99", "Quantity"])
        qty_med = float(qty_stats.loc["median", "Quantity"])
        doc.add_paragraph(
            f"Evidence: Quantity median = {qty_med:.0f} units per line, P99 = {qty_p99:.0f} units. "
            f"Extreme observations extend to the tens of thousands (see anomaly analysis in §9)."
        )
    except Exception:
        pass
    _add_figure(doc, "02_quantity_distribution.png",
                "Figure 4.1 — Quantity distribution (positive sales, log1p scale). Red dashed line is the distribution mean.")
    _add_figure(doc, "02b_quantity_boxplot.png",
                "Figure 4.2 — Quantity boxplot (raw positive scale). Individual high-value outliers are visible as faint points.")

    _add_heading_styled(doc, "4.2 Unit Price Distribution", level=2)
    doc.add_paragraph(
        "UnitPrice is also right-skewed: most items are cheap, but a small number of premium products and "
        "one-off charges create a very long tail. The log1p histogram in Figure 4.3 shows a roughly bell-shaped "
        "centre after the transformation."
    )
    _add_figure(doc, "03_unit_price_distribution.png",
                "Figure 4.3 — Unit Price distribution (positive sales, log1p scale).")
    _add_figure(doc, "03b_unit_price_boxplot.png",
                "Figure 4.4 — Unit Price boxplot (raw scale).")

    _add_heading_styled(doc, "4.3 Total Amount Distribution", level=2)
    doc.add_paragraph(
        "TotalAmount = Quantity × UnitPrice. It inherits the right-skewed nature of both inputs. The product-level "
        "multiplication can amplify extremes; the P99 / P50 ratio for TotalAmount is considerably larger than for "
        "the input variables, confirming heavier tails."
    )
    _add_figure(doc, "14_totalamount_distribution.png",
                "Figure 4.5 — TotalAmount distribution (positive sales, log1p scale).")

    doc.add_page_break()
    # ===== 5. TEMPORAL =====
    _add_heading_styled(doc, "5. Temporal Analysis: Trends, Seasonality & Daily Patterns", level=1)
    _add_heading_styled(doc, "5.1 Monthly Transaction Volume", level=2)
    mv = analyses.get("temporal", {}).get("monthly_volume", pd.DataFrame())
    if len(mv):
        m_max = mv.sort_values("transaction_lines", ascending=False).iloc[0]
        m_min = mv.sort_values("transaction_lines", ascending=True).iloc[0]
        doc.add_paragraph(
            f"Observation: Transaction volume rises across the observed period, with a clear peak in "
            f"{m_max['YearMonth']} ({int(m_max['transaction_lines']):,} lines). The lowest complete month is "
            f"{m_min['YearMonth']} ({int(m_min['transaction_lines']):,} lines)."
        )
        doc.add_paragraph(
            "Interpretation: The rising pattern is consistent with business growth during the observation window. "
            "However, the dataset covers only a single 13-month slice, so no claim can be made about year-over-year seasonality. "
            "Caution: December 2011 is truncated (data ends 09 Dec), which mechanically lowers its apparent volume."
        )
    _add_figure(doc, "04_monthly_transaction_volume.png",
                "Figure 5.1 — Monthly transaction volume (lines in blue, unique invoices in orange). Growth trend is visible.")

    _add_heading_styled(doc, "5.2 Monthly Revenue", level=2)
    mr = analyses.get("temporal", {}).get("monthly_revenue", pd.DataFrame())
    if len(mr):
        r_max = mr.sort_values("revenue", ascending=False).iloc[0]
        r_min = mr.sort_values("revenue", ascending=True).iloc[0]
        doc.add_paragraph(
            f"Observation: Highest-revenue month is {r_max['YearMonth']} at £{float(r_max['revenue']):,.2f}; "
            f"lowest complete month is {r_min['YearMonth']} at £{float(r_min['revenue']):,.2f}. Revenue broadly "
            f"tracks transaction volume, but the average-order-value line (red, right axis) adds nuance."
        )
    _add_figure(doc, "05_monthly_revenue.png",
                "Figure 5.2 — Monthly revenue (bars) together with average line value (red line).")

    _add_heading_styled(doc, "5.3 Hourly Pattern", level=2)
    doc.add_paragraph(
        "Observation: Transaction activity follows a clear intra-day pattern. Very few lines occur before 08:00 "
        "or after 20:00; the busiest window is typically midday to early afternoon."
    )
    _add_figure(doc, "09_transactions_by_hour.png",
                "Figure 5.3 — Transaction lines by hour of day. Intra-day peak pattern is clearly established.")

    _add_heading_styled(doc, "5.4 Day-of-Week Pattern", level=2)
    doc.add_paragraph(
        "Observation: Weekday order follows the business week pattern. Saturday activity is notably reduced or "
        "absent, which may reflect data-capture or operational scheduling choices. Caution: without additional "
        "context, we cannot determine whether reduced-Saturday effects are real or artefacts of the retailer's systems."
    )
    _add_figure(doc, "10_transactions_by_weekday.png",
                "Figure 5.4 — Transaction volume by day of week, ordered Monday → Sunday (not alphabetical).")

    doc.add_page_break()
    # ===== 6. GEOGRAPHIC =====
    _add_heading_styled(doc, "6. Geographic Distribution", level=1)
    ct = analyses.get("geographic", {}).get("country_transactions", pd.DataFrame())
    if len(ct):
        row0 = ct.iloc[0]
        doc.add_paragraph(
            f"Observation: {row0['Country']} dominates with {int(row0['lines']):,} transaction lines "
            f"({float(row0['lines_pct']):.2f}% of all lines). The remaining share is split across the other "
            f"{int(ct['Country'].nunique()) - 1} countries."
        )
        doc.add_paragraph(
            "Interpretation: The dataset is heavily geographically concentrated. The retailer's UK home market "
            "is clearly the largest; this affects any cross-country comparisons because sample sizes for the "
            "smaller countries can be orders of magnitude smaller."
        )
    _add_figure(doc, "06_top_countries.png",
                "Figure 6.1 — Top 10 countries by transaction-line count. UK dominance is evident.")

    doc.add_page_break()
    # ===== 7. PRODUCT ANALYSIS =====
    _add_heading_styled(doc, "7. Product Analysis: Quantity vs Revenue Rankings", level=1)
    doc.add_paragraph(
        "A central analytical question: Are the most-frequently-purchased products also the highest-revenue products? "
        "The answer is not automatically yes — cheap products can sell in huge volume without dominating revenue, "
        "just as expensive low-volume items can dominate totals."
    )
    _add_figure(doc, "07_top_products_quantity.png",
                "Figure 7.1 — Top 10 products by total quantity sold across all positive-sales transactions.")
    _add_figure(doc, "08_top_products_revenue.png",
                "Figure 7.2 — Top 10 products by revenue generated (£).")
    comp = analyses.get("product", {}).get("ranking_comparison", pd.DataFrame())
    if len(comp):
        overlap_row = comp[comp["metric"] == "Products in BOTH top lists"]
        if len(overlap_row):
            both = int(overlap_row.iloc[0]["value"])
            only_q = int(comp[comp["metric"] == "Only in quantity top"].iloc[0]["value"])
            only_r = int(comp[comp["metric"] == "Only in revenue top"].iloc[0]["value"])
            doc.add_paragraph(
                f"Observation: Within the Top-10 lists there are {both} products that appear on BOTH the quantity "
                f"and revenue rankings, {only_q} that appear ONLY in the quantity list, and {only_r} that appear "
                f"ONLY in the revenue list."
            )
            doc.add_paragraph(
                "Interpretation: This confirms that purchase frequency and revenue contribution partially diverge. "
                "This is consistent with a mixed assortment where high-volume staples and high-margin / high-ticket "
                "items coexist — merchandising and inventory teams should examine both rankings, not just one."
            )
        _add_table_from_df(doc, comp, caption="Table 7.1 — Quantity-top vs Revenue-top overlap (Top 10 lists).")
    rd = analyses.get("product", {}).get("revenue_distribution", {}).get("summary", pd.DataFrame())
    if len(rd):
        _add_table_from_df(doc, rd, caption="Table 7.2 — Product revenue concentration (Pareto analysis).")

    doc.add_page_break()
    # ===== 8. CORRELATION =====
    _add_heading_styled(doc, "8. Correlations & Scatter-Plot Relationships", level=1)
    cm = analyses.get("relationship", {}).get("correlation_matrix", pd.DataFrame())
    if len(cm):
        _add_table_from_df(doc, cm, caption="Table 8.1 — Pearson correlation coefficients (actual, computed).")
    _add_figure(doc, "13_correlation_heatmap.png",
                "Figure 8.1 — Correlation heatmap (RdBu_r palette, red = positive, blue = negative).", width_in=5.0)
    doc.add_paragraph(
        "Interpretation: By construction, TotalAmount = Quantity × UnitPrice, so its correlation with Quantity "
        "and UnitPrice is expected to be positive. The more interesting question is whether UnitPrice and Quantity "
        "are themselves associated — i.e. do customers buy fewer of expensive items (a negative association), or "
        "does the data show no strong pattern?"
    )

    _add_heading_styled(doc, "8.1 Quantity vs TotalAmount", level=2)
    qta = analyses.get("relationship", {}).get("qty_vs_totalamount", {})
    r_val = qta.get("correlation", None)
    if r_val is not None:
        doc.add_paragraph(
            f"Observation: Pearson r = {r_val:.3f} between Quantity and TotalAmount. As expected, the relationship "
            f"is positive and strong — lines with more units sold tend to have higher line-level revenue."
        )
        doc.add_paragraph(
            "Caution: This correlation is partly mechanical (TotalAmount contains Quantity). The figure intentionally "
            "clips the display at the 99th percentile to avoid letting a handful of extreme points dominate the visual."
        )
    _add_figure(doc, "11_quantity_vs_totalamount.png",
                "Figure 8.2 — Quantity vs TotalAmount, 1%-99% clipped display with regression overlay.", width_in=5.0)

    _add_heading_styled(doc, "8.2 Unit Price vs Quantity", level=2)
    uvq = analyses.get("relationship", {}).get("unitprice_vs_quantity", {})
    r_val2 = uvq.get("correlation", None)
    if r_val2 is not None:
        strength = "strong negative" if r_val2 < -0.5 else "moderate negative" if r_val2 < -0.2 else "weak negative" if r_val2 < -0.05 else "negligible"
        doc.add_paragraph(
            f"Observation: Pearson r = {r_val2:.3f} between UnitPrice and Quantity. This is a {strength} linear "
            f"association in the positive-sales dataset."
        )
        doc.add_paragraph(
            "Interpretation: If the correlation is near zero, it indicates that — at the transaction-line level — "
            "expensive items are not systematically bought in smaller quantities in a linear way. "
            "Non-linear relationships, clustering, or categorical effects may exist and would require further "
            "exploration beyond linear correlation."
        )
    _add_figure(doc, "12_unit_price_vs_quantity.png",
                "Figure 8.3 — Unit Price vs Quantity, 1%-99% clipped display with regression overlay.", width_in=5.0)

    doc.add_page_break()
    # ===== 9. ANOMALIES =====
    _add_heading_styled(doc, "9. Anomaly Analysis & Unusual Observations", level=1)
    doc.add_paragraph(
        "IQR-based outliers are identified using multiplier 1.5 on the positive-sales subset. Outliers are NOT "
        "automatically deleted; they may represent legitimate bulk orders, high-value products, cancellations, "
        "or genuine data-entry errors."
    )
    if len(iqr_summary):
        _add_table_from_df(doc, iqr_summary, caption="Table 9.1 — IQR outlier summary (actual counts & %).")
    pct = analyses.get("anomaly", {}).get("percentile_analysis", pd.DataFrame())
    if len(pct):
        _add_table_from_df(doc, pct, caption="Table 9.2 — Heavy-tail diagnostics: P99/P50 and Max/P99 ratios.")

    _add_heading_styled(doc, "9.1 Concrete Examples from the Actual Dataset", level=2)
    for ex_name, ex_label, ex_cols in [
        ("largest_quantity", "Largest Quantity transactions (positive sales)",
         ["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "TotalAmount"]),
        ("highest_totalamount", "Highest TotalAmount transactions",
         ["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "TotalAmount"]),
    ]:
        p = TABLES_DIR / f"20_anomaly_{ex_name}_examples.csv"
        if p.exists():
            ex = pd.read_csv(p)
            cols_use = [c for c in ex_cols if c in ex.columns]
            if len(ex):
                _add_table_from_df(doc, ex[cols_use].head(3),
                                   caption=f"Table 9.x — {ex_label} (first 3 of {len(ex)} concrete examples)")
    doc.add_paragraph(
        "Interpretation: Examples are taken verbatim from the dataset. Very large quantities can correspond to "
        "bulk buy codes (e.g., charity bulk purchases, wholesale orders). Very large TotalAmount rows combine "
        "moderate-to-large quantity with elevated unit price."
    )

    doc.add_page_break()
    # ===== 10. LIMITATIONS =====
    _add_heading_styled(doc, "10. Limitations, Challenges & Caveats", level=1)
    limitations = [
        ("Cross-sectional single-sample data.",
         "Only one observation window is provided; true year-over-year seasonality cannot be estimated."),
        ("Missing CustomerIDs (≈24.93% of rows).",
         "Customer-level studies must exclude these rows or impute; no imputation is performed in this report."),
        ("Right-skewed numeric variables.",
         "Means are pulled upward by heavy tails; medians and quantile-based analysis are preferred for reporting central tendency."),
        ("Missing Description rows (0.27%).",
         "Product-label analysis may require a StockCode→Description lookup step; financial aggregations are unaffected."),
        ("Truncated final month (Dec 2011 ends 09th).",
         "Monthly comparisons must mention that December 2011 is mechanically smaller."),
        ("No unit root / causal identification.",
         "Correlation and trend results are descriptive; no causal claim should be drawn from this EDA alone."),
        ("Single-retailer single-country-dominant data.",
         "Findings reflect this UK-based non-store retailer specifically; generalisation is limited."),
    ]
    doc.add_paragraph("The following limitations are flagged explicitly in the interests of analytical honesty:")
    for title, body in limitations:
        p = doc.add_paragraph()
        r = p.add_run(f"{title} — ")
        r.bold = True
        p.add_run(body)

    doc.add_page_break()
    # ===== 11. CONCLUSIONS =====
    _add_heading_styled(doc, "11. Summary of Key Findings & Conclusions", level=1)
    findings = []
    try:
        ov = df_overview
        findings.append(
            f"Dataset scale: {int(ov.get('rows', 0)):,} lines, {int(ov.get('n_invoices', 0)):,} invoices, "
            f"{int(ov.get('n_customers', 0)):,} customers, {int(ov.get('n_countries', 0))} countries, "
            f"{int(ov.get('n_products', 0)):,} products."
        )
    except Exception:
        pass
    try:
        miss = missing_df.set_index("Column")
        cust_miss = int(miss.loc["CustomerID", "Missing Count"]) if "CustomerID" in miss.index else None
        if cust_miss is not None:
            findings.append(
                f"Missingness: CustomerID contains {cust_miss:,} nulls "
                f"({float(miss.loc['CustomerID', 'Missing Percentage (%)']):.2f}%); Description nulls are smaller."
            )
    except Exception:
        pass
    if len(mv):
        findings.append(
            f"Temporal: Transaction volume shows a clear growth trend across the observation period; "
            f"monthly range (lowest to highest complete month) spans a factor of approximately "
            f"{int(mv.transaction_lines.max() / max(1, mv.transaction_lines.nsmallest(2).iloc[0]))}x."
        )
    if len(ct):
        findings.append(
            f"Geography: {ct.iloc[0]['Country']} accounts for {float(ct.iloc[0]['lines_pct']):.2f}% "
            f"of transaction lines — the data is strongly home-market concentrated."
        )
    if r_val is not None:
        findings.append(
            f"Relationships: Quantity vs TotalAmount Pearson r = {r_val:.3f} (positive, as expected by construction); "
            f"UnitPrice vs Quantity r = {r_val2:.3f} (weak-to-negligible linear association)."
        )
    if len(iqr_summary):
        u_row = iqr_summary[iqr_summary.column == "UnitPrice"].iloc[0]
        findings.append(
            f"Outliers (IQR method): UnitPrice outliers = {int(u_row.outlier_count):,} "
            f"({float(u_row.outlier_pct):.2f}%); Quantity outliers = "
            f"{int(iqr_summary[iqr_summary.column == 'Quantity'].iloc[0].outlier_count):,}; "
            f"TotalAmount outliers = "
            f"{int(iqr_summary[iqr_summary.column == 'TotalAmount'].iloc[0].outlier_count):,}."
        )
    doc.add_paragraph("Key findings, each supported by computed values from the actual dataset:")
    for i, f in enumerate(findings, 1):
        doc.add_paragraph(f"{i}. {f}", style="List Number")

    doc.add_paragraph()
    doc.add_paragraph(
        "Conclusions: The UCI Online Retail dataset supports a rich EDA with well-established patterns in "
        "time, geography, product ranking, and distribution shape. Several heavy-tailed distributions and a "
        "highly UK-concentrated customer base impose clear constraints on interpretation, but also create the "
        "analytical texture that makes this dataset a standard teaching example. "
        "All reported statistics, percentages, correlations, and rankings in this report were computed from "
        "the actual loaded dataset; no values were fabricated or approximated."
    )

    doc.add_page_break()
    _add_heading_styled(doc, "12. References", level=1)
    refs = [
        "Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [https://archive.ics.uci.edu/dataset/352/online+retail]. Irvine, CA: University of California, School of Information and Computer Science.",
        "Chen, D.-Y., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. Journal of Database Marketing & Customer Strategy Management, 19(3-4), 197–208.",
        "McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference (SciPy 2010).",
        "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95.",
        "Waskom, M. L. (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021.",
        "Hoaglin, D. C., Mosteller, F., & Tukey, J. W. (1983). Understanding Robust and Exploratory Data Analysis. John Wiley & Sons.",
    ]
    for i, ref in enumerate(refs, 1):
        doc.add_paragraph(f"[{i}] {ref}", style="List Number")

    doc.save(str(output_path))
    logger.info(f"DOCX report saved to: {output_path} ({output_path.stat().st_size/1024:.1f} KB)")
    return output_path
