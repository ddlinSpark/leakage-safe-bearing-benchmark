"""Build the revised English SCI-style manuscript as a DOCX."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

from ..paths import DOCS_DIR, OUTPUTS_DIR, PROJECT_ROOT
from .native_equations import render_native_equations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成修订版 SCI 论文 DOCX")
    parser.add_argument(
        "--results",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn_final" / "results_extended.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=OUTPUTS_DIR
        / "public_paderborn_final"
        / "analysis_extended"
        / "summary_with_ci.csv",
    )
    parser.add_argument(
        "--paired",
        type=Path,
        default=OUTPUTS_DIR
        / "public_paderborn_final"
        / "analysis_extended"
        / "paired_comparisons.csv",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn_final" / "figures",
    )
    parser.add_argument(
        "--complexity",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn_final" / "model_complexity.csv",
    )
    parser.add_argument(
        "--threshold-results",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn_thresholds" / "summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCS_DIR / "paper" / "Leakage_Safe_Spindle_Bearing_Diagnosis_SCI_Revision.docx",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def set_run_font(run, name: str = "Times New Roman", size: float = 10.5) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)


def set_paragraph_font(paragraph, size: float = 10.5) -> None:
    for run in paragraph.runs:
        set_run_font(run, size=size)


def configure_document(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for name, size in (("Title", 17), ("Heading 1", 13), ("Heading 2", 11.5), ("Heading 3", 10.5)):
        style = styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(10 if name != "Title" else 0)
        style.paragraph_format.space_after = Pt(5)
        paragraph_properties = style._element.get_or_add_pPr()
        for border in paragraph_properties.findall(qn("w:pBdr")):
            paragraph_properties.remove(border)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run()
    set_run_font(run, size=8.5)
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.extend([field_begin, instruction, field_end])


def add_heading(document: Document, text: str, level: int = 1) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    run = paragraph.add_run(text)
    set_run_font(run, size={1: 13, 2: 11.5, 3: 10.5}[level])
    run.bold = True


def add_body(document: Document, text: str, first: bool = False) -> None:
    paragraph = document.add_paragraph()
    if first:
        paragraph.paragraph_format.first_line_indent = Cm(0.75)
    run = paragraph.add_run(text)
    set_run_font(run)


def add_caption(document: Document, text: str, before: bool = False) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4 if before else 2)
    paragraph.paragraph_format.space_after = Pt(5 if before else 9)
    paragraph.paragraph_format.keep_with_next = before
    run = paragraph.add_run(text)
    set_run_font(run, size=9)
    run.italic = not before
    if before:
        run.bold = True


def shade_cell(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_border(
    cell,
    edge: str,
    size: int,
    color: str = "000000",
) -> None:
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    node = borders.find(qn(f"w:{edge}"))
    if node is None:
        node = OxmlElement(f"w:{edge}")
        borders.append(node)
    node.set(qn("w:val"), "single")
    node.set(qn("w:sz"), str(size))
    node.set(qn("w:space"), "0")
    node.set(qn("w:color"), color)


def clear_table_borders(table) -> None:
    table_properties = table._tbl.tblPr
    borders = table_properties.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        table_properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "nil")


def set_cell_margins(cell, top: int = 80, start: int = 90, bottom: int = 80, end: int = 90) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for key, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{key}"))
        if node is None:
            node = OxmlElement(f"w:{key}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def prevent_row_split(row) -> None:
    properties = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    cant_split.set(qn("w:val"), "true")
    properties.append(cant_split)


def add_table(
    document: Document,
    headers: list[str],
    rows: list[list[str]],
    widths_cm: list[float],
    caption: str,
) -> None:
    add_caption(document, caption, before=True)
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    clear_table_borders(table)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    header_cells = table.rows[0].cells
    for index, (cell, text) in enumerate(zip(header_cells, headers)):
        cell.width = Cm(widths_cm[index])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell)
        set_cell_border(cell, "top", 16)
        set_cell_border(cell, "bottom", 8)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.keep_with_next = True
        run = paragraph.add_run(text)
        set_run_font(run, size=8.5)
        run.bold = True
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    tr_pr.append(repeat)
    prevent_row_split(table.rows[0])

    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        prevent_row_split(table.rows[-1])
        for index, (cell, value) in enumerate(zip(cells, values)):
            cell.width = Cm(widths_cm[index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if row_index == len(rows) - 1:
                set_cell_border(cell, "bottom", 16)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.keep_with_next = row_index < len(rows) - 1
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if index != len(cells) - 1 else WD_ALIGN_PARAGRAPH.LEFT
            )
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(str(value))
            set_run_font(run, size=8.3)
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def add_equation(
    document: Document,
    equation,
    number: int,
) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph._p.append(equation)
    run = paragraph.add_run()
    run.add_text(f"    ({number})")
    set_run_font(run, size=9.5)


def add_figure(
    document: Document,
    image_path: Path,
    caption: str,
    width_inches: float = 6.0,
) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    shape = paragraph.add_run().add_picture(
        str(image_path),
        width=Inches(width_inches),
    )
    shape._inline.docPr.set("descr", caption)
    shape._inline.docPr.set("title", image_path.stem.replace("_", " ").title())
    add_caption(document, caption)


def row_lookup(rows: list[dict[str, str]], method: str, fraction: float) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["method"] == method and abs(float(row["label_fraction"]) - fraction) < 1e-6
    ]
    if not matches:
        raise KeyError(f"No summary row for {method}, fraction={fraction}")
    return matches[0]


def fmt_mean_std(row: dict[str, str], mean_key: str, std_key: str) -> str:
    mean = float(row[mean_key])
    std = float(row[std_key])
    return f"{mean:.3f} ± {std:.3f}"


def paired_lookup(
    paired_rows: list[dict[str, str]],
    reference: str,
    method: str,
) -> dict[str, str] | None:
    for row in paired_rows:
        if row["reference"] == reference and row["method"] == method:
            return row
        if row["reference"] == method and row["method"] == reference:
            copy = dict(row)
            copy["mean_delta_macro_f1"] = str(-float(row["mean_delta_macro_f1"]))
            copy["reference"], copy["method"] = reference, method
            return copy
    return None


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.summary)
    detailed = read_csv(args.results)
    paired = read_csv(args.paired)
    complexity = read_csv(args.complexity)
    threshold_summary = read_csv(args.threshold_results)
    equation_latex = [
        r"W_x(a,b)=\frac{1}{\sqrt{a}}\int_{-\infty}^{\infty}x(t)\psi^{*}\left(\frac{t-b}{a}\right)dt,",
        r"I_c(a,b)=\frac{|W_{x_c}(a,b)|}{\max_{a,b}|W_{x_c}(a,b)|+\epsilon},",
        r"p(y=k\mid x)=\frac{\exp(z_k/T)}{\sum_{j=1}^{K}\exp(z_j/T)},",
        r"\mathcal{S}_k=\operatorname{TopM}\left\{i\notin\mathcal{L}:\hat{y}_i=k,\;p_{i,k}\geq\tau\right\},\quad \tau=0.60,",
    ]
    native_equations = iter(render_native_equations(equation_latex))

    raw_full = row_lookup(rows, "raw_cnn", 1.0)
    cwt_full = row_lookup(rows, "cwt_cnn", 1.0)
    gru_full = row_lookup(rows, "cwt_gru", 1.0)
    no_se_full = row_lookup(rows, "cwt_gru_no_se", 1.0)
    gru_only_full = row_lookup(rows, "gru_only", 1.0)
    st_full = row_lookup(rows, "cwt_gru_selftrain", 1.0)
    st_low = row_lookup(rows, "cwt_gru_selftrain", 0.1)
    sup_low = row_lookup(rows, "cwt_gru", 0.1)
    delta_low = float(st_low["macro_f1_mean"]) - float(sup_low["macro_f1_mean"])
    delta_full = float(st_full["macro_f1_mean"]) - float(gru_full["macro_f1_mean"])
    raw_ece = sum(float(item["raw_ece"]) for item in detailed) / len(detailed)
    cal_ece = sum(float(item["cal_ece"]) for item in detailed) / len(detailed)
    full_run_count = sum(
        int(row["runs"])
        for row in rows
        if abs(float(row["label_fraction"]) - 1.0) < 1e-6
    )
    complexity_by_model = {row["model"]: row for row in complexity}
    threshold_by_value = {
        round(float(row["threshold"]), 2): row for row in threshold_summary
    }
    best_threshold = max(
        threshold_summary,
        key=lambda row: float(row["test_macro_f1_mean"]),
    )
    raw_complexity = complexity_by_model["raw_cnn"]
    gru_complexity = complexity_by_model["cwt_gru"]
    parameter_ratio = float(gru_complexity["parameters"]) / float(
        raw_complexity["parameters"]
    )
    training_latency_ratio = float(gru_complexity["training_ms_batch"]) / float(
        raw_complexity["training_ms_batch"]
    )

    protocol_figure = args.figures_dir / "figure_protocol.png"
    if not protocol_figure.is_file():
        raise FileNotFoundError(
            f"Missing protocol figure: {protocol_figure}. Run make_protocol_figure.py first."
        )
    architecture_figure = args.figures_dir / "figure_architecture.png"
    if not architecture_figure.is_file():
        raise FileNotFoundError(
            f"Missing architecture figure: {architecture_figure}. "
            "Run make_architecture_figure.py first."
        )

    document = Document()
    configure_document(document)
    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run(
        "Leakage-Safe Benchmarking of Time-Frequency Fusion and Calibrated "
        "Self-Training for Spindle Bearing Fault Diagnosis"
    )
    title_properties = title._p.get_or_add_pPr()
    for border in title_properties.findall(qn("w:pBdr")):
        title_properties.remove(border)
    set_run_font(title_run, size=17)
    title_run.bold = True

    author = document.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_run = author.add_run("Anonymous manuscript prepared for double-blind review")
    set_run_font(author_run, size=10)
    author_run.italic = True

    add_heading(document, "Abstract", 1)
    abstract = (
        "Deep-learning studies of spindle-bearing diagnosis often report near-perfect "
        "accuracy, yet those estimates can depend on overlapping windows, repeated "
        "bearing identities, and pseudo-labels that are subsequently treated as ground "
        "truth. This study re-evaluates time-frequency fusion and calibrated "
        "self-training under a leakage-safe cross-domain protocol. The Paderborn "
        "dataset supplies vibration and two phase-current channels. Non-overlapping "
        "windows are grouped by bearing, training and validation use different "
        "operating conditions, and every test fold combines a held-out bearing with a "
        f"held-out condition. Across {full_run_count} full-label runs, the raw 1D-CNN "
        f"reached a mean macro-F1 of {float(raw_full['macro_f1_mean']):.3f}, whereas "
        f"CWT-GRU reached {float(gru_full['macro_f1_mean']):.3f} with "
        f"{parameter_ratio:.1f} times as many parameters and "
        f"{training_latency_ratio:.1f} times the batch training latency. Calibrated "
        f"self-training reduced CWT-GRU macro-F1 by {abs(delta_full):.3f}. At 10% "
        "labeled windows, no pseudo-label passed the predefined 0.60 threshold, so "
        "self-training made no update. A threshold sensitivity analysis on 50% labels "
        f"showed a non-monotonic response, with the best test mean at "
        f"tau={float(best_threshold['threshold']):.2f}. Temperature scaling improved "
        f"probability calibration but did not alter class predictions. These findings "
        "show that time-frequency fusion requires explicit justification against a raw "
        "baseline and that calibration must be evaluated separately from diagnostic "
        "accuracy under distribution shift."
    )
    abstract_paragraph = document.add_paragraph()
    abstract_paragraph.paragraph_format.left_indent = Cm(0.4)
    abstract_paragraph.paragraph_format.right_indent = Cm(0.4)
    abstract_paragraph.paragraph_format.space_after = Pt(8)
    run = abstract_paragraph.add_run(abstract)
    set_run_font(run, size=9.7)

    keywords = document.add_paragraph()
    keywords.paragraph_format.space_after = Pt(9)
    label = keywords.add_run("Keywords: ")
    set_run_font(label, size=9.8)
    label.bold = True
    body = keywords.add_run(
        "spindle bearing; leakage-safe benchmark; domain generalization; continuous "
        "wavelet transform; self-training; probability calibration; class collapse"
    )
    set_run_font(body, size=9.8)

    add_heading(document, "1 Introduction", 1)
    add_body(
        document,
        "Deep-learning studies of spindle-bearing diagnosis frequently report accuracy "
        "above 99%, but several common evaluation choices can inflate that estimate "
        "[1-3,14]. Overlapping windows share samples across splits, the same bearing "
        "can appear in training and testing, and rule-generated labels can later be "
        "relabelled by the model under test. These practices measure reproduction of a "
        "convenient label process more closely than generalization to a new bearing or "
        "operating condition.",
        first=True,
    )
    add_body(
        document,
        "Vibration supplies a direct mechanical response, while motor current adds an "
        "electrical view of the drive. Combining the two signals with a time-frequency "
        "representation is plausible, but the expected gain must survive a protocol "
        "that removes adjacent-window overlap, repeated bearing identities, and "
        "test-informed pseudo-labels. Calibration changes confidence, and "
        "self-training propagates existing model decisions; neither procedure creates "
        "missing fault information.",
    )
    add_body(
        document,
        "Recent spindle-related studies combine CWT with residual shrinkage or "
        "Transformer blocks [14,15], and evaluations of CWT-based bearing diagnosis "
        "often use CWRU as a benchmark [3,16]. CWRU remains a useful scale check, but "
        "published accuracy near 99% is not directly comparable across studies when "
        "window overlap, recording splits, and model-selection data differ. The present "
        "study therefore re-runs standard baselines under one explicit split and treats "
        "the resulting ranking as the evidence for model choice.",
    )

    add_heading(document, "1.1 Contributions", 2)
    for text in [
        "A leakage-safe benchmark that combines non-overlapping windows, held-out bearing "
        "identities, and held-out operating conditions.",
        "A rigorous six-model comparison showing that sophisticated time-frequency models "
        "do not automatically outperform a raw waveform baseline under cross-domain shift.",
        "A controlled self-training and threshold analysis in which pseudo-labels can be "
        "drawn only from the unlabeled training partition.",
        "An uncertainty-aware analysis of diagnostic accuracy, class collapse, calibration, "
        "model complexity, and inference cost.",
    ]:
        paragraph = document.add_paragraph(style="List Number")
        run = paragraph.add_run(text)
        set_run_font(run)

    add_heading(document, "2 Related Work", 1)
    add_body(
        document,
        "Time-series imaging introduced a straightforward route from signal processing "
        "to convolutional networks [4]. Continuous wavelet transform adds scale-dependent "
        "time-frequency localization [5], and residual attention networks provide a "
        "stable image backbone [6-8]. CBAM and squeeze-and-excitation blocks emphasize "
        "informative spatial and channel responses, while bidirectional GRUs model "
        "ordered temporal variation [9]. These components motivate the CWT-GRU branch, "
        "but architectural sophistication alone does not establish a causal improvement. "
        "A multiscale CWT CNN-Transformer model [22] and decomposition-specific "
        "frequency/time kernels [23] represent stronger responses to the resolution "
        "and global-semantics limitations of generic CWT image classification.",
    )
    add_body(
        document,
        "Pseudo-label self-training was popularized as a simple semi-supervised "
        "procedure [10]. Its behavior depends on the reliability of the initial model "
        "and can amplify confirmation bias. Temperature scaling [11] and related "
        "post-processing methods change confidence without changing the ranking of "
        "classes. Reporting calibration as a correction for incorrect labels conflates "
        "these two mechanisms. Residual shrinkage and gated convolution have improved "
        "noise robustness in recent spindle-bearing studies [13-15], which makes strong "
        "raw and CWT baselines necessary.",
    )
    add_body(
        document,
        "Benchmark reviews show that CWRU accuracy is sensitive to data leakage and "
        "split construction [1-3]. The Paderborn dataset addresses this concern by "
        "providing healthy, artificially damaged, and lifetime-damaged bearings under "
        "several operating conditions, with synchronously measured vibration and motor "
        "currents [1,21]. The reported experiments use its bearing identifiers as groups "
        "and its operating conditions as held-out domains.",
    )
    add_table(
        document,
        ["Study", "Representative design", "Evidence boundary"],
        [
            [
                "Smith and Randall [3]",
                "Signal-diagnostic study of CWRU recordings",
                "Establishes benchmark subtleties; not a neural accuracy contest",
            ],
            [
                "Hendriks et al. [2]",
                "Leakage-aware CWRU deep-learning evaluation",
                "Shows that split construction changes reported performance",
            ],
            [
                "Li et al. [14]",
                "CWT + deep residual shrinkage on a CWRU 10-class split",
                "Random 7:3 split and noise experiments; different test isolation",
            ],
            [
                "This study",
                "Paderborn vibration/current fusion with grouped condition and bearing tests",
                "Repeated folds and seeds; no direct numeric claim against external splits",
            ],
        ],
        [3.3, 6.4, 5.5],
        "Table 1. Comparison of representative evaluation designs from prior work.",
    )

    add_heading(document, "3 Data and Evaluation Protocol", 1)
    add_body(
        document,
        "The primary benchmark contains 16 bearings from the Paderborn University "
        "Bearing Data Center. Training uses K001-K004 (healthy), KA01, KA04, and KA08 "
        "(outer-ring damage), KI01, KI04, and KI08 (inner-ring damage), and KB23 and KB24 "
        "(multiple damage). K005, KA15, KI14, and KB27 form the unseen bearing set. Each "
        "bearing was recorded under four operating conditions. For every bearing and "
        "condition, measurements 1 and 2 were used. The HostService raster contains "
        "vibration and two phase-current channels sampled at 64 kHz.",
        first=True,
    )
    add_body(
        document,
        "Data were segmented into windows of 8192 samples with a stride of 8192. Windows "
        "therefore do not overlap. Across the four folds, the test operating condition "
        "changes, and K005, KA15, KI14, and KB27 are never used for fitting. One of the "
        "remaining conditions is used for validation and the other two for training. "
        "The resulting test set has seen neither the test bearing units nor the test "
        "condition during training.",
    )
    add_table(
        document,
        ["Item", "Design choice", "Purpose"],
        [
            ["Dataset", "Paderborn University bearing data", "Verified fault labels and multimodal signals"],
            ["Signals", "Vibration + phase current 1 + phase current 2", "Mechanical and electrical views"],
            ["Window", "8192 samples, stride 8192", "No overlap between windows"],
            ["Classes", "Healthy, outer ring, inner ring, multiple", "Four bearing-health groups"],
            [
                "Bearing groups",
                "Multiple training units and one test unit per class",
                "No bearing-identity leakage",
            ],
            ["Condition groups", "One condition held out per fold", "Operating-condition generalization"],
            ["Test isolation", "Unseen bearing + unseen condition", "Strict external evaluation"],
        ],
        [3.0, 7.0, 5.0],
        "Table 2. Data and evaluation design of the public benchmark.",
    )
    add_figure(
        document,
        protocol_figure,
        "Figure 1. Revised evaluation flow. Test labels and test conditions are isolated "
        "from supervised fitting and pseudo-label generation.",
        6.4,
    )

    add_heading(document, "4 Method", 1)
    add_figure(
        document,
        architecture_figure,
        "Figure 2. CWT-GRU architecture. The time-frequency branch captures local "
        "spectral structure and the bidirectional recurrent branch retains waveform "
        "order and transient timing.",
        6.5,
    )
    add_heading(document, "4.1 Signal normalization and CWT representation", 2)
    add_body(
        document,
        "Each channel is standardized independently within a window. For channel x(t), "
        "the continuous wavelet coefficient at scale a and position b is",
    )
    add_equation(
        document,
        next(native_equations),
        1,
    )
    add_body(
        document,
        "A Morlet mother wavelet is used because it provides a balanced time-frequency "
        "response. The magnitude map is bilinearly resized to 96 × 96 and normalized by "
        "its maximum:",
    )
    add_equation(
        document,
        next(native_equations),
        2,
    )
    add_body(
        document,
        "The three-channel CWT tensor is processed by a compact residual CNN with "
        "squeeze-and-excitation attention and global average pooling. The raw waveform "
        "branch uses two-layer bidirectional GRU with 64 hidden units per direction. "
        "A two-layer projection fuses the 192-dimensional image feature and the "
        "128-dimensional recurrent feature into four class logits.",
    )

    add_heading(document, "4.2 Temperature scaling and self-training", 2)
    add_body(
        document,
        "For logits z and temperature T, the calibrated probability is",
    )
    add_equation(
        document,
        next(native_equations),
        3,
    )
    add_body(
        document,
        "Temperature is selected on the validation condition through minimizing "
        "cross-entropy. Self-training creates pseudo-labels only for windows in the "
        "unlabeled training partition. For predicted class k, candidates are ranked by "
        "confidence and limited to a class-balanced set:",
    )
    add_equation(
        document,
        next(native_equations),
        4,
    )
    add_body(
        document,
        "Here, L denotes the set of windows with verified labels, p_i,k is the "
        "temperature-scaled probability of class k for unlabeled window i, and M is the "
        "maximum number of selected windows per predicted class. The model is "
        "reinitialized after each pseudo-label round. Labeled and selected "
        "unlabeled windows are combined in a confidence-weighted cross-entropy loss. "
        "No test sample, test label, or validation label is added to the self-training "
        "pool.",
    )

    add_heading(document, "4.3 Baselines and statistics", 2)
    add_body(
        document,
        "Raw 1D-CNN, CWT 2D-CNN, and CWT-GRU use the same windows, folds, and label "
        "budgets. All models use AdamW, cosine learning-rate decay, mixed precision, and "
        "early stopping. Results are aggregated over held-out conditions and random "
        "seeds. We report macro-F1 as the primary measure because class support is not "
        "the main object of inference. Paired comparisons use the Wilcoxon signed-rank "
        "test when the number of matched runs is sufficient.",
    )
    complexity_labels = [
        ("raw_cnn", "Raw 1D-CNN"),
        ("gru_only", "GRU only"),
        ("cwt_cnn", "CWT 2D-CNN"),
        ("cwt_gru_no_se", "CWT-GRU without SE"),
        ("cwt_gru", "CWT-GRU"),
    ]
    complexity_table_rows = []
    for key, label in complexity_labels:
        item = complexity_by_model[key]
        complexity_table_rows.append(
            [
                label,
                f"{float(item['parameters']) / 1e6:.2f}",
                f"{float(item['gflops_sample']):.2f}",
                f"{float(item['inference_ms_batch']):.1f}",
                f"{float(item['training_ms_batch']):.1f}",
            ]
        )
    add_table(
        document,
        [
            "Model",
            "Parameters (M)",
            "GFLOPs/sample",
            "Inference (ms/batch)",
            "Training step (ms/batch)",
        ],
        complexity_table_rows,
        [4.3, 2.4, 2.5, 3.0, 3.4],
        "Table 3. Model complexity and latency measured on one RTX 5080 with batch size 64.",
    )
    add_body(
        document,
        "The GRU term is added analytically because PyTorch's profiler does not count "
        "recurrent matrix products. The estimate covers the three gate matrix "
        "multiplications in both directions and layers but excludes elementwise "
        "nonlinearities. Latencies include CUDA synchronization and are therefore "
        "hardware- and implementation-dependent.",
    )
    add_body(
        document,
        "The cost pattern is not explained by parameter count alone. GRU-only has the "
        "smallest parameter set (0.12 M), but its 8192 recurrent steps produce a "
        "21.7 ms inference batch and a 1.63 GFLOP estimate per sample. CWT-GRU adds "
        "the image encoder and remains at a similar latency. The image-only model has "
        "0.88 M parameters yet completes inference in 1.1 ms because its spatial "
        "operations expose more parallelism.",
    )
    add_body(
        document,
        "These measurements make the trade-off concrete. The recurrent branch consumes "
        "most of the training and inference latency, while the complete CWT-GRU does "
        "not improve macro-F1 relative to CWT-only. For latency-sensitive spindle "
        "monitoring, the raw model is therefore the stronger baseline in this study; "
        "any recurrent deployment would need a smaller sequence chunk, a stateful "
        "streaming design, or a clear accuracy benefit that the current protocol does "
        "not reveal.",
    )

    add_heading(document, "5 Experiments and Results", 1)
    add_heading(document, "5.1 Full-label comparison", 2)
    add_table(
        document,
        ["Method", "Macro-F1", "Accuracy", "Runs"],
        [
            [
                "Raw 1D-CNN",
                fmt_mean_std(raw_full, "macro_f1_mean", "macro_f1_std"),
                fmt_mean_std(raw_full, "accuracy_mean", "accuracy_std"),
                raw_full["runs"],
            ],
            [
                "CWT 2D-CNN",
                fmt_mean_std(cwt_full, "macro_f1_mean", "macro_f1_std"),
                fmt_mean_std(cwt_full, "accuracy_mean", "accuracy_std"),
                cwt_full["runs"],
            ],
            [
                "CWT-GRU",
                fmt_mean_std(gru_full, "macro_f1_mean", "macro_f1_std"),
                fmt_mean_std(gru_full, "accuracy_mean", "accuracy_std"),
                gru_full["runs"],
            ],
            [
                "CWT-GRU + calibrated self-training",
                fmt_mean_std(st_full, "macro_f1_mean", "macro_f1_std"),
                fmt_mean_std(st_full, "accuracy_mean", "accuracy_std"),
                st_full["runs"],
            ],
        ],
        [5.1, 3.8, 3.8, 2.3],
        "Table 4. Full-label results under unseen-bearing and unseen-condition testing.",
    )
    ranking = sorted(
        [
            ("raw 1D-CNN", float(raw_full["macro_f1_mean"])),
            ("CWT 2D-CNN", float(cwt_full["macro_f1_mean"])),
            ("CWT-GRU", float(gru_full["macro_f1_mean"])),
            ("CWT-GRU with calibrated self-training", float(st_full["macro_f1_mean"])),
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    add_body(
        document,
        f"The highest mean macro-F1 was obtained by {ranking[0][0]} "
        f"({ranking[0][1]:.3f}), followed by {ranking[1][0]} ({ranking[1][1]:.3f}). "
        f"Calibrated self-training changed the CWT-GRU macro-F1 by {delta_full:+.3f} "
        "under the full-label setting. This comparison is intentionally reported "
        "without selecting only a favorable fold.",
    )
    paired_gru_raw = paired_lookup(paired, "raw_cnn", "cwt_gru")
    paired_st_gru = paired_lookup(paired, "cwt_gru", "cwt_gru_selftrain")
    if paired_gru_raw and paired_st_gru:
        add_body(
            document,
            "With matched folds and seeds, CWT-GRU differed from the raw CNN by "
            f"{float(paired_gru_raw['mean_delta_macro_f1']):+.3f} macro-F1 "
            f"(Wilcoxon p = {float(paired_gru_raw['wilcoxon_p']):.3f}). One round of "
            "calibrated self-training differed from supervised CWT-GRU by "
            f"{float(paired_st_gru['mean_delta_macro_f1']):+.3f} "
            f"(p = {float(paired_st_gru['wilcoxon_p']):.3f}). These differences were "
            "not separated from run-to-run variation at the conventional 0.05 level.",
        )
    main_figure = args.figures_dir / "figure_main_comparison.png"
    if main_figure.exists():
        add_figure(
            document,
            main_figure,
            "Figure 3. Mean macro-F1 of the full-label models. Error bars show one "
            "standard deviation across matched folds and seeds.",
            5.9,
        )

    add_heading(document, "5.2 Component ablation and computational cost", 2)
    ablation_rows = [
        [
            "CWT 2D-CNN",
            fmt_mean_std(cwt_full, "macro_f1_mean", "macro_f1_std"),
            fmt_mean_std(cwt_full, "accuracy_mean", "accuracy_std"),
        ],
        [
            "GRU only",
            fmt_mean_std(gru_only_full, "macro_f1_mean", "macro_f1_std"),
            fmt_mean_std(gru_only_full, "accuracy_mean", "accuracy_std"),
        ],
        [
            "CWT-GRU without SE",
            fmt_mean_std(no_se_full, "macro_f1_mean", "macro_f1_std"),
            fmt_mean_std(no_se_full, "accuracy_mean", "accuracy_std"),
        ],
        [
            "CWT-GRU with SE",
            fmt_mean_std(gru_full, "macro_f1_mean", "macro_f1_std"),
            fmt_mean_std(gru_full, "accuracy_mean", "accuracy_std"),
        ],
    ]
    add_table(
        document,
        ["Architecture", "Macro-F1", "Accuracy"],
        ablation_rows,
        [7.0, 4.0, 4.0],
        "Table 5. Component ablation under the strict unseen-bearing and unseen-condition split.",
    )
    paired_no_se = paired_lookup(paired, "cwt_gru_no_se", "cwt_gru")
    paired_cwt_gru = paired_lookup(paired, "cwt_cnn", "cwt_gru")
    paired_cwt_gru_only = paired_lookup(paired, "cwt_cnn", "gru_only")
    if paired_no_se and paired_cwt_gru and paired_cwt_gru_only:
        add_body(
            document,
            "Removing SE changed CWT-GRU macro-F1 by "
            f"{float(paired_no_se['mean_delta_macro_f1']):+.3f} "
            f"(p = {float(paired_no_se['wilcoxon_p']):.3f}). Replacing the CWT "
            "image branch with the recurrent path changed the mean relative to CWT "
            f"by {float(paired_cwt_gru_only['mean_delta_macro_f1']):+.3f} "
            f"(p = {float(paired_cwt_gru_only['wilcoxon_p']):.3f}), while adding GRU "
            "to the CWT branch changed it by "
            f"{float(paired_cwt_gru['mean_delta_macro_f1']):+.3f} "
            f"(p = {float(paired_cwt_gru['wilcoxon_p']):.3f}). The fusion did not "
            "produce a stable gain over the CWT-only branch.",
        )
    raw_params = float(complexity_by_model["raw_cnn"]["parameters"])
    gru_params = float(complexity_by_model["cwt_gru"]["parameters"])
    raw_train = float(complexity_by_model["raw_cnn"]["training_ms_batch"])
    gru_train = float(complexity_by_model["cwt_gru"]["training_ms_batch"])
    raw_infer = float(complexity_by_model["raw_cnn"]["inference_ms_batch"])
    gru_infer = float(complexity_by_model["cwt_gru"]["inference_ms_batch"])
    add_body(
        document,
        f"CWT-GRU used {gru_params / raw_params:.1f} times as many parameters as the "
        f"raw CNN, required {gru_train / raw_train:.1f} times the measured batch "
        f"training time, and required {gru_infer / raw_infer:.1f} times the inference "
        "time. The greater cost was not accompanied by higher macro-F1 under this "
        "protocol. Table 3 also shows that the CWT image encoder itself was fast; the "
        "recurrent path introduced most of the latency because CUDA executes its time "
        "steps sequentially.",
    )

    add_heading(document, "5.3 Label efficiency and pseudo-label reliability", 2)
    budget_rows = []
    for method, label in (
        ("cwt_gru", "CWT-GRU, supervised"),
        ("cwt_gru_selftrain", "CWT-GRU, calibrated self-training"),
    ):
        candidates = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["label_fraction"]),
        )
        for row in candidates:
            budget_rows.append(
                [
                    label,
                    f"{float(row['label_fraction']) * 100:.0f}%",
                    fmt_mean_std(row, "macro_f1_mean", "macro_f1_std"),
                    row["runs"],
                ]
            )
    add_table(
        document,
        ["Model", "Labeled per class", "Macro-F1", "Runs"],
        budget_rows,
        [6.0, 3.3, 4.2, 1.8],
        "Table 6. Label-budget results. Pseudo-labels use only the unlabeled training partition.",
    )
    label_figure = args.figures_dir / "figure_label_efficiency.png"
    if label_figure.exists():
        add_figure(
            document,
            label_figure,
            "Figure 4. Macro-F1 at controlled label budgets. The difference between "
            "curves measures the observed effect of calibrated self-training.",
            5.5,
        )
    add_body(
        document,
        f"At 10% labels, supervised CWT-GRU reached {float(sup_low['macro_f1_mean']):.3f} "
        f"macro-F1 and calibrated self-training reached {float(st_low['macro_f1_mean']):.3f}. "
        f"The paired mean change was {delta_low:+.3f}. The direction and size of this "
        "change are interpreted according to the repeated-run statistics rather than "
        "as evidence that pseudo-labels recover true annotations. Across the four "
        "10% folds, no candidate passed the fixed 0.60 confidence threshold. At 50% "
        "labels, the mean increased by 0.019, but the four paired differences ranged "
        "from -0.049 to +0.175 and the Wilcoxon test did not reject zero (p = 1.000).",
    )

    add_heading(document, "5.4 Confidence threshold sensitivity", 2)
    add_body(
        document,
        "Table 7 and Figure 5 report how the threshold changes candidate coverage, "
        "pseudo-label accuracy, and post-retraining diagnostic performance.",
    )
    threshold_rows_for_table = [
        [
            f"{float(row['threshold']):.1f}",
            f"{float(row['pseudo_coverage_mean']) * 100:.1f}%",
            f"{float(row['pseudo_accuracy_mean']):.3f}",
            f"{float(row['test_macro_f1_mean']):.3f} ± "
            f"{float(row['test_macro_f1_std']):.3f}",
        ]
        for row in sorted(
            threshold_summary,
            key=lambda item: float(item["threshold"]),
        )
    ]
    add_table(
        document,
        ["Threshold", "Candidate coverage", "Pseudo-label accuracy", "Test Macro-F1"],
        threshold_rows_for_table,
        [2.8, 3.6, 4.1, 4.5],
        "Table 7. Confidence-threshold sensitivity at 50% labels over four held-out conditions.",
    )
    threshold_figure = args.figures_dir / "figure_threshold_sensitivity.png"
    if threshold_figure.exists():
        add_figure(
            document,
            threshold_figure,
            "Figure 5. Confidence-threshold sensitivity. Lower panels report pseudo-label "
            "count and classification accuracy before retraining.",
            5.7,
        )
    tau_03 = threshold_by_value[0.3]
    tau_04 = threshold_by_value[0.4]
    tau_07 = threshold_by_value[0.7]
    add_body(
        document,
        "At a threshold of 0.3, 85.6% of unlabeled windows were selected and the "
        "pseudo-label accuracy was 0.910. The corresponding retrained test macro-F1 "
        f"was {float(tau_03['test_macro_f1_mean']):.3f}. At 0.4, coverage fell slightly "
        "but pseudo-label accuracy and test macro-F1 increased to "
        f"{float(tau_04['pseudo_accuracy_mean']):.3f} and "
        f"{float(tau_04['test_macro_f1_mean']):.3f}, respectively. At 0.7, higher "
        "pseudo-label accuracy did not translate into a better classifier: coverage "
        "fell to 55.3% and test macro-F1 decreased to "
        f"{float(tau_07['test_macro_f1_mean']):.3f}. The response is non-monotonic, "
        "so a fixed threshold cannot be assumed to control either label quality or "
        "end-task performance.",
    )

    add_heading(document, "5.5 Calibration results", 2)
    add_body(
        document,
        "Table 8 and Figure 6 compare ECE before and after temperature scaling for the "
        "principal models.",
    )
    calibration_rows = []
    for method, label in (
        ("raw_cnn", "Raw 1D-CNN"),
        ("cwt_cnn", "CWT 2D-CNN"),
        ("cwt_gru", "CWT-GRU"),
        ("cwt_gru_selftrain", "CWT-GRU + calibrated self-training"),
    ):
        matched = [
            item
            for item in detailed
            if item["method"] == method and abs(float(item["label_fraction"]) - 1.0) < 1e-6
        ]
        if not matched:
            continue
        raw_error = sum(float(item["raw_ece"]) for item in matched) / len(matched)
        scaled_error = sum(float(item["cal_ece"]) for item in matched) / len(matched)
        calibration_rows.append([label, f"{raw_error:.4f}", f"{scaled_error:.4f}"])
    add_table(
        document,
        ["Method", "ECE before scaling", "ECE after scaling"],
        calibration_rows,
        [6.6, 4.2, 4.2],
        "Table 8. Expected calibration error before and after temperature scaling.",
    )
    calibration_figure = args.figures_dir / "figure_calibration.png"
    if calibration_figure.exists():
        add_figure(
            document,
            calibration_figure,
            "Figure 6. Expected calibration error before and after temperature scaling. "
            "Temperature scaling changes probabilities but not predicted classes.",
            5.6,
        )
    add_body(
        document,
        "Temperature scaling is a one-parameter monotonic transformation. It can reduce "
        "overconfidence and improve probability estimates without changing the argmax. "
        "Therefore, an accuracy change in a self-training pipeline must be traced to "
        "changed labels or decision rules rather than calibration alone.",
    )
    add_body(
        document,
        "Temperature scaling reduced ECE for every principal model, but the remaining "
        "errors were still large: 0.348 for Raw 1D-CNN, 0.392 for CWT 2D-CNN, 0.385 "
        "for CWT-GRU, and 0.425 for self-training. A monotonic rescaling cannot repair "
        "a probability mass assigned to the wrong class. This distinction matters for "
        "industrial alarms because a well-calibrated wrong prediction can still trigger "
        "an incorrect maintenance action.",
    )
    add_body(
        document,
        "The training logs also show why validation-based temperature selection can "
        "remain optimistic. Validation measurements came from the same training bearing "
        "units, while the test condition used a different bearing and operating point. "
        "A temperature that matches in-domain confidence therefore does not guarantee "
        "calibration after a domain shift. Future work should fit and report calibration "
        "under grouped or leave-domain-out validation and include a reliability diagram "
        "for each held-out condition.",
    )
    add_body(
        document,
        "ECE is also sensitive to binning and class prevalence. The present comparison "
        "uses 15 equal-width confidence bins on the same test folds, which supports "
        "the within-study ranking but does not make the values directly comparable to "
        "publications using different bins. Future reports should add reliability "
        "diagrams and at least one proper scoring rule, such as negative log-likelihood "
        "or the Brier score, alongside ECE.",
    )

    add_heading(document, "5.6 Class collapse and error structure", 2)
    add_body(
        document,
        "Figure 7 aggregates all 12 prediction files of each principal model over the "
        "four folds and three seeds, avoiding conclusions based on one favorable run.",
    )
    confusion_figure = args.figures_dir / "figure_confusion_matrix.png"
    if confusion_figure.exists():
        add_figure(
            document,
            confusion_figure,
            "Figure 7. Row-normalized confusion matrices aggregated across folds and "
            "seeds for the four principal models.",
            6.3,
        )
    add_body(
        document,
        "The aggregated matrices show a persistent inner-ring attractor. The raw CNN "
        "classified only 28% of inner-ring windows as inner-ring and sent 48% to the "
        "outer-ring class. CWT-GRU improved the inner-ring diagonal to 55%, but sent "
        "49% of outer-ring windows to inner-ring and 42% of inner-ring windows to "
        "healthy. Self-training increased the multiple-damage-to-inner-ring rate to "
        "51%. These exchanges match the fall in macro-F1: calibration reallocates "
        "confidence among similar domains without creating a representation that "
        "separates them.",
    )

    add_heading(document, "6 Discussion", 1)
    add_heading(document, "6.1 The validation-to-test gap reflects domain generalization", 2)
    add_body(
        document,
        "The large validation-to-test gap is one of the clearest findings. Raw 1D-CNN, "
        "CWT 2D-CNN, and CWT-GRU reached mean validation macro-F1 values of 0.861, "
        "0.816, and 0.823, but their test values were 0.308, 0.291, and 0.284. The "
        "model-selection data and the test data differ in both bearing identity and "
        "operating condition. Consequently, a score around 0.3 should be read as an "
        "out-of-distribution diagnostic result, not as evidence that the architecture "
        "is intrinsically incapable of learning the training distribution.",
        first=True,
    )
    add_heading(document, "6.2 Time-frequency images lose information that the waveform retains", 2)
    add_body(
        document,
        "The implemented CWT branch uses coefficient magnitude and therefore discards "
        "phase. It also resamples a 48-scale, 8192-sample time-frequency field to a "
        "96 x 96 image. Phase relations between current and vibration channels, short "
        "impact timing, and narrow sidebands can be weakened by these operations. The "
        "raw 1D-CNN, in contrast, receives the ordered samples directly. Wu and Yu [23] "
        "make a related point: conventional small 2D kernels can fail to capture the "
        "global semantics of multi-component time-frequency spectrograms, and "
        "decomposition-specific frequency and time kernels may be more appropriate. "
        "The present negative result should therefore constrain future CWT claims: a "
        "generic image classifier is not a neutral substitute for phase-preserving or "
        "direction-aware signal processing.",
    )
    add_heading(document, "6.3 Self-training fails for two different reasons", 2)
    add_body(
        document,
        "At 10% labels, the initial CWT-GRU was too uncertain for any candidate to pass "
        "the predefined 0.60 threshold after validation temperature scaling. The "
        "self-training code therefore returned the initial model unchanged. At 50% "
        "labels, many candidates passed, but the threshold response was non-monotonic: "
        "0.4 produced the highest mean test macro-F1, while 0.7 produced the most "
        "accurate selected labels but a much smaller candidate pool and a lower test "
        "score. Label precision alone is insufficient when selection coverage and "
        "class composition change.",
    )
    add_body(
        document,
        "Full-label self-training supplied pseudo-labels for 324 to 496 windows per "
        "fold even though those windows were not part of the balanced labeled subset. "
        "Replacing true labels with confident predictions increased the inner-ring "
        "attractor and reduced macro-F1. This is confirmation bias at the class level: "
        "the model is most consistent on the classes it already recognizes, and the "
        "class-balanced cap cannot restore information about systematically confused "
        "classes.",
    )
    add_heading(document, "6.4 Accuracy does not justify recurrent fusion by default", 2)
    add_body(
        document,
        f"CWT-GRU had {gru_params:.0f} parameters and required "
        f"{gru_train:.1f} ms per training batch, compared with {raw_params:.0f} "
        f"parameters and {raw_train:.1f} ms for Raw 1D-CNN. The recurrent model also "
        f"required {gru_infer:.1f} ms versus {raw_infer:.1f} ms for inference. It did "
        "not produce a statistically distinguishable accuracy gain. The high recurrent "
        "latency is consistent with the sequential dependency across 8192 time steps "
        "on a GPU. A model with lower FLOPs is not necessarily faster when its kernels "
        "expose limited parallelism.",
    )
    add_heading(document, "6.5 A reporting standard for future comparisons", 2)
    add_body(
        document,
        "The benchmark supports four concrete reporting requirements. First, windows "
        "must be non-overlapping across evaluation boundaries or grouped by source "
        "recording. Second, bearing, machine, and operating-condition identifiers must "
        "be reported separately from random-window splits. Third, self-training must "
        "state the unlabeled source, confidence rule, candidate count, pseudo-label "
        "accuracy when labels are available for audit, and the number of update rounds. "
        "Fourth, calibration metrics should be accompanied by diagnostic accuracy and "
        "not interpreted as evidence of corrected labels. Studies using CWRU or other "
        "saturated benchmarks should include a cross-domain or grouped evaluation "
        "before claiming industrial reliability.",
    )

    add_heading(document, "7 Threats to Validity", 1)
    add_body(
        document,
        "The public benchmark isolates bearings on a controlled test rig, not a complete "
        "machine-tool spindle assembly. It cannot reproduce installation stiffness, "
        "tool-holder dynamics, coolant effects, or production load cycles. The 16-bearing "
        "subset keeps computation tractable but limits the number of independent units; "
        "the four held-out conditions should not be interpreted as coverage of all spindle "
        "operating states. The held-out fault bearings KA15, KI14, and KB27 belong to "
        "the lifetime-damage group, while several training bearings are artificially "
        "damaged. This distinction improves the test realism relative to a fully "
        "seeded-fault benchmark, but the dataset still does not reproduce long-term "
        "wear progression or complete spindle assembly dynamics.",
        first=True,
    )
    add_body(
        document,
        "Only 16 bearings are used, and each condition contains two measurement "
        "repetitions. This keeps the experiment tractable but limits the number of "
        "independent units and prevents strong claims about rare faults. External "
        "studies are discussed qualitatively because their splits, window overlap, and "
        "test-label provenance differ from the present protocol. Direct numeric "
        "comparison with a published 99% score would therefore be misleading.",
    )

    add_heading(document, "8 Conclusion", 1)
    add_body(
        document,
        "This study provides a leakage-safe public benchmark and an explicit comparison "
        "against standard waveform and time-frequency baselines. Under unseen-bearing "
        "and unseen-condition testing, "
        "the complex CWT-GRU model did not outperform the waveform baseline by a "
        "statistically distinguishable margin, and calibrated self-training reduced "
        "macro-F1 in the repeated full-label experiment. Temperature scaling improved "
        "probability calibration without changing predicted classes. The study therefore "
        "supports strict evaluation and separate reporting of calibration, pseudo-labeling, "
        "and diagnostic accuracy. In this benchmark, a simpler Raw 1D-CNN provided a "
        "stronger accuracy-cost trade-off than CWT-GRU, and thresholded self-training "
        "did not compensate for the distribution shift.",
    )

    add_heading(document, "Data and Code Availability", 1)
    add_body(
        document,
        "CWRU and Paderborn data are available from their public data centers [20,21]. "
        "The benchmark code defines the exact file identifiers, non-overlapping windows, "
        "bearing groups, condition folds, baselines, pseudo-label selection, and statistical "
        "summaries. The archive IDs and split manifests accompany the source code; no data "
        "or model weight is redistributed by this manuscript.",
    )

    add_heading(document, "References", 1)
    references = [
        "C. Lessmeier, J. K. Kimotho, D. Zimmer, and W. Sextro, \"Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: A benchmark data set for data-driven classification,\" in Proc. European Conf. PHM Soc., vol. 3, no. 1, 2016, doi: 10.36001/phme.2016.v3i1.1577.",
        "J. Hendriks, P. Dumond, and D. A. Knox, \"Towards better benchmarking using the CWRU bearing fault dataset,\" Mech. Syst. Signal Process., vol. 169, Art. no. 108732, Apr. 2022, doi: 10.1016/j.ymssp.2021.108732.",
        "W. A. Smith and R. B. Randall, \"Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study,\" Mech. Syst. Signal Process., vol. 64-65, pp. 100-131, 2015.",
        "Z. Wang and T. Oates, \"Imaging time-series to improve classification and imputation,\" in Proc. IJCAI, 2015, pp. 2627-2633.",
        "C. Torrence and G. P. Compo, \"A practical guide to wavelet analysis,\" Bull. Amer. Meteor. Soc., vol. 79, no. 1, pp. 61-78, 1998.",
        "J. Hu, L. Shen, and G. Sun, \"Squeeze-and-excitation networks,\" in Proc. IEEE CVPR, 2018, pp. 7132-7141.",
        "S. Woo, J. Park, J. Y. Lee, and I. S. Kweon, \"CBAM: Convolutional block attention module,\" in Proc. ECCV, 2018, pp. 3-19.",
        "K. He, X. Zhang, S. Ren, and J. Sun, \"Deep residual learning for image recognition,\" in Proc. IEEE CVPR, 2016, pp. 770-778.",
        "K. Cho, B. van Merrienboer, C. Gulcehre, et al., \"Learning phrase representations using RNN encoder-decoder for statistical machine translation,\" in Proc. EMNLP, 2014, pp. 1724-1734.",
        "D.-H. Lee, \"Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks,\" in Proc. ICML Workshop Challenges Representation Learning, 2013.",
        "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, \"On calibration of modern neural networks,\" in Proc. ICML, 2017, pp. 1321-1330.",
        "T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, \"Focal loss for dense object detection,\" in Proc. IEEE ICCV, 2017, pp. 2980-2988.",
        "M. Zhao, S. Zhong, X. Fu, B. Tang, and M. Pecht, \"Deep residual shrinkage networks for fault diagnosis,\" IEEE Trans. Ind. Informat., vol. 16, no. 7, pp. 4681-4690, Jul. 2020.",
        "X. Li, J. Wang, J. Wang, J. Wang, J. Chen, and X. Yu, \"Research on CNC machine tool spindle fault diagnosis method based on DRSN-GCE model,\" Algorithms, vol. 18, no. 6, Art. no. 304, 2025, doi: 10.3390/a18060304.",
        "X. Li, J. Chen, J. Wang, et al., \"Research on fault diagnosis method of bearings in the spindle system for CNC machine tools based on DRSN-Transformer,\" IEEE Access, vol. 12, pp. 74586-74595, 2024, doi: 10.1109/ACCESS.2024.3404968.",
        "Y. Zhang, L. Lin, J. Wang, et al., \"Attention activation network for bearing fault diagnosis under various noise environments,\" Sci. Rep., vol. 15, Art. no. 977, 2025, doi: 10.1038/s41598-025-85275-w.",
        "D. P. Kingma and J. Ba, \"Adam: A method for stochastic optimization,\" in Proc. ICLR, 2015.",
        "I. Loshchilov and F. Hutter, \"Decoupled weight decay regularization,\" in Proc. ICLR, 2019.",
        "Z. Liu, Y. Lin, Y. Cao, et al., \"Swin Transformer: Hierarchical vision transformer using shifted windows,\" in Proc. IEEE ICCV, 2021, pp. 10012-10022.",
        "Case Western Reserve University, \"Bearing Data Center,\" https://engineering.case.edu/bearingdatacenter (accessed Oct. 27, 2026).",
        "Paderborn University, \"Bearing Data Center,\" https://groups.uni-paderborn.de/kat/BearingDataCenter/ (accessed Oct. 27, 2026).",
        "B. Fang, Y. Hu, G. Zheng, X. Zhang, and L. Xie, \"Multiscale time-frequency CNN-transformer model for bearing fault diagnosis: A comprehensive feature extraction approach,\" Eng. Fail. Anal., vol. 184, Art. no. 110312, 2026, doi: 10.1016/j.engfailanal.2025.110312.",
        "B. Wu and G. Yu, \"Time-frequency decomposition and convolution strategy for fault diagnosis of rotating machinery,\" Eng. Fail. Anal., vol. 191, Art. no. 110793, 2026, doi: 10.1016/j.engfailanal.2026.110793.",
    ]
    for index, reference in enumerate(references, 1):
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.7)
        paragraph.paragraph_format.first_line_indent = Cm(-0.7)
        paragraph.paragraph_format.space_after = Pt(2)
        run = paragraph.add_run(f"[{index}] {reference}")
        set_run_font(run, size=8.6)

    document.core_properties.title = title_run.text
    document.core_properties.subject = (
        "Leakage-safe public benchmark, component ablation, and calibrated self-training"
    )
    document.core_properties.author = "Anonymous"
    document.core_properties.keywords = (
        "bearing fault diagnosis; spindle; CWT; self-training; calibration"
    )
    document.save(args.output)
    print(f"Saved manuscript to {args.output}")


if __name__ == "__main__":
    main()
