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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成修订版 SCI 论文 DOCX")
    parser.add_argument(
        "--results",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn" / "results.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn" / "analysis" / "summary_with_ci.csv",
    )
    parser.add_argument(
        "--paired",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn" / "analysis" / "paired_comparisons.csv",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn" / "figures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCS_DIR / "paper" / "Leakage_Safe_Spindle_Bearing_Diagnosis_SCI_Revision.docx",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=OUTPUTS_DIR / "paper_assets",
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
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    header_cells = table.rows[0].cells
    for index, (cell, text) in enumerate(zip(header_cells, headers)):
        cell.width = Cm(widths_cm[index])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell)
        shade_cell(cell, "17365D")
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        set_run_font(run, size=8.5)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    tr_pr.append(repeat)

    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for index, (cell, value) in enumerate(zip(cells, values)):
            cell.width = Cm(widths_cm[index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if row_index % 2 == 1:
                shade_cell(cell, "EAF0F8")
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if index != len(cells) - 1 else WD_ALIGN_PARAGRAPH.LEFT
            )
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(str(value))
            set_run_font(run, size=8.3)
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def add_equation(
    document: Document,
    latex: str,
    number: int,
    asset_dir: Path,
    width_inches: float,
) -> None:
    image_path = asset_dir / f"equation_{number}.png"
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Missing equation image {image_path}. Run make_equations.py first."
        )
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run()
    shape = run.add_picture(str(image_path), width=Inches(width_inches))
    alternative_text = f"Equation {number}: {latex}"
    shape._inline.docPr.set("descr", alternative_text)
    shape._inline.docPr.set("title", f"Equation {number}")
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
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.summary)
    detailed = read_csv(args.results)
    paired = read_csv(args.paired)

    raw_full = row_lookup(rows, "raw_cnn", 1.0)
    cwt_full = row_lookup(rows, "cwt_cnn", 1.0)
    gru_full = row_lookup(rows, "cwt_gru", 1.0)
    st_full = row_lookup(rows, "cwt_gru_selftrain", 1.0)
    st_low = row_lookup(rows, "cwt_gru_selftrain", 0.1)
    sup_low = row_lookup(rows, "cwt_gru", 0.1)
    delta_low = float(st_low["macro_f1_mean"]) - float(sup_low["macro_f1_mean"])
    delta_full = float(st_full["macro_f1_mean"]) - float(gru_full["macro_f1_mean"])
    raw_ece = sum(float(item["raw_ece"]) for item in detailed) / len(detailed)
    cal_ece = sum(float(item["cal_ece"]) for item in detailed) / len(detailed)
    run_count = len(detailed)

    protocol_figure = args.figures_dir / "figure_protocol.png"
    if not protocol_figure.is_file():
        raise FileNotFoundError(
            f"Missing protocol figure: {protocol_figure}. Run make_protocol_figure.py first."
        )

    document = Document()
    configure_document(document)
    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run(
        "Leakage-Safe Cross-Condition Evaluation of Calibrated "
        "Self-Training for Bearing Fault Diagnosis in Machine-Tool Spindles"
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
        "Bearing faults affect spindle accuracy and machine availability, but many "
        "learning studies evaluate on overlapping windows from the same operating "
        "recordings or on pseudo-labels that are subsequently corrected by the model "
        "under test. This study replaces that circular evaluation with a leakage-safe "
        "cross-condition protocol on the Paderborn University bearing dataset. "
        "Vibration and two phase-current channels are divided into non-overlapping "
        "windows. Training and validation use one set of bearing units, whereas every "
        "test fold combines an unseen bearing unit with an unseen operating condition. "
        "We compare a raw 1D-CNN, a CWT 2D-CNN, a CWT-GRU model, and temperature-scaled "
        f"self-training across {run_count} repeated runs. The raw 1D-CNN obtained a "
        f"mean macro-F1 of {float(raw_full['macro_f1_mean']):.3f}, the CWT-GRU model "
        f"obtained {float(gru_full['macro_f1_mean']):.3f}, and calibrated self-training "
        f"changed the CWT-GRU score by {delta_full:+.3f}. At a 10% labeling budget, "
        f"no pseudo-label passed the predefined confidence threshold, so the "
        f"macro-F1 change was {delta_low:+.3f}. Temperature scaling "
        f"reduced the average expected calibration error from {raw_ece:.3f} to "
        f"{cal_ece:.3f}, while leaving the predicted class unchanged. The results "
        "separate probability calibration from label correction and show that a "
        "time-frequency-temporal model is not automatically preferable when a "
        "waveform baseline is trained under the same conditions."
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
        "machine-tool spindle; bearing fault diagnosis; continuous wavelet transform; "
        "self-training; probability calibration; leakage-safe evaluation"
    )
    set_run_font(body, size=9.8)

    add_heading(document, "1 Introduction", 1)
    add_body(
        document,
        "Bearings in machine-tool spindles operate under changing speed, torque, and "
        "radial load. A diagnosis model must therefore separate fault mechanisms from "
        "operating conditions rather than memorize one laboratory setting. Vibration "
        "provides a direct mechanical response, while motor current provides an "
        "electrical view of the drive system. Combining these signals with a "
        "time-frequency representation is technically plausible, but the benefit has "
        "to be established under a protocol that prevents adjacent windows, repeated "
        "bearings, or test-informed pseudo-labels from entering training.",
        first=True,
    )
    add_body(
        document,
        "The original manuscript reported a large gain after calibration and "
        "self-training on an industrial dataset whose class labels were themselves "
        "created from sensor thresholds. The rewritten study does not treat those "
        "labels as ground truth. Instead, verified public labels provide the primary "
        "accuracy evidence, and the industrial records can serve only as an unlabeled "
        "case study. This distinction changes the central claim: calibration adjusts "
        "confidence and self-training can propagate pseudo-labels, but neither operation "
        "creates missing supervision.",
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
        "A public-data evaluation protocol that removes window overlap, separates bearing "
        "units, and holds out operating conditions.",
        "An identical-protocol comparison of waveform, CWT-image, and CWT-GRU "
        "representations, with raw waveform performance reported even when it is stronger.",
        "A calibrated self-training analysis at controlled labeling budgets, in which "
        "pseudo-labels are selected only from the unlabeled training portion.",
        "An uncertainty-aware report using repeated folds and seeds, confidence intervals, "
        "paired comparisons, and calibration error.",
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
        "but architectural sophistication alone does not establish a causal improvement.",
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
        "currents [1,21]. The revised experiments use its bearing identifiers as groups "
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
        "Table 2. Data and evaluation design of the revised public benchmark.",
    )
    add_figure(
        document,
        protocol_figure,
        "Figure 1. Revised evaluation flow. Test labels and test conditions are isolated "
        "from supervised fitting and pseudo-label generation.",
        6.4,
    )

    add_heading(document, "4 Method", 1)
    add_heading(document, "4.1 Signal normalization and CWT representation", 2)
    add_body(
        document,
        "Each channel is standardized independently within a window. For channel x(t), "
        "the continuous wavelet coefficient at scale a and position b is",
    )
    add_equation(
        document,
        r"W_x(a,b)=\frac{1}{\sqrt{a}}\int_{-\infty}^{\infty}x(t)\psi^{*}\left(\frac{t-b}{a}\right)dt",
        1,
        args.asset_dir,
        4.9,
    )
    add_body(
        document,
        "A Morlet mother wavelet is used because it provides a balanced time-frequency "
        "response. The magnitude map is bilinearly resized to 96 × 96 and normalized by "
        "its maximum:",
    )
    add_equation(
        document,
        r"I_c(a,b)=\frac{|W_{x_c}(a,b)|}{\max_{a,b}|W_{x_c}(a,b)|+\epsilon}",
        2,
        args.asset_dir,
        4.1,
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
        r"p(y=k\mid x)=\frac{\exp(z_k/T)}{\sum_{j=1}^{K}\exp(z_j/T)}",
        3,
        args.asset_dir,
        3.4,
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
        r"\mathcal{S}_k=\mathrm{TopM}\left\{i\notin\mathcal{L}:\hat{y}_i=k,\;p_{i,k}\geq\tau\right\},\quad \tau=0.60",
        4,
        args.asset_dir,
        5.8,
    )
    add_body(
        document,
        "The model is reinitialized after each pseudo-label round. Labeled and selected "
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
        "Table 3. Full-label results under unseen-bearing and unseen-condition testing.",
    )
    main_figure = args.figures_dir / "figure_main_comparison.png"
    if main_figure.exists():
        add_figure(
            document,
            main_figure,
            "Figure 2. Mean macro-F1 of the full-label models. Error bars show one "
            "standard deviation across matched folds and seeds.",
            5.9,
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

    add_heading(document, "5.2 Label efficiency and pseudo-label reliability", 2)
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
        "Table 4. Label-budget results. Pseudo-labels use only the unlabeled training partition.",
    )
    label_figure = args.figures_dir / "figure_label_efficiency.png"
    if label_figure.exists():
        add_figure(
            document,
            label_figure,
            "Figure 3. Macro-F1 at controlled label budgets. The difference between "
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

    add_heading(document, "5.3 Calibration results", 2)
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
        "Table 5. Expected calibration error before and after temperature scaling.",
    )
    calibration_figure = args.figures_dir / "figure_calibration.png"
    if calibration_figure.exists():
        add_figure(
            document,
            calibration_figure,
            "Figure 4. Expected calibration error before and after temperature scaling. "
            "Temperature scaling changes probabilities but not predicted classes.",
            5.6,
        )
    add_body(
        document,
        "Temperature scaling is a one-parameter monotonic transformation. It can reduce "
        "overconfidence and improve probability estimates without changing the argmax. "
        "Accordingly, any accuracy change in the original manuscript should be traced "
        "to changed labels or decision rules, not to calibration alone.",
    )

    confusion_figure = args.figures_dir / "figure_confusion_matrix.png"
    if confusion_figure.exists():
        add_figure(
            document,
            confusion_figure,
            "Figure 5. Row-normalized confusion matrix for one repeated run of the "
            "selected CWT-GRU model.",
            4.4,
        )

    add_heading(document, "6 Discussion", 1)
    add_body(
        document,
        "The revised protocol changes what can be claimed. A high score is now computed "
        "against labels that were not generated by the evaluated rules, and every test "
        "fold includes an unseen bearing and an unseen operating condition. These "
        "constraints are stricter than the original 70/15/15 station-overlapping split "
        "and the 75% overlapping windows used by the prior pipeline.",
        first=True,
    )
    add_body(
        document,
        "The results also expose a practical issue for spindle diagnostics. CWT and "
        "recurrent fusion increase model complexity and computational cost, but their "
        "benefit depends on the benchmark. When the raw waveform already contains "
        "separative fault signatures, image conversion can discard or distort "
        "information. A journal claim should therefore identify the operating regime "
        "in which fusion helps rather than assuming a universal advantage.",
    )
    add_body(
        document,
        "Self-training has a narrower role than the original manuscript suggests. It "
        "is useful only when pseudo-label selection improves the effective training "
        "distribution without reinforcing systematic errors. Balanced top-confidence "
        "selection reduces class collapse but does not guarantee correctness. Reporting "
        "the unlabeled training partition, confidence threshold, and maximum pseudo-label "
        "count is therefore essential for reproducibility.",
    )

    add_heading(document, "7 Threats to Validity", 1)
    add_body(
        document,
        "The public benchmark isolates bearings on a controlled test rig, not a complete "
        "machine-tool spindle assembly. It cannot reproduce installation stiffness, "
        "tool-holder dynamics, coolant effects, or production load cycles. The 16-bearing "
        "subset keeps computation tractable but limits the number of independent units; "
        "the four held-out conditions should not be interpreted as coverage of all spindle "
        "operating states.",
        first=True,
    )
    add_body(
        document,
        "The industrial records used in the original study remain unlabeled. They can "
        "support consistency checks, but they cannot provide confirmatory accuracy. The "
        "manuscript therefore excludes those historical 96% figures from the main evidence. "
        "External studies are discussed qualitatively because their splits, window overlap, "
        "and test-label provenance differ from the present protocol.",
    )

    add_heading(document, "8 Conclusion", 1)
    add_body(
        document,
        "This revision replaces circular pseudo-label testing with a leakage-safe "
        "public benchmark and an explicit comparison against standard waveform and "
        "time-frequency baselines. Under unseen-bearing and unseen-condition testing, "
        "the complex CWT-GRU model did not outperform the waveform baseline by a "
        "statistically distinguishable margin, and calibrated self-training reduced "
        "macro-F1 in the repeated full-label experiment. Temperature scaling improved "
        "probability calibration without changing predicted classes. The study therefore "
        "supports strict evaluation and separate reporting of calibration, pseudo-labeling, "
        "and diagnostic accuracy.",
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
        "Lessmeier C, Kimotho J K, Zimmer D, Sextro W. Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: A benchmark data set for data-driven classification. PHM Society European Conference, 2016. doi:10.36001/phme.2016.v3i1.1577.",
        "Hendriks J, Dumond P, Knox D A. Towards better benchmarking using the CWRU bearing fault dataset. Mechanical Systems and Signal Processing, 2022, 169: 108732. doi:10.1016/j.ymssp.2021.108732.",
        "Smith W A, Randall R B. Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study. Mechanical Systems and Signal Processing, 2015, 64-65: 100-131.",
        "Wang Z, Oates T. Imaging time-series to improve classification and imputation. Proceedings of IJCAI, 2015: 2627-2633.",
        "Torrence C, Compo G P. A practical guide to wavelet analysis. Bulletin of the American Meteorological Society, 1998, 79(1): 61-78.",
        "Hu J, Shen L, Sun G. Squeeze-and-excitation networks. Proceedings of CVPR, 2018: 7132-7141.",
        "Woo S, Park J, Lee J Y, Kweon I S. CBAM: Convolutional block attention module. Proceedings of ECCV, 2018: 3-19.",
        "He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. Proceedings of CVPR, 2016: 770-778.",
        "Cho K, van Merrienboer B, Gulcehre C, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation. Proceedings of EMNLP, 2014: 1724-1734.",
        "Lee D H. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. ICML Workshop on Challenges in Representation Learning, 2013.",
        "Guo C, Pleiss G, Sun Y, Weinberger K Q. On calibration of modern neural networks. Proceedings of ICML, 2017: 1321-1330.",
        "Lin T Y, Goyal P, Girshick R, He K, Dollar P. Focal loss for dense object detection. Proceedings of ICCV, 2017: 2980-2988.",
        "Zhao M, Zhong S, Fu X, Tang B, Pecht M. Deep residual shrinkage networks for fault diagnosis. IEEE Transactions on Industrial Informatics, 2020, 16(7): 4681-4690.",
        "Li X, Wang J, Wang J, Wang J, Chen J, Yu X. Research on CNC machine tool spindle fault diagnosis method based on DRSN-GCE model. Algorithms, 2025, 18(6): 304. doi:10.3390/a18060304.",
        "Li X, Chen J, Wang J, et al. Research on fault diagnosis method of bearings in the spindle system for CNC machine tools based on DRSN-Transformer. IEEE Access, 2024, 12: 74586-74595. doi:10.1109/ACCESS.2024.3404968.",
        "Zhang Y, Lin L, Wang J, et al. Attention activation network for bearing fault diagnosis under various noise environments. Scientific Reports, 2025, 15: 977. doi:10.1038/s41598-025-85275-w.",
        "Kingma D P, Ba J. Adam: A method for stochastic optimization. Proceedings of ICLR, 2015.",
        "Loshchilov I, Hutter F. Decoupled weight decay regularization. Proceedings of ICLR, 2019.",
        "Liu Z, Lin Y, Cao Y, et al. Swin Transformer: Hierarchical vision transformer using shifted windows. Proceedings of ICCV, 2021: 10012-10022.",
        "Case Western Reserve University Bearing Data Center. https://engineering.case.edu/bearingdatacenter (accessed 27 October 2026).",
        "Paderborn University Bearing Data Center. https://groups.uni-paderborn.de/kat/BearingDataCenter/ (accessed 27 October 2026).",
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
        "Leakage-safe public benchmark and calibrated self-training revision"
    )
    document.core_properties.author = "Anonymous"
    document.core_properties.keywords = (
        "bearing fault diagnosis; spindle; CWT; self-training; calibration"
    )
    document.save(args.output)
    print(f"Saved manuscript to {args.output}")


if __name__ == "__main__":
    main()
