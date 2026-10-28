"""Assemble the submission manuscript PDF and a standalone supplement."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
TITLE = (
    "Leakage-Safe Cross-Domain Benchmarking for Bearing Fault Diagnosis: "
    "Time-Frequency Fusion, Calibration, and Self-Training"
)
SUPPLEMENTARY_FIGURES = [
    (
        "figure_14_per_fold_reliability.png",
        "Figure S1. Per-fold reliability diagrams for CWT-GRU before and after "
        "validation-fitted temperature scaling.",
    ),
    (
        "figure_15_temperature_distribution.png",
        "Figure S2. Distribution of validation-fitted temperatures for the "
        "principal retained models.",
    ),
    (
        "figure_20_foldwise_paired_differences.png",
        "Figure S3. Fold-level paired Macro-F1 differences for the principal "
        "comparisons.",
    ),
    (
        "figure_21_training_budget.png",
        "Figure S4. Validation and held-out-test effects of extending the "
        "training opportunity from 12 to 30 epochs.",
    ),
    (
        "figure_24_normalization_sensitivity.png",
        "Figure S5. Window-wise and training-statistic normalization "
        "sensitivity.",
    ),
]


def add_page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.setFont("Helvetica", 8)
    text = f"Supplementary material | {document.page}"
    canvas.drawRightString(
        A4[0] - 18 * mm,
        12 * mm,
        text,
    )
    canvas.restoreState()


def build_supplement(figures_dir: Path, output: Path) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SupplementTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.black,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    body_style = ParagraphStyle(
        "SupplementBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.black,
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    caption_style = ParagraphStyle(
        "SupplementCaption",
        parent=body_style,
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        spaceBefore=6,
    )
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="Supplementary Material: Leakage-Safe Bearing Benchmark",
        author="Anonymous",
        subject="Supplementary figures for the leakage-safe bearing benchmark",
    )
    story = [
        Paragraph("Supplementary Material", title_style),
        Paragraph(TITLE, body_style),
        Spacer(1, 4 * mm),
        Paragraph(
            "This supplement contains the five diagnostic figures referenced "
            "from the manuscript. Each figure is generated from retained "
            "result files in the accompanying repository; the supplement does "
            "not introduce a new model-selection experiment.",
            body_style,
        ),
        Spacer(1, 3 * mm),
    ]
    available_width = A4[0] - 36 * mm
    available_height = A4[1] - 58 * mm
    for index, (name, caption) in enumerate(SUPPLEMENTARY_FIGURES):
        path = figures_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        image = Image(str(path))
        scale = min(
            available_width / image.imageWidth,
            (available_height - 18 * mm) / image.imageHeight,
        )
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        story.extend(
            [
                image,
                Paragraph(caption, caption_style),
            ]
        )
        if index < len(SUPPLEMENTARY_FIGURES) - 1:
            story.append(PageBreak())
    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manuscript-pdf",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "local"
            / "render"
            / "final_release"
            / "Leakage_Safe_Bearing_Benchmark_SCI_final.pdf"
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "figures" / "publication",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "manuscript",
    )
    args = parser.parse_args()
    if not args.manuscript_pdf.is_file():
        raise FileNotFoundError(args.manuscript_pdf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manuscript = args.output_dir / "manuscript.pdf"
    shutil.copy2(args.manuscript_pdf, manuscript)
    supplement = args.output_dir / "supplementary.pdf"
    build_supplement(args.figures_dir.resolve(), supplement)
    print(manuscript)
    print(supplement)


if __name__ == "__main__":
    main()
