"""Formatting helpers for the SCI manuscript."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from docx.document import Document as DocumentType
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


FONT_NAME = "Times New Roman"
BODY_PT = 10.5
TABLE_FONT_PT = 8.2
BLACK = RGBColor(0, 0, 0)
HEADER_FILL = "DCE6F1"
ALTERNATE_FILL = "F4F7FA"
BORDER_DARK = "404040"


def set_run_font(
    run,
    *,
    size: float = BODY_PT,
    bold: bool | None = None,
    italic: bool | None = None,
) -> None:
    run.font.name = FONT_NAME
    run.font.size = Pt(size)
    run.font.color.rgb = BLACK
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT_NAME)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT_NAME)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT_NAME)


def configure_document(document: DocumentType) -> None:
    section = document.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(2.1)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.header_distance = Cm(1.0)
    section.footer_distance = Cm(0.9)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = FONT_NAME
    normal.font.size = Pt(BODY_PT)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.12
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)

    for name, size, before, after in (
        ("Heading 1", 14, 14, 6),
        ("Heading 2", 12, 11, 5),
        ("Heading 3", 10.5, 9, 4),
    ):
        style = styles[name]
        style.font.name = FONT_NAME
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)

    title = styles["Title"]
    title.font.name = FONT_NAME
    title.font.size = Pt(18)
    title.font.bold = True
    title.font.color.rgb = BLACK
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(10)
    title._element.get_or_add_pPr().remove_all("w:pBdr")

    add_page_number_footer(document)


def add_page_number_footer(document: DocumentType) -> None:
    footer = document.sections[0].footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    set_run_font(run, size=8.5)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, text, end])


def add_title(document: DocumentType, title_text: str) -> None:
    paragraph = document.add_paragraph(style="Title")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(title_text)
    set_run_font(run, size=18, bold=True)
    paragraph.paragraph_format.keep_with_next = True


def add_body(
    document: DocumentType,
    text: str,
    *,
    first: bool = False,
    italic: bool = False,
    bold_lead: str | None = None,
) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Cm(0.55) if first else Cm(0)
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        set_run_font(lead, bold=True)
        body = paragraph.add_run(text[len(bold_lead) :])
        set_run_font(body, italic=italic)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, italic=italic)


def add_small_text(document: DocumentType, text: str, *, italic: bool = False) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    set_run_font(run, size=9, italic=italic)
    paragraph.paragraph_format.space_after = Pt(8)


def add_caption(document: DocumentType, text: str, *, above: bool) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6 if above else 3)
    paragraph.paragraph_format.space_after = Pt(4 if above else 8)
    paragraph.paragraph_format.keep_with_next = above
    run = paragraph.add_run(text)
    set_run_font(run, size=9, italic=False)


def add_figure(
    document: DocumentType,
    path: Path,
    caption: str,
    *,
    width_inches: float,
    alt_text: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run()
    inline = run.add_picture(str(path), width=Inches(width_inches))
    inline._inline.docPr.set("descr", alt_text)
    add_caption(document, caption, above=False)


def _set_cell_margins(cell, top: int = 80, start: int = 90, bottom: int = 80, end: int = 90) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin_name, value in (
        ("top", top),
        ("start", start),
        ("bottom", bottom),
        ("end", end),
    ):
        node = tc_mar.find(qn(f"w:{margin_name}"))
        if node is None:
            node = OxmlElement(f"w:{margin_name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def _set_cell_border(cell, **edges: dict[str, str | int]) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    border = tc_pr.first_child_found_in("w:tcBorders")
    if border is None:
        border = OxmlElement("w:tcBorders")
        tc_pr.append(border)
    for edge_name, attributes in edges.items():
        tag = f"w:{edge_name}"
        element = border.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            border.append(element)
        for key, value in attributes.items():
            element.set(qn(f"w:{key}"), str(value))


def _shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.first_child_found_in("w:shd")
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)
    shading.set(qn("w:val"), "clear")


def _repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    tr_pr.append(repeat)


def _keep_row_together(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def _set_no_wrap(cell) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    no_wrap = OxmlElement("w:noWrap")
    tc_pr.append(no_wrap)


def add_three_line_table(
    document: DocumentType,
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    widths_cm: Sequence[float],
    caption: str,
) -> None:
    add_caption(document, caption, above=True)
    rows = list(rows)
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.allow_autofit = False
    header = table.rows[0]
    _repeat_header(header)
    _keep_row_together(header)
    for index, (cell, text) in enumerate(zip(header.cells, headers, strict=True)):
        cell.width = Cm(widths_cm[index])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        _set_cell_margins(cell)
        _shade_cell(cell, HEADER_FILL)
        _set_cell_border(
            cell,
            top={"val": "single", "sz": "12", "color": BORDER_DARK},
            bottom={"val": "single", "sz": "6", "color": BORDER_DARK},
            left={"val": "nil"},
            right={"val": "nil"},
        )
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.keep_with_next = True
        run = paragraph.add_run(str(text))
        set_run_font(run, size=TABLE_FONT_PT, bold=True)

    for row_index, values in enumerate(rows):
        row = table.add_row()
        _keep_row_together(row)
        for column_index, (cell, value) in enumerate(
            zip(row.cells, values, strict=True)
        ):
            cell.width = Cm(widths_cm[column_index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            _set_cell_margins(cell)
            if row_index % 2 == 1:
                _shade_cell(cell, ALTERNATE_FILL)
            bottom = (
                {"val": "single", "sz": "12", "color": BORDER_DARK}
                if row_index == len(rows) - 1
                else {"val": "nil"}
            )
            _set_cell_border(
                cell,
                top={"val": "nil"},
                bottom=bottom,
                left={"val": "nil"},
                right={"val": "nil"},
            )
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.keep_together = True
            paragraph.paragraph_format.keep_with_next = row_index < len(rows) - 1
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if column_index == 0
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            run = paragraph.add_run(str(value))
            set_run_font(run, size=TABLE_FONT_PT)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def add_numbered_items(document: DocumentType, items: Sequence[str]) -> None:
    for item in items:
        paragraph = document.add_paragraph(style="List Number")
        paragraph.paragraph_format.left_indent = Cm(0.65)
        paragraph.paragraph_format.first_line_indent = Cm(-0.45)
        paragraph.paragraph_format.space_after = Pt(4)
        run = paragraph.add_run(item)
        set_run_font(run)


def add_bullet_items(document: DocumentType, items: Sequence[str]) -> None:
    for item in items:
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.left_indent = Cm(0.65)
        paragraph.paragraph_format.first_line_indent = Cm(-0.4)
        paragraph.paragraph_format.space_after = Pt(3)
        run = paragraph.add_run(item)
        set_run_font(run)


def add_reference(document: DocumentType, number: int, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.left_indent = Cm(0.7)
    paragraph.paragraph_format.first_line_indent = Cm(-0.7)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(1.0)
    paragraph.paragraph_format.line_spacing = 0.98
    run = paragraph.add_run(f"[{number}] {text}")
    set_run_font(run, size=8.2)


def add_page_break(document: DocumentType) -> None:
    document.add_page_break()
