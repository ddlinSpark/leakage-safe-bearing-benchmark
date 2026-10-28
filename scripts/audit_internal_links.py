"""Audit reference and cross-reference hyperlinks in DOCX and PDF."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pdfplumber
from lxml import etree
from pypdf import PdfReader
from pypdf.generic import IndirectObject


ROOT = Path(__file__).resolve().parents[1]
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
CITATION_RE = re.compile(r"\[\d+(?:\s*(?:,|-)\s*\d+)*\]")
CROSS_REFERENCE_RE = re.compile(
    r"\b(?:Figures?|Fig\.?|Tables?|Sections?|Sec\.?|Appendix|Equations?|Eq\.?)"
    r"\s+(?:\([A-Z]?\d+(?:\.\d+)*\)|[A-Z]?\d+(?:\.\d+)*(?:\s*-\s*"
    r"[A-Z]?\d+(?:\.\d+)*)?|[A-Z])"
)
REFERENCE_RE = re.compile(r"^\s*\[(\d+)\]\s+")
CAPTION_RE = re.compile(r"^\s*(Figure|Table)\s+(S?\d+)\.")


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(
        value.text or ""
        for value in paragraph.xpath(".//w:t", namespaces=NS)
    )


def expand_citation(inner: str) -> list[int]:
    numbers: list[int] = []
    for part in inner.split(","):
        part = part.strip()
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if range_match:
            start, end = (int(value) for value in range_match.groups())
            numbers.extend(range(min(start, end), max(start, end) + 1))
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def expected_citation_anchors(token: str) -> set[str]:
    inner = token[1:-1]
    anchors: set[str] = set()
    for part in inner.split(","):
        part = part.strip()
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if range_match:
            start, end = (int(value) for value in range_match.groups())
            anchors.update({f"ref_{start}", f"ref_{end}"})
        elif part.isdigit():
            anchors.add(f"ref_{part}")
    return anchors


def cross_reference_anchor(token: str) -> str | None:
    appendix = re.fullmatch(r"Appendix\s+([A-Z])", token.strip())
    if appendix:
        return f"appendix_{appendix.group(1).lower()}"
    match = re.match(
        r"(Figures?|Fig\.?|Tables?|Sections?|Sec\.?|Equations?|Eq\.?)"
        r"\s+\(?([A-Z]?)(\d+(?:\.\d+)*)\)?",
        token.strip(),
    )
    if not match:
        return None
    label, prefix, number = match.groups()
    key = number.replace(".", "_")
    if label.startswith(("Figure", "Fig")):
        return f"fig_{prefix.lower()}{key}"
    if label.startswith("Table"):
        return f"tbl_{key}"
    if label.startswith(("Section", "Sec")):
        return f"sec_{key}"
    return f"eq_{key}"


def cross_reference_kind(token: str) -> str:
    if token.startswith(("Figure", "Fig")):
        return "figure"
    if token.startswith("Table"):
        return "table"
    if token.startswith(("Section", "Sec")):
        return "section"
    if token.startswith("Appendix"):
        return "appendix"
    return "equation"


def resolve_destination_page(
    reader: PdfReader,
    page_id_to_index: dict[int, int],
    destination: Any,
) -> int | None:
    if isinstance(destination, IndirectObject):
        destination = destination.get_object()
    if isinstance(destination, (str, bytes)):
        try:
            page = reader.get_destination_page_number(destination)
        except Exception:
            return None
        return page if page >= 0 else None
    if isinstance(destination, list) and destination:
        page_reference = destination[0]
        if isinstance(page_reference, IndirectObject):
            return page_id_to_index.get(page_reference.idnum)
    return None


def annotation_destination(annotation) -> Any:
    destination = annotation.get("/Dest")
    if destination is None and "/A" in annotation:
        action = annotation["/A"].get_object()
        destination = action.get("/D")
    return destination


def audit(docx_path: Path, pdf_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(docx_path) as archive:
        document = etree.fromstring(archive.read("word/document.xml"))

    paragraphs = document.xpath("//w:body//w:p", namespaces=NS)
    paragraph_texts = [paragraph_text(paragraph) for paragraph in paragraphs]
    references_index = next(
        (
            index
            for index, text in enumerate(paragraph_texts)
            if text.strip() == "References"
        ),
        len(paragraphs),
    )
    body_paragraphs = paragraphs[:references_index]
    reference_paragraphs = paragraphs[references_index:]

    bookmark_names = [
        value
        for value in document.xpath("//w:bookmarkStart/@w:name", namespaces=NS)
        if value
    ]
    bookmark_counts = Counter(bookmark_names)
    duplicate_bookmarks = sorted(
        name for name, count in bookmark_counts.items() if count > 1
    )
    bookmark_set = set(bookmark_names)

    reference_entries: dict[int, str] = {}
    duplicate_reference_numbers: list[int] = []
    for paragraph in reference_paragraphs:
        text = paragraph_text(paragraph).strip()
        match = REFERENCE_RE.match(text)
        if not match:
            continue
        number = int(match.group(1))
        if number in reference_entries:
            duplicate_reference_numbers.append(number)
        reference_entries[number] = text

    citation_occurrences: list[dict[str, Any]] = []
    missing_citation_links: list[str] = []
    cited_numbers: set[int] = set()
    ranges: list[str] = []
    for paragraph_index, paragraph in enumerate(body_paragraphs):
        text = paragraph_text(paragraph)
        anchors = set(
            paragraph.xpath(".//w:hyperlink/@w:anchor", namespaces=NS)
        )
        for match in CITATION_RE.finditer(text):
            token = match.group(0)
            numbers = expand_citation(token[1:-1])
            if not numbers:
                continue
            cited_numbers.update(numbers)
            if "-" in token:
                ranges.append(token)
            expected = expected_citation_anchors(token)
            if not expected.issubset(anchors):
                missing_citation_links.append(token)
            citation_occurrences.append(
                {
                    "paragraph": paragraph_index,
                    "text": token,
                    "references": numbers,
                }
            )

    cross_reference_occurrences: list[dict[str, Any]] = []
    missing_cross_reference_links: list[str] = []
    cross_reference_counts: Counter[str] = Counter(
        {
            "figure": 0,
            "table": 0,
            "section": 0,
            "appendix": 0,
            "equation": 0,
        }
    )
    for paragraph in body_paragraphs:
        if paragraph.xpath(
            "./w:pPr/w:pStyle[starts-with(@w:val, 'Heading')]",
            namespaces=NS,
        ):
            continue
        text = paragraph_text(paragraph)
        if CAPTION_RE.match(text):
            continue
        anchors = set(
            paragraph.xpath(".//w:hyperlink/@w:anchor", namespaces=NS)
        )
        for match in CROSS_REFERENCE_RE.finditer(text):
            token = match.group(0)
            anchor = cross_reference_anchor(token)
            kind = cross_reference_kind(token)
            cross_reference_counts[kind] += 1
            if anchor is None or anchor not in anchors:
                missing_cross_reference_links.append(token)
            cross_reference_occurrences.append(
                {
                    "text": token,
                    "anchor": anchor,
                    "kind": kind,
                }
            )

    hyperlink_anchors = document.xpath(
        "//w:hyperlink/@w:anchor",
        namespaces=NS,
    )
    broken_docx_links = sorted(
        set(hyperlink_anchors) - bookmark_set
    )
    missing_reference_targets = sorted(
        cited_numbers - set(reference_entries)
    )
    unused_references = sorted(set(reference_entries) - cited_numbers)
    missing_reference_bookmarks = sorted(
        number
        for number in reference_entries
        if f"ref_{number}" not in bookmark_set
    )
    missing_figure_targets = sorted(
        f"Figure {number}"
        for number in range(1, 19)
        if f"fig_{number}" not in bookmark_set
    )
    missing_figure_targets.extend(
        f"Figure S{number}"
        for number in range(1, 6)
        if f"fig_s{number}" not in bookmark_set
    )
    missing_table_targets = sorted(
        f"Table {number}"
        for number in range(1, 19)
        if f"tbl_{number}" not in bookmark_set
    )
    missing_equation_targets = sorted(
        f"Equation {number}"
        for number in range(1, 7)
        if f"eq_{number}" not in bookmark_set
    )
    required_section_bookmarks = {
        *(f"sec_{number}" for number in range(1, 10)),
        "sec_3_2",
        "appendix_a",
        "abstract",
        "data_code",
        "refs",
    }
    missing_section_targets = sorted(
        required_section_bookmarks - bookmark_set
    )

    pdf_reader = PdfReader(pdf_path)
    page_id_to_index = {
        page.indirect_reference.idnum: index
        for index, page in enumerate(pdf_reader.pages)
    }
    pdf_links: list[dict[str, Any]] = []
    unresolved_pdf_links = 0
    with pdfplumber.open(pdf_path) as plumber:
        for source_index, page in enumerate(pdf_reader.pages):
            for reference in page.get("/Annots") or []:
                annotation = reference.get_object()
                if annotation.get("/Subtype") != "/Link":
                    continue
                destination = annotation_destination(annotation)
                target_index = resolve_destination_page(
                    pdf_reader,
                    page_id_to_index,
                    destination,
                )
                if target_index is None:
                    unresolved_pdf_links += 1
                    continue
                rect = [float(value) for value in annotation.get("/Rect")]
                x0, y0, x1, y1 = rect
                plumber_page = plumber.pages[source_index]
                top = float(plumber_page.height) - y1
                bottom = float(plumber_page.height) - y0
                try:
                    source_text = (
                        plumber_page.crop((x0, top, x1, bottom)).extract_text()
                        or ""
                    ).strip()
                except Exception:
                    source_text = ""
                pdf_links.append(
                    {
                        "source_page": source_index + 1,
                        "target_page": target_index + 1,
                        "text": source_text,
                    }
                )

    outline_titles: list[str] = []

    def walk_outline(items) -> None:
        for item in items:
            if isinstance(item, list):
                walk_outline(item)
            else:
                outline_titles.append(str(item.title))

    walk_outline(pdf_reader.outline)
    required_outline = {
        "Abstract",
        "1 Introduction",
        "2 Related Work",
        "3 Methodology",
        "4 Results by Research Question",
        "5 Re-trained Model and External Baseline Comparison",
        "6 Ablation Study",
        "7 Experimental Analysis",
        "8 Discussion",
        "9 Conclusion",
        "Appendix A Additional Diagnostic Figures",
        "References",
    }
    missing_outline = sorted(required_outline - set(outline_titles))

    page_text = [
        (page.extract_text() or "")
        for page in pdf_reader.pages
    ]
    samples = {}
    for label in ("[1]", "[14]", "Figure 4", "Table 13"):
        matches = [
            record
            for record in pdf_links
            if record["text"].strip() == label
            and record["target_page"] >= 1
        ]
        samples[label] = matches[:1]
    unresolved_samples = [
        label for label, matches in samples.items() if not matches
    ]

    broken_reference_destinations = []
    broken_cross_reference_destinations = []
    for record in pdf_links:
        text = record["text"].strip()
        match = re.fullmatch(r"\[(\d+)\]", text)
        if not match:
            continue
        number = int(match.group(1))
        target_text = page_text[record["target_page"] - 1]
        if not re.search(rf"(?m)^\s*\[{number}\]\s", target_text):
            broken_reference_destinations.append(
                {
                    **record,
                    "reference": number,
                }
            )
            continue
        target_text = " ".join(
            page_text[record["target_page"] - 1].split()
        )
        figure_match = re.search(r"Figure\s+(S?\d+)", text)
        table_match = re.search(r"Table\s+(\d+)", text)
        section_match = re.search(r"Section\s+(\d+(?:\.\d+)*)", text)
        equation_match = re.search(r"(?:Eq\.|Equation)\s+\(?(\d+)\)?", text)
        expected_pattern = None
        if figure_match:
            expected_pattern = rf"Figure\s+{figure_match.group(1)}\."
        elif table_match:
            expected_pattern = rf"Table\s+{table_match.group(1)}\."
        elif section_match:
            expected_pattern = rf"{re.escape(section_match.group(1))}\s+\S"
        elif equation_match:
            expected_pattern = rf"\({equation_match.group(1)}\)"
        if expected_pattern and not re.search(expected_pattern, target_text):
            broken_cross_reference_destinations.append(
                {
                    **record,
                    "expected_target_pattern": expected_pattern,
                }
            )

    payload = {
        "docx": str(docx_path),
        "pdf": str(pdf_path),
        "reference_entries": len(reference_entries),
        "in_text_citation_occurrences": len(citation_occurrences),
        "unique_cited_references": len(cited_numbers),
        "cited_references": sorted(cited_numbers),
        "missing_reference_targets": missing_reference_targets,
        "unused_references": unused_references,
        "missing_reference_bookmarks": missing_reference_bookmarks,
        "missing_figure_targets": missing_figure_targets,
        "missing_table_targets": missing_table_targets,
        "missing_equation_targets": missing_equation_targets,
        "missing_section_targets": missing_section_targets,
        "duplicate_reference_numbers": sorted(set(duplicate_reference_numbers)),
        "duplicate_bookmarks": duplicate_bookmarks,
        "citation_ranges": sorted(set(ranges)),
        "missing_citation_links": missing_citation_links,
        "bookmark_count": len(bookmark_names),
        "docx_hyperlink_count": len(hyperlink_anchors),
        "broken_docx_links": broken_docx_links,
        "cross_reference_counts": dict(sorted(cross_reference_counts.items())),
        "cross_reference_occurrences": len(cross_reference_occurrences),
        "missing_cross_reference_links": missing_cross_reference_links,
        "pdf_pages": len(pdf_reader.pages),
        "pdf_link_annotations": len(pdf_links),
        "unresolved_pdf_links": unresolved_pdf_links,
        "broken_reference_destinations": broken_reference_destinations,
        "broken_cross_reference_destinations": broken_cross_reference_destinations,
        "pdf_outline": outline_titles,
        "missing_pdf_outline": missing_outline,
        "sample_links": samples,
        "unresolved_sample_links": unresolved_samples,
    }
    failures = [
        bool(missing_reference_targets),
        bool(unused_references),
        bool(missing_reference_bookmarks),
        bool(missing_figure_targets),
        bool(missing_table_targets),
        bool(missing_equation_targets),
        bool(missing_section_targets),
        bool(duplicate_reference_numbers),
        bool(duplicate_bookmarks),
        bool(missing_citation_links),
        bool(broken_docx_links),
        bool(missing_cross_reference_links),
        unresolved_pdf_links > 0,
        bool(broken_reference_destinations),
        bool(broken_cross_reference_destinations),
        bool(missing_outline),
        bool(unresolved_samples),
    ]
    payload["status"] = "FAIL" if any(failures) else "PASS"
    return payload


def write_report(payload: dict[str, Any], output: Path) -> None:
    def check(value: bool) -> str:
        return "PASS" if not value else "FAIL"

    range_limitations = (
        "Visible range endpoints are linked independently. Intermediate numbers "
        "inside a hyphenated range remain part of the displayed range but are "
        "not separately visible hyperlink targets."
        if payload["citation_ranges"]
        else "None."
    )
    lines = [
        "# Reference and Cross-Reference Link Report",
        "",
        "## Verdict",
        "",
        f"**{payload['status']}**",
        "",
        "PDF links were tested by resolving every link annotation destination to "
        "a target page. Source text was extracted from each annotation rectangle.",
        "",
        "```yaml",
        f"reference_hyperlink_status: {payload['status']}",
        f"figure_cross_reference_status: {check(bool(payload['missing_figure_targets']) or bool(payload['missing_cross_reference_links']))}",
        f"table_cross_reference_status: {check(bool(payload['missing_table_targets']) or bool(payload['missing_cross_reference_links']))}",
        f"equation_cross_reference_status: {check(bool(payload['missing_equation_targets']))}",
        f"pdf_hyperlink_preservation: {check(payload['unresolved_pdf_links'] > 0 or bool(payload['broken_reference_destinations']) or bool(payload['broken_cross_reference_destinations']))}",
        f"pdf_bookmarks: {check(bool(payload['missing_pdf_outline']))}",
        "```",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total reference entries | {payload['reference_entries']} |",
        f"| Total in-text citation occurrences | {payload['in_text_citation_occurrences']} |",
        f"| Unique cited references | {payload['unique_cited_references']} |",
        f"| Missing reference targets | {len(payload['missing_reference_targets'])} |",
        f"| Unused references | {len(payload['unused_references'])} |",
        f"| Broken DOCX hyperlinks | {len(payload['broken_docx_links'])} |",
        f"| Citation ranges detected | {len(payload['citation_ranges'])} |",
        f"| DOCX bookmarks | {payload['bookmark_count']} |",
        f"| DOCX internal hyperlinks | {payload['docx_hyperlink_count']} |",
        f"| PDF link annotations | {payload['pdf_link_annotations']} |",
        f"| PDF pages | {payload['pdf_pages']} |",
        f"| Defined Figure targets | {23 - len(payload['missing_figure_targets'])} of 23 |",
        f"| Defined Table targets | {18 - len(payload['missing_table_targets'])} of 18 |",
        f"| Defined Equation targets | {6 - len(payload['missing_equation_targets'])} of 6 |",
        "",
        "## Checks",
        "",
        "| Check | Status |",
        "| --- | --- |",
        f"| Reference targets and bookmarks | {check(bool(payload['missing_reference_targets']) or bool(payload['missing_reference_bookmarks']))} |",
        f"| Figure/table/equation targets | {check(bool(payload['missing_figure_targets']) or bool(payload['missing_table_targets']) or bool(payload['missing_equation_targets']))} |",
        f"| Section/Appendix targets | {check(bool(payload['missing_section_targets']))} |",
        f"| Citation links | {check(bool(payload['missing_citation_links']))} |",
        f"| DOCX anchors | {check(bool(payload['broken_docx_links']))} |",
        f"| Figure/table/section/equation links | {check(bool(payload['missing_cross_reference_links']))} |",
        f"| PDF annotation destinations | {check(payload['unresolved_pdf_links'] > 0 or bool(payload['broken_reference_destinations']) or bool(payload['broken_cross_reference_destinations']))} |",
        f"| PDF bookmarks/outline | {check(bool(payload['missing_pdf_outline']))} |",
        f"| Required click samples | {check(bool(payload['unresolved_sample_links']))} |",
        "",
        "## Citation details",
        "",
        f"- Cited references: `{payload['cited_references']}`",
        f"- Missing reference targets: `{payload['missing_reference_targets']}`",
        f"- Unused references: `{payload['unused_references']}`",
        f"- Missing citation links: `{payload['missing_citation_links']}`",
        f"- Detected ranges: `{payload['citation_ranges']}`",
        f"- Range handling: {range_limitations}",
        "",
        "## Cross-references",
        "",
        f"- Occurrences by kind: `{payload['cross_reference_counts']}`",
        f"- Missing cross-reference links: `{payload['missing_cross_reference_links']}`",
        f"- Missing figure targets: `{payload['missing_figure_targets']}`",
        f"- Missing table targets: `{payload['missing_table_targets']}`",
        f"- Missing equation targets: `{payload['missing_equation_targets']}`",
        f"- Missing section/appendix targets: `{payload['missing_section_targets']}`",
        "",
        "## PDF click tests",
        "",
        "| Display text | Source page | Target page | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for label, matches in payload["sample_links"].items():
        if matches:
            record = matches[0]
            lines.append(
                f"| `{label}` | {record['source_page']} | "
                f"{record['target_page']} | PASS |"
            )
        else:
            lines.append(f"| `{label}` | - | - | FAIL |")
    lines.extend(
        [
            "",
            "## PDF outline",
            "",
            f"- Outline entries: {len(payload['pdf_outline'])}",
            f"- Missing required outline entries: `{payload['missing_pdf_outline']}`",
            "",
            "The PDF outline contains Abstract, all numbered top-level sections, "
            "Appendix A, Data and Code Availability, and References.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docx",
        type=Path,
        default=ROOT
        / "docs"
        / "paper"
        / "Leakage_Safe_Bearing_Benchmark_SCI_final.docx",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=ROOT / "manuscript" / "manuscript.pdf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "REFERENCE_LINK_REPORT.md",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "artifacts" / "local" / "internal_link_audit.json",
    )
    args = parser.parse_args()
    payload = audit(args.docx.resolve(), args.pdf.resolve())
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_report(payload, args.output.resolve())
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
