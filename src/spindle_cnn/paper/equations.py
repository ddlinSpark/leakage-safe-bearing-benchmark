"""Convert small LaTeX expressions to native Word OMML equations."""

from __future__ import annotations

import copy
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

from docx.document import Document as DocumentType
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from lxml import etree


MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _pandoc_executable() -> Path:
    configured = os.environ.get("PANDOC_EXE")
    candidates = [
        Path(configured) if configured else None,
        Path(r"D:\App\anaconda3\Scripts\pandoc.exe"),
        Path("pandoc"),
    ]
    for candidate in candidates:
        if candidate is not None and (
            candidate.is_file() or candidate.name.lower() == "pandoc"
        ):
            return candidate
    raise FileNotFoundError("Pandoc is required to create native Word equations")


def latex_to_omml(latex: str) -> etree._Element:
    with tempfile.TemporaryDirectory(prefix="spindle_equation_") as temporary:
        root = Path(temporary)
        markdown = root / "equation.md"
        output = root / "equation.docx"
        markdown.write_text(f"$$\n{latex}\n$$\n", encoding="utf-8")
        completed = subprocess.run(
            [
                str(_pandoc_executable()),
                str(markdown),
                "-f",
                "markdown",
                "-t",
                "docx",
                "-o",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Pandoc equation conversion failed: {completed.stderr}")
        with zipfile.ZipFile(output) as archive:
            document_xml = archive.read("word/document.xml")
        xml = etree.fromstring(document_xml)
        equations = xml.xpath("//m:oMath", namespaces={"m": MATH_NS})
        if not equations:
            raise RuntimeError(f"No OMML equation produced for: {latex}")
        return copy.deepcopy(equations[0])


def add_equation(
    document: DocumentType,
    latex: str,
    number: int,
) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_together = True
    paragraph._p.append(latex_to_omml(latex))
    number_run = paragraph.add_run(f"    ({number})")
    number_run.font.name = "Times New Roman"
    number_run.font.size = Pt(10)
