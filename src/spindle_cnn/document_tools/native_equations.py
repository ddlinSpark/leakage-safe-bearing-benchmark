"""Create native Word OMML equations through Pandoc."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

from lxml import etree


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _pandoc_executable() -> str:
    explicit = os.environ.get("PANDOC_EXE")
    if explicit:
        return explicit
    executable = shutil.which("pandoc")
    if not executable:
        raise FileNotFoundError(
            "Pandoc is required to author native Word equations. "
            "Set PANDOC_EXE or install Pandoc."
        )
    return executable


def render_native_equations(equations: list[str]) -> list[etree._Element]:
    markdown_parts: list[str] = []
    for index, expression in enumerate(equations, 1):
        markdown_parts.extend([f"Equation {index}", "", f"$${expression}$$", ""])

    with tempfile.TemporaryDirectory(prefix="spindle_equations_") as temp_dir:
        temp_path = Path(temp_dir)
        markdown_path = temp_path / "equations.md"
        docx_path = temp_path / "equations.docx"
        markdown_path.write_text("\n".join(markdown_parts), encoding="utf-8")
        subprocess.run(
            [
                _pandoc_executable(),
                "-f",
                "markdown",
                "-t",
                "docx",
                str(markdown_path),
                "-o",
                str(docx_path),
            ],
            check=True,
            capture_output=True,
        )
        with ZipFile(docx_path) as archive:
            document_xml = archive.read("word/document.xml")

    root = etree.fromstring(document_xml)
    math_elements = root.xpath("//m:oMath", namespaces={"m": MATH_NS})
    if len(math_elements) != len(equations):
        raise RuntimeError(
            f"Expected {len(equations)} native equations, got {len(math_elements)}"
        )
    return [deepcopy(element) for element in math_elements]
