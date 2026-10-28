"""Tests for the formal paper-repository layout and result catalogs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_readme_is_concise_and_contains_required_sections() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 250
    for heading in (
        "## Key Findings",
        "## Generalization Gap",
        "## Experiment Status",
        "## Leakage-Safe Protocol",
        "## Quick Start",
        "## Data Availability",
        "## Reproducibility",
        "## Scope and Limitations",
    ):
        assert heading in text
    assert "not all evaluated methods produced positive" in text


def test_formal_documentation_and_entry_points_exist() -> None:
    for relative in (
        "docs/DATASET.md",
        "docs/PROTOCOL.md",
        "docs/EXPERIMENTS.md",
        "docs/RESULTS.md",
        "docs/REPRODUCIBILITY.md",
        "scripts/prepare_data.py",
        "scripts/run_main_benchmark.py",
        "scripts/generate_statistics.py",
        "scripts/generate_figures.py",
        "scripts/reproduce_all.py",
        "scripts/smoke_test.py",
        "src/spindle_cnn/visualization/style.py",
    ):
        assert (ROOT / relative).is_file(), relative


def test_canonical_and_split_manifests_match_files() -> None:
    canonical = json.loads(
        (ROOT / "results" / "canonical" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for relative, record in canonical["files"].items():
        path = ROOT / relative
        assert path.stat().st_size == record["bytes"], relative
        assert sha256(path) == record["sha256"], relative

    splits = json.loads(
        (ROOT / "splits" / "manifest.json").read_text(encoding="utf-8")
    )
    assert splits["authoritative_directory"] == "results/splits"
    for relative, record in splits["files"].items():
        path = ROOT / relative
        assert path.stat().st_size == record["bytes"], relative
        assert sha256(path) == record["sha256"], relative
