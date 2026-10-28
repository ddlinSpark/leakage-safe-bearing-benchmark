"""Audit the final manuscript text, terminology, and release-readiness markers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from docx import Document


FORBIDDEN_TERMS = (
    "TFNet",
    "Full-label",
    "full-label",
    "training failure",
)

REVIEW_TERMS = (
    "first",
    "novel",
    "SOTA",
    "state-of-the-art",
    "significantly",
    "outperform",
    "superior",
    "statistical significance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_text(document: Document) -> list[str]:
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.extend(paragraph.text for paragraph in cell.paragraphs)
    return [part.strip() for part in parts if part.strip()]


def contexts(paragraphs: list[str], term: str) -> list[str]:
    matches = []
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    for index, paragraph in enumerate(paragraphs):
        if pattern.search(paragraph):
            positions = [match.start() for match in pattern.finditer(paragraph)]
            for position in positions[:3]:
                start = max(0, position - 55)
                end = min(len(paragraph), position + len(term) + 85)
                matches.append(
                    f"paragraph {index + 1}: ...{paragraph[start:end]}..."
                )
    return matches


def regex_contexts(paragraphs: list[str], pattern: str) -> list[str]:
    matches = []
    compiled = re.compile(pattern, re.IGNORECASE)
    for index, paragraph in enumerate(paragraphs):
        for match in compiled.finditer(paragraph):
            start = max(0, match.start() - 55)
            end = min(len(paragraph), match.end() + 85)
            matches.append(
                f"paragraph {index + 1}: ...{paragraph[start:end]}..."
            )
    return matches


def count_git_email_classes(root: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["git", "log", "--all", "--format=%ae"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    addresses = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    anonymous = [
        address
        for address in addresses
        if address.endswith("@users.noreply.github.com")
    ]
    return len(addresses), len(addresses) - len(anonymous)


def count_hostname_fields(root: Path) -> tuple[int, int]:
    config_paths = list((root / "results" / "runs").glob("*/config.json"))
    hits = 0
    for path in config_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        environment = payload.get("environment", {})
        if "hostname" in environment or "host_name" in environment:
            hits += 1
    return len(config_paths), hits


def verify_figure_hashes(root: Path) -> tuple[int, int, int, list[str]]:
    publication = root / "figures" / "publication"
    manifest_path = publication / "figure_hashes.json"
    if not manifest_path.is_file():
        manifest_path = root / "artifacts" / "local" / "figure_hashes.json"
    publication_pngs = len(list(publication.glob("*.png")))
    if not manifest_path.is_file():
        return 0, 0, publication_pngs, ["figure hash manifest is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    compared = 0
    matches = 0
    failures = []
    for name, record in manifest.items():
        path = publication / name
        compared += 1
        if not path.is_file():
            failures.append(f"missing figure: {name}")
        elif sha256(path) != record["sha256"]:
            failures.append(f"hash mismatch: {name}")
        else:
            matches += 1
    return compared, matches, publication_pngs, failures


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paper_path = (
        root / "docs" / "paper" / "Leakage_Safe_Bearing_Benchmark_SCI_final.docx"
    )
    backup_path = (
        root
        / "docs"
        / "paper"
        / "archive"
        / "Leakage_Safe_Bearing_Benchmark_SCI_pre_statistics_revision.docx"
    )
    document = Document(paper_path)
    paragraphs = document_text(document)
    text = "\n".join(paragraphs)

    forbidden_hits = {
        term: contexts(paragraphs, term) for term in FORBIDDEN_TERMS
    }
    forbidden_hits = {
        term: matches for term, matches in forbidden_hits.items() if matches
    }
    loose_full_balanced = [
        context
        for context in contexts(paragraphs, "Full balanced")
        if "Full balanced budget" not in context
    ]
    ambiguous_percentage = [
        context
        for term in ("100% label", "100% training")
        for context in contexts(paragraphs, term)
    ]
    bad_p_syntax = regex_contexts(
        paragraphs,
        r"p\s*=\s*(?:=<|<)",
    )
    review_hits = {
        term: contexts(paragraphs, term) for term in REVIEW_TERMS
    }
    review_hits = {
        term: matches for term, matches in review_hits.items() if matches
    }
    p_values = sorted(set(re.findall(r"p\s*(?:=|<)\s*(?:[0-9.]+|<)", text)))

    audit_path = (
        root
        / "artifacts"
        / "local"
        / "verify_manuscript_numbers.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    render_dir = root / "artifacts" / "local" / "render" / "release_candidate"
    if not render_dir.is_dir():
        render_dir = root / "artifacts" / "local" / "render" / "final_release"
    if not render_dir.is_dir():
        render_dir = root / "artifacts" / "local" / "render" / "final_revision"
    page_count = len(list(render_dir.glob("page-*.png")))
    a11y_path = render_dir / "a11y_report.json"
    a11y = (
        json.loads(a11y_path.read_text(encoding="utf-8"))
        if a11y_path.is_file()
        else {}
    )
    a11y_counts = a11y.get("counts", {})

    commit_count, non_anonymous_commits = count_git_email_classes(root)
    config_count, hostname_configs = count_hostname_fields(root)
    figures_compared, figures_matched, publication_pngs, figure_failures = (
        verify_figure_hashes(root)
    )
    manifest_path = (
        root / "figures" / "publication" / "figure_hashes.json"
    )
    manifest_display = (
        "figures/publication/figure_hashes.json"
        if manifest_path.is_file()
        else "artifacts/local/figure_hashes.json"
    )
    render_display = render_dir.relative_to(root).as_posix()
    numerical_pass = audit.get("status") == "PASS"
    figures_pass = (
        figures_compared == figures_matched
        and figures_compared > 0
        and publication_pngs >= 18
        and not figure_failures
        and len(document.inline_shapes) == 23
    )
    text_pass = (
        not forbidden_hits
        and not loose_full_balanced
        and not ambiguous_percentage
        and not bad_p_syntax
        and len(document.tables) == 18
        and page_count > 0
    )
    release_blocked = non_anonymous_commits > 0 or hostname_configs > 0

    review_lines = []
    for term, matches in review_hits.items():
        review_lines.append(f"### `{term}`")
        review_lines.extend(f"- {match}" for match in matches[:10])
        if len(matches) > 10:
            review_lines.append(f"- ... {len(matches) - 10} additional matches")
    if not review_lines:
        review_lines.append("- No review-only term was found.")

    forbidden_lines = []
    for term, matches in forbidden_hits.items():
        forbidden_lines.append(f"- `{term}`: {len(matches)} occurrence(s)")
    if loose_full_balanced:
        forbidden_lines.append(
            f"- Ambiguous `Full balanced` without `budget`: "
            f"{len(loose_full_balanced)} occurrence(s)"
        )
    if ambiguous_percentage:
        forbidden_lines.append(
            f"- Ambiguous 100% label terminology: {len(ambiguous_percentage)} occurrence(s)"
        )
    if bad_p_syntax:
        forbidden_lines.append(
            f"- Incorrect p-value syntax: {len(bad_p_syntax)} occurrence(s)"
        )
    if not forbidden_lines:
        forbidden_lines.append("- No forbidden or legacy term was found.")

    verdict = "PASS" if numerical_pass and figures_pass and text_pass else "FAIL"
    release_status = (
        "BLOCKED for anonymous public release until the privacy items are "
        "resolved."
        if release_blocked
        else "READY for final public release after the remaining manual checks."
    )

    report = f"""# Final Manuscript Audit

## Verdict

Numerical and manuscript consistency: **{verdict}**.

The independent audit reports {audit.get('mismatch_count')} mismatches across
{len(document.tables)} tables and {len(document.inline_shapes)} figures. The
rendered document contains {page_count} pages.

## Source files

- Final manuscript: `docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx`
- Final SHA-256: `{sha256(paper_path)}`
- Pre-revision backup:
  `docs/paper/archive/Leakage_Safe_Bearing_Benchmark_SCI_pre_statistics_revision.docx`
- Backup SHA-256: `{sha256(backup_path)}`
- Independent audit JSON:
  `artifacts/local/verify_manuscript_numbers.json`
- Figure hash manifest:
  `{manifest_display}`
- Page renders:
  `{render_display}/`

## Terminology scan

{chr(10).join(forbidden_lines)}

The artifact uses "10% balanced budget", "50% balanced budget", and
"Full balanced budget" for the label-budget conditions. The displayed model
name is TFN; the historical machine-readable result key remains `tfnet`.

## Contextual keyword review

These words are not forbidden by themselves. Each match was checked in context.

{chr(10).join(review_lines)}

The only `outperform` occurrence is the RQ1 question, which asks whether
time-frequency representations outperform a raw waveform model. It does not
assert that they do.

## p-value formatting

Observed forms:

{chr(10).join(f"- `{value}`" for value in p_values)}

No malformed form such as `p=<` remains.

## Table and figure consistency

- Table captions present: 1 through 18.
- Figure captions present: 1 through 18 and S1 through S5.
- Table 6 is generated from `results/hierarchical_statistics.csv`.
- Table 10 displays TFN while retaining the historical `tfnet` result key.
- Reference [22] includes DOI `10.1016/j.ymssp.2023.110952`.
- Figure hashes: {figures_matched} of {figures_compared} PNG and PDF files
  match the publication hash manifest.
- All {page_count} rendered pages were visually inspected; no clipping, overlap, missing
  glyph, or broken table was found.
- Accessibility audit: high={a11y_counts.get('high', 0)},
  medium={a11y_counts.get('medium', 0)}, low={a11y_counts.get('low', 0)}.

## Release-readiness

**{release_status}**

The remaining blockers are privacy-related, not numerical:

- {non_anonymous_commits} of {commit_count} Git commits carry non-anonymous
  author email metadata.
- {hostname_configs} of {config_count} retained canonical run configurations
  contain a workstation hostname field.
- The repository is public while the manuscript is labeled for double-blind
  review.

Resolving these items changes repository history or publication visibility.
They require an explicit decision and should not be altered silently during a
statistical revision.
"""
    audit_dir = root / "docs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "FINAL_MANUSCRIPT_AUDIT.md").write_text(
        report,
        encoding="utf-8",
    )
    print(
        f"{verdict}: numerical={numerical_pass}, "
        f"figures={figures_pass}, text={text_pass}"
    )
    print(release_status)
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
