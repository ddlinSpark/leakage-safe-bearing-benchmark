"""Reproduce the formal release from retained results or a full training run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(arguments: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Train the canonical, label, threshold, and external matrices first.",
    )
    parser.add_argument(
        "--include-domain-shift",
        action="store_true",
        help="Require the original-amplitude cache and rebuild Figure 13.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest and the lightweight smoke test.",
    )
    parser.add_argument(
        "--pdf-validation",
        type=Path,
        help="Optional ignored PDF path for internal-link destination testing.",
    )
    args = parser.parse_args()

    if args.run_benchmark:
        run(
            [
                "scripts/run_main_benchmark.py",
                "--experiment",
                "all",
                "--no-resume",
            ]
        )
        run(["scripts/run_external_baselines.py", "--no-resume"])

    if not args.skip_tests:
        run(["-m", "pytest", "-q"])
        run(["scripts/smoke_test.py"])

    run(["scripts/generate_statistics.py"])
    run(["scripts/build_paper_metrics.py"])
    figure_command = ["scripts/generate_figures.py"]
    if args.include_domain_shift:
        figure_command.append("--include-domain-shift")
    run(figure_command)
    run(["scripts/build_results_catalog.py"])
    run(
        [
            "-m",
            "spindle_cnn.paper.build_docx",
            "--results-dir",
            "results",
            "--figures-dir",
            "figures/publication",
            "--output",
            "docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx",
        ]
    )
    run(["scripts/verify_manuscript_numbers.py"])
    run(["scripts/check_manuscript_consistency.py"])
    run(["scripts/check_figure_palette.py"])
    run(["scripts/check_readme_links.py"])
    if args.pdf_validation:
        run(
            [
                "scripts/audit_internal_links.py",
                "--pdf",
                str(args.pdf_validation),
            ]
        )
    print("Reproduction workflow complete.")


if __name__ == "__main__":
    main()
