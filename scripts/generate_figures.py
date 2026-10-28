"""Regenerate every publication figure from retained result files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_module(module: str, arguments: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "figures" / "publication",
    )
    parser.add_argument(
        "--include-domain-shift",
        action="store_true",
        help="Rebuild Figure 13 from the original-amplitude cache.",
    )
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=ROOT
        / "data"
        / "public"
        / "paderborn"
        / "pu_4class_w8192_raw.npz",
    )
    args = parser.parse_args()
    results = str(args.results_dir.resolve())
    figures = str(args.figures_dir.resolve())

    commands = [
        (
            "spindle_cnn.experiments.make_method_figures",
            ["--output-dir", figures],
        ),
        (
            "spindle_cnn.experiments.make_figures",
            ["--results-dir", results, "--output-dir", figures],
        ),
        (
            "spindle_cnn.analysis.label_granularity",
            ["--results-dir", results, "--figures-dir", figures],
        ),
        (
            "spindle_cnn.analysis.cwt_sensitivity",
            ["--results-dir", results, "--figures-dir", figures],
        ),
        (
            "spindle_cnn.analysis.self_training",
            ["--results-dir", results, "--figures-dir", figures],
        ),
        (
            "spindle_cnn.analysis.feature_space",
            [
                "--results-dir",
                results,
                "--figures-dir",
                figures,
                "--reuse-embeddings",
            ],
        ),
        (
            "spindle_cnn.analysis.calibration_audit",
            ["--results-dir", results, "--figures-dir", figures],
        ),
        (
            "spindle_cnn.analysis.baseline_tuning",
            ["--results-dir", results, "--figures-dir", figures],
        ),
        (
            "spindle_cnn.analysis.foldwise",
            ["--results-dir", results, "--figures-dir", figures],
        ),
        (
            "spindle_cnn.analysis.training_budget",
            ["--results-dir", results, "--figures-dir", figures],
        ),
        (
            "spindle_cnn.analysis.normalization_sensitivity",
            ["--results-dir", results, "--figures-dir", figures],
        ),
    ]
    if args.include_domain_shift:
        if not args.raw_cache.is_file():
            raise FileNotFoundError(
                f"Original-amplitude cache is required: {args.raw_cache}"
            )
        commands.append(
            (
                "spindle_cnn.analysis.domain_shift",
                [
                    "--raw-cache",
                    str(args.raw_cache.resolve()),
                    "--results-dir",
                    results,
                    "--figures-dir",
                    figures,
                ],
            )
        )
    for module, arguments in commands:
        print(f"[figures] {module}")
        run_module(module, arguments)

    run_module(
        "scripts.hash_publication_figures",
        ["--figures-dir", figures],
    )
    print(f"Figure regeneration complete: {args.figures_dir}")


if __name__ == "__main__":
    main()
