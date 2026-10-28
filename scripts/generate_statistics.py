"""Regenerate canonical summaries, paired tests, and hierarchical intervals."""

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
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    results = args.results_dir.resolve()
    run_module(
        "spindle_cnn.analysis.statistics",
        [
            "--results",
            str(results / "raw_runs.csv"),
            "--output-dir",
            str(results),
        ],
    )
    run_module(
        "spindle_cnn.analysis.hierarchical",
        [
            "--results-dir",
            str(results),
            "--external-dir",
            str(results / "sota"),
            "--output",
            str(results / "hierarchical_statistics.csv"),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
        ],
    )
    run_module(
        "spindle_cnn.analysis.sota",
        [
            "--results-dir",
            str(results),
            "--external-dir",
            str(results / "sota"),
        ],
    )
    print(f"Statistics regeneration complete: {results}")


if __name__ == "__main__":
    main()
