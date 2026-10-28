"""Run CWT-representation and normalization sensitivity experiments."""

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
        "--kind",
        choices=["cwt", "normalization", "all"],
        default="all",
    )
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    common = ["--workers", str(args.workers)]
    if args.device:
        common.extend(["--device", args.device])
    if args.allow_dirty:
        common.append("--allow-dirty")
    if args.no_resume:
        common.append("--no-resume")
    if args.prepare_only:
        common.append("--prepare-only")
    if args.kind in {"cwt", "all"}:
        run_module("spindle_cnn.experiments.run_cwt_sensitivity", common)
    if args.kind in {"normalization", "all"}:
        run_module(
            "spindle_cnn.experiments.run_normalization_sensitivity",
            common,
        )


if __name__ == "__main__":
    main()
