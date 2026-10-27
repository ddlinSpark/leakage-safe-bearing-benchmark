"""Threshold-sensitivity entry point."""

from __future__ import annotations

import sys

from .run_benchmark import main as benchmark_main


def main() -> None:
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", "threshold"])
    benchmark_main()


if __name__ == "__main__":
    main()
