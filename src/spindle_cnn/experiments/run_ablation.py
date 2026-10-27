"""Ablation runner built on the unified benchmark executor."""

from __future__ import annotations

import argparse
import sys

from .run_benchmark import main as benchmark_main


def main() -> None:
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", "main"])
    if "--methods" not in sys.argv:
        sys.argv.extend(
            [
                "--methods",
                ",".join(
                    [
                        "cwt_cnn",
                        "gru_only",
                        "cwt_gru_no_se",
                        "cwt_gru",
                        "cwt_gru_temp",
                        "cwt_gru_selftrain",
                    ]
                ),
            ]
        )
    benchmark_main()


if __name__ == "__main__":
    main()
