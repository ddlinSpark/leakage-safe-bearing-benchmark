"""Download and prepare the canonical Paderborn benchmark cache."""

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
        "--archive-dir",
        type=Path,
        default=ROOT / "data" / "public" / "paderborn" / "rars",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=ROOT / "data" / "public" / "paderborn" / "extracted",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data"
        / "public"
        / "paderborn"
        / "pu_4class_w8192.npz",
    )
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=8192)
    parser.add_argument("--scales", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-measurements", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--seven-zip", type=Path)
    args = parser.parse_args()

    prepare = [
        "--archive-dir",
        str(args.archive_dir),
        "--extract-dir",
        str(args.extract_dir),
        "--workers",
        str(args.workers),
    ]
    if args.skip_download:
        prepare.append("--skip-download")
    if args.seven_zip:
        prepare.extend(["--seven-zip", str(args.seven_zip)])
    run_module("spindle_cnn.public_benchmark.prepare_paderborn", prepare)

    run_module(
        "spindle_cnn.public_benchmark.pu_data",
        [
            "--data-root",
            str(args.extract_dir),
            "--output",
            str(args.output),
            "--window-size",
            str(args.window_size),
            "--stride",
            str(args.stride),
            "--scales",
            str(args.scales),
            "--image-size",
            str(args.image_size),
            "--max-measurements",
            str(args.max_measurements),
            "--workers",
            str(args.workers),
        ],
    )


if __name__ == "__main__":
    main()
