"""Move selected generated result families into the tracked results tree."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    "baseline_tuning_lr001": "baseline_tuning",
    "training_budget_30": "training_budget_30",
    "calibration_recheck": "calibration_recheck",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = root / "artifacts" / "local"
    result_root = root / "results"
    for source_name, target_name in FAMILIES.items():
        source = source_root / source_name
        target = result_root / target_name
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, target, dirs_exist_ok=True)
        raw_path = target / "raw_runs.csv"
        if raw_path.is_file():
            frame = pd.read_csv(raw_path)
            old_prefix = (
                Path("artifacts") / "local" / source_name
            ).as_posix()
            new_prefix = (Path("results") / target_name).as_posix()
            frame["config_path"] = frame["config_path"].astype(str).str.replace(
                old_prefix,
                new_prefix,
                regex=False,
            )
            frame.to_csv(raw_path, index=False, lineterminator="\n")
        print(f"Finalized {target.relative_to(root)}")


if __name__ == "__main__":
    main()
