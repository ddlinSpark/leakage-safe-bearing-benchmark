"""Build one machine-readable registry for canonical and sensitivity runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def discover_result_tables(root: Path) -> list[Path]:
    candidates = [
        root / "results" / "raw_runs.csv",
        root / "results" / "sota" / "raw_runs.csv",
        root / "results" / "self_training_sensitivity" / "raw_runs.csv",
        root / "results" / "normalization_sensitivity" / "raw_runs.csv",
        root / "results" / "baseline_tuning" / "raw_runs.csv",
        root / "results" / "calibration_recheck" / "raw_runs.csv",
        root / "results" / "training_budget_30" / "raw_runs.csv",
    ]
    candidates.extend(
        sorted((root / "results" / "cwt_sensitivity").glob("*/raw_runs.csv"))
    )
    return [path for path in candidates if path.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "experiment_registry.csv",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    frames = []
    for path in discover_result_tables(root):
        frame = pd.read_csv(path)
        relative = path.relative_to(root).as_posix()
        frame["result_table"] = relative
        frame["result_family"] = path.parent.relative_to(root).as_posix()
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No raw_runs.csv tables were found")
    raw = pd.concat(frames, ignore_index=True)
    registry = pd.DataFrame(
        {
            "experiment_id": raw["run_id"],
            "result_family": raw["result_family"],
            "result_table": raw["result_table"],
            "experiment_type": raw["experiment_type"],
            "method": raw["method"],
            "model_key": raw["model_key"],
            "fold": raw["fold"],
            "seed": raw["seed"],
            "train_budget": raw["label_ratio"],
            "threshold": raw["threshold"],
            "configuration": raw["config_path"],
            "configuration_hash": raw["config_hash"],
            "selected_epoch": raw["best_epoch"],
            "validation_metric": raw["best_val_macro_f1"],
            "test_metric": raw["macro_f1"],
            "checkpoint": "",
            "status": raw["status"],
            "code_commit": raw["git_commit"],
            "git_dirty": raw["git_dirty"],
            "split_hash": raw["split_hash"],
            "dataset_hash": raw["dataset_hash"],
            "config_path": raw["config_path"],
        }
    )
    duplicate_ids = registry["experiment_id"].duplicated(keep=False)
    if duplicate_ids.any():
        raise ValueError(
            "Experiment IDs collide across result families: "
            f"{registry.loc[duplicate_ids, 'experiment_id'].head().tolist()}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(args.output, index=False, lineterminator="\n")
    print(
        f"Wrote {len(registry)} runs from {len(frames)} tables to {args.output}"
    )


if __name__ == "__main__":
    main()
