"""Independent test-metric recomputation from saved per-run logits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.metrics import metrics_from_logits
from ..experiments.common import atomic_write_json, read_results
from ..paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute test metrics from a saved run's predictions"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Recompute all rows in raw_runs.csv.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-8)
    return parser.parse_args()


def recheck_one(
    results_dir: Path,
    row: pd.Series,
    tolerance: float,
) -> dict[str, object]:
    run_id = str(row["run_id"])
    prediction_path = results_dir / "runs" / run_id / "predictions.npz"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    with np.load(prediction_path) as prediction:
        logits = prediction["logits"]
        labels = prediction["labels"]
        predictions = prediction["predictions"]
        temperature = float(prediction["temperature"])

    raw_metrics = metrics_from_logits(logits, labels, temperature=1.0)
    calibrated_metrics = metrics_from_logits(
        logits,
        labels,
        temperature=temperature,
    )
    observed = {
        "macro_f1": float(calibrated_metrics["macro_f1"]),
        "accuracy": float(calibrated_metrics["accuracy"]),
        "ece": float(calibrated_metrics["ece"]),
        "nll": float(calibrated_metrics["nll"]),
        "brier": float(calibrated_metrics["brier"]),
        "raw_macro_f1": float(raw_metrics["macro_f1"]),
        "raw_accuracy": float(raw_metrics["accuracy"]),
        "raw_ece": float(raw_metrics["ece"]),
        "raw_nll": float(raw_metrics["nll"]),
        "raw_brier": float(raw_metrics["brier"]),
    }
    comparisons = {}
    for key, value in observed.items():
        recorded = float(row[key])
        difference = abs(value - recorded)
        comparisons[key] = {
            "recorded": recorded,
            "recomputed": value,
            "absolute_difference": difference,
            "passed": bool(difference <= tolerance),
        }
    if not all(item["passed"] for item in comparisons.values()):
        failed = [key for key, item in comparisons.items() if not item["passed"]]
        raise AssertionError(f"Metric mismatch for {run_id}: {failed}")
    if not np.array_equal(
        predictions,
        np.argmax(logits, axis=1),
    ):
        raise AssertionError(f"Saved decision mismatch for {run_id}")
    return {
        "run_id": run_id,
        "status": "passed",
        "temperature": temperature,
        "metrics": comparisons,
    }


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    rows = read_results(results_dir / "raw_runs.csv")
    if args.all:
        selected = rows
    elif args.run_id:
        selected = rows.loc[rows["run_id"].astype(str).eq(args.run_id)]
    else:
        raise ValueError("Specify --run-id or --all")
    if selected.empty:
        raise ValueError("No matching run was found")

    reports = [
        recheck_one(results_dir, row, args.tolerance)
        for _, row in selected.iterrows()
    ]
    output = results_dir / "rechecks" / (
        "all_runs.json" if args.all else f"{args.run_id}.json"
    )
    atomic_write_json(
        output,
        {
            "passed": len(reports),
            "runs": reports,
        },
    )
    print(f"Rechecked {len(reports)} runs; all metrics match within {args.tolerance}")


if __name__ == "__main__":
    main()
