"""Build the single numerical JSON source used by manuscript verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
        payload = payload.replace(b"\r\n", b"\n")
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def summary_records(frame: pd.DataFrame, group_columns: list[str]) -> list[dict[str, Any]]:
    records = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {column: value for column, value in zip(group_columns, keys, strict=True)}
        record.update(
            {
                "runs": int(len(group)),
                "macro_f1_mean": float(group["macro_f1"].mean()),
                "macro_f1_sd": float(group["macro_f1"].std(ddof=1)),
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_sd": float(group["accuracy"].std(ddof=1)),
                "validation_macro_f1_mean": float(group["best_val_macro_f1"].mean()),
                "validation_macro_f1_sd": float(group["best_val_macro_f1"].std(ddof=1)),
            }
        )
        records.append(record)
    return records


def optional_csv(root: Path, path: str) -> list[dict[str, Any]]:
    full_path = root / path
    if not full_path.is_file():
        return []
    frame = pd.read_csv(full_path)
    frame = frame.replace({np.nan: None})
    return frame.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "paper_metrics.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results"
    raw = pd.read_csv(results / "raw_runs.csv")
    external = pd.read_csv(results / "sota" / "raw_runs.csv")
    hierarchical = pd.read_csv(results / "hierarchical_statistics.csv")
    source_paths = [
        results / "raw_runs.csv",
        results / "sota" / "raw_runs.csv",
        results / "hierarchical_statistics.csv",
        results / "domain_gap_summary.csv",
        results / "summary_with_ci.csv",
        results / "sota" / "summary_with_ci.csv",
        results / "calibration_per_run.csv",
        results / "label_granularity_sensitivity.csv",
        results / "training_diagnostics_summary.csv",
        results / "foldwise_paired_differences.csv",
        results / "win_tie_loss.csv",
        results / "cwt_sensitivity_summary.csv",
        results / "cwt_sensitivity_comparisons.csv",
        results / "normalization_sensitivity_summary.csv",
        results / "normalization_sensitivity_comparisons.csv",
        results / "self_training_sensitivity_summary.csv",
        results / "self_training_sensitivity_comparisons.csv",
        results / "pseudo_label_diagnostics.csv",
        results / "external_baseline_tuning.csv",
        results / "external_baseline_tuning_summary.csv",
        results / "training_budget_comparison.csv",
        results / "feature_space_clustering.csv",
        results / "threshold_per_class_summary.csv",
        results / "environment.json",
        root / "data" / "manifests" / "paderborn_selected_bearings.csv",
        root / "data" / "manifests" / "paderborn_archives_sha256.csv",
        root / "data" / "manifests" / "cwt_sensitivity_caches.csv",
    ]
    source_hashes = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in source_paths
        if path.is_file()
    }
    main = raw.loc[raw["experiment_type"].eq("main")]
    external_main = external.loc[external["status"].eq("success")]
    payload = {
        "schema_version": 1,
        "purpose": "Single numerical source for manuscript tables and claim checks",
        "source_sha256": source_hashes,
        "counts": {
            "canonical_rows": len(raw),
            "main_rows": len(main),
            "external_rows": len(external),
            "experiment_types": raw.groupby("experiment_type").size().to_dict(),
            "seeds": sorted(int(value) for value in raw["seed"].unique()),
            "folds": sorted(int(value) for value in raw["fold"].unique()),
        },
        "main_models": summary_records(main, ["method"]),
        "external_models": summary_records(external_main, ["method"]),
        "hierarchical_comparisons": [
            {
                key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
                for key, value in row.items()
            }
            for row in hierarchical.to_dict(orient="records")
        ],
        "domain_gap": optional_csv(root, "results/domain_gap_summary.csv"),
        "calibration_summary": optional_csv(root, "results/calibration_summary.csv"),
        "label_granularity": optional_csv(
            root, "results/label_granularity_sensitivity.csv"
        ),
        "training_diagnostics": optional_csv(
            root, "results/training_diagnostics_summary.csv"
        ),
        "foldwise_differences": optional_csv(
            root, "results/foldwise_paired_differences.csv"
        ),
        "win_tie_loss": optional_csv(root, "results/win_tie_loss.csv"),
        "sensitivity": {
            "cwt": optional_csv(root, "results/cwt_sensitivity_summary.csv"),
            "normalization": optional_csv(
                root, "results/normalization_sensitivity_summary.csv"
            ),
            "self_training": optional_csv(
                root, "results/self_training_sensitivity_summary.csv"
            ),
            "baseline_tuning": optional_csv(
                root, "results/external_baseline_tuning_summary.csv"
            ),
            "training_budget": optional_csv(
                root, "results/training_budget_comparison.csv"
            ),
            "feature_space": optional_csv(
                root, "results/feature_space_clustering.csv"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {args.output} with {len(raw) + len(external)} registry rows")


if __name__ == "__main__":
    main()
