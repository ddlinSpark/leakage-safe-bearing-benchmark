"""Audit sensitivity result completeness without using test performance."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def audit_frame(
    name: str,
    path: Path,
    *,
    expected_rows: int,
    expected_methods: set[str] | None = None,
) -> dict[str, object]:
    if not path.is_file():
        return {
            "result_family": name,
            "path": path.relative_to(ROOT).as_posix(),
            "rows": 0,
            "status": "REPRODUCTION FAILURE",
            "note": "missing result table",
        }
    frame = pd.read_csv(path)
    duplicate_keys = int(
        frame.duplicated(
            ["experiment_type", "method", "fold", "seed", "label_ratio", "threshold"],
            keep=False,
        ).sum()
    )
    statuses = set(frame["status"].astype(str).unique()) if "status" in frame else set()
    methods = set(frame["method"].astype(str).unique()) if "method" in frame else set()
    passed = (
        len(frame) == expected_rows
        and duplicate_keys == 0
        and statuses <= {"success"}
        and (expected_methods is None or methods == expected_methods)
    )
    return {
        "result_family": name,
        "path": path.relative_to(ROOT).as_posix(),
        "rows": len(frame),
        "expected_rows": expected_rows,
        "methods": ";".join(sorted(methods)),
        "duplicate_experiment_keys": duplicate_keys,
        "statuses": ";".join(sorted(statuses)),
        "git_dirty_rows": (
            int(frame["git_dirty"].astype(str).str.casefold().eq("true").sum())
            if "git_dirty" in frame
            else 0
        ),
        "status": "VERIFIED" if passed else "REPRODUCTION FAILURE",
        "note": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = [
        audit_frame(
            "coverage_self_training",
            root / "results" / "self_training_sensitivity" / "raw_runs.csv",
            expected_rows=40,
            expected_methods={"cwt_gru_selftrain_coverage"},
        ),
        audit_frame(
            "normalization",
            root / "results" / "normalization_sensitivity" / "raw_runs.csv",
            expected_rows=60,
            expected_methods={"raw_cnn", "cwt_cnn", "cwt_gru"},
        ),
        audit_frame(
            "baseline_tuning",
            root / "results" / "baseline_tuning" / "raw_runs.csv",
            expected_rows=80,
            expected_methods={"raw_cnn", "cwt_cnn", "drsn", "tfnet"},
        ),
        audit_frame(
            "training_budget",
            root / "results" / "training_budget_30" / "raw_runs.csv",
            expected_rows=80,
            expected_methods={"raw_cnn", "cwt_cnn", "cwt_gru_no_se", "cwt_gru"},
        ),
        audit_frame(
            "calibration_recheck",
            root / "results" / "calibration_recheck" / "raw_runs.csv",
            expected_rows=80,
            expected_methods={
                "raw_cnn",
                "cwt_cnn",
                "cwt_gru",
                "cwt_gru_selftrain",
            },
        ),
    ]
    variants = [
        "no_max_norm",
        "log_magnitude",
        "resolution_128",
        "scales_32",
        "scales_64",
        "real_imag",
    ]
    for variant in variants:
        rows.append(
            audit_frame(
                f"cwt_{variant}",
                root / "results" / "cwt_sensitivity" / variant / "raw_runs.csv",
                expected_rows=20,
                expected_methods={"cwt_cnn"},
            )
        )
    threshold_diagnostics = root / "results" / "threshold_per_class_diagnostics.csv"
    if threshold_diagnostics.is_file():
        threshold = pd.read_csv(threshold_diagnostics)
        passed = (
            len(threshold) == 100
            and not threshold.duplicated(["fold", "seed", "threshold"]).any()
            and threshold["stored_pseudo_count_matched"].astype(str).str.lower().eq("true").all()
        )
        rows.append(
            {
                "result_family": "threshold_per_class_diagnostics",
                "path": threshold_diagnostics.relative_to(root).as_posix(),
                "rows": len(threshold),
                "expected_rows": 100,
                "methods": "cwt_gru_selftrain",
                "duplicate_experiment_keys": int(
                    threshold.duplicated(["fold", "seed", "threshold"]).sum()
                ),
                "statuses": "success",
                "git_dirty_rows": 0,
                "status": "VERIFIED" if passed else "REPRODUCTION FAILURE",
                "note": "",
            }
        )
    else:
        rows.append(
            {
                "result_family": "threshold_per_class_diagnostics",
                "path": "results/threshold_per_class_diagnostics.csv",
                "rows": 0,
                "expected_rows": 100,
                "methods": "",
                "duplicate_experiment_keys": 0,
                "statuses": "",
                "git_dirty_rows": 0,
                "status": "REPRODUCTION FAILURE",
                "note": "missing result table",
            }
        )
    report = pd.DataFrame(rows)
    output = root / "reports" / "sensitivity_completeness.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False, lineterminator="\n")
    print(report.to_string(index=False))
    if report["status"].ne("VERIFIED").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
