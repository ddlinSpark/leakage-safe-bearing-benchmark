"""Check result registries, statistics, and the manuscript as one evidence chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = {42, 123, 2024, 2025, 3407}
EXPECTED_FOLDS = {0, 1, 2, 3}
EXPECTED_MAIN_METHODS = {
    "raw_cnn",
    "gru_only",
    "cwt_cnn",
    "cwt_gru_no_se",
    "cwt_gru",
    "cwt_gru_temp",
    "cwt_gru_selftrain",
}
EXPECTED_EXTERNAL_METHODS = {"drsn", "tfnet"}


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }
    )


def expected_keys() -> set[tuple[int, int]]:
    return {
        (fold, seed)
        for fold in EXPECTED_FOLDS
        for seed in EXPECTED_SEEDS
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_matches(root: Path, manifest_path: Path) -> tuple[bool, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, record in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
        elif path.stat().st_size != record["bytes"]:
            failures.append(f"size mismatch {relative}")
        elif file_sha256(path) != record["sha256"]:
            failures.append(f"hash mismatch {relative}")
    return not failures, "; ".join(failures) if failures else f"{len(manifest['files'])} files"


def validate_run_matrix(
    frame: pd.DataFrame,
    methods: set[str],
    *,
    label: str,
) -> tuple[bool, str]:
    errors = []
    actual_methods = set(frame["method"].astype(str))
    if actual_methods != methods:
        errors.append(
            f"methods expected {sorted(methods)}, found {sorted(actual_methods)}"
        )
    for method, group in frame.groupby("method"):
        keys = set(
            zip(
                group["fold"].astype(int),
                group["seed"].astype(int),
                strict=True,
            )
        )
        if len(group) != 20 or keys != expected_keys():
            errors.append(
                f"{method}: {len(group)} rows and {len(keys)} unique fold-seed keys"
            )
    return not errors, "; ".join(errors) if errors else label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "CONSISTENCY_REPORT.md",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "artifacts" / "local" / "consistency_report.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results"
    checks: list[dict[str, Any]] = []

    raw = pd.read_csv(results / "raw_runs.csv")
    external = pd.read_csv(results / "sota" / "raw_runs.csv")
    registry = pd.read_csv(results / "experiment_registry.csv")
    summary = pd.read_csv(results / "summary.csv")
    external_summary = pd.read_csv(results / "sota" / "summary_with_ci.csv")
    hierarchical = pd.read_csv(results / "hierarchical_statistics.csv")
    complexity = pd.read_csv(results / "model_complexity.csv")
    metrics = json.loads(
        (results / "paper_metrics.json").read_text(encoding="utf-8")
    )

    counts = raw["experiment_type"].value_counts().to_dict()
    add_check(
        checks,
        "Canonical run counts",
        len(raw) == 320
        and counts.get("main", 0) == 140
        and counts.get("label_efficiency", 0) == 80
        and counts.get("threshold", 0) == 100,
        f"320 rows: {counts}",
    )
    add_check(
        checks,
        "External adaptation counts",
        len(external) == 40,
        f"{len(external)} rows",
    )
    add_check(
        checks,
        "Consolidated experiment registry",
        len(registry) == 820,
        f"{len(registry)} rows",
    )
    canonical_ok, canonical_detail = manifest_matches(
        root,
        results / "canonical" / "manifest.json",
    )
    add_check(
        checks,
        "Canonical result catalog",
        canonical_ok,
        canonical_detail,
    )
    split_ok, split_detail = manifest_matches(
        root,
        root / "splits" / "manifest.json",
    )
    add_check(
        checks,
        "Frozen split catalog",
        split_ok,
        split_detail,
    )
    add_check(
        checks,
        "Unique canonical run identifiers",
        raw["run_id"].is_unique and external["run_id"].is_unique,
        f"canonical={raw['run_id'].nunique()}, external={external['run_id'].nunique()}",
    )
    add_check(
        checks,
        "Completed canonical runs",
        raw["status"].eq("success").all()
        and external["status"].eq("success").all(),
        "all retained canonical and external rows report success",
    )
    add_check(
        checks,
        "Clean canonical provenance",
        not raw["git_dirty"].astype(bool).any()
        and not external["git_dirty"].astype(bool).any(),
        "no canonical run was produced from a dirty worktree",
    )

    main_rows = raw.loc[raw["experiment_type"].eq("main")].copy()
    main_ok, main_detail = validate_run_matrix(
        main_rows,
        EXPECTED_MAIN_METHODS,
        label="7 methods x 4 folds x 5 seeds = 140 matched runs",
    )
    add_check(checks, "Main model matrix", main_ok, main_detail)
    external_ok, external_detail = validate_run_matrix(
        external,
        EXPECTED_EXTERNAL_METHODS,
        label="2 adaptations x 4 folds x 5 seeds = 40 matched runs",
    )
    add_check(checks, "External model matrix", external_ok, external_detail)

    expected_summary = (
        main_rows.groupby("method", as_index=False)
        .agg(
            runs=("run_id", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            accuracy_mean=("accuracy", "mean"),
            validation_macro_f1_mean=("best_val_macro_f1", "mean"),
        )
        .set_index("method")
    )
    recorded_summary = summary.loc[summary["experiment_type"].eq("main")].set_index(
        "method"
    )
    summary_ok = set(expected_summary.index) == set(recorded_summary.index)
    summary_detail = "all seven main methods are present"
    if summary_ok:
        for row in expected_summary.itertuples():
            recorded = recorded_summary.loc[row.Index]
            if (
                int(recorded["Runs"]) != int(row.runs)
                or not np.isclose(
                    float(recorded["Macro-F1 Mean"]),
                    float(row.macro_f1_mean),
                    atol=1e-12,
                )
                or not np.isclose(
                    float(recorded["Accuracy Mean"]),
                    float(row.accuracy_mean),
                    atol=1e-12,
                )
            ):
                summary_ok = False
                summary_detail = f"summary mismatch for {row.Index}"
                break
    add_check(
        checks,
        "Macro-F1 and accuracy aggregation",
        summary_ok,
        summary_detail,
    )

    readme = (root / "README.md").read_text(encoding="utf-8")
    readme_labels = {
        "raw_cnn": "Raw 1D-CNN",
        "gru_only": "GRU only",
        "cwt_cnn": "CWT 2D-CNN",
        "cwt_gru_no_se": "CWT-GRU without SE",
        "cwt_gru": "CWT-GRU",
        "cwt_gru_selftrain": (
            "CWT-GRU + temperature scaling + self-training"
        ),
    }
    missing_readme_values = []
    for method, label in readme_labels.items():
        row = expected_summary.loc[method]
        expected_row = (
            f"| {label} | {row['macro_f1_mean']:.3f} +/- "
            f"{main_rows.loc[main_rows['method'].eq(method), 'macro_f1'].std():.3f} "
            f"| {row['accuracy_mean']:.3f} +/- "
            f"{main_rows.loc[main_rows['method'].eq(method), 'accuracy'].std():.3f} |"
        )
        if expected_row not in readme:
            missing_readme_values.append(method)
    add_check(
        checks,
        "README key findings",
        not missing_readme_values,
        (
            "six principal result rows match canonical means and sample SDs"
            if not missing_readme_values
            else f"mismatching methods: {missing_readme_values}"
        ),
    )

    metric_main = {
        str(row["method"]): row for row in metrics["main_models"]
    }
    metrics_ok = set(metric_main) == EXPECTED_MAIN_METHODS
    if metrics_ok:
        for method, row in expected_summary.iterrows():
            record = metric_main[method]
            if not np.isclose(
                float(record["macro_f1_mean"]),
                float(row["macro_f1_mean"]),
                atol=1e-12,
            ):
                metrics_ok = False
                break
    add_check(
        checks,
        "paper_metrics model means",
        metrics_ok,
        "main model means match raw_runs.csv exactly",
    )

    external_metric = {
        str(row["method"]): row for row in metrics["external_models"]
    }
    external_metrics_ok = set(external_metric) == EXPECTED_EXTERNAL_METHODS
    if external_metrics_ok:
        for method, group in external.groupby("method"):
            external_metrics_ok &= bool(
                np.isclose(
                    float(external_metric[method]["macro_f1_mean"]),
                    float(group["macro_f1"].mean()),
                    atol=1e-12,
                )
            )
    add_check(
        checks,
        "paper_metrics external means",
        external_metrics_ok,
        "DRSN and TFN adaptation means match raw_runs.csv exactly",
    )

    hierarchical_ok = (
        len(hierarchical) == 19
        and hierarchical["n_folds"].eq(4).all()
        and hierarchical["n_seeds"].eq(5).all()
        and hierarchical["n_pairs"].eq(20).all()
        and hierarchical["bootstrap_samples"].eq(20_000).all()
        and (
            hierarchical["hierarchical_ci_low"]
            <= hierarchical["hierarchical_ci_high"]
        ).all()
        and (hierarchical["fold_aware_p"] >= 0.125 - 1e-12).all()
    )
    add_check(
        checks,
        "Fold-first hierarchical inference",
        hierarchical_ok,
        "19 comparisons retain 4 folds, 5 seeds, 20 pairs, 20000 bootstrap samples, ordered CIs, and p >= 0.125",
    )

    parameter_ok = (
        complexity["method"].nunique() == 7
        and complexity["parameters"].gt(0).all()
        and complexity.groupby("method")["parameters"].nunique().eq(1).all()
    )
    add_check(
        checks,
        "Parameter-count registry",
        parameter_ok,
        ", ".join(
            f"{row.method}={int(row.parameters)}"
            for row in complexity.drop_duplicates("method").itertuples()
        ),
    )
    add_check(
        checks,
        "Seed and fold registry",
        set(raw["seed"].astype(int)) == EXPECTED_SEEDS
        and set(raw["fold"].astype(int)) == EXPECTED_FOLDS,
        f"seeds={sorted(EXPECTED_SEEDS)}, folds={sorted(EXPECTED_FOLDS)}",
    )

    audit_json = root / "artifacts" / "local" / "consistency_submission_audit.json"
    completed = subprocess.run(
        [
            "python",
            str(root / "scripts" / "audit_submission.py"),
            "--json-output",
            str(audit_json),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if audit_json.is_file():
        audit = json.loads(audit_json.read_text(encoding="utf-8"))
        audit_pass = audit.get("status") == "PASS"
        audit_detail = (
            f"independent DOCX/table audit: {audit.get('mismatch_count')} mismatches"
        )
    else:
        audit_pass = False
        audit_detail = completed.stderr.strip() or "audit produced no JSON"
    add_check(
        checks,
        "Manuscript and table consistency",
        audit_pass,
        audit_detail,
    )

    failures = [row for row in checks if row["status"] == "FAIL"]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failure_count": len(failures),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Consistency Report",
        "",
        "## Verdict",
        "",
        f"**{payload['status']}**. {len(checks) - len(failures)} of "
        f"{len(checks)} checks passed.",
        "",
        "The check reads canonical CSV/JSON results and runs the independent "
        "DOCX/table audit. It never rewrites an experimental value.",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in checks:
        detail = str(row["detail"]).replace("|", "\\|")
        lines.append(f"| {row['check']} | {row['status']} | {detail} |")
    lines.extend(["", "## Mismatches", ""])
    if failures:
        for row in failures:
            lines.append(f"- **{row['check']}**: {row['detail']}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Source precedence",
            "",
            "`results/raw_runs.csv` and the retained per-run predictions remain "
            "the numerical sources of truth. The DOCX is checked against those "
            "files and is never used to overwrite them.",
            "",
        ]
    )
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"{payload['status']}: {len(checks) - len(failures)}/{len(checks)} checks passed")
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
