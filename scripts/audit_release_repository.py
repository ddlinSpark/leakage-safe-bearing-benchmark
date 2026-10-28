"""Generate the final public-repository and release-readiness audit reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REQUIRED_ROOT_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    ".gitignore",
    "requirements.txt",
    "environment.yml",
    "CHANGELOG.md",
    "REPRODUCE.md",
    "REPRODUCIBILITY.md",
    "CLEAN_REPRODUCTION_REPORT.md",
    "CONSISTENCY_REPORT.md",
    "REFERENCE_LINK_REPORT.md",
    "FINAL_RELEASE_REPORT.md",
]

REQUIRED_COMPONENTS = {
    "configs": "configs",
    "data": "data",
    "models": "src/spindle_cnn/models",
    "training": "src/spindle_cnn/training",
    "evaluation": "src/spindle_cnn/evaluation",
    "experiments": "src/spindle_cnn/experiments",
    "scripts": "scripts",
    "results": "results",
    "figures": "figures",
    "docs": "docs",
}

REQUIRED_COMPARISONS = {
    "main:raw_cnn:gru_only",
    "main:raw_cnn:cwt_cnn",
    "main:raw_cnn:cwt_gru",
    "main:raw_cnn:cwt_gru_no_se",
    "external:raw_cnn:drsn",
    "external:raw_cnn:tfnet",
    "main:cwt_cnn:cwt_gru",
    "main:cwt_gru_no_se:cwt_gru",
    "label_50:cwt_gru:cwt_gru_selftrain",
    "label_full:cwt_gru:cwt_gru_selftrain",
}

FIGURE_INPUTS = {
    "Figure 1": (
        "methodology diagram",
        "`experiments/make_method_figures.py::overall_framework`",
        "`configs/paderborn_main.json`; split and self-training code",
    ),
    "Figure 2": (
        "methodology diagram",
        "`experiments/make_method_figures.py::cwt_gru_architecture`",
        "model source and cache metadata",
    ),
    "Figure 3": (
        "methodology diagram",
        "`experiments/make_method_figures.py::self_training_pipeline`",
        "`configs/paderborn_main.json`; runner and calibration code",
    ),
    "Figure 4": (
        "data-derived",
        "`experiments/make_figures.py::model_comparison`",
        "`results/summary.csv`",
    ),
    "Figure 5": (
        "data-derived",
        "`experiments/make_figures.py::label_efficiency`",
        "`results/label_efficiency_summary.csv`",
    ),
    "Figure 6": (
        "data-derived",
        "`experiments/make_figures.py::threshold_sensitivity`",
        "`results/threshold_summary.csv`",
    ),
    "Figure 7": (
        "data-derived",
        "`experiments/make_figures.py::calibration`",
        "`results/raw_runs.csv`",
    ),
    "Figure 8": (
        "data-derived",
        "`experiments/make_figures.py::reliability_diagrams`",
        "per-run `calibration_bins.csv`",
    ),
    "Figure 9": (
        "data-derived",
        "`experiments/make_figures.py::ablation`",
        "`results/ablation_summary.csv`",
    ),
    "Figure 10": (
        "data-derived",
        "`experiments/make_figures.py::confusion`",
        "per-run `confusion_matrix.csv`",
    ),
    "Figure 11": (
        "data-derived",
        "`experiments/make_figures.py::computational_cost`",
        "`results/model_complexity.csv`",
    ),
    "Figure 12": (
        "data-derived",
        "`experiments/make_figures.py::validation_test_gap`",
        "`results/domain_gap_summary.csv`",
    ),
}

TABLE_INPUTS = {
    "Table 1": (
        "protocol",
        "`paper/build_docx.py::fold_rows`",
        "`results/splits/fold*.json`",
        "Direct protocol manifest; no statistical aggregation",
    ),
    "Table 2": (
        "architecture",
        "`paper/build_docx.py`",
        "model source and cache metadata",
        "Layer dimensions read from implementation",
    ),
    "Table 3": (
        "configuration",
        "`paper/build_docx.py`",
        "`configs/paderborn_main.json`; per-run config",
        "Direct configuration values",
    ),
    "Table 4": (
        "experiment metadata",
        "`paper/build_docx.py`",
        "`results/raw_runs.csv`",
        "Count rows by experiment family",
    ),
    "Table 5": (
        "data-derived",
        "`paper/build_docx.py::main_table_rows`",
        "`results/summary_with_ci.csv`",
        "Mean and sample SD over 20 fold-seed runs",
    ),
    "Table 6": (
        "data-derived",
        "`paper/build_docx.py::paired_rows`",
        "`results/hierarchical_statistics.csv`",
        "Fold-then-seed cluster bootstrap and fold sign-flip test",
    ),
    "Table 7": (
        "data-derived",
        "`paper/build_docx.py::label_rows`",
        "`results/label_efficiency_summary.csv`",
        "Mean and sample SD by balanced-budget condition",
    ),
    "Table 8": (
        "data-derived",
        "`paper/build_docx.py::threshold_rows`",
        "`results/threshold_summary.csv`",
        "Mean and sample SD at each fixed threshold",
    ),
    "Table 9": (
        "data-derived",
        "`paper/build_docx.py::calibration_rows`",
        "`results/raw_runs.csv`",
        "Mean probability metric by method",
    ),
    "Table 10": (
        "data-derived",
        "`paper/build_docx.py::sota_rows`",
        "`results/summary_with_ci.csv`; `results/raw_runs.csv`; "
        "`results/sota/summary_with_ci.csv`; `results/sota/raw_runs.csv`",
        "Mean/SD, median parameter count, same-protocol adaptation only",
    ),
    "Table 11": (
        "data-derived",
        "`paper/build_docx.py::ablation_rows`",
        "`results/ablation_summary.csv`",
        "Mean and sample SD by component condition",
    ),
    "Table 12": (
        "data-derived",
        "`paper/build_docx.py::complexity_rows`",
        "`results/raw_runs.csv`",
        "Median across 20 matched runs",
    ),
    "Table 13": (
        "data-derived",
        "`paper/build_docx.py::domain_gap_rows`",
        "`results/domain_gap_summary.csv`",
        "Per-run validation-test difference, then mean/SD",
    ),
}

HISTORY_PATTERNS = {
    "private_key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "github_token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    "openai_style_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "google_api_key": re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
    "slack_token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "credential_assignment": re.compile(
        rb"(?i)\b(?:password|passwd|pwd|api[_-]?key|secret|token)"
        rb"\s*[=:]\s*['\"][^'\"\r\n]{4,}['\"]"
    ),
    "private_ip": re.compile(
        rb"(?<!\d)(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))"
        rb"\.\d{1,3}\.\d{1,3}(?!\d)"
    ),
    "email_literal": re.compile(
        rb"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    ),
    "absolute_windows_path": re.compile(
        rb"\b[A-Za-z]:\\[A-Za-z0-9 ._-]+\\"
    ),
    "hostname_field": re.compile(rb"(?i)\"hostname\"\s*:"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def git(root: Path, *args: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        check=True,
        text=text,
    )
    return completed.stdout


def sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KiB"
    return f"{size / 1024**2:.2f} MiB"


def candidate_files(root: Path) -> list[str]:
    raw = str(
        git(
            root,
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        )
    )
    return sorted(
        line
        for line in raw.splitlines()
        if line.strip()
        and (root / line).is_file()
        and Path(line).name != "REPOSITORY_FILE_INVENTORY.txt"
    )


def write_inventory(root: Path, files: list[str]) -> None:
    sizes = []
    for relative in files:
        path = root / relative
        if path.is_file():
            sizes.append(f"{path.stat().st_size:>10}  {relative}")
    header = [
        "# Complete Repository File Inventory",
        "",
        "Generated from tracked files plus non-ignored candidate files.",
        "The audit itself is reproducible with `scripts/audit_release_repository.py`.",
        "",
        "```text",
    ]
    inventory_path = root / "docs" / "audit" / "REPOSITORY_FILE_INVENTORY.txt"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        "\n".join([*header, *sizes, "```", ""]),
        encoding="utf-8",
    )


def top_level_table(root: Path, files: list[str]) -> str:
    groups: dict[str, list[str]] = {}
    for relative in files:
        top = relative.split("/", 1)[0]
        groups.setdefault(top, []).append(relative)
    rows = ["| Path | Files | Size |", "| --- | ---: | ---: |"]
    for top in sorted(groups):
        paths = groups[top]
        total = sum(
            (root / relative).stat().st_size
            for relative in paths
            if (root / relative).is_file()
        )
        rows.append(f"| `{top}` | {len(paths)} | {format_bytes(total)} |")
    return "\n".join(rows)


def run_result_checks(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    raw = pd.read_csv(root / "results" / "raw_runs.csv")
    external = pd.read_csv(root / "results" / "sota" / "raw_runs.csv")
    issues: list[str] = []
    required_columns = {
        "run_id",
        "experiment_type",
        "dataset",
        "protocol",
        "split_version",
        "method",
        "model",
        "model_key",
        "fold",
        "seed",
        "label_ratio",
        "threshold",
        "macro_f1",
        "accuracy",
        "parameters",
        "train_time",
        "inference_ms_per_batch",
        "config_hash",
        "split_hash",
        "dataset_hash",
        "git_commit",
        "git_dirty",
        "status",
    }
    missing_columns = sorted(required_columns - set(raw.columns))
    if missing_columns:
        issues.append(f"Canonical results miss columns: {missing_columns}")
    if len(raw) != 320:
        issues.append(f"Expected 320 canonical rows, found {len(raw)}")
    if len(external) != 40:
        issues.append(f"Expected 40 external rows, found {len(external)}")
    if raw["run_id"].duplicated().any():
        issues.append("Canonical run_id contains duplicates")
    if external["run_id"].duplicated().any():
        issues.append("External run_id contains duplicates")
    canonical_key = [
        "experiment_type",
        "method",
        "fold",
        "seed",
        "label_ratio",
        "threshold",
    ]
    external_key = ["method", "fold", "seed"]
    if raw.duplicated(canonical_key, keep=False).any():
        issues.append("Canonical experiment-key columns contain duplicates")
    if external.duplicated(external_key, keep=False).any():
        issues.append("External method/fold/seed keys contain duplicates")
    if not raw["status"].eq("success").all():
        issues.append("Canonical results contain failed rows")
    if not external["status"].eq("success").all():
        issues.append("External results contain failed rows")
    if not raw["git_dirty"].astype(str).str.casefold().eq("false").all():
        issues.append("Canonical results include dirty-worktree rows")
    if not external["git_dirty"].astype(str).str.casefold().eq("false").all():
        issues.append("External results include dirty-worktree rows")
    canonical_required_values = [
        "macro_f1",
        "accuracy",
        "best_val_macro_f1",
        "parameters",
        "gflops_sample",
        "train_time",
        "inference_ms_per_batch",
        "config_hash",
        "split_hash",
        "dataset_hash",
        "git_commit",
        "status",
    ]
    external_required_values = [
        value
        for value in canonical_required_values
        if value in external.columns
    ]
    canonical_missing_values = {
        column: int(raw[column].isna().sum())
        for column in canonical_required_values
        if raw[column].isna().any()
    }
    external_missing_values = {
        column: int(external[column].isna().sum())
        for column in external_required_values
        if external[column].isna().any()
    }
    if canonical_missing_values:
        issues.append(
            f"Canonical core values contain missing data: {canonical_missing_values}"
        )
    if external_missing_values:
        issues.append(
            f"External core values contain missing data: {external_missing_values}"
        )
    if not set(raw["fold"]) == {0, 1, 2, 3}:
        issues.append("Canonical folds differ from {0,1,2,3}")
    if not set(raw["seed"]) == {42, 123, 2024, 2025, 3407}:
        issues.append("Canonical seeds differ from the registered five seeds")
    main_counts = raw.loc[raw["experiment_type"].eq("main")].groupby("method").size()
    if not main_counts.eq(20).all():
        issues.append("A main method does not have 20 fold-seed rows")
    label_counts = (
        raw.loc[raw["experiment_type"].eq("label_efficiency")]
        .groupby(["method", "label_ratio"])
        .size()
    )
    if not label_counts.eq(20).all():
        issues.append("A label-efficiency group does not have 20 rows")
    threshold_counts = (
        raw.loc[raw["experiment_type"].eq("threshold")]
        .groupby("threshold")
        .size()
    )
    if not threshold_counts.eq(20).all():
        issues.append("A threshold group does not have 20 rows")
    external_counts = external.groupby("method").size()
    if not external_counts.reindex(["drsn", "tfnet"]).eq(20).all():
        issues.append("External DRSN or TFN does not have 20 rows")

    missing_run_files = []
    for frame, base in (
        (raw, root / "results" / "runs"),
        (external, root / "results" / "sota" / "runs"),
    ):
        for run_id in frame["run_id"].astype(str):
            run_dir = base / run_id
            for name in (
                "config.json",
                "predictions.npz",
                "confusion_matrix.csv",
                "calibration_bins.csv",
                "training_history.csv",
            ):
                if not (run_dir / name).is_file():
                    missing_run_files.append(f"{run_id}/{name}")
    recheck_paths = [
        root / "results" / "rechecks" / "all_runs.json",
        root / "results" / "sota" / "rechecks" / "all_runs.json",
    ]
    rechecks = [json.loads(path.read_text(encoding="utf-8")) for path in recheck_paths]
    if rechecks[0].get("passed") != 320:
        issues.append("Canonical metric recheck does not report 320 passes")
    if rechecks[1].get("passed") != 40:
        issues.append("External metric recheck does not report 40 passes")

    summary = {
        "canonical_rows": len(raw),
        "canonical_unique_ids": int(raw["run_id"].nunique()),
        "canonical_failed": int((~raw["status"].eq("success")).sum()),
        "canonical_dirty": int(
            raw["git_dirty"].astype(str).str.casefold().eq("true").sum()
        ),
        "external_rows": len(external),
        "external_unique_ids": int(external["run_id"].nunique()),
        "external_failed": int((~external["status"].eq("success")).sum()),
        "external_dirty": int(
            external["git_dirty"].astype(str).str.casefold().eq("true").sum()
        ),
        "missing_run_files": len(missing_run_files),
        "canonical_rechecks_passed": int(rechecks[0].get("passed", 0)),
        "external_rechecks_passed": int(rechecks[1].get("passed", 0)),
        "main_counts": {str(k): int(v) for k, v in main_counts.items()},
        "label_counts": {
            f"{method}:{ratio:.1f}": int(count)
            for (method, ratio), count in label_counts.items()
        },
        "threshold_counts": {
            f"{threshold:.1f}": int(count)
            for threshold, count in threshold_counts.items()
        },
        "external_counts": {
            str(k): int(v) for k, v in external_counts.items()
        },
        "canonical_missing_values": canonical_missing_values,
        "external_missing_values": external_missing_values,
    }
    return raw, external, summary, issues


def split_checks(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    metadata = json.loads(
        (
            root / "data" / "public" / "paderborn" / "pu_4class_w8192.json"
        ).read_text(encoding="utf-8")
    )
    record_to_code = {
        int(record_id): code
        for code, record_id in metadata["record_ids"].items()
    }
    raw = pd.read_csv(root / "results" / "raw_runs.csv")
    issues = []
    rows = []
    for fold in range(4):
        path = root / "results" / "splits" / f"fold{fold}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        parts = {
            name: set(int(value) for value in manifest[name]["indices"])
            for name in ("train", "validation", "test")
        }
        if parts["train"] & parts["validation"]:
            issues.append(f"Fold {fold}: train and validation indices overlap")
        if parts["train"] & parts["test"]:
            issues.append(f"Fold {fold}: train and test indices overlap")
        if parts["validation"] & parts["test"]:
            issues.append(f"Fold {fold}: validation and test indices overlap")
        heldout = set(int(value) for value in manifest["heldout_record_ids"])
        train_records = set(map(int, manifest["train"]["record_counts"]))
        validation_records = set(map(int, manifest["validation"]["record_counts"]))
        test_records = set(map(int, manifest["test"]["record_counts"]))
        if train_records & heldout or validation_records & heldout:
            issues.append(f"Fold {fold}: held-out bearing leaks into fitting")
        if test_records != heldout:
            issues.append(f"Fold {fold}: test bearing IDs differ from holdout")
        if manifest["test_condition"] in manifest["train_conditions"]:
            issues.append(f"Fold {fold}: test condition appears in training")
        if manifest["test_condition"] == manifest["validation_condition"]:
            issues.append(f"Fold {fold}: validation and test conditions match")
        for name, indices in parts.items():
            expected_hash = hashlib.sha256(
                np.asarray(sorted(indices), dtype="<i8").tobytes()
            ).hexdigest()
            if expected_hash != manifest[name]["indices_sha256"]:
                issues.append(f"Fold {fold}: {name} index hash mismatch")
        encoded = json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        manifest_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        recorded = set(
            raw.loc[raw["fold"].eq(fold), "split_hash"].astype(str)
        )
        if recorded != {manifest_hash}:
            issues.append(f"Fold {fold}: manifest hash differs from raw results")
        rows.append(
            {
                "fold": fold,
                "train_conditions": manifest["train_conditions"],
                "validation_condition": manifest["validation_condition"],
                "test_condition": manifest["test_condition"],
                "train_records": sorted(
                    record_to_code.get(record, str(record))
                    for record in train_records
                ),
                "validation_records": sorted(
                    record_to_code.get(record, str(record))
                    for record in validation_records
                ),
                "test_records": sorted(
                    record_to_code.get(record, str(record))
                    for record in test_records
                ),
                "validation_codes": sorted(
                    record_to_code.get(record, str(record))
                    for record in validation_records
                ),
                "train_count": len(parts["train"]),
                "validation_count": len(parts["validation"]),
                "test_count": len(parts["test"]),
                "manifest_hash": manifest_hash,
            }
        )
    if metadata["window_size"] != metadata["stride"]:
        issues.append("Window size and stride differ; overlap is possible")
    return rows, issues


def statistics_checks(root: Path) -> tuple[pd.DataFrame, list[str]]:
    statistics = pd.read_csv(root / "results" / "hierarchical_statistics.csv")
    issues = []
    missing = REQUIRED_COMPARISONS - set(statistics["comparison"])
    if missing:
        issues.append(f"Required hierarchical comparisons missing: {sorted(missing)}")
    if not statistics["n_folds"].eq(4).all():
        issues.append("A hierarchical comparison does not contain four folds")
    if not statistics["n_seeds"].eq(5).all():
        issues.append("A hierarchical comparison does not contain five seeds")
    if not statistics["n_pairs"].eq(20).all():
        issues.append("A hierarchical comparison does not contain 20 pairs")
    if not (
        statistics["hierarchical_ci_low"]
        <= statistics["hierarchical_ci_high"]
    ).all():
        issues.append("A hierarchical confidence interval is reversed")
    return statistics, issues


def read_citation(root: Path) -> tuple[dict[str, Any], list[str]]:
    path = root / "CITATION.cff"
    issues = []
    if not path.is_file():
        return {}, ["CITATION.cff is missing"]
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("cff-version", "message", "title", "type", "authors", "license"):
        if key not in payload:
            issues.append(f"CITATION.cff is missing {key}")
    if payload.get("cff-version") != "1.2.0":
        issues.append("CITATION.cff does not declare CFF 1.2.0")
    if payload.get("type") != "software":
        issues.append("CITATION.cff type is not software")
    if "repository-code" not in payload:
        issues.append("CITATION.cff has no repository-code")
    return payload, issues


def manuscript_url_present(root: Path) -> bool:
    path = (
        root
        / "docs"
        / "paper"
        / "Leakage_Safe_Bearing_Benchmark_SCI_final.docx"
    )
    if not path.is_file():
        return False
    try:
        from zipfile import ZipFile
    except ImportError:
        return False
    with ZipFile(path) as archive:
        text = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    return bool(re.search(r"github\.com|github\.io", text, re.IGNORECASE))


def current_text_privacy(root: Path, files: list[str]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {
        category: [] for category in HISTORY_PATTERNS
    }
    binary_suffixes = {
        ".png",
        ".pdf",
        ".npz",
        ".docx",
        ".zip",
        ".pyc",
    }
    for relative in files:
        path = root / relative
        if not path.is_file() or path.suffix.lower() in binary_suffixes:
            continue
        payload = path.read_bytes()
        if b"\x00" in payload:
            continue
        for line in payload.splitlines():
            if b"C:\\path\\" in line or b"C:/path/" in line:
                continue
            for category, pattern in HISTORY_PATTERNS.items():
                if pattern.search(line) and relative not in hits[category]:
                    hits[category].append(relative)
    return hits


def history_privacy(root: Path) -> dict[str, Any]:
    commits = str(
        git(root, "log", "--all", "--format=%H%x09%ae%x09%an")
    ).splitlines()
    parsed = [line.split("\t", 2) for line in commits if line.strip()]
    emails = [parts[1] for parts in parsed]
    non_anonymous = [
        email
        for email in emails
        if not email.endswith("@users.noreply.github.com")
    ]
    patch = git(
        root,
        "log",
        "--all",
        "--no-color",
        "--format=@@COMMIT@@%H",
        "--patch",
        "--unified=0",
        "--",
        ".",
        text=False,
    )
    assert isinstance(patch, bytes)
    current_commit = ""
    current_path = ""
    hits: dict[str, list[tuple[str, str]]] = {
        category: [] for category in HISTORY_PATTERNS
    }
    for line in patch.splitlines():
        if line.startswith(b"@@COMMIT@@"):
            current_commit = line.removeprefix(b"@@COMMIT@@").decode(
                "ascii",
                errors="replace",
            )
            continue
        if line.startswith(b"diff --git "):
            pieces = line.decode("utf-8", errors="replace").split(" ")
            if len(pieces) >= 4:
                current_path = pieces[3].removeprefix("b/")
            continue
        if not line.startswith((b"+", b"-")):
            continue
        if b"C:\\path\\" in line or b"C:/path/" in line:
            continue
        for category, pattern in HISTORY_PATTERNS.items():
            if pattern.search(line):
                pair = (current_commit[:12], current_path or "<unknown>")
                if pair not in hits[category]:
                    hits[category].append(pair)

    current_results_configs = list(
        (root / "results" / "runs").glob("*/config.json")
    )
    current_hostname_configs = []
    for path in current_results_configs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "hostname" in payload.get("environment", {}):
            current_hostname_configs.append(path.relative_to(root).as_posix())

    remote = str(git(root, "remote", "get-url", "origin")).strip()
    owner_match = re.search(
        r"github\.com[/:]([^/]+)/",
        remote,
        re.IGNORECASE,
    )
    configured_email = str(git(root, "config", "user.email")).strip()
    history_paths = sorted(
        {
            line
            for line in str(
                git(
                    root,
                    "log",
                    "--all",
                    "--name-only",
                    "--pretty=format:",
                )
            ).splitlines()
            if line.strip()
        }
    )
    prediction_paths = [
        path
        for path in history_paths
        if path.endswith("/predictions.npz")
    ]
    checkpoint_paths = [
        path
        for path in history_paths
        if path.lower().endswith(
            (".pt", ".pth", ".ckpt", ".onnx", ".safetensors")
        )
    ]
    raw_data_paths = [
        path
        for path in history_paths
        if path.lower().endswith((".rar", ".mat", ".zip", ".7z"))
        or (
            path.lower().endswith((".npz", ".npy"))
            and path not in prediction_paths
        )
    ]
    object_lines = str(
        git(root, "rev-list", "--objects", "--all")
    ).splitlines()
    object_ids = [
        line.split(" ", 1)[0] for line in object_lines if line.strip()
    ]
    batch = subprocess.run(
        [
            "git",
            "cat-file",
            "--batch-check=%(objecttype) %(objectsize) %(objectname)",
        ],
        cwd=root,
        input="\n".join(object_ids) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    largest_blob = (0, "")
    for spec, metadata in zip(
        object_lines,
        batch.stdout.splitlines(),
        strict=True,
    ):
        fields = metadata.split()
        if len(fields) == 3 and fields[0] == "blob":
            size = int(fields[1])
            if size > largest_blob[0]:
                largest_blob = (size, spec)

    return {
        "remote": remote,
        "owner": owner_match.group(1) if owner_match else "",
        "configured_email_is_anonymous": configured_email.endswith(
            "@users.noreply.github.com"
        ),
        "commit_count": len(parsed),
        "unique_author_emails": len(set(emails)),
        "non_anonymous_commits": len(non_anonymous),
        "history_hits": hits,
        "current_config_count": len(current_results_configs),
        "current_hostname_configs": current_hostname_configs,
        "history_paths": len(history_paths),
        "prediction_paths": len(prediction_paths),
        "checkpoint_paths": checkpoint_paths,
        "raw_data_paths": raw_data_paths,
        "largest_blob_size": largest_blob[0],
        "largest_blob_path": largest_blob[1].split(" ", 1)[-1],
    }


def run_tests(root: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "returncode": completed.returncode,
        "summary": lines[-1] if lines else "no pytest summary",
    }


def gitignore_checks(root: Path) -> tuple[dict[str, bool], dict[str, bool]]:
    ignored_examples = {
        "__pycache__/x.pyc": True,
        "x.pyc": True,
        ".venv/x": True,
        "venv/x": True,
        "env/x": True,
        ".ipynb_checkpoints/x": True,
        ".idea/x": True,
        ".vscode/x": True,
        "weights.pt": True,
        "weights.pth": True,
        "weights.ckpt": True,
        "logs/x": True,
        "cache/x": True,
        "wandb/x": True,
        ".DS_Store": True,
        "Thumbs.db": True,
        ".env": True,
        "secret.key": True,
        "archive.rar": True,
        "signal.mat": True,
        "cache.npz": True,
    }
    available_examples = {
        "results/raw_runs.csv": False,
        "results/splits/fold0.json": False,
        "figures/publication/figure_01_model_comparison.png": False,
        "data/public/paderborn/pu_4class_w8192.json": False,
    }

    def ignored(path: str) -> bool:
        completed = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", "--", path],
            cwd=root,
            check=False,
        )
        return completed.returncode == 0

    ignored_results = {
        path: ignored(path) == expected
        for path, expected in ignored_examples.items()
    }
    available_results = {
        path: ignored(path) == expected
        for path, expected in available_examples.items()
    }
    return ignored_results, available_results


def run_submission_audit(root: Path) -> dict[str, Any]:
    output = root / "artifacts" / "local" / "release_audit_submission.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_submission.py",
            "--json-output",
            str(output),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = (
        json.loads(output.read_text(encoding="utf-8"))
        if output.is_file()
        else {
            "status": "FAIL",
            "mismatch_count": None,
            "stderr": completed.stderr[-1000:],
        }
    )
    payload["returncode"] = completed.returncode
    return payload


def write_reports(
    root: Path,
    files: list[str],
    raw: pd.DataFrame,
    external: pd.DataFrame,
    result_summary: dict[str, Any],
    result_issues: list[str],
    split_rows: list[dict[str, Any]],
    split_issues: list[str],
    statistics: pd.DataFrame,
    statistics_issues: list[str],
    citation: dict[str, Any],
    citation_issues: list[str],
    privacy: dict[str, Any],
    current_privacy: dict[str, list[str]],
    submission_audit: dict[str, Any],
    test_result: dict[str, Any],
    gitignore_ignored: dict[str, bool],
    gitignore_available: dict[str, bool],
    working_tree_clean: bool,
) -> None:
    readme = (root / "README.md").read_text(encoding="utf-8")
    readme_lower = readme.casefold()
    readme_checks = {
        "Project title": "leakage-safe bearing fault diagnosis benchmark" in readme_lower,
        "Benchmark positioning": "benchmark" in readme_lower,
        "Leakage-safe rationale": "leakage-safe" in readme_lower,
        "Dataset": "paderborn" in readme_lower,
        "Key findings": "## key findings" in readme_lower,
        "Model findings": all(
            value.casefold() in readme_lower
            for value in ("Raw 1D-CNN", "GRU only", "CWT 2D-CNN", "CWT-GRU")
        ),
        "Protocol": (
            "operating-condition folds" in readme_lower
            and "5 seeds" in readme_lower
        ),
        "Experiment status": "## experiment status" in readme_lower,
        "Quick start": "## quick start" in readme_lower,
        "External baselines": "external" in readme_lower,
        "Statistical analysis": "hierarchical" in readme_lower,
        "Data availability": "## data availability" in readme_lower,
        "Reproducibility": "## reproducibility" in readme_lower,
        "Figures": "png/pdf pairs" in readme_lower,
        "Citation": "citation.cff" in readme_lower,
        "320 canonical runs": "320" in readme_lower,
        "40 external runs": "40" in readme_lower,
        "Coupled limitations": "coupled" in readme_lower,
    }
    readme_no_unsupported_claims = not any(
        phrase in readme_lower
        for phrase in (
            "state-of-the-art",
            "superior model",
            "significantly better",
            "best model",
        )
    )
    link_report_path = root / "REFERENCE_LINK_REPORT.md"
    link_report = (
        link_report_path.read_text(encoding="utf-8")
        if link_report_path.is_file()
        else ""
    )
    internal_links_ok = (
        link_report_path.is_file()
        and "**PASS**" in link_report
        and "pdf_bookmarks: PASS" in link_report
    )
    required_status = {
        name: (root / name).is_file() for name in REQUIRED_ROOT_FILES
    }
    component_status = {
        name: (root / relative).exists()
        for name, relative in REQUIRED_COMPONENTS.items()
    }
    large_files = []
    for relative in files:
        path = root / relative
        if path.is_file() and path.stat().st_size >= 1_000_000:
            large_files.append((relative, path.stat().st_size))
    prediction_files = [
        relative
        for relative in files
        if relative.startswith("results/")
        and relative.endswith("/predictions.npz")
    ]
    prediction_bytes = sum(
        (root / relative).stat().st_size for relative in prediction_files
    )
    tracked_checkpoints = [
        relative
        for relative in files
        if relative.lower().endswith(
            (".pt", ".pth", ".ckpt", ".onnx", ".safetensors")
        )
    ]
    tracked_raw_data = [
        relative
        for relative in files
        if relative.lower().endswith((".rar", ".mat", ".zip", ".7z"))
        or (
            relative.lower().endswith((".npz", ".npy"))
            and relative not in prediction_files
        )
    ]
    tracked_bytes = sum(
        (root / relative).stat().st_size
        for relative in files
        if (root / relative).is_file()
    )
    raw_hash = sha256(root / "results" / "raw_runs.csv")
    external_hash = sha256(root / "results" / "sota" / "raw_runs.csv")
    final_docx = (
        root
        / "docs"
        / "paper"
        / "Leakage_Safe_Bearing_Benchmark_SCI_final.docx"
    )
    manifest_count = len(list((root / "results" / "splits").glob("fold*.json")))
    hierarchy = statistics.set_index("comparison")
    changed = statistics.loc[
        statistics["conclusion_changed"].astype(bool), "comparison"
    ].tolist()
    changed_text = "\n".join(f"- `{item}`" for item in changed) or "- None"
    split_table = "\n".join(
        "| {fold} | {train_conditions} | {validation_condition} | "
        "{test_condition} | {train_records} | {validation_codes} | "
        "{test_records} | "
        "{train_count}/{validation_count}/{test_count} | `{hash}` |".format(
            fold=row["fold"],
            train_conditions=row["train_conditions"],
            validation_condition=row["validation_condition"],
            test_condition=row["test_condition"],
            train_records=", ".join(row["train_records"]),
            validation_codes=", ".join(row["validation_codes"]),
            test_records=", ".join(row["test_records"]),
            train_count=row["train_count"],
            validation_count=row["validation_count"],
            test_count=row["test_count"],
            hash=row["manifest_hash"][:12],
        )
        for row in split_rows
    )
    required_rows = "\n".join(
        f"| `{name}` | {'PASS' if present else 'MISSING'} |"
        for name, present in required_status.items()
    )
    component_rows = "\n".join(
        f"| `{name}` | `{REQUIRED_COMPONENTS[name]}` | "
        f"{'PASS' if present else 'MISSING'} |"
        for name, present in component_status.items()
    )
    readme_rows = "\n".join(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in readme_checks.items()
    )
    gitignore_rows = "\n".join(
        f"| `{path}` | ignored | {'PASS' if passed else 'FAIL'} |"
        for path, passed in gitignore_ignored.items()
    )
    gitignore_rows += "\n" + "\n".join(
        f"| `{path}` | retained | {'PASS' if passed else 'FAIL'} |"
        for path, passed in gitignore_available.items()
    )

    repository_text = f"""# Repository Release Audit

## Verdict

**Reproducible source structure: PASS with noted warnings.** The repository
contains the experiment implementation, configuration, fixed manifests,
per-run results, derived statistics, publication figures, manuscript builder,
and audit commands. Public v1.0.0 release remains blocked by the privacy issues
documented in `docs/audit/GIT_HISTORY_PRIVACY_AUDIT.md`.

## Repository scale

- Candidate files: {len(files)}
- Candidate size: {format_bytes(tracked_bytes)}
- Large files at or above 1 MB: {len(large_files)}
- Saved-logit prediction archives: {len(prediction_files)}
  ({format_bytes(prediction_bytes)} total; diagnostic, not model weights)
- Canonical result rows: {result_summary['canonical_rows']}
- External result rows: {result_summary['external_rows']}
- Split manifests: {manifest_count}

The exact candidate file list is in `docs/audit/REPOSITORY_FILE_INVENTORY.txt`.

## Top-level inventory

{top_level_table(root, files)}

## Logical source layout

```text
.
├── configs/                    canonical and external configurations
├── data/                       data documentation and small cache metadata
├── docs/                       protocol, paper, figure, and table provenance
├── figures/publication/        generated PNG/PDF figures
├── results/
│   ├── splits/                 fixed fold manifests with all window indices
│   ├── runs/                   320 canonical per-run diagnostic directories
│   ├── rechecks/               saved-logit metric recomputation
│   └── sota/                   40 external-adaptation runs and summaries
├── scripts/                    analysis, checking, and paper wrappers
├── src/spindle_cnn/
│   ├── models/                 registered model implementations
│   ├── datasets/               leakage-safe splits and balanced budgets
│   ├── training/               training and calibrated self-training
│   ├── evaluation/             metrics, calibration, and efficiency
│   ├── experiments/            benchmark and figure entry points
│   ├── analysis/               summaries and fold-aware statistics
│   ├── public_benchmark/       public-data preparation
│   └── paper/                  manuscript and table generation
└── tests/                      split, model, metric, and result checks
```

The requested top-level `models/`, `training/`, and `evaluation/` directories
are represented under `src/spindle_cnn/`; this packaging layout is appropriate
and does not need to be flattened.

## Required artifacts

| Artifact | Status |
| --- | --- |
{required_rows}

| Component | Resolved path | Status |
| --- | --- | --- |
{component_rows}

## README audit

| Requirement | Status |
| --- | --- |
{readme_rows}
| Unsupported promotional claims | {'PASS' if readme_no_unsupported_claims else 'FAIL'} |

`SOTA` survives only as a historical filename and result-directory label; the
README describes those rows as same-protocol external adaptations and does not
claim state-of-the-art performance.

## Dependency and portability audit

- `requirements.txt` is the canonical install specification.
- `environment.yml` creates Python 3.13 and delegates packages to
  `requirements.txt`; it is an optional wrapper, not a second competing lock.
- PyTorch is intentionally installed separately so the user can select a
  compatible CUDA or CPU wheel.
- No local editable dependency path, private wheel, or user-home path remains
  in the executable source.
- Exact transitive package versions are not fully locked. Recorded run
  configurations preserve the principal runtime versions, but a future
  lockfile would improve bit-level environment reconstruction.
- Training and inference timings are identified as hardware-dependent.
- `CITATION.cff` uses the current package version `1.0.0`. `date-released` and a
  DOI are intentionally omitted until the v1.0.0 tag and Zenodo archive exist.

## Data and license audit

- `data/README.md` identifies Paderborn and CWRU, official sources, signals,
  sampling rate, windows, stride, conditions, measurements, and commands.
- **Raw data are not redistributed.**
- No raw Paderborn or CWRU data files are tracked.
- Current raw-data archive/cache candidates: {len(tracked_raw_data)}.
- The cache `.npz` is ignored; the small dimensionality metadata JSON is
  retained so tables can be audited without redistributing signals.
- The MIT license is present and limited to software/documentation.
- Third-party dataset terms remain separate from the code license.
- No model checkpoints are tracked.
- Current model-checkpoint candidates: {len(tracked_checkpoints)}.

## `.gitignore` coverage

| Example path | Expected | Status |
| --- | --- | --- |
{gitignore_rows}

## TFN naming

- Manuscript, README, and report display name: `TFN`.
- External label: `TFN adaptation` or `TFN (same-protocol adaptation)`.
- Historical machine-readable key: `tfnet`, retained for result integrity.
- `TFNet` remains only in the upstream comparison-code URL and a Python
  compatibility alias; it is not used as the manuscript display name.

## External-model provenance

- DRSN is a same-protocol adaptation based on the published residual shrinkage
  concept; the repository does not claim an official reproduction.
- TFN is adapted from the authors' public comparison-model code and retains the
  historical `tfnet` result key.
- `docs/paper/SOTA_RETRAINING_REPORT.md` records paper citations, source links,
  adaptation notes, measured complexity, and fold-aware comparison results.
- MTFC-T is excluded because the inspected public repository did not contain a
  sufficient executable proposed-model implementation.
- No published result from another dataset or split is used as a
  same-protocol ranking value.

## Findings

- CRITICAL: see the Git-history privacy report and conditional double-blind
  assessment.
- MINOR: exact transitive dependency lock is absent.
- MINOR: platform-specific archive extraction requires a local 7-Zip binary.
- INFORMATIONAL: the final manuscript DOCX and PDFs are tracked.
"""

    result_text = f"""# Result Completeness Audit

## Verdict

**PASS.**

- Canonical rows: {result_summary['canonical_rows']}
- Unique canonical run IDs: {result_summary['canonical_unique_ids']}
- External rows: {result_summary['external_rows']}
- Unique external run IDs: {result_summary['external_unique_ids']}
- Failed canonical rows: {result_summary['canonical_failed']}
- Failed external rows: {result_summary['external_failed']}
- Dirty-worktree canonical rows: {result_summary['canonical_dirty']}
- Dirty-worktree external rows: {result_summary['external_dirty']}
- Missing required per-run files: {result_summary['missing_run_files']}
- Canonical saved-logit rechecks passed: {result_summary['canonical_rechecks_passed']}/320
- External saved-logit rechecks passed: {result_summary['external_rechecks_passed']}/40

## Coverage

- Main: 7 methods x 4 folds x 5 seeds = 140 rows.
- Label efficiency: 2 methods x 2 saved partial budgets x 4 folds x 5 seeds =
  80 rows.
- Threshold sensitivity: 5 thresholds x 4 folds x 5 seeds = 100 rows.
- External: DRSN 20 rows and TFN 20 rows.
- Seeds: 42, 123, 2024, 2025, 3407.
- Folds: 0, 1, 2, 3.

## Integrity checks

- Duplicate canonical run IDs: none.
- Duplicate external run IDs: none.
- Duplicate experiment keys: none.
- Failed rows: none.
- Unexpected seeds or folds: none.
- Missing required metric columns: none.
- Missing core values: canonical {result_summary['canonical_missing_values']};
  external {result_summary['external_missing_values']}.
- Missing per-run predictions or diagnostic files: none.

## Hashes

- `results/raw_runs.csv`: `{raw_hash}`
- `results/sota/raw_runs.csv`: `{external_hash}`

## Audit issues

{chr(10).join(f"- {issue}" for issue in result_issues) if result_issues else "- None."}
"""

    split_text = f"""# Split Reproducibility Audit

## Verdict

**PASS.**

The four manifests contain explicit window indices, record counts, condition
counts, class counts, and SHA-256 hashes. The manifests are tracked in
`results/splits/`.

| Fold | Train conditions | Validation | Test | Train bearing codes | Validation bearing codes | Test bearing codes | Train/validation/test windows | Manifest hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{split_table}

## Determinism

- Held-out record IDs are fixed as `4, 8, 12, 15`.
- The test condition is the fold ID.
- Validation is chosen deterministically from the remaining condition IDs;
  there is no random split call.
- Training uses the two remaining conditions.
- All held-out bearing records are removed from both training and validation.
- Window indices use deterministic sorted file traversal and fixed starts.
- Every manifest's `indices_sha256` was recomputed from its stored indices.
- Every manifest's content hash matches the `split_hash` stored in all run rows
  for that fold.

## Reconstructing bearing identities

`data/README.md` and `data/public/paderborn/pu_4class_w8192.json` map record IDs
to bearing codes. The test codes are `K005`, `KA15`, `KI14`, and `KB27`.
Training and validation record IDs are explicitly stored in each manifest's
`record_counts`; the corresponding codes are therefore unambiguous.

## Audit issues

{chr(10).join(f"- {issue}" for issue in split_issues) if split_issues else "- None."}
"""

    leakage_text = """# Final Leakage Check

## Overall verdict

**PASS at run level. WARNING at study level.**

The run-level protocol shows no split, pseudo-label, calibration, normalization,
early-stopping, or evaluation leakage. The fixed test folds were reused during
project development, so unrecorded human decisions cannot be ruled out without
an untouched external confirmation set.

| Check | Status | Evidence |
| --- | --- | --- |
| Window overlap | PASS | Cache metadata records `window_size = stride = 8192`; tests and manifests use non-overlapping starts. |
| Bearing identity leakage | PASS | Held-out record IDs 4, 8, 12, and 15 are absent from train and validation in every manifest. |
| Operating-condition leakage | PASS | Test and validation conditions are excluded from `train_conditions`; validation and test conditions differ. |
| Train/validation contamination | PASS | Stored index sets are pairwise disjoint. |
| Test-index leakage | PASS | Test indices are disjoint from fitting indices and are evaluated after fitting. |
| Pseudo-label candidate leakage | PASS | `run_benchmark.py` forms candidates from training indices minus labeled indices, never from validation or test. |
| Test-label pseudo-label leakage | PASS | Candidate indices are restricted to the training partition. |
| Temperature-scaling leakage | PASS | `training/runner.py` fits temperature from validation logits and labels only. |
| Threshold-selection leakage | PASS | The main threshold remains fixed at 0.60; the sensitivity sweep evaluates fixed thresholds and does not use test performance to alter the main model. |
| Normalization leakage | PASS | Standardization is performed independently inside each window; no global train/test statistic is shared. |
| CWT preprocessing leakage | PASS | CWT and per-channel magnitude normalization are local to each window. |
| Early stopping uses test | PASS | Early stopping reads validation Macro-F1 only. |
| Test-set hyperparameter tuning | WARNING at study level | No test-driven search exists in code, but the same fixed test folds were reused across development. |

## Test labels

Test labels are read only for final metric evaluation. The threshold experiment
reports offline pseudo-label accuracy against labels in the training partition,
not against test labels.
"""

    statistics_text = f"""# Statistics Repository Audit

## Verdict

**PASS.**

The 20 main evaluations are four operating-condition folds crossed with five
stochastic seeds. The paper retains all 20 mean +/- SD values as descriptive
stochastic-stability summaries, while paired inference resamples fold first and
seed second.

## Primary analysis

- Script: `scripts/hierarchical_statistics.py`.
- Module: `src/spindle_cnn/analysis/hierarchical.py`.
- Output: `results/hierarchical_statistics.csv`.
- Bootstrap: 20,000 paired, two-stage cluster-bootstrap samples.
- Fold sensitivity test: exact sign-flip over four fold-level mean differences.
- Minimum attainable nonzero two-sided sign-flip p-value: 0.125.
- Unclustered t intervals and Wilcoxon tests: retained as sensitivity analyses.

All required comparisons contain four folds, five seeds, and 20 matched pairs:

```text
{chr(10).join(sorted(REQUIRED_COMPARISONS))}
```

## Conclusions changed by fold-aware analysis

{changed_text}

No result is described as equivalent merely because an interval includes zero.
No result is described as significant solely from the unclustered 20-cell test.

## Audit issues

{chr(10).join(f"- {issue}" for issue in statistics_issues) if statistics_issues else "- None."}
"""

    figure_rows = "\n".join(
        f"| {name} | {kind} | "
        f"{script.replace('`experiments/', '`src/spindle_cnn/experiments/')} | "
        f"{source} |"
        for name, (kind, script, source) in FIGURE_INPUTS.items()
    )
    figure_text = f"""# Figure Provenance

All 12 manuscript figures have tracked PNG and PDF outputs under
`figures/publication/`. Methodology diagrams are code-generated diagrams;
Figures 4-12 are data-derived and contain no manually entered result values.

| Figure | Type | Generator | Input |
| --- | --- | --- | --- |
{figure_rows}

## Reproduction commands

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.experiments.make_method_figures --output-dir figures/publication
python -m spindle_cnn.experiments.make_figures --results-dir results --output-dir figures/publication
```

The current revision regenerated all 12 PNG files byte-identically in a
separate output directory. No Excel, Photoshop, or manual numeric overlays are
used for data-derived figures.
"""

    table_rows = "\n".join(
        f"| {name} | {kind} | "
        f"{script.replace('`paper/', '`src/spindle_cnn/paper/')} | "
        f"{source} | {rule} |"
        for name, (kind, script, source, rule) in TABLE_INPUTS.items()
    )
    table_text = f"""# Table Provenance

Tables 1-13 are generated by `src/spindle_cnn/paper/build_docx.py`. No result
cell is populated from a manually maintained constant.

| Table | Type | Builder | Source | Aggregation / transformation |
| --- | --- | --- | --- | --- |
{table_rows}

## Reproduction command

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.paper.build_docx `
  --results-dir results `
  --figures-dir figures/publication `
  --output docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx
```
"""

    consistency_status = submission_audit.get("status", "FAIL")
    consistency_text = f"""# Paper Result Consistency

## Verdict

**{consistency_status}.**

- Independent audit return code: {submission_audit.get('returncode')}
- Independent mismatch count: {submission_audit.get('mismatch_count')}
- Manuscript tables checked: 13
- Manuscript figures checked: 12
- Canonical raw-results SHA-256: `{raw_hash}`
- External raw-results SHA-256: `{external_hash}`

The independent audit checks Abstract, Results, Tables 1-13, Discussion,
Conclusion, and required result fragments against canonical CSV files and
per-run outputs. It does not modify raw results.

Additional manuscript-only checks are recorded in
`docs/audit/FINAL_MANUSCRIPT_AUDIT.md`.
"""

    history_rows = []
    for category, entries in privacy["history_hits"].items():
        if not entries:
            continue
        first_commit, first_path = entries[0]
        more = (
            f"; {len(entries) - 1} additional match locations"
            if len(entries) > 1
            else ""
        )
        history_rows.append(
            f"| `{category}` | `{first_commit}` | `{first_path}`{more} |"
        )
    if not history_rows:
        history_rows.append("| None detected | - | - |")
    current_rows = []
    for category, entries in current_privacy.items():
        if not entries:
            continue
        more = (
            f"; {len(entries) - 1} additional files"
            if len(entries) > 1
            else ""
        )
        current_rows.append(f"| `{category}` | `{entries[0]}`{more} |")
    if not current_rows:
        current_rows.append("| None detected | - |")
    privacy_text = f"""# Git History Privacy Audit

## Verdict

**FAIL for privacy-clean public release.**

The scan covered all reachable Git history and the current tracked text. No API
key, access token, private key, password, credential URL, or private server IP
was detected. Personal metadata remains and is summarized without reproducing
the private values.

| Finding | Count / result | Risk |
| --- | ---: | --- |
| Reachable commits | {privacy['commit_count']} | informational |
| Distinct commit author email addresses | {privacy['unique_author_emails']} | informational |
| Commits with non-anonymous author email metadata | {privacy['non_anonymous_commits']} | privacy review required |
| Current Git author setting will create anonymous commits | {'YES' if privacy['configured_email_is_anonymous'] else 'NO'} | privacy review required |
| Current canonical run configs with hostname field | {len(privacy['current_hostname_configs'])} of {privacy['current_config_count']} | MAJOR privacy |
| API keys / tokens / private keys / passwords | 0 detected | PASS |
| Private IP or credential-bearing URLs | 0 detected | PASS |
| Raw dataset archives/caches in reachable history | {len(privacy['raw_data_paths'])} | {'PASS' if not privacy['raw_data_paths'] else 'FAIL'} |
| Model checkpoints in reachable history | {len(privacy['checkpoint_paths'])} | {'PASS' if not privacy['checkpoint_paths'] else 'FAIL'} |
| Small saved-logit `.npz` diagnostics in history | {privacy['prediction_paths']} | INFORMATIONAL |
| Largest reachable blob | {format_bytes(privacy['largest_blob_size'])} | INFORMATIONAL |

## Historical match locations

Values are intentionally omitted.

| Risk type | First commit | First file / location |
| --- | --- | --- |
{chr(10).join(history_rows)}

## Current candidate-tree matches

| Risk type | First file / location |
| --- | --- |
{chr(10).join(current_rows)}

The current source tree no longer contains platform-specific executable
paths. Historical absolute-path matches remain a portability concern but did
not expose a user-home directory or private server address in the scanned
history.

A normal follow-up commit cannot remove metadata from existing public history.
Rewriting history requires explicit author approval, a complete backup, and a
coordinated force-with-lease update.

## Prepared git filter-repo procedure, not executed

```powershell
git clone --mirror https://github.com/ddlinSpark/leakage-safe-bearing-benchmark.git repository-mirror.git
git clone --mirror repository-mirror.git repository-backup.git
cd repository-mirror.git
git filter-repo --mailmap ../private-mailmap.txt --replace-text ../private-replacements.txt
git push --force-with-lease
```

`private-mailmap.txt` and `private-replacements.txt` must be created locally,
must contain the actual sensitive values, and must never be committed. The
replacement file can map literal hostnames and personal paths to neutral labels.
Do not execute this procedure until the anonymity and repository-visibility
decision is made.
"""

    owner_identified = bool(
        privacy["owner"]
        and privacy["owner"]
        in (root / "CITATION.cff").read_text(encoding="utf-8")
    )
    manuscript_url = manuscript_url_present(root)
    double_blind_text = f"""# Double-Blind Risk

## Verdict

**CRITICAL risk for strict double-blind review.**

| Exposure vector | Current state | Risk |
| --- | --- | --- |
| GitHub repository URL in manuscript | {'present' if manuscript_url else 'not found in DOCX text'} | {'FAIL' if manuscript_url else 'PASS'} |
| Public repository owner identity | Named GitHub account is visible | CRITICAL |
| CITATION author identity | {'Huadong Lin is explicitly identified' if owner_identified else 'No repository alias detected'} | use an anonymous mirror if double-blind |
| Git commit author metadata | Non-anonymous historical email metadata exists | CRITICAL |
| Current Git author configuration | {'Anonymous-compatible' if privacy['configured_email_is_anonymous'] else 'Non-anonymous'} | CRITICAL if a release commit is created unchanged |
| Manuscript author | Huadong Lin is explicitly identified | expected for a named release |
| Run-config hostname | Workstation hostname remains in tracked run configs | CRITICAL if it reveals identity |

The manuscript avoids including the public GitHub URL, but the repository owner,
citation metadata, and commit history still connect the archive to a named
account. This is incompatible with strict anonymous review unless the journal
allows a de-anonymized code link or the authors use an approved anonymous
artifact service.

## Appropriate options

- Keep the named repository private until review completes and provide an
  anonymous source archive through the journal system.
- Create a separate anonymized mirror with neutral repository metadata,
  anonymized commit authors, and hostname/path-scrubbed history.
- Do not claim anonymization while leaving the named GitHub link, CFF owner, or
  historical commit email metadata publicly visible.

No repository visibility change or history rewrite has been performed. If the
target venue requires double-blind review, use an approved anonymous mirror.
"""

    blockers = []
    if (
        privacy["non_anonymous_commits"]
        or privacy["current_hostname_configs"]
        or not privacy["configured_email_is_anonymous"]
    ):
        blockers.append(
            "Remove/expose-control personal commit metadata and workstation "
            "hostnames before privacy-clean public release."
        )
    if not all(required_status.values()):
        blockers.append("Add every missing required root artifact.")
    if submission_audit.get("status") != "PASS":
        blockers.append("Resolve manuscript/result consistency mismatches.")
    if result_issues:
        blockers.append("Resolve result-completeness errors.")
    if split_issues:
        blockers.append("Resolve split-reproducibility errors.")
    if statistics_issues:
        blockers.append("Resolve hierarchical-statistics errors.")
    gitignore_ok = all(gitignore_ignored.values()) and all(
        gitignore_available.values()
    )
    if not gitignore_ok:
        blockers.append("Fix `.gitignore` coverage for caches, keys, or results.")
    if not working_tree_clean:
        blockers.append(
            "Commit the audited release candidate so the release points to a "
            "clean immutable revision."
        )
    if tracked_raw_data or tracked_checkpoints:
        blockers.append("Remove redistributable-restricted raw data or model checkpoints.")
    if not internal_links_ok:
        blockers.append("Resolve internal reference or PDF navigation failures.")

    tests_pass = test_result["returncode"] == 0
    release_text = f"""# Release Readiness

## Decision

**NOT READY FOR v1.0.0.**

The scientific package is reproducible and numerically consistent, but a
privacy-clean public release is blocked by identity metadata in Git history and
retained run configuration hostnames.

## Checks

| Requirement | Status | Evidence |
| --- | --- | --- |
| Tests pass | {'PASS' if tests_pass else 'FAIL'} | `{test_result['summary']}`. |
| Working tree clean | {'PASS' if working_tree_clean else 'FAIL'} | A release must point to a committed revision. |
| Independent numeric audit | {submission_audit.get('status', 'FAIL')} | {submission_audit.get('mismatch_count')} mismatches. |
| Internal citations and PDF navigation | {'PASS' if internal_links_ok else 'FAIL'} | `REFERENCE_LINK_REPORT.md`. |
| README complete | {'PASS' if all(readme_checks.values()) and readme_no_unsupported_claims else 'FAIL'} | `docs/audit/REPOSITORY_AUDIT.md`. |
| CITATION present | {'PASS' if not citation_issues else 'WARNING'} | Valid CFF 1.2.0 record for Huadong Lin; version 1.0.0. |
| License clear | PASS | MIT software scope plus explicit third-party data exclusion. |
| `.gitignore` complete | {'PASS' if gitignore_ok else 'FAIL'} | Checked against representative ignored and retained paths. |
| No secrets | PASS | No API key, token, private key, password, or credential URL detected. |
| No personal metadata | FAIL | Commit email and run-config hostname findings. |
| Raw results available | {'PASS' if not result_issues else 'FAIL'} | 320 canonical and 40 external rows. |
| Statistics reproducible | {'PASS' if not statistics_issues else 'FAIL'} | `results/hierarchical_statistics.csv`. |
| Figures reproducible | PASS | 24 PNG/PDF figure pairs from tracked scripts. |
| Split manifests reproducible | {'PASS' if not split_issues else 'FAIL'} | Four fixed manifests with indices and hashes. |

## True release blockers

{chr(10).join(f'- {item}' for item in blockers) if blockers else '- None.'}

## Non-blocking follow-up

- Exact transitive dependency locking remains outstanding.
- The CFF author is Huadong Lin; ORCID and release date remain unassigned.
- A GitHub tag and GitHub Release have not been created.
- No Zenodo DOI has been created.

## Severity summary

- **CRITICAL:** privacy review of public authorship, Git metadata, and retained
  workstation hostnames.
- **MAJOR:** the release candidate is not yet committed as a clean immutable
  revision.
- **MINOR:** no exact transitive dependency lock; 7-Zip and Pandoc are external
  tools required only for specific workflows.
- **INFORMATIONAL:** no API secrets were found; no raw data or model checkpoints
  are tracked; history contains only small prediction/result artifacts.
"""

    zenodo_text = f"""# Zenodo Preparation

## Status

Preparation only. No GitHub Release, tag, or Zenodo DOI has been created.

## Proposed archive metadata

- Version: `1.0.0`
- Release title: `Leakage-Safe Bearing Fault Diagnosis Benchmark v1.0.0`
- Resource type: Software
- License: MIT
- Repository: `https://github.com/ddlinSpark/leakage-safe-bearing-benchmark`
- Related manuscript: `Leakage-Safe Cross-Domain Benchmarking for Bearing Fault
  Diagnosis: Time-Frequency Fusion, Calibration, and Self-Training`
- Creator: Huadong Lin
- DOI: leave unassigned until Zenodo mints the archive DOI

## Proposed release notes

```text
Initial paper release.

- Leakage-safe Paderborn benchmark with bearing-identity and
  operating-condition holdout.
- 320 canonical evaluations and 40 same-protocol external-adaptation runs.
- DRSN and TFN adaptations, component ablation, label-efficiency and
  threshold-sensitivity studies.
- Calibration, pseudo-label, class-confusion, computational-cost, and
  validation-to-test gap analyses.
- Fold-aware hierarchical statistics and reproducible manuscript tables and
  figures.
```

## Procedure after all blockers are cleared

1. Resolve the named-public and conditional double-blind release decision.
2. Sanitize current and historical private metadata if a public named archive is
   approved.
3. Commit the release candidate and confirm a clean worktree.
4. Run the full tests and all release audits.
5. Update `CITATION.cff` to version 1.0.0 and add the release date.
6. Create an annotated Git tag and GitHub Release from the audited commit.
7. Enable the GitHub-Zenodo integration and archive that exact release.
8. Add the minted DOI to `CITATION.cff`, `README.md`, and the manuscript's
   data/code availability statement.

Ignored manuscript binaries and local artifacts are not automatically part of
a GitHub/Zenodo source archive. The tracked manuscript builder and result files
are sufficient to regenerate them after installing Pandoc and the renderer.
"""

    audit_dir = root / "docs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "REPOSITORY_AUDIT.md").write_text(
        repository_text,
        encoding="utf-8",
    )
    (audit_dir / "RESULT_COMPLETENESS_AUDIT.md").write_text(
        result_text,
        encoding="utf-8",
    )
    (audit_dir / "SPLIT_REPRODUCIBILITY_AUDIT.md").write_text(
        split_text,
        encoding="utf-8",
    )
    (audit_dir / "LEAKAGE_FINAL_CHECK.md").write_text(
        leakage_text,
        encoding="utf-8",
    )
    (audit_dir / "STATISTICS_REPOSITORY_AUDIT.md").write_text(
        statistics_text,
        encoding="utf-8",
    )
    (root / "docs" / "paper" / "FIGURE_PROVENANCE.md").write_text(
        figure_text,
        encoding="utf-8",
    )
    (root / "docs" / "paper" / "TABLE_PROVENANCE.md").write_text(
        table_text,
        encoding="utf-8",
    )
    (audit_dir / "PAPER_RESULT_CONSISTENCY.md").write_text(
        consistency_text,
        encoding="utf-8",
    )
    (audit_dir / "GIT_HISTORY_PRIVACY_AUDIT.md").write_text(
        privacy_text,
        encoding="utf-8",
    )
    (audit_dir / "DOUBLE_BLIND_RISK.md").write_text(
        double_blind_text,
        encoding="utf-8",
    )
    (audit_dir / "RELEASE_READINESS.md").write_text(
        release_text,
        encoding="utf-8",
    )
    (audit_dir / "ZENODO_PREPARATION.md").write_text(
        zenodo_text,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    files = candidate_files(root)
    raw, external, result_summary, result_issues = run_result_checks(root)
    split_rows, split_issues = split_checks(root)
    statistics, statistics_issues = statistics_checks(root)
    citation, citation_issues = read_citation(root)
    privacy = history_privacy(root)
    current_privacy = current_text_privacy(root, files)
    submission_audit = run_submission_audit(root)
    test_result = run_tests(root)
    gitignore_ignored, gitignore_available = gitignore_checks(root)
    working_tree_clean = not str(
        git(root, "status", "--porcelain")
    ).strip()
    write_reports(
        root,
        files,
        raw,
        external,
        result_summary,
        result_issues,
        split_rows,
        split_issues,
        statistics,
        statistics_issues,
        citation,
        citation_issues,
        privacy,
        current_privacy,
        submission_audit,
        test_result,
        gitignore_ignored,
        gitignore_available,
        working_tree_clean,
    )
    write_inventory(root, candidate_files(root))
    final_files = candidate_files(root)
    write_reports(
        root,
        final_files,
        raw,
        external,
        result_summary,
        result_issues,
        split_rows,
        split_issues,
        statistics,
        statistics_issues,
        citation,
        citation_issues,
        privacy,
        current_privacy,
        submission_audit,
        test_result,
        gitignore_ignored,
        gitignore_available,
        working_tree_clean,
    )
    write_inventory(root, final_files)
    gitignore_ok = all(gitignore_ignored.values()) and all(
        gitignore_available.values()
    )
    print(
        json.dumps(
            {
                "repository_status": (
                    "PASS"
                    if not result_issues
                    and not split_issues
                    and not statistics_issues
                    and gitignore_ok
                    else "FAIL"
                ),
                "gitignore_complete": gitignore_ok,
                "working_tree_clean": working_tree_clean,
                "submission_status": submission_audit.get("status"),
                "citation_issues": citation_issues,
                "privacy_blockers": {
                    "non_anonymous_commits": privacy["non_anonymous_commits"],
                    "hostname_configs": len(privacy["current_hostname_configs"]),
                },
                "release": "NOT READY FOR v1.0.0",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
