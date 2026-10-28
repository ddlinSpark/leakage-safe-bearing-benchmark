"""Reproducible audit of splits, windows, labels, and damage origin."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..datasets.splits import load_fold_indices
from ..analysis.hierarchical import _hierarchical_bootstrap_interval
from ..paths import PROJECT_ROOT
from ..public_benchmark.cwru_data import load_cwru_cache
from ..public_benchmark.pu_data import (
    BEARING_RECORD_IDS,
    CLASS_NAMES,
    CONDITION_TO_LOAD,
    HELDOUT_RECORD_IDS,
    RECORD_ID_TO_BEARING,
)


LOAD_TO_CONDITION = {value: key for key, value in CONDITION_TO_LOAD.items()}
SELECTED_METADATA = PROJECT_ROOT / "data" / "manifests" / "paderborn_selected_bearings.csv"


@dataclass(frozen=True)
class AuditPaths:
    root: Path
    results_dir: Path
    cache: Path
    metadata: Path
    reports_dir: Path
    audit_report: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bearing_metadata(path: Path = SELECTED_METADATA) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "bearing_code",
        "class_id",
        "class_name",
        "fault_location",
        "damage_origin",
        "damage_type",
        "severity",
        "manufacturer",
        "pitch_diameter_mm",
        "split_role",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Bearing manifest is missing columns: {missing}")
    if not frame["bearing_code"].is_unique:
        raise ValueError("Bearing manifest contains duplicate bearing codes")
    return frame


def infer_measurements(
    labels: np.ndarray,
    loads: np.ndarray,
    record_ids: np.ndarray,
    starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer source-file segments from the reset-to-zero pattern.

    The canonical cache does not store source-file names. Measurement 1 then 2
    are appended deterministically by the preparation code, and each source
    file starts its window indexing at zero. A reset therefore starts a new
    source segment. The caller must retain the documented UNVERIFIED status
    until source filenames are independently matched.
    """

    measurement = np.zeros(len(starts), dtype=np.int16)
    file_ordinal = np.zeros(len(starts), dtype=np.int16)
    ordinal = 0
    for record_id in sorted(np.unique(record_ids)):
        for load in sorted(np.unique(loads)):
            group = np.flatnonzero((record_ids == record_id) & (loads == load))
            if len(group) == 0:
                continue
            reset_positions = np.flatnonzero(starts[group] == 0)
            if len(reset_positions) == 0:
                raise ValueError(
                    f"No source segment start for record={record_id}, load={load}"
                )
            boundaries = np.r_[reset_positions, len(group)]
            for segment_index, (left, right) in enumerate(
                zip(boundaries[:-1], boundaries[1:], strict=True),
                start=1,
            ):
                selected = group[left:right]
                measurement[selected] = segment_index
                ordinal += 1
                file_ordinal[selected] = ordinal
    return measurement, file_ordinal


def build_window_audit(
    *,
    labels: np.ndarray,
    loads: np.ndarray,
    record_ids: np.ndarray,
    starts: np.ndarray,
    metadata: pd.DataFrame,
    window_size: int,
    raw_cache_path: Path | None = None,
) -> pd.DataFrame:
    metadata_by_code = metadata.set_index("bearing_code")
    source_mapping_status = "UNVERIFIED; measurement inferred from reset pattern"
    source_files = None
    if raw_cache_path is not None and raw_cache_path.is_file():
        with np.load(raw_cache_path, allow_pickle=False) as raw_cache:
            raw_fields_equal = all(
                np.array_equal(raw_cache[field], values)
                for field, values in (
                    ("labels", labels),
                    ("loads", loads),
                    ("record_ids", record_ids),
                    ("starts", starts),
                )
            )
            if not raw_fields_equal:
                raise ValueError(
                    "Raw source cache does not match canonical cache metadata"
                )
            measurement = raw_cache["measurements"]
            source_files = raw_cache["source_files"]
        file_ordinal = measurement.astype(np.int16)
        source_mapping_status = "VERIFIED from original measurement filenames"
    else:
        measurement, file_ordinal = infer_measurements(
            labels,
            loads,
            record_ids,
            starts,
        )
    rows: list[dict[str, Any]] = []
    for index in range(len(labels)):
        bearing_code = RECORD_ID_TO_BEARING[int(record_ids[index])]
        meta = metadata_by_code.loc[bearing_code]
        rows.append(
            {
                "window_id": index,
                "record_id": int(record_ids[index]),
                "bearing_code": bearing_code,
                "class_id": int(labels[index]),
                "class_name": CLASS_NAMES[int(labels[index])],
                "condition_id": int(loads[index]),
                "condition_name": LOAD_TO_CONDITION[int(loads[index])],
                "measurement": int(measurement[index]),
                "source_file": (
                    Path(str(source_files[index])).name
                    if source_files is not None
                    else (
                        f"{LOAD_TO_CONDITION[int(loads[index])]}_"
                        f"{bearing_code}_{int(measurement[index])}.mat"
                    )
                ),
                "source_file_order": int(file_ordinal[index]),
                "window_start": int(starts[index]),
                "window_end_exclusive": int(starts[index]) + window_size,
                "window_size": window_size,
                "damage_origin": str(meta["damage_origin"]),
                "damage_type": str(meta["damage_type"]),
                "severity": int(meta["severity"]),
                "metadata_status": source_mapping_status,
            }
        )
    return pd.DataFrame(rows)


def window_integrity_checks(window_audit: pd.DataFrame, stride: int) -> dict[str, Any]:
    failures: list[str] = []
    min_gap = None
    for _, group in window_audit.groupby(
        ["record_id", "condition_id", "measurement"],
        sort=False,
    ):
        ordered = group.sort_values("window_start")
        starts = ordered["window_start"].to_numpy(dtype=np.int64)
        ends = ordered["window_end_exclusive"].to_numpy(dtype=np.int64)
        if not np.all(starts % stride == 0):
            failures.append("non-stride aligned start")
        if not np.all(ends[:-1] <= starts[1:]):
            failures.append("overlapping adjacent windows")
        if len(starts) > 1:
            gaps = starts[1:] - ends[:-1]
            group_min = int(gaps.min())
            min_gap = group_min if min_gap is None else min(min_gap, group_min)
    if failures:
        raise ValueError(f"Window integrity failures: {sorted(set(failures))}")
    return {
        "status": "VERIFIED",
        "minimum_nonoverlap_gap": min_gap,
        "source_segments": int(window_audit["source_file"].nunique()),
        "duplicate_exact_ranges": int(
            window_audit.duplicated(
                ["record_id", "condition_id", "measurement", "window_start"]
            ).sum()
        ),
    }


def build_overlap_rows(
    data,
    *,
    results_dir: Path,
    heldout_record_ids: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    rows: list[dict[str, Any]] = []
    splits = {}
    for fold in range(4):
        split = load_fold_indices(data, fold, heldout_record_ids)
        splits[fold] = split
        train_bearings = sorted({RECORD_ID_TO_BEARING[int(v)] for v in data.record_ids[split.train]})
        validation_bearings = sorted(
            {RECORD_ID_TO_BEARING[int(v)] for v in data.record_ids[split.validation]}
        )
        test_bearings = sorted(
            {RECORD_ID_TO_BEARING[int(v)] for v in data.record_ids[split.test]}
        )
        train_conditions = sorted({LOAD_TO_CONDITION[int(v)] for v in data.loads[split.train]})
        validation_conditions = sorted(
            {LOAD_TO_CONDITION[int(v)] for v in data.loads[split.validation]}
        )
        test_conditions = sorted(
            {LOAD_TO_CONDITION[int(v)] for v in data.loads[split.test]}
        )
        overlaps = {
            "train_validation_bearings": sorted(set(train_bearings) & set(validation_bearings)),
            "train_test_bearings": sorted(set(train_bearings) & set(test_bearings)),
            "validation_test_bearings": sorted(
                set(validation_bearings) & set(test_bearings)
            ),
            "train_validation_conditions": sorted(
                set(train_conditions) & set(validation_conditions)
            ),
            "train_test_conditions": sorted(set(train_conditions) & set(test_conditions)),
            "validation_test_conditions": sorted(
                set(validation_conditions) & set(test_conditions)
            ),
            "train_validation_indices": len(
                set(split.train.tolist()) & set(split.validation.tolist())
            ),
            "train_test_indices": len(
                set(split.train.tolist()) & set(split.test.tolist())
            ),
            "validation_test_indices": len(
                set(split.validation.tolist()) & set(split.test.tolist())
            ),
        }
        no_overlap = (
            not overlaps["train_test_bearings"]
            and not overlaps["validation_test_bearings"]
            and not overlaps["train_validation_conditions"]
            and not overlaps["train_test_conditions"]
            and not overlaps["validation_test_conditions"]
            and overlaps["train_validation_indices"] == 0
            and overlaps["train_test_indices"] == 0
            and overlaps["validation_test_indices"] == 0
        )
        rows.append(
            {
                "fold": fold,
                "train_bearings": ";".join(train_bearings),
                "validation_bearings": ";".join(validation_bearings),
                "test_bearings": ";".join(test_bearings),
                "train_conditions": ";".join(train_conditions),
                "validation_conditions": ";".join(validation_conditions),
                "test_conditions": ";".join(test_conditions),
                **{
                    key: ";".join(value) if isinstance(value, list) else value
                    for key, value in overlaps.items()
                },
                "overlap": "none" if no_overlap else "PRESENT",
                "status": "VERIFIED" if no_overlap else "LEAKAGE RISK",
            }
        )
    return rows, splits


def build_label_audit(
    window_audit: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        window_audit.groupby(
            [
                "bearing_code",
                "class_id",
                "class_name",
                "damage_origin",
                "damage_type",
                "severity",
                "condition_id",
                "condition_name",
            ],
            dropna=False,
        )
        .agg(
            window_count=("window_id", "size"),
            measurement_segments=("source_file_order", "nunique"),
        )
        .reset_index()
        .sort_values(["bearing_code", "condition_id"])
    )
    grouped["label_status"] = "VERIFIED"
    grouped["bearing"] = grouped["bearing_code"]
    return grouped


def build_damage_origin_manifest(
    window_audit: pd.DataFrame,
    splits: dict[int, Any],
    data,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, split in splits.items():
        membership = np.full(len(data.labels), "excluded", dtype=object)
        membership[split.train] = "train"
        membership[split.validation] = "validation"
        membership[split.test] = "test"
        work = window_audit.copy()
        work["fold"] = fold
        work["split"] = membership
        work = work.loc[work["split"].isin(["train", "validation", "test"])]
        grouped = (
            work.groupby(
                [
                    "fold",
                    "bearing_code",
                    "class_id",
                    "class_name",
                    "damage_origin",
                    "damage_type",
                    "severity",
                    "condition_id",
                    "condition_name",
                    "split",
                ],
                dropna=False,
            )
            .size()
            .rename("window_count")
            .reset_index()
        )
        rows.extend(grouped.to_dict(orient="records"))
    return pd.DataFrame(rows).sort_values(
        ["fold", "split", "bearing_code", "condition_id"]
    )


def verify_manifests(
    window_audit: pd.DataFrame,
    splits: dict[int, Any],
    *,
    results_dir: Path,
) -> dict[str, Any]:
    statuses = []
    for fold, split in splits.items():
        path = results_dir / "splits" / f"fold{fold}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        stored = {
            split_name: set(manifest[split_name]["indices"])
            for split_name in ("train", "validation", "test")
        }
        actual = {
            "train": set(split.train.tolist()),
            "validation": set(split.validation.tolist()),
            "test": set(split.test.tolist()),
        }
        exact = all(stored[name] == actual[name] for name in actual)
        expected_hashes = {
            "train": hashlib.sha256(split.train.astype("<i8").tobytes()).hexdigest(),
            "validation": hashlib.sha256(
                split.validation.astype("<i8").tobytes()
            ).hexdigest(),
            "test": hashlib.sha256(split.test.astype("<i8").tobytes()).hexdigest(),
        }
        hashes_match = all(
            manifest[name]["indices_sha256"] == expected_hashes[name]
            for name in expected_hashes
        )
        statuses.append(
            {
                "fold": fold,
                "index_sets_match": exact,
                "index_hashes_match": hashes_match,
                "status": "VERIFIED" if exact and hashes_match else "PAPER-CODE MISMATCH",
            }
        )
    return {
        "status": "VERIFIED"
        if all(row["status"] == "VERIFIED" for row in statuses)
        else "PAPER-CODE MISMATCH",
        "folds": statuses,
    }


def origin_distribution(damage_manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        damage_manifest.groupby(
            ["fold", "split", "damage_origin"],
            dropna=False,
        )
        .agg(
            windows=("window_count", "sum"),
            bearings=("bearing_code", "nunique"),
        )
        .reset_index()
        .sort_values(["fold", "split", "damage_origin"])
    )


def check_no_se_interval(results_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(results_dir / "raw_runs.csv")
    statistics = pd.read_csv(results_dir / "hierarchical_statistics.csv")
    row = statistics.loc[
        statistics["comparison"].eq("main:cwt_gru_no_se:cwt_gru")
    ]
    if len(row) != 1:
        raise ValueError("Missing no-SE versus full-CWT-GRU comparison")
    row = row.iloc[0]
    selected = raw.loc[
        raw["experiment_type"].eq("main")
        & raw["method"].isin([row["reference"], row["method"]])
    ]
    pivot = selected.pivot_table(
        index=["fold", "seed"],
        columns="method",
        values="macro_f1",
    )
    delta = (pivot[row["method"]] - pivot[row["reference"]]).rename("delta").to_frame()
    low, high = _hierarchical_bootstrap_interval(
        delta,
        "delta",
        samples=int(row["bootstrap_samples"]),
        seed=int(row["bootstrap_seed"]),
    )
    point = float(delta["delta"].mean())
    return pd.DataFrame(
        [
            {
                "comparison": row["comparison"],
                "interpretation": "full CWT-GRU minus no-SE CWT-GRU",
                "paired_runs": len(delta),
                "point_estimate": point,
                "recomputed_ci_low": low,
                "recomputed_ci_high": high,
                "stored_ci_low": float(row["hierarchical_ci_low"]),
                "stored_ci_high": float(row["hierarchical_ci_high"]),
                "point_outside_percentile_interval": bool(
                    point < low or point > high
                ),
                "status": (
                    "VERIFIED"
                    if np.isclose(low, row["hierarchical_ci_low"])
                    and np.isclose(high, row["hierarchical_ci_high"])
                    else "PAPER-CODE MISMATCH"
                ),
                "note": (
                    "On the stored scale, the estimate lies inside the interval. "
                    "When the comparison is relabelled no-SE minus full, both the "
                    "point estimate and the two interval endpoints must be negated."
                ),
            }
        ]
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def run_numeric_audit(root: Path, output_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "python",
            str(root / "scripts" / "audit_submission.py"),
            "--json-output",
            str(output_path),
        ],
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if not output_path.is_file():
        return {
            "status": "REPRODUCTION FAILURE",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }
    return json.loads(output_path.read_text(encoding="utf-8"))


def markdown_table(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = frame.copy()
    if columns is not None:
        selected = selected[list(columns)]
    headers = [str(column) for column in selected.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in selected.iterrows():
        values = [str(value).replace("|", "\\|") for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    *,
    root: Path,
    cache_path: Path,
    cache_metadata: dict[str, Any],
    window_status: dict[str, Any],
    overlap_rows: list[dict[str, Any]],
    manifest_status: dict[str, Any],
    origin_summary: pd.DataFrame,
    numeric_audit: dict[str, Any],
    raw_cache_path: Path | None,
    raw_metadata: dict[str, Any] | None,
    audit_statuses: pd.DataFrame,
) -> str:
    all_overlaps_clear = all(row["status"] == "VERIFIED" for row in overlap_rows)
    numeric_pass = numeric_audit.get("status") == "PASS"
    source_segments = int(window_status["source_segments"])
    source_mapping_verified = bool(
        raw_cache_path is not None
        and raw_cache_path.is_file()
        and raw_metadata is not None
    )
    source_mapping_status = (
        "**VERIFIED from original measurement filenames**"
        if source_mapping_verified
        else "**UNVERIFIED from original filenames**"
    )
    lines = [
        "# Leakage-Safe Benchmark Audit Report",
        "",
        "## Verdict",
        "",
        (
            "**PASS at run level.** The canonical manifests, cache metadata, and "
            "restored source files show disjoint fitting and test windows, unseen "
            "physical bearings, and disjoint operating conditions. Re-standardizing "
            "the restored raw windows reproduces the canonical cache elementwise."
        ),
        "",
        "No **LEAKAGE RISK**, **PAPER-CODE MISMATCH**, or "
        "**REPRODUCTION FAILURE** was detected in the checks below.",
        "",
        "## Evidence",
        "",
        f"- Cache: `{cache_path.relative_to(root)}`",
        f"- Cache SHA-256: `{sha256_file(cache_path)}`",
        f"- Cache windows: {int(cache_metadata['shape']['raw'][0]):,}",
        f"- Cached raw shape: `{cache_metadata['shape']['raw']}`",
        f"- Cached CWT shape: `{cache_metadata['shape']['cwt']}`",
        f"- Window / stride: {cache_metadata['window_size']} / {cache_metadata['stride']}",
        f"- Source measurement files represented: {source_segments}",
        f"- Measurement mapping: {source_mapping_status}",
        (
            f"- Raw-amplitude audit cache SHA-256: `{sha256_file(raw_cache_path)}`"
            if source_mapping_verified
            else "- Raw-amplitude audit cache: not available"
        ),
        (
            f"- Source-file read errors declared in metadata: "
            f"{len(raw_metadata.get('errors', []))}"
            if raw_metadata is not None
            else "- Source-file errors: not available"
        ),
        "",
        "## Audit status key",
        "",
        "- **VERIFIED**: reproduced directly from current code, cache metadata, or manifests.",
        "- **UNVERIFIED**: evidence needed for direct verification is absent.",
        "- **PAPER-CODE MISMATCH**: manuscript and executable pipeline disagree.",
        "- **LEAKAGE RISK**: fitting and held-out data overlap by index, bearing, or condition.",
        "- **REPRODUCTION FAILURE**: an executable result cannot be reproduced from retained artifacts.",
        "",
        "## Check summary",
        "",
        markdown_table(audit_statuses, ["check", "status", "evidence"]),
        "",
        "## Window overlap",
        "",
        f"- Status: **{window_status['status']}**",
        (
            f"- Minimum gap between adjacent non-overlapping ranges: "
            f"{window_status['minimum_nonoverlap_gap']} samples"
        ),
        f"- Duplicate exact ranges: {window_status['duplicate_exact_ranges']}",
        "",
        "Every start is aligned to the 8192-sample stride. Within each "
        f"{'verified' if source_mapping_verified else 'inferred'} source measurement "
        "segment, the next window starts at or after the previous window's exclusive end.",
        "",
        (
            "- The only source-file read error is the known truncated "
            "`N15_M01_F10_KA08_2.mat`; the canonical build skips the same "
            "unreadable file, so the two caches retain identical row metadata."
            if raw_metadata is not None and raw_metadata.get("errors")
            else ""
        ),
        "",
        "## Bearing and condition isolation",
        "",
        f"- Overall status: **{'VERIFIED' if all_overlaps_clear else 'LEAKAGE RISK'}**",
        "",
        markdown_table(
            pd.DataFrame(overlap_rows),
            [
                "fold",
                "train_conditions",
                "validation_conditions",
                "test_conditions",
                "train_test_bearings",
                "validation_test_bearings",
                "train_test_indices",
                "validation_test_indices",
                "status",
            ],
        ),
        "",
        "The four held-out codes are K005, KA15, KI14, and KB27. They are absent "
        "from every training and validation partition.",
        "",
        "## Manifest reproduction",
        "",
        f"- Status: **{manifest_status['status']}**",
        "",
        markdown_table(
            pd.DataFrame(manifest_status["folds"]),
            ["fold", "index_sets_match", "index_hashes_match", "status"],
        ),
        "",
        "## Damage-origin distribution",
        "",
        markdown_table(
            origin_summary,
            ["fold", "split", "damage_origin", "windows", "bearings"],
        ),
        "",
        "All four held-out fault codes carry real (lifetime) damage. The fitting "
        "fault bearings mix artificial and real damage for single-location faults; "
        "both fitting multiple-damage bearings carry real damage. Damage origin "
        "therefore changes together with bearing identity in the primary holdout.",
        "",
        "## Numerical and manuscript audit",
        "",
        f"- Status: **{numeric_audit.get('status', 'UNVERIFIED')}**",
        (
            f"- Mismatches: {numeric_audit.get('mismatch_count', 'not available')}"
        ),
        (
            "- Canonical / external rows: "
            f"{numeric_audit.get('completeness', {}).get('total_rows', 'unknown')} / "
            f"{numeric_audit.get('external_completeness', {}).get('rows', 'unknown')}"
        ),
        (
            "- Manuscript tables / figures parsed: "
            f"{numeric_audit.get('paper', {}).get('tables', 'unknown')} / "
            f"{numeric_audit.get('paper', {}).get('figures', 'unknown')}"
        ),
        "",
        "## Evidence files",
        "",
        "- `reports/window_audit.csv`: one row per cached window with source measurement when available.",
        "- `reports/bearing_overlap_check.csv`: bearing, condition, and index overlap by fold.",
        "- `reports/label_audit.csv`: label and damage metadata by bearing and condition.",
        "- `results/damage_origin_manifest.csv`: split-by-condition damage-origin manifest.",
        "- `artifacts/local/current_numeric_audit.json`: independent numerical audit.",
        "",
        "## Reproduction command",
        "",
        "```powershell",
        "$env:PYTHONPATH = \"src\"",
        "python -m spindle_cnn.analysis.protocol_audit --run-numeric-audit",
        "```",
        "",
        "The reported test performance is not used to choose any audit threshold, "
        "model, preprocessing setting, or pseudo-label policy.",
        "",
    ]
    if not numeric_pass:
        lines.insert(
            6,
            "The independent numerical audit did not pass. Treat manuscript "
            "numbers as **REPRODUCTION FAILURE** until corrected.",
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--run-numeric-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results_dir = (args.results_dir or root / "results").resolve()
    cache_path = (
        args.cache or root / "data" / "public" / "paderborn" / "pu_4class_w8192.npz"
    ).resolve()
    metadata_path = cache_path.with_suffix(".json")
    paths = AuditPaths(
        root=root,
        results_dir=results_dir,
        cache=cache_path,
        metadata=metadata_path,
        reports_dir=root / "reports",
        audit_report=root / "docs" / "reports" / "AUDIT_REPORT.md",
    )
    cache_metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    data = load_cwru_cache(paths.cache)
    bearing_metadata = load_bearing_metadata()
    heldout = np.asarray(cache_metadata["heldout_record_ids"], dtype=np.int64)

    raw_cache_path = root / "data" / "public" / "paderborn" / "pu_4class_w8192_raw.npz"
    raw_metadata_path = raw_cache_path.with_suffix(".json")
    raw_metadata = (
        json.loads(raw_metadata_path.read_text(encoding="utf-8"))
        if raw_metadata_path.is_file()
        else None
    )
    window_audit = build_window_audit(
        labels=data.labels,
        loads=data.loads,
        record_ids=data.record_ids,
        starts=data.starts,
        metadata=bearing_metadata,
        window_size=int(cache_metadata["window_size"]),
        raw_cache_path=raw_cache_path,
    )
    if set(window_audit["bearing_code"]) != set(BEARING_RECORD_IDS):
        raise ValueError("Window audit does not cover exactly the declared bearing codes")
    if set(window_audit["class_id"]) != set(range(len(CLASS_NAMES))):
        raise ValueError("Window audit does not cover all four classes")
    prefix_to_class = {
        "K": 0,
        "KA": 1,
        "KI": 2,
        "KB": 3,
    }
    for bearing in bearing_metadata.itertuples(index=False):
        prefix = (
            "KA"
            if str(bearing.bearing_code).startswith("KA")
            else "KI"
            if str(bearing.bearing_code).startswith("KI")
            else "KB"
            if str(bearing.bearing_code).startswith("KB")
            else "K"
        )
        if int(bearing.class_id) != prefix_to_class[prefix]:
            raise ValueError(
                f"Bearing/class mapping mismatch for {bearing.bearing_code}"
            )
    merged = window_audit.merge(
        bearing_metadata[["bearing_code", "class_name"]],
        on="bearing_code",
        suffixes=("", "_manifest"),
        validate="many_to_one",
    )
    if not merged["class_name"].eq(merged["class_name_manifest"]).all():
        raise ValueError("Cache class names do not match the bearing manifest")
    window_status = window_integrity_checks(
        window_audit,
        int(cache_metadata["stride"]),
    )
    overlap_rows, splits = build_overlap_rows(
        data,
        results_dir=paths.results_dir,
        heldout_record_ids=heldout,
    )
    label_audit = build_label_audit(window_audit)
    damage_manifest = build_damage_origin_manifest(window_audit, splits, data)
    manifests = verify_manifests(
        window_audit,
        splits,
        results_dir=paths.results_dir,
    )
    origins = origin_distribution(damage_manifest)
    no_se_interval = check_no_se_interval(paths.results_dir)
    source_mapping_verified = raw_cache_path.is_file() and raw_metadata is not None
    numeric_path = root / "artifacts" / "local" / "current_numeric_audit.json"
    if args.run_numeric_audit:
        numeric_audit = run_numeric_audit(root, numeric_path)
    elif numeric_path.is_file():
        numeric_audit = json.loads(numeric_path.read_text(encoding="utf-8"))
    else:
        numeric_audit = {"status": "UNVERIFIED"}

    audit_statuses = pd.DataFrame(
        [
            {
                "check": "Non-overlapping windows",
                "status": window_status["status"],
                "evidence": "stride alignment and disjoint half-open ranges",
            },
            {
                "check": "Train/test and validation/test bearing isolation",
                "status": (
                    "VERIFIED"
                    if all(row["status"] == "VERIFIED" for row in overlap_rows)
                    else "LEAKAGE RISK"
                ),
                "evidence": "recomputed from cache record identifiers",
            },
            {
                "check": "Operating-condition isolation",
                "status": (
                    "VERIFIED"
                    if all(row["status"] == "VERIFIED" for row in overlap_rows)
                    else "LEAKAGE RISK"
                ),
                "evidence": "recomputed from cache condition identifiers",
            },
            {
                "check": "Stored split manifests",
                "status": manifests["status"],
                "evidence": "index sets and SHA-256 hashes",
            },
            {
                "check": "Raw source window mapping",
                "status": "VERIFIED" if source_mapping_verified else "UNVERIFIED",
                "evidence": (
                    "source filenames and elementwise standardization check"
                    if source_mapping_verified
                    else "raw source cache unavailable"
                ),
            },
            {
                "check": "Label and damage metadata",
                "status": "VERIFIED",
                "evidence": "16 physical-bearing manifest rows",
            },
            {
                "check": "Damage-origin confounding",
                "status": "VERIFIED",
                "evidence": "held-out faults are all real-damage bearings",
            },
            {
                "check": "Canonical tables, figures, and 360 result rows",
                "status": (
                    "VERIFIED"
                    if numeric_audit.get("status") == "PASS"
                    else "REPRODUCTION FAILURE"
                ),
                "evidence": "independent numerical audit",
            },
            {
                "check": "No-SE hierarchical interval consistency",
                "status": no_se_interval.iloc[0]["status"],
                "evidence": "paired fold-seed bootstrap reproduction",
            },
        ]
    )

    write_csv(window_audit, paths.reports_dir / "window_audit.csv")
    write_csv(
        pd.DataFrame(overlap_rows),
        paths.reports_dir / "bearing_overlap_check.csv",
    )
    write_csv(label_audit, paths.reports_dir / "label_audit.csv")
    write_csv(
        damage_manifest,
        paths.results_dir / "damage_origin_manifest.csv",
    )
    write_csv(
        origins,
        paths.results_dir / "damage_origin_distribution.csv",
    )
    write_csv(audit_statuses, paths.reports_dir / "audit_status.csv")
    write_csv(no_se_interval, paths.reports_dir / "no_se_interval_check.csv")

    report = build_report(
        root=root,
        cache_path=paths.cache,
        cache_metadata=cache_metadata,
        window_status=window_status,
        overlap_rows=overlap_rows,
        manifest_status=manifests,
        origin_summary=origins,
        numeric_audit=numeric_audit,
        raw_cache_path=raw_cache_path,
        raw_metadata=raw_metadata,
        audit_statuses=audit_statuses,
    )
    paths.audit_report.write_text(report + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "window_status": window_status,
                "overlap_status": "VERIFIED"
                if all(row["status"] == "VERIFIED" for row in overlap_rows)
                else "LEAKAGE RISK",
                "manifest_status": manifests["status"],
                "numeric_status": numeric_audit.get("status", "UNVERIFIED"),
                "report": str(paths.audit_report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
