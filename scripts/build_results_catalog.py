"""Build immutable indexes for canonical results and split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def normalized_payload(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
        payload = payload.replace(b"\r\n", b"\n")
    return payload


def sha256(path: Path) -> str:
    payload = normalized_payload(path)
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def main() -> None:
    canonical_dir = ROOT / "results" / "canonical"
    splits_dir = ROOT / "splits"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    canonical_files = [
        "results/raw_runs.csv",
        "results/summary.csv",
        "results/summary_with_ci.csv",
        "results/statistical_tests.csv",
        "results/hierarchical_statistics.csv",
        "results/domain_gap_summary.csv",
        "results/label_efficiency_summary.csv",
        "results/threshold_summary.csv",
        "results/ablation_summary.csv",
        "results/model_complexity.csv",
        "results/paper_metrics.json",
        "results/sota/raw_runs.csv",
        "results/sota/statistical_tests.csv",
    ]
    canonical = {
        "schema_version": 1,
        "status": "canonical",
        "digest_scope": "text normalized to LF",
        "row_contract": {
            "canonical_rows": 320,
            "external_rows": 40,
            "main_rows": 140,
            "label_efficiency_rows": 80,
            "threshold_rows": 100,
            "folds": [0, 1, 2, 3],
            "seeds": [42, 123, 2024, 2025, 3407],
        },
        "files": {},
    }
    for relative in canonical_files:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        record = {
            "bytes": len(normalized_payload(path)),
            "sha256": sha256(path),
        }
        if path.suffix == ".csv":
            record["rows"] = len(pd.read_csv(path))
        canonical["files"][relative] = record
    (canonical_dir / "manifest.json").write_text(
        json.dumps(canonical, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    split_files = {}
    for fold in range(4):
        path = ROOT / "results" / "splits" / f"fold{fold}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        split_files[f"results/splits/fold{fold}.json"] = {
            "bytes": len(normalized_payload(path)),
            "sha256": sha256(path),
        }
    split_manifest = {
        "schema_version": 1,
        "status": "frozen",
        "authoritative_directory": "results/splits",
        "digest_scope": "text normalized to LF",
        "files": split_files,
    }
    (splits_dir / "manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Indexed {len(canonical_files)} canonical files and 4 split manifests.")


if __name__ == "__main__":
    main()
