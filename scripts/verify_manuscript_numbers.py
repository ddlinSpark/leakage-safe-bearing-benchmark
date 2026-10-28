"""Verify paper_metrics.json and the final DOCX against canonical results."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    metrics_path = root / "results" / "paper_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    raw = pd.read_csv(root / "results" / "raw_runs.csv")
    external = pd.read_csv(root / "results" / "sota" / "raw_runs.csv")
    errors: list[str] = []
    if metrics["counts"]["canonical_rows"] != len(raw):
        errors.append("paper_metrics canonical row count differs from raw_runs.csv")
    if metrics["counts"]["external_rows"] != len(external):
        errors.append("paper_metrics external row count differs from external raw_runs.csv")
    for record in metrics["main_models"]:
        group = raw.loc[
            raw["experiment_type"].eq("main")
            & raw["method"].eq(record["method"])
        ]
        if not np.isclose(record["macro_f1_mean"], group["macro_f1"].mean(), atol=1e-12):
            errors.append(f"main metric mismatch: {record['method']}")
    for record in metrics["external_models"]:
        group = external.loc[external["method"].eq(record["method"])]
        if not np.isclose(record["macro_f1_mean"], group["macro_f1"].mean(), atol=1e-12):
            errors.append(f"external metric mismatch: {record['method']}")

    audit_path = root / "artifacts" / "local" / "verify_manuscript_numbers.json"
    completed = subprocess.run(
        [
            "python",
            str(root / "scripts" / "audit_submission.py"),
            "--json-output",
            str(audit_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if not audit_path.is_file():
        errors.append("independent manuscript audit did not produce JSON")
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            errors.append(
                f"independent manuscript audit failed with {audit.get('mismatch_count')} mismatches"
            )
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        "PASS: paper_metrics.json matches canonical result tables and the "
        "independent DOCX/table audit reports zero mismatches."
    )


if __name__ == "__main__":
    main()
