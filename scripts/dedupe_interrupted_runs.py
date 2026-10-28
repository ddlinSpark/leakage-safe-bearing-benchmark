"""Resolve duplicate rows created by an interrupted/resumed configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_runs", type=Path)
    args = parser.parse_args()
    path = args.raw_runs.resolve()
    frame = pd.read_csv(path)
    key_columns = ["experiment_type", "method", "fold", "seed", "label_ratio", "threshold"]
    duplicates = frame.loc[frame.duplicated(key_columns, keep=False)].copy()
    if duplicates.empty:
        print("No duplicate experiment keys found.")
        return
    for key, group in duplicates.groupby(key_columns, dropna=False):
        metrics = group[["macro_f1", "accuracy", "ece", "nll", "brier"]].to_numpy(
            dtype=float
        )
        if not np.allclose(metrics, metrics[0], rtol=0, atol=1e-12):
            raise ValueError(
                "Duplicate rows disagree on metrics; human review is required "
                f"for key {key}"
            )
    archive = path.with_name("raw_runs_all_attempts.csv")
    frame.to_csv(archive, index=False, lineterminator="\n")
    frame["_timestamp"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = (
        frame.sort_values("_timestamp")
        .drop_duplicates(key_columns, keep="last")
        .drop(columns="_timestamp")
        .sort_values(["method", "fold", "seed"])
    )
    frame.to_csv(path, index=False, lineterminator="\n")
    print(
        f"Archived {len(duplicates)} duplicate attempts and retained "
        f"{len(frame)} unique runs. Duplicate metrics were identical."
    )


if __name__ == "__main__":
    main()
