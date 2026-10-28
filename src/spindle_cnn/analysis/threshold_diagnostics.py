"""Reconstruct per-class pseudo-label selections for the threshold sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..datasets.splits import balanced_indices, load_fold_indices, set_global_seed
from ..models.registry import METHODS
from ..paths import PROJECT_ROOT
from ..public_benchmark.cwru_data import load_cwru_cache
from ..training.runner import select_pseudo_labels, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paderborn_main.json",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import json

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = load_cwru_cache(
        (PROJECT_ROOT / config["dataset"]["cache"]).resolve()
    )
    raw = pd.read_csv(args.results_dir / "raw_runs.csv")
    threshold_rows = raw.loc[raw["experiment_type"].eq("threshold")]
    thresholds = sorted(threshold_rows["threshold"].dropna().unique())
    fraction = float(config["experiments"]["threshold"]["fraction"])
    heldout = np.asarray(config["dataset"]["heldout_record_ids"], dtype=np.int64)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    rows = []
    for fold in config["folds"]:
        split = load_fold_indices(data, int(fold), heldout)
        for seed in config["seeds"]:
            set_global_seed(int(seed), bool(config["training"]["deterministic"]))
            rng = np.random.default_rng(int(seed) + int(fold) * 1000)
            labeled = balanced_indices(split.train, data.labels, fraction, rng)
            candidate_mask = np.ones(len(split.train), dtype=bool)
            candidate_mask[np.isin(split.train, labeled)] = False
            candidates = split.train[candidate_mask]
            state = train_model(
                data,
                labeled,
                data.labels[labeled],
                np.ones(len(labeled), dtype=np.float32),
                split.validation,
                METHODS["cwt_gru"],
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                learning_rate=config["training"]["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
                patience=config["training"]["patience"],
                num_workers=config["training"]["num_workers"],
                device=device,
                seed=int(seed),
            )
            max_per_class = max(
                1,
                len(labeled) // (int(data.labels.max()) + 1) * 2,
            )
            for threshold in thresholds:
                selection = select_pseudo_labels(
                    state.model,
                    data,
                    candidates,
                    state.temperature,
                    max_per_class=max_per_class,
                    confidence_threshold=float(threshold),
                    batch_size=config["training"]["batch_size"],
                    num_workers=config["training"]["num_workers"],
                    device=device,
                )
                selected = selection.selected_candidate_positions
                predicted = selection.candidate_predictions[selected]
                confidence = selection.candidate_confidence[selected]
                true_labels = data.labels[selection.indices]
                baseline = threshold_rows.loc[
                    threshold_rows["fold"].eq(fold)
                    & threshold_rows["seed"].eq(seed)
                    & threshold_rows["threshold"].round(6).eq(round(float(threshold), 6))
                ]
                if len(baseline) != 1:
                    raise ValueError(
                        f"Expected one stored threshold row for fold={fold}, "
                        f"seed={seed}, threshold={threshold}"
                    )
                baseline = baseline.iloc[0]
                if int(baseline["pseudo_count"]) != len(selection.indices):
                    raise ValueError(
                        "Threshold selection count does not reproduce the stored "
                        f"row: fold={fold}, seed={seed}, threshold={threshold}, "
                        f"stored={baseline['pseudo_count']}, "
                        f"recomputed={len(selection.indices)}"
                    )
                record = {
                    "fold": int(fold),
                    "seed": int(seed),
                    "threshold": float(threshold),
                    "candidate_pool_size": len(candidates),
                    "selected_count": len(selection.indices),
                    "selected_fraction": len(selection.indices) / len(candidates),
                    "confidence_mean": (
                        float(confidence.mean()) if len(confidence) else np.nan
                    ),
                    "confidence_median": (
                        float(np.median(confidence)) if len(confidence) else np.nan
                    ),
                    "pseudo_accuracy": (
                        float((data.labels[selection.indices] == selection.labels).mean())
                        if len(selection.indices)
                        else np.nan
                    ),
                    "stored_pseudo_count_matched": True,
                }
                for class_id in range(int(data.labels.max()) + 1):
                    mask = predicted == class_id
                    record[f"predicted_candidates_class_{class_id}"] = int(
                        (selection.candidate_predictions == class_id).sum()
                    )
                    record[f"selected_class_{class_id}"] = int(mask.sum())
                    record[f"coverage_class_{class_id}"] = float(
                        mask.sum()
                        / max(
                            int(
                                (
                                    selection.candidate_predictions == class_id
                                ).sum()
                            ),
                            1,
                        )
                    )
                    record[f"pseudo_accuracy_class_{class_id}"] = (
                        float((true_labels[mask] == class_id).mean())
                        if mask.any()
                        else np.nan
                    )
                rows.append(record)
    per_run = pd.DataFrame(rows)
    per_run.to_csv(
        args.results_dir / "threshold_per_class_diagnostics.csv",
        index=False,
        lineterminator="\n",
    )
    coverage_columns = [
        column
        for column in per_run.columns
        if column.startswith("coverage_class_")
    ]
    accuracy_columns = [
        column
        for column in per_run.columns
        if column.startswith("pseudo_accuracy_class_")
    ]
    summary = (
        per_run.groupby("threshold")
        .agg(
            runs=("seed", "count"),
            selected_fraction_mean=("selected_fraction", "mean"),
            pseudo_accuracy_mean=("pseudo_accuracy", "mean"),
            **{
                column: (column, "mean")
                for column in coverage_columns + accuracy_columns
            },
        )
        .reset_index()
    )
    summary.to_csv(
        args.results_dir / "threshold_per_class_summary.csv",
        index=False,
        lineterminator="\n",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
