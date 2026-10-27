"""Sensitivity analysis for pseudo-label confidence thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from ..paths import DATA_DIR, OUTPUTS_DIR
from .cwru_data import load_cwru_cache
from .train_cwru import (
    WindowSet,
    balanced_indices,
    evaluate_model,
    load_fold_indices,
    select_pseudo_labels,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描自训练伪标签置信阈值")
    parser.add_argument(
        "--cache",
        type=Path,
        default=DATA_DIR / "public" / "paderborn" / "pu_4class_w8192.npz",
    )
    parser.add_argument(
        "--heldout-record-ids",
        default="4,8,12,15",
    )
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn_thresholds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = load_cwru_cache(args.cache)
    heldout = np.asarray(
        [int(value) for value in args.heldout_record_ids.split(",") if value.strip()],
        dtype=np.int64,
    )
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    num_classes = int(data.labels.max()) + 1

    for fold in folds:
        train_idx_all, val_idx, test_idx = load_fold_indices(data, fold, heldout)
        for seed in seeds:
            set_seed(seed)
            rng = np.random.default_rng(seed + fold * 1000)
            labeled_idx = balanced_indices(
                train_idx_all,
                data.labels,
                args.fraction,
                rng,
            )
            candidate_mask = np.ones(len(train_idx_all), dtype=bool)
            candidate_mask[np.isin(train_idx_all, labeled_idx)] = False
            candidate_idx = train_idx_all[candidate_mask]
            print(
                f"fold={fold} seed={seed} labeled={len(labeled_idx)} "
                f"unlabeled={len(candidate_idx)}"
            )
            state = train_model(
                data,
                labeled_idx,
                data.labels[labeled_idx],
                np.ones(len(labeled_idx), dtype=np.float32),
                val_idx,
                "cwt_gru",
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                num_workers=0,
                device=device,
                seed=seed,
            )
            baseline_raw, baseline_cal, _, _ = evaluate_model(
                state,
                data,
                test_idx,
                args.batch_size,
                0,
                device,
            )
            max_per_class = max(1, len(labeled_idx) // num_classes * 2)

            for threshold in thresholds:
                pseudo_idx, pseudo_labels, pseudo_weights = select_pseudo_labels(
                    state.model,
                    data,
                    candidate_idx,
                    state.temperature,
                    max_per_class=max_per_class,
                    confidence_threshold=threshold,
                    batch_size=args.batch_size,
                    num_workers=0,
                    device=device,
                )
                pseudo_count = len(pseudo_idx)
                pseudo_accuracy = (
                    float(accuracy_score(data.labels[pseudo_idx], pseudo_labels))
                    if pseudo_count
                    else float("nan")
                )
                pseudo_macro_f1 = (
                    float(
                        f1_score(
                            data.labels[pseudo_idx],
                            pseudo_labels,
                            average="macro",
                            zero_division=0,
                        )
                    )
                    if pseudo_count
                    else float("nan")
                )
                if pseudo_count == 0:
                    trained_on_pseudo = False
                    raw_metrics = baseline_raw
                    calibrated_metrics = baseline_cal
                else:
                    combined_idx = np.concatenate([labeled_idx, pseudo_idx])
                    combined_labels = np.concatenate(
                        [data.labels[labeled_idx], pseudo_labels]
                    )
                    combined_weights = np.concatenate(
                        [np.ones(len(labeled_idx), dtype=np.float32), pseudo_weights]
                    )
                    retrained = train_model(
                        data,
                        combined_idx,
                        combined_labels,
                        combined_weights,
                        val_idx,
                        "cwt_gru_selftrain",
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        patience=args.patience,
                        num_workers=0,
                        device=device,
                        seed=seed + 1,
                    )
                    raw_metrics, calibrated_metrics, _, _ = evaluate_model(
                        retrained,
                        data,
                        test_idx,
                        args.batch_size,
                        0,
                        device,
                    )
                    trained_on_pseudo = True

                rows.append(
                    {
                        "test_condition": fold,
                        "seed": seed,
                        "label_fraction": args.fraction,
                        "threshold": threshold,
                        "pseudo_count": pseudo_count,
                        "pseudo_coverage": pseudo_count / len(candidate_idx),
                        "pseudo_accuracy": pseudo_accuracy,
                        "pseudo_macro_f1": pseudo_macro_f1,
                        "trained_on_pseudo": trained_on_pseudo,
                        "baseline_accuracy": baseline_raw["accuracy"],
                        "baseline_macro_f1": baseline_raw["macro_f1"],
                        "test_accuracy": raw_metrics["accuracy"],
                        "test_macro_f1": raw_metrics["macro_f1"],
                        "calibrated_ece": calibrated_metrics["ece"],
                    }
                )
                print(
                    f"  tau={threshold:.2f} pseudo={pseudo_count} "
                    f"pseudo_acc={pseudo_accuracy:.3f} "
                    f"test_f1={raw_metrics['macro_f1']:.3f}"
                )
                pd.DataFrame(rows).to_csv(args.output_dir / "results.csv", index=False)

    pd.DataFrame(rows).to_csv(args.output_dir / "results.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
