"""Unified runner for main, ablation, label-budget, and threshold experiments."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from ..datasets.splits import (
    balanced_indices,
    build_split_manifest,
    load_fold_indices,
    set_global_seed,
    write_json,
)
from ..evaluation.calibration import probabilities_from_logits
from ..evaluation.efficiency import count_parameters, estimate_model_gflops
from ..evaluation.metrics import calibration_bin_table
from ..models import build_model
from ..models.registry import METHODS, MethodSpec, validate_methods
from ..paths import PROJECT_ROOT
from ..public_benchmark.cwru_data import load_cwru_cache
from ..training.runner import (
    TrainState,
    evaluate_model,
    run_self_training,
    select_pseudo_labels,
    train_model,
)
from .common import (
    RESULT_COLUMNS,
    append_result,
    atomic_write_csv,
    atomic_write_json,
    configure_logging,
    ensure_clean_worktree,
    environment_snapshot,
    file_sha256,
    has_exact_run,
    load_json_config,
    read_results,
    relative_or_absolute,
    sanitize_identifier,
    stable_json_hash,
    utc_now,
)


MAIN_METHODS = [
    "raw_cnn",
    "gru_only",
    "cwt_cnn",
    "cwt_gru_no_se",
    "cwt_gru",
    "cwt_gru_temp",
    "cwt_gru_selftrain",
]

ABLATION_METHODS = [
    "cwt_cnn",
    "gru_only",
    "cwt_gru_no_se",
    "cwt_gru",
    "cwt_gru_temp",
    "cwt_gru_selftrain",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the leakage-safe Paderborn benchmark"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paderborn_main.json",
    )
    parser.add_argument(
        "--experiment",
        choices=["main", "label_efficiency", "threshold", "ablation", "all"],
        default="main",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds")
    parser.add_argument("--folds")
    parser.add_argument("--methods")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse only exact run_id + config_hash matches.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Mark exploratory runs from a dirty worktree.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_csv_ints(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    if args.smoke:
        config["seeds"] = [42]
        config["folds"] = [3]
        config["training"]["epochs"] = 1
        config["training"]["patience"] = 1
        config["training"]["pseudo_rounds"] = 1
        config["experiments"]["label_efficiency"]["fractions"] = [0.1]
        config["experiments"]["threshold"]["thresholds"] = [0.4, 0.6]
    if args.seeds:
        config["seeds"] = parse_csv_ints(args.seeds)
    if args.folds:
        config["folds"] = parse_csv_ints(args.folds)
    if args.methods and args.experiment != "threshold":
        config["experiments"][
            "main" if args.experiment in {"main", "ablation"} else "label_efficiency"
        ]["methods"] = [
            value.strip() for value in args.methods.split(",") if value.strip()
        ]
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.patience is not None:
        config["training"]["patience"] = args.patience
    if args.device:
        config["training"]["device"] = args.device
    return config


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def model_signature(
    method_spec: MethodSpec,
    *,
    data,
    batch_size: int,
    device: torch.device,
) -> tuple[int, float]:
    model = build_model(
        method_spec.model_key,
        num_classes=int(data.labels.max()) + 1,
        raw_channels=data.raw.shape[1],
        image_channels=data.cwt.shape[1],
    ).to(device)
    raw = torch.from_numpy(data.raw[:1]).to(device)
    cwt = torch.from_numpy(data.cwt[:1]).float().to(device)
    gflops = estimate_model_gflops(
        model,
        raw,
        cwt,
        model_key=method_spec.model_key,
    )
    return count_parameters(model), gflops


def run_id_for(
    *,
    experiment_type: str,
    fold: int,
    seed: int,
    method: str,
    label_ratio: float,
    threshold: float | None,
    config_hash: str,
) -> str:
    pieces = [
        experiment_type,
        f"f{fold}",
        f"s{seed}",
        method,
        f"l{label_ratio:.4f}",
    ]
    if threshold is not None and math.isfinite(threshold):
        pieces.append(f"t{threshold:.4f}")
    pieces.append(config_hash[:10])
    return sanitize_identifier("__".join(pieces))


def build_run_config(
    *,
    config: dict[str, Any],
    experiment_type: str,
    method: str,
    method_spec: MethodSpec,
    fold: int,
    seed: int,
    label_ratio: float,
    threshold: float | None,
    split_hash: str,
    dataset_hash: str,
    provenance: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config_schema_version": 1,
        "experiment_type": experiment_type,
        "dataset": config["dataset"],
        "experiment": config["experiments"].get(experiment_type, {}),
        "run": {
            "method": method,
            "model_key": method_spec.model_key,
            "uses_temperature": method_spec.uses_temperature,
            "uses_self_training": method_spec.uses_self_training,
            "fold": fold,
            "seed": seed,
            "label_ratio": label_ratio,
            "threshold": threshold,
            "split_hash": split_hash,
            "dataset_hash": dataset_hash,
        },
        "training": config["training"],
        "provenance": provenance,
        "environment": environment,
    }


def write_diagnostics(
    run_dir: Path,
    *,
    state: TrainState,
    logits: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    run_id: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(run_dir / "training_history.csv", pd.DataFrame(state.history))
    np.savez_compressed(
        run_dir / "predictions.npz",
        logits=logits,
        labels=labels,
        predictions=predictions,
        temperature=state.temperature,
    )

    probabilities = probabilities_from_logits(logits, state.temperature)
    raw_probabilities = probabilities_from_logits(logits, 1.0)
    calibration_rows = []
    for source, values in (("raw", raw_probabilities), ("temperature", probabilities)):
        for row in calibration_bin_table(values, labels):
            calibration_rows.append({"run_id": run_id, "source": source, **row})
    atomic_write_csv(
        run_dir / "calibration_bins.csv",
        pd.DataFrame(calibration_rows),
    )

    num_classes = logits.shape[1]
    counts = confusion_matrix(labels, predictions, labels=np.arange(num_classes))
    confusion_rows = [
        {
            "run_id": run_id,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "count": int(counts[true_class, predicted_class]),
        }
        for true_class in range(num_classes)
        for predicted_class in range(num_classes)
    ]
    atomic_write_csv(run_dir / "confusion_matrix.csv", pd.DataFrame(confusion_rows))


def result_row(
    *,
    config: dict[str, Any],
    experiment_type: str,
    method: str,
    method_spec: MethodSpec,
    fold: int,
    seed: int,
    label_ratio: float,
    threshold: float | None,
    raw_metrics: dict[str, float],
    calibrated_metrics: dict[str, float],
    state: TrainState,
    train_time_s: float,
    timing: dict[str, float],
    parameters: int,
    gflops_sample: float,
    labeled_windows: int,
    unlabeled_windows: int,
    train_windows: int,
    validation_windows: int,
    test_windows: int,
    pseudo_metadata: dict[str, float | int],
    config_hash: str,
    split_hash: str,
    dataset_hash: str,
    provenance: dict[str, Any],
    config_path: str,
) -> dict[str, Any]:
    run_id = run_id_for(
        experiment_type=experiment_type,
        fold=fold,
        seed=seed,
        method=method,
        label_ratio=label_ratio,
        threshold=threshold,
        config_hash=config_hash,
    )
    calibrated_metrics = dict(calibrated_metrics)
    base = {
        "run_id": run_id,
        "experiment_type": experiment_type,
        "dataset": config["dataset"]["name"],
        "protocol": config["dataset"]["protocol"],
        "split_version": config["dataset"]["split_version"],
        "method": method,
        "model": method_spec.model_key,
        "model_key": method_spec.model_key,
        "evaluation_mode": (
            "temperature_scaling"
            if method_spec.uses_temperature
            else "uncalibrated_decision"
        ),
        "fold": fold,
        "seed": seed,
        "label_ratio": label_ratio,
        "threshold": np.nan if threshold is None else threshold,
        "macro_f1": calibrated_metrics["macro_f1"],
        "accuracy": calibrated_metrics["accuracy"],
        "macro_precision": calibrated_metrics["macro_precision"],
        "macro_recall": calibrated_metrics["macro_recall"],
        "ece": calibrated_metrics["ece"],
        "nll": calibrated_metrics["nll"],
        "brier": calibrated_metrics["brier"],
        "raw_macro_f1": raw_metrics["macro_f1"],
        "raw_accuracy": raw_metrics["accuracy"],
        "raw_ece": raw_metrics["ece"],
        "raw_nll": raw_metrics["nll"],
        "raw_brier": raw_metrics["brier"],
        "train_time_s": train_time_s,
        "train_time": train_time_s,
        **timing,
        "inference_time": timing["inference_time_ms"],
        "parameters": parameters,
        "gflops_sample": gflops_sample,
        "temperature": state.temperature,
        "train_loss": state.last_train_loss,
        "best_val_loss": state.best_val_loss,
        "best_val_macro_f1": state.best_val_f1,
        "epochs_ran": state.epochs_ran,
        "best_epoch": state.best_epoch,
        "labeled_windows": labeled_windows,
        "unlabeled_candidate_windows": unlabeled_windows,
        "train_windows": train_windows,
        "validation_windows": validation_windows,
        "test_windows": test_windows,
        "pseudo_count": pseudo_metadata.get("pseudo_count", 0),
        "pseudo_coverage": pseudo_metadata.get("pseudo_coverage", 0.0),
        "pseudo_accuracy": pseudo_metadata.get("pseudo_accuracy", np.nan),
        "pseudo_macro_f1": pseudo_metadata.get("pseudo_macro_f1", np.nan),
        "self_training_rounds_completed": pseudo_metadata.get(
            "self_training_rounds_completed",
            0,
        ),
        "config_hash": config_hash,
        "split_hash": split_hash,
        "dataset_hash": dataset_hash,
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "config_path": config_path,
        "status": "success",
        "failure_reason": "",
        "timestamp_utc": utc_now(),
    }
    return {column: base.get(column, np.nan) for column in RESULT_COLUMNS}


def failed_result_row(
    *,
    config: dict[str, Any],
    experiment_type: str,
    method: str,
    method_spec: MethodSpec,
    fold: int,
    seed: int,
    label_ratio: float,
    threshold: float | None,
    config_hash: str,
    split_hash: str,
    dataset_hash: str,
    provenance: dict[str, Any],
    config_path: str,
    reason: str,
) -> dict[str, Any]:
    run_id = run_id_for(
        experiment_type=experiment_type,
        fold=fold,
        seed=seed,
        method=method,
        label_ratio=label_ratio,
        threshold=threshold,
        config_hash=config_hash,
    )
    base = {
        "run_id": run_id,
        "experiment_type": experiment_type,
        "dataset": config["dataset"]["name"],
        "protocol": config["dataset"]["protocol"],
        "split_version": config["dataset"]["split_version"],
        "method": method,
        "model": method_spec.model_key,
        "model_key": method_spec.model_key,
        "evaluation_mode": (
            "temperature_scaling"
            if method_spec.uses_temperature
            else "uncalibrated_decision"
        ),
        "fold": fold,
        "seed": seed,
        "label_ratio": label_ratio,
        "threshold": np.nan if threshold is None else threshold,
        "config_hash": config_hash,
        "split_hash": split_hash,
        "dataset_hash": dataset_hash,
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "config_path": config_path,
        "status": "failed",
        "failure_reason": reason,
        "timestamp_utc": utc_now(),
    }
    return {column: base.get(column, np.nan) for column in RESULT_COLUMNS}


def train_and_evaluate(
    *,
    data,
    split,
    experiment_type: str,
    method: str,
    method_spec: MethodSpec,
    fold: int,
    seed: int,
    label_ratio: float,
    confidence_threshold: float,
    config: dict[str, Any],
    device: torch.device,
    results_path: Path,
    runs_dir: Path,
    split_hash: str,
    dataset_hash: str,
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], TrainState]:
    set_global_seed(seed, config["training"]["deterministic"])
    rng = np.random.default_rng(seed + fold * 1000)
    labeled_idx = balanced_indices(split.train, data.labels, label_ratio, rng)
    candidate_mask = np.ones(len(split.train), dtype=bool)
    candidate_mask[np.isin(split.train, labeled_idx)] = False
    candidate_idx = split.train[candidate_mask]

    threshold = confidence_threshold if method_spec.uses_self_training else None
    provisional_config = {
        "config": config,
        "method": method,
        "fold": fold,
        "seed": seed,
        "label_ratio": label_ratio,
        "threshold": threshold,
        "split_hash": split_hash,
        "dataset_hash": dataset_hash,
        "provenance": provenance,
    }
    config_hash = stable_json_hash(provisional_config)
    run_id = run_id_for(
        experiment_type=experiment_type,
        fold=fold,
        seed=seed,
        method=method,
        label_ratio=label_ratio,
        threshold=threshold,
        config_hash=config_hash,
    )
    run_dir = runs_dir / run_id
    run_config = build_run_config(
        config=config,
        experiment_type=experiment_type,
        method=method,
        method_spec=method_spec,
        fold=fold,
        seed=seed,
        label_ratio=label_ratio,
        threshold=threshold,
        split_hash=split_hash,
        dataset_hash=dataset_hash,
        provenance=provenance,
        environment=environment_snapshot(device),
    )
    config_path = run_dir / "config.json"
    atomic_write_json(config_path, run_config)

    if method_spec.uses_self_training:
        start = time.perf_counter()
        state, pseudo_metadata = run_self_training(
            data,
            labeled_idx,
            candidate_idx,
            split.validation,
            method_spec,
            rounds=config["training"]["pseudo_rounds"],
            confidence_threshold=confidence_threshold,
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            patience=config["training"]["patience"],
            num_workers=config["training"]["num_workers"],
            device=device,
            seed=seed,
        )
        train_time_s = time.perf_counter() - start
    else:
        start = time.perf_counter()
        state = train_model(
            data,
            labeled_idx,
            data.labels[labeled_idx],
            np.ones(len(labeled_idx), dtype=np.float32),
            split.validation,
            method_spec,
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            patience=config["training"]["patience"],
            num_workers=config["training"]["num_workers"],
            device=device,
            seed=seed,
        )
        train_time_s = time.perf_counter() - start
        pseudo_metadata = {}

    parameters = count_parameters(state.model)
    sample_raw = torch.from_numpy(data.raw[:1]).to(device)
    sample_cwt = torch.from_numpy(data.cwt[:1]).float().to(device)
    gflops_sample = estimate_model_gflops(
        state.model,
        sample_raw,
        sample_cwt,
        model_key=method_spec.model_key,
    )
    (
        raw_metrics,
        calibrated_metrics,
        logits,
        labels,
        predictions,
        timing,
    ) = evaluate_model(
        state,
        data,
        split.test,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        device=device,
    )
    write_diagnostics(
        run_dir,
        state=state,
        logits=logits,
        labels=labels,
        predictions=predictions,
        run_id=run_id,
    )
    row = result_row(
        config=config,
        experiment_type=experiment_type,
        method=method,
        method_spec=method_spec,
        fold=fold,
        seed=seed,
        label_ratio=label_ratio,
        threshold=threshold,
        raw_metrics=raw_metrics,
        calibrated_metrics=calibrated_metrics,
        state=state,
        train_time_s=train_time_s,
        timing=timing,
        parameters=parameters,
        gflops_sample=gflops_sample,
        labeled_windows=len(labeled_idx),
        unlabeled_windows=len(candidate_idx),
        train_windows=len(split.train),
        validation_windows=len(split.validation),
        test_windows=len(split.test),
        pseudo_metadata=pseudo_metadata,
        config_hash=config_hash,
        split_hash=split_hash,
        dataset_hash=dataset_hash,
        provenance=provenance,
        config_path=relative_or_absolute(config_path),
    )
    append_result(results_path, row)
    return row, state


def summarize_results(output_dir: Path) -> None:
    raw = read_results(output_dir / "raw_runs.csv")
    if raw.empty:
        return
    main_full = raw.loc[
        raw["experiment_type"].eq("main")
        & raw["label_ratio"].eq(1.0)
        & raw["method"].isin(["cwt_gru", "cwt_gru_selftrain"])
    ].copy()
    main_full["experiment_type"] = "label_efficiency"
    main_full["run_id"] = main_full["run_id"].astype(str) + "__label100"
    label_standard = pd.concat(
        [
            raw.loc[raw["experiment_type"].eq("main")],
            raw.loc[raw["experiment_type"].eq("label_efficiency")],
            main_full,
        ],
        ignore_index=True,
    ).drop_duplicates("run_id")
    standard = label_standard[
        [
            "experiment_type",
            "method",
            "model_key",
            "label_ratio",
            "macro_f1",
            "accuracy",
        ]
    ].copy()
    standard = standard.rename(columns={"macro_f1": "macro_f1_primary"})
    grouped = (
        standard.groupby(
            ["experiment_type", "method", "model_key", "label_ratio"],
            dropna=False,
        )
        .agg(
            Macro_F1_Mean=("macro_f1_primary", "mean"),
            Macro_F1_Std=("macro_f1_primary", "std"),
            Accuracy_Mean=("accuracy", "mean"),
            Accuracy_Std=("accuracy", "std"),
            Runs=("macro_f1_primary", "count"),
        )
        .reset_index()
        .rename(columns={"method": "Model"})
    )
    atomic_write_csv(output_dir / "summary.csv", grouped)

    threshold = raw.loc[raw["experiment_type"].eq("threshold")].copy()
    if not threshold.empty:
        threshold_summary = (
            threshold.groupby(["method", "threshold"], dropna=False)
            .agg(
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                ece_mean=("ece", "mean"),
                ece_std=("ece", "std"),
                pseudo_count_mean=("pseudo_count", "mean"),
                pseudo_count_std=("pseudo_count", "std"),
                pseudo_coverage_mean=("pseudo_coverage", "mean"),
                pseudo_coverage_std=("pseudo_coverage", "std"),
                pseudo_accuracy_mean=("pseudo_accuracy", "mean"),
                pseudo_accuracy_std=("pseudo_accuracy", "std"),
                runs=("seed", "count"),
            )
            .reset_index()
            .sort_values("threshold")
        )
        atomic_write_csv(output_dir / "threshold_summary.csv", threshold_summary)

    complexity = (
        raw.groupby(["method", "model_key"], dropna=False)
        .agg(
            parameters=("parameters", "median"),
            gflops_sample=("gflops_sample", "median"),
            train_time_s_median=("train_time_s", "median"),
            inference_ms_per_batch_median=("inference_ms_per_batch", "median"),
            runs=("seed", "count"),
        )
        .reset_index()
    )
    atomic_write_csv(output_dir / "model_complexity.csv", complexity)


def run_main_experiment(
    config: dict[str, Any],
    *,
    methods: list[str],
    fractions: list[float],
    experiment_type: str,
    output_dir: Path,
    data,
    device: torch.device,
    provenance: dict[str, Any],
    dataset_hash: str,
    resume: bool,
) -> None:
    validate_methods(methods)
    results_path = output_dir / "raw_runs.csv"
    results = read_results(results_path)
    runs_dir = output_dir / "runs"
    heldout = np.asarray(config["dataset"]["heldout_record_ids"], dtype=np.int64)
    for fold in parse_csv_ints(config["folds"]):
        split = load_fold_indices(data, fold, heldout)
        manifest = build_split_manifest(
            data,
            split,
            dataset_name=config["dataset"]["name"],
            split_version=config["dataset"]["split_version"],
            fold=fold,
            heldout_record_ids=heldout,
        )
        split_path = output_dir / "splits" / f"fold{fold}.json"
        split_hash = write_json(split_path, manifest)

        for seed in parse_csv_ints(config["seeds"]):
            computed_cwt_gru: TrainState | None = None
            full_methods = list(methods)
            for method in full_methods:
                spec = METHODS[method]
                for label_ratio in fractions:
                    # Temperature scaling is an evaluation of the same trained
                    # network, not a new architectural training run.
                    reuse_base = (
                        method == "cwt_gru_temp"
                        and "cwt_gru" in full_methods
                        and computed_cwt_gru is not None
                    )
                    provisional = {
                        "config": config,
                        "method": method,
                        "fold": fold,
                        "seed": seed,
                        "label_ratio": label_ratio,
                        "threshold": (
                            config["training"]["pseudo_threshold"]
                            if spec.uses_self_training
                            else None
                        ),
                        "split_hash": split_hash,
                        "dataset_hash": dataset_hash,
                        "provenance": provenance,
                    }
                    config_hash = stable_json_hash(provisional)
                    run_id = run_id_for(
                        experiment_type=experiment_type,
                        fold=fold,
                        seed=seed,
                        method=method,
                        label_ratio=label_ratio,
                        threshold=provisional["threshold"],
                        config_hash=config_hash,
                    )
                    if resume and has_exact_run(
                        results,
                        run_id=run_id,
                        config_hash=config_hash,
                    ):
                        logging.info("Reusing exact run %s", run_id)
                        continue
                    if reuse_base:
                        logging.info(
                            "Reusing CWT-GRU weights for temperature-scaled evaluation"
                        )
                        # For the current main set, the base row and temperature
                        # row share numerical outputs. Re-evaluate the saved state
                        # and write the separately auditable row.
                        state = computed_cwt_gru
                        run_dir = runs_dir / run_id
                        run_config = build_run_config(
                            config=config,
                            experiment_type=experiment_type,
                            method=method,
                            method_spec=spec,
                            fold=fold,
                            seed=seed,
                            label_ratio=label_ratio,
                            threshold=provisional["threshold"],
                            split_hash=split_hash,
                            dataset_hash=dataset_hash,
                            provenance=provenance,
                            environment=environment_snapshot(device),
                        )
                        config_path = run_dir / "config.json"
                        atomic_write_json(config_path, run_config)
                        (
                            raw_metrics,
                            calibrated_metrics,
                            logits,
                            labels,
                            predictions,
                            timing,
                        ) = evaluate_model(
                            state,
                            data,
                            split.test,
                            batch_size=config["training"]["batch_size"],
                            num_workers=config["training"]["num_workers"],
                            device=device,
                        )
                        write_diagnostics(
                            run_dir,
                            state=state,
                            logits=logits,
                            labels=labels,
                            predictions=predictions,
                            run_id=run_id,
                        )
                        baseline = results.loc[
                            results["run_id"].astype(str).str.startswith(
                                run_id_for(
                                    experiment_type=experiment_type,
                                    fold=fold,
                                    seed=seed,
                                    method="cwt_gru",
                                    label_ratio=label_ratio,
                                    threshold=None,
                                    config_hash=stable_json_hash(
                                        {
                                            "config": config,
                                            "method": "cwt_gru",
                                            "fold": fold,
                                            "seed": seed,
                                            "label_ratio": label_ratio,
                                            "threshold": None,
                                            "split_hash": split_hash,
                                            "dataset_hash": dataset_hash,
                                            "provenance": provenance,
                                        }
                                    ),
                                )
                            )
                        ]
                        train_time_s = (
                            float(baseline.iloc[-1]["train_time_s"])
                            if not baseline.empty
                            else float("nan")
                        )
                        parameters = count_parameters(state.model)
                        sample_raw = torch.from_numpy(data.raw[:1]).to(device)
                        sample_cwt = torch.from_numpy(data.cwt[:1]).float().to(device)
                        gflops_sample = estimate_model_gflops(
                            state.model,
                            sample_raw,
                            sample_cwt,
                            model_key=spec.model_key,
                        )
                        rng = np.random.default_rng(seed + fold * 1000)
                        labeled_idx = balanced_indices(
                            split.train,
                            data.labels,
                            label_ratio,
                            rng,
                        )
                        candidate_count = len(split.train) - len(labeled_idx)
                        row = result_row(
                            config=config,
                            experiment_type=experiment_type,
                            method=method,
                            method_spec=spec,
                            fold=fold,
                            seed=seed,
                            label_ratio=label_ratio,
                            threshold=provisional["threshold"],
                            raw_metrics=raw_metrics,
                            calibrated_metrics=calibrated_metrics,
                            state=state,
                            train_time_s=train_time_s,
                            timing=timing,
                            parameters=parameters,
                            gflops_sample=gflops_sample,
                            labeled_windows=len(labeled_idx),
                            unlabeled_windows=candidate_count,
                            train_windows=len(split.train),
                            validation_windows=len(split.validation),
                            test_windows=len(split.test),
                            pseudo_metadata={},
                            config_hash=config_hash,
                            split_hash=split_hash,
                            dataset_hash=dataset_hash,
                            provenance=provenance,
                            config_path=relative_or_absolute(config_path),
                        )
                        append_result(results_path, row)
                        results = read_results(results_path)
                        continue

                    try:
                        row, state = train_and_evaluate(
                            data=data,
                            split=split,
                            experiment_type=experiment_type,
                            method=method,
                            method_spec=spec,
                            fold=fold,
                            seed=seed,
                            label_ratio=label_ratio,
                            confidence_threshold=config["training"]["pseudo_threshold"],
                            config=config,
                            device=device,
                            results_path=results_path,
                            runs_dir=runs_dir,
                            split_hash=split_hash,
                            dataset_hash=dataset_hash,
                            provenance=provenance,
                        )
                    except Exception as exc:
                        logging.exception(
                            "failed experiment=%s fold=%d seed=%d method=%s "
                            "label_ratio=%.2f",
                            experiment_type,
                            fold,
                            seed,
                            method,
                            label_ratio,
                        )
                        failure = failed_result_row(
                            config=config,
                            experiment_type=experiment_type,
                            method=method,
                            method_spec=spec,
                            fold=fold,
                            seed=seed,
                            label_ratio=label_ratio,
                            threshold=provisional["threshold"],
                            config_hash=config_hash,
                            split_hash=split_hash,
                            dataset_hash=dataset_hash,
                            provenance=provenance,
                            config_path=relative_or_absolute(
                                runs_dir / run_id / "config.json"
                            ),
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                        append_result(results_path, failure)
                        results = read_results(results_path)
                        continue
                    results = read_results(results_path)
                    if method == "cwt_gru":
                        computed_cwt_gru = state
                    logging.info(
                        "completed experiment=%s fold=%d seed=%d method=%s "
                        "label_ratio=%.2f macro_f1=%.4f",
                        experiment_type,
                        fold,
                        seed,
                        method,
                        label_ratio,
                        row["macro_f1"],
                    )


def run_threshold_experiment(
    config: dict[str, Any],
    *,
    output_dir: Path,
    data,
    device: torch.device,
    provenance: dict[str, Any],
    dataset_hash: str,
    resume: bool,
) -> None:
    results_path = output_dir / "raw_runs.csv"
    results = read_results(results_path)
    runs_dir = output_dir / "runs"
    heldout = np.asarray(config["dataset"]["heldout_record_ids"], dtype=np.int64)
    base_spec = METHODS["cwt_gru"]
    training = config["training"]
    threshold_config = config["experiments"]["threshold"]
    label_ratio = float(threshold_config["fraction"])
    thresholds = parse_csv_floats(threshold_config["thresholds"])

    for fold in parse_csv_ints(config["folds"]):
        split = load_fold_indices(data, fold, heldout)
        manifest = build_split_manifest(
            data,
            split,
            dataset_name=config["dataset"]["name"],
            split_version=config["dataset"]["split_version"],
            fold=fold,
            heldout_record_ids=heldout,
        )
        split_hash = write_json(
            output_dir / "splits" / f"fold{fold}.json",
            manifest,
        )
        for seed in parse_csv_ints(config["seeds"]):
            set_global_seed(seed, training["deterministic"])
            rng = np.random.default_rng(seed + fold * 1000)
            labeled_idx = balanced_indices(split.train, data.labels, label_ratio, rng)
            candidate_mask = np.ones(len(split.train), dtype=bool)
            candidate_mask[np.isin(split.train, labeled_idx)] = False
            candidate_idx = split.train[candidate_mask]

            missing_thresholds: list[float] = []
            hashes: dict[float, str] = {}
            for threshold in thresholds:
                provisional = {
                    "config": config,
                    "method": "cwt_gru_selftrain",
                    "fold": fold,
                    "seed": seed,
                    "label_ratio": label_ratio,
                    "threshold": threshold,
                    "split_hash": split_hash,
                    "dataset_hash": dataset_hash,
                    "provenance": provenance,
                }
                config_hash = stable_json_hash(provisional)
                hashes[threshold] = config_hash
                run_id = run_id_for(
                    experiment_type="threshold",
                    fold=fold,
                    seed=seed,
                    method="cwt_gru_selftrain",
                    label_ratio=label_ratio,
                    threshold=threshold,
                    config_hash=config_hash,
                )
                if not (
                    resume
                    and has_exact_run(
                        results,
                        run_id=run_id,
                        config_hash=config_hash,
                    )
                ):
                    missing_thresholds.append(threshold)
            if not missing_thresholds:
                logging.info(
                    "Reusing all threshold runs fold=%d seed=%d",
                    fold,
                    seed,
                )
                continue

            start = time.perf_counter()
            try:
                base_state = train_model(
                    data,
                    labeled_idx,
                    data.labels[labeled_idx],
                    np.ones(len(labeled_idx), dtype=np.float32),
                    split.validation,
                    base_spec,
                    epochs=training["epochs"],
                    batch_size=training["batch_size"],
                    learning_rate=training["learning_rate"],
                    weight_decay=training["weight_decay"],
                    patience=training["patience"],
                    num_workers=training["num_workers"],
                    device=device,
                    seed=seed,
                )
            except Exception as exc:
                logging.exception(
                    "failed threshold base training fold=%d seed=%d", fold, seed
                )
                for threshold in missing_thresholds:
                    config_hash = hashes[threshold]
                    run_id = run_id_for(
                        experiment_type="threshold",
                        fold=fold,
                        seed=seed,
                        method="cwt_gru_selftrain",
                        label_ratio=label_ratio,
                        threshold=threshold,
                        config_hash=config_hash,
                    )
                    append_result(
                        results_path,
                        failed_result_row(
                            config=config,
                            experiment_type="threshold",
                            method="cwt_gru_selftrain",
                            method_spec=METHODS["cwt_gru_selftrain"],
                            fold=fold,
                            seed=seed,
                            label_ratio=label_ratio,
                            threshold=threshold,
                            config_hash=config_hash,
                            split_hash=split_hash,
                            dataset_hash=dataset_hash,
                            provenance=provenance,
                            config_path=relative_or_absolute(
                                runs_dir / run_id / "config.json"
                            ),
                            reason=f"{type(exc).__name__}: {exc}",
                        ),
                    )
                results = read_results(results_path)
                continue
            base_train_time = time.perf_counter() - start
            max_per_class = max(
                1,
                len(labeled_idx) // (int(data.labels.max()) + 1) * 2,
            )

            for threshold in missing_thresholds:
                config_hash = hashes[threshold]
                pseudo_idx, pseudo_labels, pseudo_weights, _ = select_pseudo_labels(
                    base_state.model,
                    data,
                    candidate_idx,
                    base_state.temperature,
                    max_per_class=max_per_class,
                    confidence_threshold=threshold,
                    batch_size=training["batch_size"],
                    num_workers=training["num_workers"],
                    device=device,
                )
                pseudo_count = len(pseudo_idx)
                pseudo_accuracy = float("nan")
                pseudo_macro_f1 = float("nan")
                if pseudo_count:
                    from sklearn.metrics import accuracy_score, f1_score

                    pseudo_accuracy = float(
                        accuracy_score(data.labels[pseudo_idx], pseudo_labels)
                    )
                    pseudo_macro_f1 = float(
                        f1_score(
                            data.labels[pseudo_idx],
                            pseudo_labels,
                            average="macro",
                            zero_division=0,
                        )
                    )
                    retrain_start = time.perf_counter()
                    combined_idx = np.concatenate([labeled_idx, pseudo_idx])
                    combined_labels = np.concatenate(
                        [data.labels[labeled_idx], pseudo_labels]
                    )
                    combined_weights = np.concatenate(
                        [
                            np.ones(len(labeled_idx), dtype=np.float32),
                            pseudo_weights,
                        ]
                    )
                    state = train_model(
                        data,
                        combined_idx,
                        combined_labels,
                        combined_weights,
                        split.validation,
                        METHODS["cwt_gru_selftrain"],
                        epochs=training["epochs"],
                        batch_size=training["batch_size"],
                        learning_rate=training["learning_rate"],
                        weight_decay=training["weight_decay"],
                        patience=training["patience"],
                        num_workers=training["num_workers"],
                        device=device,
                        seed=seed + 1,
                    )
                    train_time_s = base_train_time + (
                        time.perf_counter() - retrain_start
                    )
                else:
                    state = base_state
                    train_time_s = base_train_time

                (
                    raw_metrics,
                    calibrated_metrics,
                    logits,
                    labels,
                    predictions,
                    timing,
                ) = evaluate_model(
                    state,
                    data,
                    split.test,
                    batch_size=training["batch_size"],
                    num_workers=training["num_workers"],
                    device=device,
                )
                run_id = run_id_for(
                    experiment_type="threshold",
                    fold=fold,
                    seed=seed,
                    method="cwt_gru_selftrain",
                    label_ratio=label_ratio,
                    threshold=threshold,
                    config_hash=config_hash,
                )
                run_dir = runs_dir / run_id
                run_config = build_run_config(
                    config=config,
                    experiment_type="threshold",
                    method="cwt_gru_selftrain",
                    method_spec=METHODS["cwt_gru_selftrain"],
                    fold=fold,
                    seed=seed,
                    label_ratio=label_ratio,
                    threshold=threshold,
                    split_hash=split_hash,
                    dataset_hash=dataset_hash,
                    provenance=provenance,
                    environment=environment_snapshot(device),
                )
                config_path = run_dir / "config.json"
                atomic_write_json(config_path, run_config)
                write_diagnostics(
                    run_dir,
                    state=state,
                    logits=logits,
                    labels=labels,
                    predictions=predictions,
                    run_id=run_id,
                )
                parameters = count_parameters(state.model)
                sample_raw = torch.from_numpy(data.raw[:1]).to(device)
                sample_cwt = torch.from_numpy(data.cwt[:1]).float().to(device)
                gflops_sample = estimate_model_gflops(
                    state.model,
                    sample_raw,
                    sample_cwt,
                    model_key="cwt_gru",
                )
                row = result_row(
                    config=config,
                    experiment_type="threshold",
                    method="cwt_gru_selftrain",
                    method_spec=METHODS["cwt_gru_selftrain"],
                    fold=fold,
                    seed=seed,
                    label_ratio=label_ratio,
                    threshold=threshold,
                    raw_metrics=raw_metrics,
                    calibrated_metrics=calibrated_metrics,
                    state=state,
                    train_time_s=train_time_s,
                    timing=timing,
                    parameters=parameters,
                    gflops_sample=gflops_sample,
                    labeled_windows=len(labeled_idx),
                    unlabeled_windows=len(candidate_idx),
                    train_windows=len(split.train),
                    validation_windows=len(split.validation),
                    test_windows=len(split.test),
                    pseudo_metadata={
                        "pseudo_count": pseudo_count,
                        "pseudo_coverage": pseudo_count / max(len(candidate_idx), 1),
                        "pseudo_accuracy": pseudo_accuracy,
                        "pseudo_macro_f1": pseudo_macro_f1,
                        "self_training_rounds_completed": 1 if pseudo_count else 0,
                    },
                    config_hash=config_hash,
                    split_hash=split_hash,
                    dataset_hash=dataset_hash,
                    provenance=provenance,
                    config_path=relative_or_absolute(config_path),
                )
                append_result(results_path, row)
                results = read_results(results_path)
                logging.info(
                    "completed threshold fold=%d seed=%d threshold=%.2f "
                    "pseudo=%d macro_f1=%.4f",
                    fold,
                    seed,
                    threshold,
                    pseudo_count,
                    row["macro_f1"],
                )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = apply_overrides(load_json_config(config_path), args)
    output_dir = (args.output_dir or PROJECT_ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "run_benchmark.log")
    provenance = ensure_clean_worktree(args.allow_dirty)
    if provenance["git_dirty"]:
        logging.warning("Running from a dirty worktree; results are exploratory")
    atomic_write_json(output_dir / "config_used.json", config)

    cache_path = (PROJECT_ROOT / config["dataset"]["cache"]).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Dataset cache not found: {cache_path}. Run the documented data "
            "preparation command first."
        )
    dataset_hash = file_sha256(cache_path)
    data = load_cwru_cache(cache_path)
    device = resolve_device(config["training"]["device"])

    logging.info("dataset=%s windows=%d", config["dataset"]["name"], len(data.labels))
    logging.info("git_commit=%s", provenance["git_commit"])
    logging.info("dataset_sha256=%s", dataset_hash)
    logging.info(
        "seeds=%s folds=%s",
        config["seeds"],
        config["folds"],
    )

    experiments = (
        ["main", "label_efficiency", "threshold"]
        if args.experiment == "all"
        else ["main" if args.experiment == "ablation" else args.experiment]
    )
    for experiment in experiments:
        if experiment == "main":
            configured_methods = config["experiments"].get("main", {}).get(
                "methods",
                MAIN_METHODS,
            )
            if args.experiment == "ablation" and not args.methods:
                configured_methods = ABLATION_METHODS
            run_main_experiment(
                config,
                methods=list(configured_methods),
                fractions=[1.0],
                experiment_type="main",
                output_dir=output_dir,
                data=data,
                device=device,
                provenance=provenance,
                dataset_hash=dataset_hash,
                resume=args.resume,
            )
        elif experiment == "label_efficiency":
            run_main_experiment(
                config,
                methods=config["experiments"]["label_efficiency"]["methods"],
                fractions=[
                    float(value)
                    for value in config["experiments"]["label_efficiency"][
                        "fractions"
                    ]
                    if float(value) < 1.0
                ],
                experiment_type="label_efficiency",
                output_dir=output_dir,
                data=data,
                device=device,
                provenance=provenance,
                dataset_hash=dataset_hash,
                resume=args.resume,
            )
        elif experiment == "threshold":
            run_threshold_experiment(
                config,
                output_dir=output_dir,
                data=data,
                device=device,
                provenance=provenance,
                dataset_hash=dataset_hash,
                resume=args.resume,
            )
    summarize_results(output_dir)


if __name__ == "__main__":
    main()
