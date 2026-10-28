"""Training diagnostics and small-subset optimization sanity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..datasets.splits import balanced_indices, load_fold_indices, set_global_seed
from ..models.registry import METHODS
from ..paths import PROJECT_ROOT
from ..public_benchmark.cwru_data import load_cwru_cache
from ..training.runner import train_model


SANITY_METHODS = ["raw_cnn", "cwt_cnn", "cwt_gru"]


def read_history(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def analyze_history(
    row: pd.Series,
    results_dir: Path,
    max_epochs: int,
) -> dict[str, Any]:
    history = read_history(
        results_dir / "runs" / str(row["run_id"]) / "training_history.csv"
    )
    if history.empty:
        return {
            "run_id": row["run_id"],
            "method": row["method"],
            "fold": int(row["fold"]),
            "seed": int(row["seed"]),
            "status": "REPRODUCTION FAILURE",
            "reason": "missing training_history.csv",
        }
    train_loss = history["train_loss"].to_numpy(dtype=float)
    val_loss = history["val_loss"].to_numpy(dtype=float)
    val_f1 = history["val_macro_f1"].to_numpy(dtype=float)
    finite = bool(
        np.isfinite(train_loss).all()
        and np.isfinite(val_loss).all()
        and np.isfinite(val_f1).all()
    )
    cap_selected = int(row["best_epoch"]) == int(row["epochs_ran"]) == max_epochs
    loss_explosion = bool(
        len(val_loss) > 1 and np.nanmax(val_loss) > max(50.0, 20 * np.nanmin(val_loss))
    )
    no_learning = bool(
        len(train_loss) > 1
        and train_loss[-1] > 0.98 * train_loss[0]
        and np.nanmax(val_f1) <= 0.30
    )
    if not finite or loss_explosion:
        status = "OPTIMIZATION WARNING"
        reason = "non-finite history or explosive validation loss"
    elif no_learning:
        status = "OPTIMIZATION WARNING"
        reason = "training loss and validation Macro-F1 did not improve materially"
    elif cap_selected:
        status = "CAP SELECTED"
        reason = "best epoch equals the maximum epoch budget"
    else:
        status = "VERIFIED"
        reason = "early stopping or stable non-cap selection"
    return {
        "run_id": row["run_id"],
        "method": row["method"],
        "fold": int(row["fold"]),
        "seed": int(row["seed"]),
        "status": status,
        "reason": reason,
        "epochs_recorded": len(history),
        "best_epoch": int(row["best_epoch"]),
        "max_epochs": max_epochs,
        "train_loss_first": float(train_loss[0]),
        "train_loss_last": float(train_loss[-1]),
        "train_loss_min": float(train_loss.min()),
        "val_loss_best": float(val_loss.min()),
        "val_loss_last": float(val_loss[-1]),
        "val_macro_f1_max": float(val_f1.max()),
        "val_macro_f1_last": float(val_f1[-1]),
        "gradient_norm_last": (
            float(history["gradient_norm"].iloc[-1])
            if "gradient_norm" in history
            else np.nan
        ),
    }


def analyze_existing(
    results_dir: Path,
    *,
    max_epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(results_dir / "raw_runs.csv")
    main = raw.loc[
        raw["experiment_type"].eq("main")
        & raw["method"].isin(
            [
                "raw_cnn",
                "gru_only",
                "cwt_cnn",
                "cwt_gru_no_se",
                "cwt_gru",
                "cwt_gru_selftrain",
            ]
        )
    ]
    per_run = pd.DataFrame(
        [analyze_history(row, results_dir, max_epochs) for _, row in main.iterrows()]
    )
    summary = (
        per_run.groupby("method")
        .agg(
            runs=("run_id", "count"),
            verified_runs=("status", lambda values: int((values == "VERIFIED").sum())),
            cap_selected_runs=("status", lambda values: int((values == "CAP SELECTED").sum())),
            optimization_warnings=(
                "status",
                lambda values: int((values == "OPTIMIZATION WARNING").sum()),
            ),
            mean_best_epoch=("best_epoch", "mean"),
            min_best_epoch=("best_epoch", "min"),
            max_best_epoch=("best_epoch", "max"),
            mean_train_loss_last=("train_loss_last", "mean"),
            mean_val_loss_best=("val_loss_best", "mean"),
            mean_val_macro_f1_max=("val_macro_f1_max", "mean"),
        )
        .reset_index()
    )
    return per_run, summary


def run_overfit_sanity(
    *,
    config_path: Path,
    output_path: Path,
    epochs: int,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache = (PROJECT_ROOT / config["dataset"]["cache"]).resolve()
    data = load_cwru_cache(cache)
    heldout = np.asarray(config["dataset"]["heldout_record_ids"], dtype=np.int64)
    split = load_fold_indices(data, 0, heldout)
    rng = np.random.default_rng(seed)
    subset = balanced_indices(split.train, data.labels, fraction, rng)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for method in SANITY_METHODS:
        set_global_seed(seed)
        state = train_model(
            data,
            subset,
            data.labels[subset],
            np.ones(len(subset), dtype=np.float32),
            subset,
            METHODS[method],
            epochs=epochs,
            batch_size=min(16, len(subset)),
            learning_rate=0.002,
            weight_decay=0.0001,
            patience=epochs,
            num_workers=0,
            device=device,
            seed=seed,
        )
        history = pd.DataFrame(state.history)
        rows.append(
            {
                "method": method,
                "fold": 0,
                "seed": seed,
                "subset_fraction_requested": fraction,
                "subset_windows": len(subset),
                "epochs_requested": epochs,
                "epochs_ran": state.epochs_ran,
                "best_epoch": state.best_epoch,
                "subset_macro_f1": state.best_val_f1,
                "reached_0_95": bool(state.best_val_f1 >= 0.95),
                "reached_0_99": bool(state.best_val_f1 >= 0.99),
                "final_train_loss": state.last_train_loss,
                "final_train_macro_f1": (
                    float(history["train_macro_f1"].iloc[-1])
                    if "train_macro_f1" in history
                    else np.nan
                ),
                "test_data_used": False,
                "role": "engineering sanity check only",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False, lineterminator="\n")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paderborn_main.json",
    )
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--run-overfit", action="store_true")
    parser.add_argument("--overfit-epochs", type=int, default=80)
    parser.add_argument("--overfit-fraction", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_run, summary = analyze_existing(
        args.results_dir,
        max_epochs=args.max_epochs,
    )
    per_run.to_csv(
        args.results_dir / "training_diagnostics.csv",
        index=False,
        lineterminator="\n",
    )
    summary.to_csv(
        args.results_dir / "training_diagnostics_summary.csv",
        index=False,
        lineterminator="\n",
    )
    print(summary.to_string(index=False))
    if args.run_overfit:
        overfit = run_overfit_sanity(
            config_path=args.config,
            output_path=args.results_dir / "overfit_sanity.csv",
            epochs=args.overfit_epochs,
            fraction=args.overfit_fraction,
            seed=args.seed,
        )
        print(overfit.to_string(index=False))


if __name__ == "__main__":
    main()
