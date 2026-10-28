"""Validation-only tuning-budget audit for four representative baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.hierarchical import _hierarchical_bootstrap_interval
from ..paths import PROJECT_ROOT


LABELS = {
    "raw_cnn": "Raw 1D-CNN",
    "cwt_cnn": "CWT 2D-CNN",
    "drsn": "DRSN adaptation",
    "tfnet": "TFN adaptation",
}


def default_rows(results_dir: Path) -> pd.DataFrame:
    main = pd.read_csv(results_dir / "raw_runs.csv")
    external = pd.read_csv(results_dir / "sota" / "raw_runs.csv")
    selected = pd.concat(
        [
            main.loc[
                main["experiment_type"].eq("main")
                & main["method"].isin(["raw_cnn", "cwt_cnn"])
            ],
            external.loc[external["method"].isin(["drsn", "tfnet"])],
        ],
        ignore_index=True,
    )
    selected["candidate"] = "lr0.002_wd0.0001"
    selected["selection_source"] = "predeclared_default"
    return selected


def alternate_rows(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "raw_runs.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Baseline tuning results not found: {path}. "
            "Run the paderborn_baseline_tuning_lr001 configuration first."
        )
    frame = pd.read_csv(path)
    frame["candidate"] = "lr0.001_wd0.0001"
    frame["selection_source"] = "predeclared_alternative"
    return frame


def select_by_validation(frame: pd.DataFrame) -> pd.DataFrame:
    fold_validation = (
        frame.groupby(["method", "fold", "candidate"])
        .agg(mean_validation_macro_f1=("best_val_macro_f1", "mean"))
        .reset_index()
    )
    selected = (
        fold_validation.sort_values(
            ["method", "fold", "mean_validation_macro_f1", "candidate"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["method", "fold"])
        .rename(columns={"candidate": "selected_configuration"})
    )
    frame = frame.merge(
        selected[["method", "fold", "selected_configuration"]],
        on=["method", "fold"],
        how="left",
        validate="many_to_one",
    )
    frame["selected"] = frame["candidate"].eq(frame["selected_configuration"])
    if frame["selected"].groupby([frame["method"], frame["fold"]]).sum().ne(5).any():
        raise ValueError("Every fold must select exactly one candidate over five seeds")
    return frame


def compare_selected(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seed_id = 20261028
    for method, group in frame.groupby("method"):
        default = group.loc[group["candidate"].eq("lr0.002_wd0.0001")].set_index(
            ["fold", "seed"]
        )["macro_f1"]
        tuned = (
            group.loc[group["selected"]]
            .set_index(["fold", "seed"])["macro_f1"]
            .rename("delta")
            .to_frame()
        )
        tuned["default"] = default.reindex(tuned.index)
        tuned["delta"] = tuned["delta"] - tuned["default"]
        folded = tuned["delta"].rename("delta").to_frame()
        low, high = _hierarchical_bootstrap_interval(
            folded,
            "delta",
            samples=20_000,
            seed=seed_id + int(method == "cwt_cnn") + int(method == "drsn") + int(method == "tfnet"),
        )
        fold_means = tuned.groupby(level="fold")["delta"].mean()
        selected_by_fold = (
            group.loc[group["selected"]]
            .drop_duplicates(["fold"])
            .set_index("fold")["candidate"]
        )
        rows.append(
            {
                "method": method,
                "model_label": LABELS[method],
                "default_mean_macro_f1": float(default.mean()),
                "validation_selected_mean_macro_f1": float(
                    tuned["default"].add(tuned["delta"]).mean()
                ),
                "mean_tuned_minus_default": float(tuned["delta"].mean()),
                "hierarchical_ci_low": low,
                "hierarchical_ci_high": high,
                "fold_wins": int((fold_means > 1e-12).sum()),
                "fold_ties": int((fold_means.abs() <= 1e-12).sum()),
                "fold_losses": int((fold_means < -1e-12).sum()),
                "selected_configurations": ";".join(
                    f"{fold}:{selected_by_fold.loc[fold]}"
                    for fold in sorted(selected_by_fold.index)
                ),
            }
        )
    return pd.DataFrame(rows)


def candidate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["method", "model_label", "candidate"])
        .agg(
            runs=("run_id", "nunique"),
            parameter_count=("parameters", "median"),
            validation_macro_f1_mean=("best_val_macro_f1", "mean"),
            validation_macro_f1_sd=("best_val_macro_f1", "std"),
            test_macro_f1_mean=("macro_f1", "mean"),
            test_macro_f1_sd=("macro_f1", "std"),
            mean_best_epoch=("best_epoch", "mean"),
        )
        .reset_index()
    )


def save_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    methods = list(LABELS)
    candidates = ["lr0.002_wd0.0001", "lr0.001_wd0.0001"]
    colors = ["#4D4D4D", "#B3B3B3"]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    x = np.arange(len(methods))
    width = 0.34
    for axis, metric, title in (
        (axes[0], "validation_macro_f1_mean", "Validation Macro-F1"),
        (axes[1], "test_macro_f1_mean", "Held-out test Macro-F1"),
    ):
        for index, (candidate, color) in enumerate(zip(candidates, colors, strict=True)):
            selected = (
                summary.loc[summary["candidate"].eq(candidate)]
                .set_index("method")
                .reindex(methods)
            )
            axis.bar(
                x + (index - 0.5) * width,
                selected[metric],
                width=width,
                color=color,
                edgecolor="black",
                linewidth=0.45,
                label=candidate,
            )
        axis.set_xticks(x, [LABELS[method] for method in methods], rotation=18, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(title, loc="left")
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=7)
    png = output_dir / "figure_18_baseline_tuning.png"
    pdf = output_dir / "figure_18_baseline_tuning.pdf"
    figure.savefig(png, dpi=320, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--alternative-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline_tuning",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "publication",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default = default_rows(args.results_dir)
    alternate = alternate_rows(args.alternative_dir)
    combined = pd.concat([default, alternate], ignore_index=True)
    combined["model_label"] = combined["method"].map(LABELS)
    combined["parameter_count"] = combined["parameters"]
    combined = select_by_validation(combined)
    output_columns = [
        "method",
        "model_label",
        "parameter_count",
        "candidate",
        "fold",
        "seed",
        "best_val_macro_f1",
        "macro_f1",
        "selected",
        "selection_source",
        "selected_configuration",
        "run_id",
    ]
    output = combined[output_columns].rename(
        columns={
            "method": "architecture_key",
            "model_label": "architecture",
            "best_val_macro_f1": "validation_macro_f1",
            "macro_f1": "test_macro_f1",
        }
    )
    output.to_csv(
        args.results_dir / "external_baseline_tuning.csv",
        index=False,
        lineterminator="\n",
    )
    summary = candidate_summary(combined)
    summary.to_csv(
        args.results_dir / "external_baseline_tuning_summary.csv",
        index=False,
        lineterminator="\n",
    )
    comparisons = compare_selected(combined)
    comparisons.to_csv(
        args.results_dir / "external_baseline_tuning_comparisons.csv",
        index=False,
        lineterminator="\n",
    )
    save_figure(summary, args.figures_dir)
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
