"""Summarize per-window versus training-statistic normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.hierarchical import _hierarchical_bootstrap_interval
from ..paths import PROJECT_ROOT
from ..visualization.style import (
    AUXILIARY_COLORS,
    GRID_COLOR,
    INK_COLOR,
    apply_paper_style,
)


METHODS = ["raw_cnn", "cwt_cnn", "cwt_gru"]
LABELS = {
    "raw_cnn": "Raw 1D-CNN",
    "cwt_cnn": "CWT 2D-CNN",
    "cwt_gru": "CWT-GRU",
}


def load_frames(
    results_dir: Path,
    sensitivity_dir: Path,
) -> pd.DataFrame:
    canonical = pd.read_csv(results_dir / "raw_runs.csv")
    canonical = canonical.loc[
        canonical["experiment_type"].eq("main")
        & canonical["method"].isin(METHODS)
    ].copy()
    canonical["normalization"] = "A_window_zscore"
    sensitivity_path = sensitivity_dir / "raw_runs.csv"
    if not sensitivity_path.is_file():
        raise FileNotFoundError(sensitivity_path)
    sensitivity = pd.read_csv(sensitivity_path)
    sensitivity["normalization"] = "B_training_statistics"
    return pd.concat([canonical, sensitivity], ignore_index=True)


def save_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style(font_size=8.5)
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), constrained_layout=True)
    width = 0.34
    x = np.arange(len(METHODS))
    for offset, normalization, color, label in (
        (
            -0.18,
            "A_window_zscore",
            AUXILIARY_COLORS["reference"],
            "A: window z-score",
        ),
        (
            0.18,
            "B_training_statistics",
            AUXILIARY_COLORS["secondary"],
            "B: training statistics",
        ),
    ):
        selected = (
            summary.loc[summary["normalization"].eq(normalization)]
            .set_index("method")
            .reindex(METHODS)
        )
        axes[0].bar(
            x + offset,
            selected["test_macro_f1_mean"],
            yerr=selected["test_macro_f1_sd"],
            width=width,
            capsize=2,
            color=color,
            edgecolor=INK_COLOR,
            linewidth=0.45,
            label=label,
        )
        axes[1].bar(
            x + offset,
            selected["validation_test_gap_mean"],
            yerr=selected["validation_test_gap_sd"],
            width=width,
            capsize=2,
            color=color,
            edgecolor=INK_COLOR,
            linewidth=0.45,
        )
        axes[2].bar(
            x + offset,
            selected["relative_drop_mean"],
            width=width,
            color=color,
            edgecolor=INK_COLOR,
            linewidth=0.45,
        )
    for axis in axes:
        axis.set_xticks(x, [LABELS[method] for method in METHODS], rotation=20, ha="right")
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    axes[0].set_ylabel("Test Macro-F1")
    axes[0].set_ylim(0, 0.55)
    axes[1].set_ylabel("Validation-test gap")
    axes[2].set_ylabel("Relative drop")
    axes[0].legend(frameon=False, fontsize=6.8)
    png = output_dir / "figure_24_normalization_sensitivity.png"
    pdf = output_dir / "figure_24_normalization_sensitivity.pdf"
    figure.savefig(png, dpi=320, bbox_inches="tight")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--sensitivity-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "normalization_sensitivity",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "publication",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_frames(args.results_dir, args.sensitivity_dir)
    frame["validation_test_gap"] = (
        frame["best_val_macro_f1"] - frame["macro_f1"]
    )
    frame["relative_drop"] = (
        frame["validation_test_gap"] / frame["best_val_macro_f1"].clip(lower=1e-12)
    )
    summary = (
        frame.groupby(["method", "normalization"])
        .agg(
            runs=("run_id", "nunique"),
            test_macro_f1_mean=("macro_f1", "mean"),
            test_macro_f1_sd=("macro_f1", "std"),
            validation_macro_f1_mean=("best_val_macro_f1", "mean"),
            validation_macro_f1_sd=("best_val_macro_f1", "std"),
            validation_test_gap_mean=("validation_test_gap", "mean"),
            validation_test_gap_sd=("validation_test_gap", "std"),
            relative_drop_mean=("relative_drop", "mean"),
            relative_drop_sd=("relative_drop", "std"),
        )
        .reset_index()
    )
    comparisons = []
    for index, method in enumerate(METHODS):
        selected = frame.loc[frame["method"].eq(method)]
        pivot = selected.pivot_table(
            index=["fold", "seed"],
            columns="normalization",
            values="macro_f1",
        )
        delta = (
            pivot["B_training_statistics"] - pivot["A_window_zscore"]
        ).rename("delta").to_frame()
        low, high = _hierarchical_bootstrap_interval(
            delta,
            "delta",
            samples=20_000,
            seed=20261028 + index,
        )
        comparisons.append(
            {
                "method": method,
                "model_label": LABELS[method],
                "n_pairs": len(delta),
                "mean_delta_normalization_b_minus_a": float(delta["delta"].mean()),
                "hierarchical_ci_low": low,
                "hierarchical_ci_high": high,
                "fold_wins_b": int(
                    (
                        delta.groupby(level="fold")["delta"].mean() > 1e-12
                    ).sum()
                ),
                "fold_losses_b": int(
                    (
                        delta.groupby(level="fold")["delta"].mean() < -1e-12
                    ).sum()
                ),
            }
        )
    summary.to_csv(
        args.results_dir / "normalization_sensitivity_summary.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(comparisons).to_csv(
        args.results_dir / "normalization_sensitivity_comparisons.csv",
        index=False,
        lineterminator="\n",
    )
    save_figure(summary, args.figures_dir)
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
