"""Compare the canonical 12-epoch budget with a 30-epoch opportunity."""

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


METHOD_LABELS = {
    "raw_cnn": "Raw 1D-CNN",
    "cwt_cnn": "CWT 2D-CNN",
    "cwt_gru_no_se": "CWT-GRU without SE",
    "cwt_gru": "CWT-GRU",
}


def paired_delta(
    baseline: pd.DataFrame,
    extended: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    left = baseline.set_index(["fold", "seed"])[value_column].rename("budget_12")
    right = extended.set_index(["fold", "seed"])[value_column].rename("budget_30")
    paired = pd.concat([left, right], axis=1).dropna()
    if len(paired) != 20:
        raise ValueError(f"Expected 20 pairs, found {len(paired)}")
    return (paired["budget_30"] - paired["budget_12"]).rename("delta").to_frame()


def save_figure(comparison: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    methods = list(METHOD_LABELS)
    x = np.arange(len(methods))
    axis.axhline(0, color="0.55", linewidth=0.8, linestyle="--")
    for offset, metric, color, label in (
        (-0.18, "validation_delta_mean", "#4D4D4D", "Validation"),
        (0.18, "test_delta_mean", "#B0B0B0", "Held-out test"),
    ):
        selected = comparison.set_index("method").reindex(methods)
        axis.bar(
            x + offset,
            selected[metric],
            yerr=selected[metric.replace("_mean", "_sd")],
            capsize=2,
            width=0.34,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            label=label,
        )
    axis.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=18, ha="right")
    axis.set_ylabel("30-epoch minus 12-epoch Macro-F1")
    axis.grid(axis="y", color="0.9", linewidth=0.6)
    axis.legend(frameon=False)
    png = output_dir / "figure_21_training_budget.png"
    pdf = output_dir / "figure_21_training_budget.pdf"
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
        "--extended-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "training_budget_30",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "publication",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = pd.read_csv(args.results_dir / "raw_runs.csv")
    canonical = canonical.loc[
        canonical["experiment_type"].eq("main")
        & canonical["method"].isin(METHOD_LABELS)
    ]
    extended_path = args.extended_dir / "raw_runs.csv"
    if not extended_path.is_file():
        raise FileNotFoundError(extended_path)
    extended = pd.read_csv(extended_path)
    rows = []
    for index, method in enumerate(METHOD_LABELS):
        baseline = canonical.loc[canonical["method"].eq(method)]
        opportunity = extended.loc[extended["method"].eq(method)]
        validation = paired_delta(baseline, opportunity, "best_val_macro_f1")
        test = paired_delta(baseline, opportunity, "macro_f1")
        val_low, val_high = _hierarchical_bootstrap_interval(
            validation,
            "delta",
            samples=20_000,
            seed=20261028 + index,
        )
        test_low, test_high = _hierarchical_bootstrap_interval(
            test,
            "delta",
            samples=20_000,
            seed=20361028 + index,
        )
        rows.append(
            {
                "method": method,
                "model_label": METHOD_LABELS[method],
                "validation_delta_mean": validation["delta"].mean(),
                "validation_delta_sd": validation["delta"].std(ddof=1),
                "validation_ci_low": val_low,
                "validation_ci_high": val_high,
                "test_delta_mean": test["delta"].mean(),
                "test_delta_sd": test["delta"].std(ddof=1),
                "test_ci_low": test_low,
                "test_ci_high": test_high,
                "budget_30_best_epoch_mean": opportunity["best_epoch"].mean(),
                "budget_30_best_epoch_max": opportunity["best_epoch"].max(),
                "budget_30_cap_selected_runs": int(
                    ((opportunity["best_epoch"] == 30) & (opportunity["epochs_ran"] == 30)).sum()
                ),
                "validation_improved_runs": int((validation["delta"] > 1e-12).sum()),
                "n_pairs": len(validation),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        args.results_dir / "training_budget_comparison.csv",
        index=False,
        lineterminator="\n",
    )
    save_figure(comparison, args.figures_dir)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
