"""Analyze the coverage-controlled self-training sensitivity experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.hierarchical import (
    _fold_signflip_p_value,
    _hierarchical_bootstrap_interval,
)
from ..paths import PROJECT_ROOT
from ..visualization.style import (
    GRID_COLOR,
    INK_COLOR,
    apply_paper_style,
    method_color,
)


BUDGETS = {
    "50_percent": 0.5,
    "full_balanced": 1.0,
}


def load_canonical_reference(results_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(results_dir / "raw_runs.csv")
    supervised_main = raw.loc[
        raw["experiment_type"].eq("main") & raw["method"].eq("cwt_gru")
    ].copy()
    supervised_main["budget"] = "full_balanced"
    supervised_half = raw.loc[
        raw["experiment_type"].eq("label_efficiency")
        & raw["method"].eq("cwt_gru")
        & np.isclose(raw["label_ratio"].to_numpy(dtype=float), 0.5)
    ].copy()
    supervised_half["budget"] = "50_percent"
    fixed_main = raw.loc[
        raw["experiment_type"].eq("main")
        & raw["method"].eq("cwt_gru_selftrain")
    ].copy()
    fixed_main["budget"] = "full_balanced"
    fixed_half = raw.loc[
        raw["experiment_type"].eq("label_efficiency")
        & raw["method"].eq("cwt_gru_selftrain")
        & np.isclose(raw["label_ratio"].to_numpy(dtype=float), 0.5)
    ].copy()
    fixed_half["budget"] = "50_percent"
    supervised = pd.concat([supervised_main, supervised_half], ignore_index=True)
    fixed = pd.concat([fixed_main, fixed_half], ignore_index=True)
    return pd.concat(
        [
            supervised.assign(condition="supervised"),
            fixed.assign(condition="fixed_threshold_0.60"),
        ],
        ignore_index=True,
    )


def load_coverage(sensitivity_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(sensitivity_dir / "raw_runs.csv")
    if raw.empty or raw["status"].ne("success").any():
        raise ValueError("Coverage self-training results are incomplete or failed")
    rows = []
    for _, row in raw.iterrows():
        budget = "50_percent" if np.isclose(float(row["label_ratio"]), 0.5) else "full_balanced"
        if budget not in BUDGETS:
            continue
        diagnostic_path = (
            sensitivity_dir / "runs" / str(row["run_id"]) / "pseudo_label_diagnostics.csv"
        )
        diagnostics = pd.read_csv(diagnostic_path)
        final_round = diagnostics.iloc[-1]
        rows.append(
            {
                "run_id": row["run_id"],
                "method": "cwt_gru_selftrain_coverage",
                "condition": "coverage_controlled_0.50",
                "budget": budget,
                "fold": int(row["fold"]),
                "seed": int(row["seed"]),
                "macro_f1": float(row["macro_f1"]),
                "accuracy": float(row["accuracy"]),
                "pseudo_count": int(row["pseudo_count"]),
                "pseudo_coverage": float(row["pseudo_coverage"]),
                "pseudo_accuracy": float(row["pseudo_accuracy"]),
                "rounds_completed": int(row["self_training_rounds_completed"]),
                "selected_confidence_mean": float(
                    final_round["selected_confidence_mean"]
                ),
                "selected_confidence_median": float(
                    final_round["selected_confidence_median"]
                ),
                **{
                    column: final_round[column]
                    for column in final_round.index
                    if column.startswith(
                        ("selected_class_", "pseudo_accuracy_class_")
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def compare_conditions(
    frame: pd.DataFrame,
    *,
    budget: str,
    left: str,
    right: str,
    samples: int = 20_000,
    seed: int = 20261028,
) -> dict[str, Any]:
    selected = frame.loc[frame["budget"].eq(budget)]
    pivot = selected.pivot_table(
        index=["fold", "seed"],
        columns="condition",
        values="macro_f1",
    )
    paired = pivot[[left, right]].dropna()
    if len(paired) != 20:
        raise ValueError(
            f"Expected 20 matched pairs for {budget}/{left}/{right}, got {len(paired)}"
        )
    delta = (paired[right] - paired[left]).rename("delta").to_frame()
    fold_means = delta.groupby(level="fold")["delta"].mean()
    low, high = _hierarchical_bootstrap_interval(
        delta,
        "delta",
        samples=samples,
        seed=seed,
    )
    wins = int((fold_means > 1e-12).sum())
    losses = int((fold_means < -1e-12).sum())
    ties = int(len(fold_means) - wins - losses)
    return {
        "budget": budget,
        "comparison": f"{right} minus {left}",
        "left": left,
        "right": right,
        "n_pairs": len(paired),
        "mean_delta": float(delta["delta"].mean()),
        "hierarchical_ci_low": low,
        "hierarchical_ci_high": high,
        "fold_signflip_p": _fold_signflip_p_value(delta, "delta"),
        "fold_wins": wins,
        "fold_ties": ties,
        "fold_losses": losses,
        "fold_deltas": ";".join(f"{value:.6f}" for value in fold_means.tolist()),
    }


def save_figure(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style(font_size=8.5)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    conditions = [
        "supervised",
        "fixed_threshold_0.60",
        "coverage_controlled_0.50",
    ]
    labels = ["Supervised", "Fixed tau=0.60", "Coverage 50%"]
    colors = [
        method_color("cwt_gru"),
        method_color("cwt_gru_temp"),
        method_color("cwt_gru_selftrain_coverage"),
    ]
    for axis, (budget, budget_label) in zip(
        axes,
        [("50_percent", "50% balanced budget"), ("full_balanced", "Full balanced budget")],
        strict=True,
    ):
        selected = summary.loc[summary["budget"].eq(budget)].set_index("condition")
        x = np.arange(len(conditions))
        values = selected.reindex(conditions)
        axis.bar(
            x,
            values["macro_f1_mean"],
            yerr=values["macro_f1_sd"],
            capsize=2,
            color=colors,
            edgecolor=INK_COLOR,
            linewidth=0.45,
            width=0.62,
        )
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylim(0, 0.55)
        axis.set_ylabel("Macro-F1")
        axis.set_title(budget_label, loc="left")
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    png = output_dir / "figure_17_self_training_sensitivity.png"
    figure.savefig(png, dpi=320, bbox_inches="tight")
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
        default=PROJECT_ROOT / "results" / "self_training_sensitivity",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "publication",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = load_canonical_reference(args.results_dir)
    coverage = load_coverage(args.sensitivity_dir)
    combined = pd.concat([canonical, coverage], ignore_index=True)
    summary = (
        combined.groupby(["budget", "condition"])
        .agg(
            runs=("macro_f1", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
        )
        .reset_index()
    )
    comparisons = []
    for budget in BUDGETS:
        comparisons.append(
            compare_conditions(
                combined,
                budget=budget,
                left="supervised",
                right="coverage_controlled_0.50",
                samples=args.bootstrap_samples,
            )
        )
        comparisons.append(
            compare_conditions(
                combined,
                budget=budget,
                left="fixed_threshold_0.60",
                right="coverage_controlled_0.50",
                samples=args.bootstrap_samples,
            )
        )
    comparison_frame = pd.DataFrame(comparisons)
    diagnostics = coverage.loc[
        :,
        [
            column
            for column in coverage.columns
            if column.startswith(
                (
                    "fold",
                    "seed",
                    "budget",
                    "run_id",
                    "pseudo_",
                    "selected_",
                    "rounds_",
                )
            )
        ],
    ]
    coverage.to_csv(
        args.results_dir / "self_training_coverage_per_run.csv",
        index=False,
        lineterminator="\n",
    )
    diagnostics.to_csv(
        args.results_dir / "pseudo_label_diagnostics.csv",
        index=False,
        lineterminator="\n",
    )
    summary.to_csv(
        args.results_dir / "self_training_sensitivity_summary.csv",
        index=False,
        lineterminator="\n",
    )
    comparison_frame.to_csv(
        args.results_dir / "self_training_sensitivity_comparisons.csv",
        index=False,
        lineterminator="\n",
    )
    save_figure(summary, comparison_frame, args.figures_dir)
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
