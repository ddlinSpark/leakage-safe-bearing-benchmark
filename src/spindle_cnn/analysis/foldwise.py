"""Fold-wise paired effects and descriptive win/tie/loss counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import PROJECT_ROOT
from ..plotting.style import (
    GRID_COLOR,
    INK_COLOR,
    MUTED_COLOR,
    PAPER_COLORS,
    apply_paper_style,
)


KEY_COMPARISONS = [
    "main:raw_cnn:cwt_cnn",
    "main:raw_cnn:cwt_gru",
    "main:cwt_gru_no_se:cwt_gru",
    "label_50:cwt_gru:cwt_gru_selftrain",
    "label_full:cwt_gru:cwt_gru_selftrain",
]

LABELS = {
    "main:raw_cnn:cwt_cnn": "CWT 2D-CNN - Raw",
    "main:raw_cnn:cwt_gru": "CWT-GRU - Raw",
    "main:cwt_gru_no_se:cwt_gru": "Full CWT-GRU - no-SE",
    "label_50:cwt_gru:cwt_gru_selftrain": "Self-training - supervised (50%)",
    "label_full:cwt_gru:cwt_gru_selftrain": "Self-training - supervised (Full)",
}


def comparison_frame(
    raw: pd.DataFrame,
    statistics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for comparison in KEY_COMPARISONS:
        row = statistics.loc[statistics["comparison"].eq(comparison)]
        if len(row) != 1:
            raise ValueError(f"Expected one statistics row for {comparison}")
        row = row.iloc[0]
        if comparison.startswith("label_50:"):
            selected = raw.loc[
                raw["experiment_type"].eq("label_efficiency")
                & np.isclose(raw["label_ratio"].to_numpy(dtype=float), 0.5)
                & raw["method"].isin([row["reference"], row["method"]])
            ]
        elif comparison.startswith("label_full:"):
            selected = raw.loc[
                raw["experiment_type"].eq("main")
                & np.isclose(raw["label_ratio"].to_numpy(dtype=float), 1.0)
                & raw["method"].isin([row["reference"], row["method"]])
            ]
        else:
            selected = raw.loc[
                raw["experiment_type"].eq("main")
                & raw["method"].isin([row["reference"], row["method"]])
            ]
        pivot = selected.pivot_table(
            index=["fold", "seed"],
            columns="method",
            values="macro_f1",
        )
        delta = pivot[row["method"]] - pivot[row["reference"]]
        for fold, values in delta.groupby(level="fold"):
            rows.append(
                {
                    "comparison": comparison,
                    "label": LABELS[comparison],
                    "reference": row["reference"],
                    "method": row["method"],
                    "fold": int(fold),
                    "n_seeds": len(values),
                    "mean_delta": float(values.mean()),
                    "sd_delta": float(values.std(ddof=1)),
                    **{
                        f"seed_{seed}": float(value)
                        for seed, value in values.droplevel("fold").items()
                    },
                }
            )
    return pd.DataFrame(rows)


def win_table(foldwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison, group in foldwise.groupby("comparison", sort=False):
        values = group["mean_delta"].to_numpy(dtype=float)
        rows.append(
            {
                "comparison": comparison,
                "label": group["label"].iloc[0],
                "fold_wins": int((values > 1e-12).sum()),
                "fold_ties": int((np.abs(values) <= 1e-12).sum()),
                "fold_losses": int((values < -1e-12).sum()),
                "mean_fold_delta": float(values.mean()),
                "minimum_fold_delta": float(values.min()),
                "maximum_fold_delta": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def save_figure(foldwise: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style(font_size=8.5)
    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    comparisons = [
        comparison
        for comparison in KEY_COMPARISONS
        if comparison in set(foldwise["comparison"])
    ]
    axis.axhline(0, color=MUTED_COLOR, linewidth=0.8, linestyle="--")
    for position, comparison in enumerate(comparisons):
        group = foldwise.loc[foldwise["comparison"].eq(comparison)]
        axis.scatter(
            np.full(len(group), position),
            group["mean_delta"],
            facecolors=[
                PAPER_COLORS[3] if value >= 0 else PAPER_COLORS[10]
                for value in group["mean_delta"]
            ],
            edgecolors=INK_COLOR,
            linewidths=0.5,
            s=34,
            zorder=3,
        )
        for _, row in group.iterrows():
            axis.text(
                position + 0.08,
                row["mean_delta"],
                f"F{int(row['fold'])}",
                va="center",
                fontsize=6.5,
            )
    axis.set_xticks(
        np.arange(len(comparisons)),
        [LABELS[comparison] for comparison in comparisons],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Fold mean paired Macro-F1 difference")
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    png = output_dir / "figure_20_foldwise_paired_differences.png"
    pdf = output_dir / "figure_20_foldwise_paired_differences.pdf"
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
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "publication",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.results_dir / "raw_runs.csv")
    statistics = pd.read_csv(args.results_dir / "hierarchical_statistics.csv")
    foldwise = comparison_frame(raw, statistics)
    wins = win_table(foldwise)
    foldwise.to_csv(
        args.results_dir / "foldwise_paired_differences.csv",
        index=False,
        lineterminator="\n",
    )
    wins.to_csv(
        args.results_dir / "win_tie_loss.csv",
        index=False,
        lineterminator="\n",
    )
    save_figure(foldwise, args.figures_dir)
    print(wins.to_string(index=False))


if __name__ == "__main__":
    main()
