"""Summarize CWT preprocessing sensitivity experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

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
from ..public_benchmark.cwt_sensitivity import VARIANTS


VARIANTS["baseline"] = {
    "description": "48 linear magnitude scales, 96 x 96, per-channel max normalization"
}
LABELS = {
    "baseline": "Baseline magnitude CWT",
    "no_max_norm": "No max normalization",
    "log_magnitude": "Log magnitude",
    "resolution_128": "128 x 128 resolution",
    "scales_32": "32 scales",
    "scales_64": "64 scales",
    "real_imag": "Real + imaginary",
}


def load_results(results_dir: Path) -> pd.DataFrame:
    canonical = pd.read_csv(results_dir / "raw_runs.csv")
    baseline = canonical.loc[
        canonical["experiment_type"].eq("main")
        & canonical["method"].eq("cwt_cnn")
    ].copy()
    baseline["variant"] = "baseline"
    frames = [baseline]
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        path = results_dir / "cwt_sensitivity" / variant / "raw_runs.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame["variant"] = variant
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def comparisons(
    frame: pd.DataFrame,
    *,
    raw_reference: pd.DataFrame,
) -> pd.DataFrame:
    baseline = frame.loc[frame["variant"].eq("baseline")].set_index(
        ["fold", "seed"]
    )["macro_f1"]
    raw = raw_reference.set_index(["fold", "seed"])["macro_f1"]
    rows = []
    for reference_name, reference_values in (
        ("baseline_cwt_48", baseline),
        ("raw_1d_cnn", raw),
    ):
        for index, variant in enumerate(VARIANTS):
            if variant == "baseline":
                continue
            part = frame.loc[frame["variant"].eq(variant)]
            if part.empty:
                continue
            pivot = (
                part.set_index(["fold", "seed"])["macro_f1"]
                .rename("variant")
                .to_frame()
            )
            pivot["reference"] = reference_values.reindex(pivot.index)
            pivot["delta"] = pivot["variant"] - pivot["reference"]
            delta = pivot["delta"].rename("delta").to_frame()
            low, high = _hierarchical_bootstrap_interval(
                delta,
                "delta",
                samples=20_000,
                seed=20261028 + index + 100 * (reference_name == "raw_1d_cnn"),
            )
            fold_means = delta.groupby(level="fold")["delta"].mean()
            rows.append(
                {
                    "variant": variant,
                    "reference": reference_name,
                    "n_pairs": len(delta),
                    "variant_mean_macro_f1": float(pivot["variant"].mean()),
                    "reference_mean_macro_f1": float(pivot["reference"].mean()),
                    "mean_delta": float(delta["delta"].mean()),
                    "hierarchical_ci_low": low,
                    "hierarchical_ci_high": high,
                    "fold_signflip_p": _fold_signflip_p_value(delta, "delta"),
                    "fold_wins": int((fold_means > 1e-12).sum()),
                    "fold_ties": int((fold_means.abs() <= 1e-12).sum()),
                    "fold_losses": int((fold_means < -1e-12).sum()),
                }
            )
    return pd.DataFrame(rows)


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
    variants = [variant for variant in VARIANTS if variant in set(summary["variant"])]
    figure, axis = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    x = np.arange(len(variants))
    axis.bar(
        x,
        summary.set_index("variant").reindex(variants)["macro_f1_mean"],
        yerr=summary.set_index("variant").reindex(variants)["macro_f1_sd"],
        capsize=2,
        color=["#3D3D3D" if value == "baseline" else "#B5B5B5" for value in variants],
        edgecolor="black",
        linewidth=0.45,
        width=0.66,
    )
    axis.set_xticks(
        x,
        [LABELS[variant] for variant in variants],
        rotation=20,
        ha="right",
    )
    axis.set_ylim(0, 0.55)
    axis.set_ylabel("Held-out test Macro-F1")
    axis.grid(axis="y", color="0.9", linewidth=0.6)
    png = output_dir / "figure_19_cwt_sensitivity.png"
    pdf = output_dir / "figure_19_cwt_sensitivity.pdf"
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
    frame = load_results(args.results_dir)
    canonical_raw = pd.read_csv(args.results_dir / "raw_runs.csv")
    raw_reference = canonical_raw.loc[
        canonical_raw["experiment_type"].eq("main")
        & canonical_raw["method"].eq("raw_cnn")
    ]
    summary = (
        frame.groupby("variant")
        .agg(
            runs=("run_id", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            mean_best_epoch=("best_epoch", "mean"),
            mean_validation_macro_f1=("best_val_macro_f1", "mean"),
        )
        .reset_index()
    )
    comparison = comparisons(frame, raw_reference=raw_reference)
    frame.to_csv(
        args.results_dir / "cwt_sensitivity_per_run.csv",
        index=False,
        lineterminator="\n",
    )
    summary.to_csv(
        args.results_dir / "cwt_sensitivity_summary.csv",
        index=False,
        lineterminator="\n",
    )
    comparison.to_csv(
        args.results_dir / "cwt_sensitivity_comparisons.csv",
        index=False,
        lineterminator="\n",
    )
    save_figure(summary, args.figures_dir)
    print(summary.to_string(index=False))
    if not comparison.empty:
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
