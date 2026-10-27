"""Generate self-contained SCI manuscript figures from benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from ..paths import OUTPUTS_DIR


METHOD_LABELS = {
    "raw_cnn": "Raw 1D-CNN",
    "gru_only": "GRU only",
    "cwt_cnn": "CWT 2D-CNN",
    "cwt_gru_no_se": "CWT-GRU without SE",
    "cwt_gru": "CWT-GRU",
    "cwt_gru_selftrain": "CWT-GRU + calibrated self-training",
}
CLASS_LABELS = ["Healthy", "Outer ring", "Inner ring", "Multiple damage"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成公共基准论文图表")
    parser.add_argument(
        "--results",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn_final" / "results_extended.csv",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=OUTPUTS_DIR
        / "public_paderborn_final"
        / "analysis_extended"
        / "summary_with_ci.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn_final" / "predictions",
    )
    parser.add_argument(
        "--threshold-results",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn_thresholds" / "summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn_final" / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results)
    summary = pd.read_csv(args.analysis)

    full = summary[np.isclose(summary["label_fraction"], 1.0)].copy()
    full = full[full["method"].isin(METHOD_LABELS)]
    full["method_label"] = full["method"].map(METHOD_LABELS)
    full = full.sort_values("macro_f1_mean")
    colors = plt.cm.tab20(np.linspace(0.0, 0.9, len(full)))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(full))
    errors = np.nan_to_num(full["macro_f1_std"].to_numpy())
    ax.bar(x, full["macro_f1_mean"], yerr=errors, capsize=4, color=colors)
    ax.set_xticks(x, full["method_label"], rotation=22, ha="right")
    upper = float(np.nanmax(full["macro_f1_mean"] + full["macro_f1_std"]))
    ax.set_ylim(0, min(1.0, upper * 1.25))
    ax.set_ylabel("Macro-F1 (higher is better)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.savefig(args.output_dir / "figure_main_comparison.png")
    plt.close(fig)

    budget_methods = [
        method
        for method in ("cwt_gru", "cwt_gru_selftrain")
        if method in set(summary["method"])
    ]
    fig, ax = plt.subplots(figsize=(6.1, 4.0))
    for idx, method in enumerate(budget_methods):
        subset = summary[summary["method"] == method].sort_values("label_fraction")
        ax.errorbar(
            subset["label_fraction"] * 100,
            subset["macro_f1_mean"],
            yerr=np.nan_to_num(subset["macro_f1_std"]),
            marker="o" if idx == 0 else "s",
            capsize=4,
            linewidth=1.8,
            color="#2F5597" if idx == 0 else "#C55A11",
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Labeled training windows per class (%)")
    ax.set_ylabel("Macro-F1 (higher is better)")
    budget_upper = float(
        np.nanmax(
            summary.loc[
                summary["method"].isin(budget_methods),
                "macro_f1_mean",
            ]
            + summary.loc[
                summary["method"].isin(budget_methods),
                "macro_f1_std",
            ]
        )
    )
    ax.set_ylim(0, min(1.0, budget_upper * 1.2))
    ax.grid(color="#E5E5E5", linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(args.output_dir / "figure_label_efficiency.png")
    plt.close(fig)

    calibration_results = results[
        np.isclose(results["label_fraction"], 1.0)
        & results["method"].isin(
            ["raw_cnn", "cwt_cnn", "cwt_gru", "cwt_gru_selftrain"]
        )
    ]
    calibration = (
        calibration_results.groupby("method")[["raw_ece", "cal_ece"]]
        .mean()
        .reindex(
            [
                method
                for method in (
                    "raw_cnn",
                    "cwt_cnn",
                    "cwt_gru",
                    "cwt_gru_selftrain",
                )
                if method in set(calibration_results["method"])
            ]
        )
    )
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    x = np.arange(len(calibration))
    width = 0.36
    ax.bar(
        x - width / 2,
        calibration["raw_ece"],
        width,
        label="Before scaling",
        color="#8C8C8C",
    )
    ax.bar(
        x + width / 2,
        calibration["cal_ece"],
        width,
        label="Temperature scaled",
        color="#2F5597",
    )
    ax.set_xticks(
        x,
        [METHOD_LABELS[item] for item in calibration.index],
        rotation=22,
        ha="right",
    )
    ax.set_ylabel("Expected calibration error (lower is better)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False)
    fig.savefig(args.output_dir / "figure_calibration.png")
    plt.close(fig)

    principal_methods = ["raw_cnn", "cwt_cnn", "cwt_gru", "cwt_gru_selftrain"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2))
    for ax, method in zip(axes.flatten(), principal_methods):
        count_matrix = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int64)
        for path in sorted(
            args.predictions.glob(f"*_{method}_fraction1.00_seed*.npz")
        ):
            with np.load(path) as prediction:
                logits = prediction["logits"]
                labels = prediction["labels"]
                temperature = float(prediction["temperature"])
            probabilities = np.exp(logits / temperature)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            count_matrix += confusion_matrix(
                labels,
                probabilities.argmax(axis=1),
                labels=np.arange(len(CLASS_LABELS)),
            )
        normalized = count_matrix / np.maximum(
            count_matrix.sum(axis=1, keepdims=True),
            1,
        )
        ConfusionMatrixDisplay(normalized, display_labels=CLASS_LABELS).plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format=".2f",
        )
        ax.set_title(METHOD_LABELS[method], pad=8)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        plt.setp(ax.get_xticklabels(), rotation=24, ha="right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_confusion_matrix.png")
    plt.close(fig)

    if args.threshold_results.is_file():
        threshold_summary = pd.read_csv(args.threshold_results)
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(6.3, 6.2),
            sharex=True,
            gridspec_kw={"height_ratios": [1.35, 1.0]},
        )
        ax1.errorbar(
            threshold_summary["threshold"],
            threshold_summary["test_macro_f1_mean"],
            yerr=threshold_summary["test_macro_f1_std"],
            marker="o",
            linewidth=1.8,
            capsize=4,
            color="#2F5597",
            label="Test Macro-F1",
        )
        ax1.set_ylabel("Test Macro-F1")
        ax1.set_ylim(
            0,
            min(
                1.0,
                float(
                    np.nanmax(
                        threshold_summary["test_macro_f1_mean"]
                        + threshold_summary["test_macro_f1_std"]
                    )
                )
                * 1.35,
            ),
        )
        ax1.grid(color="#E5E5E5", linewidth=0.6)
        ax1.legend(frameon=False, loc="upper right")

        ax2.bar(
            threshold_summary["threshold"],
            threshold_summary["pseudo_count_mean"],
            width=0.035,
            color="#D9E2F3",
            edgecolor="#2F5597",
            label="Pseudo-label count",
        )
        ax2.set_ylabel("Pseudo-label count")
        ax2.set_xlabel("Confidence threshold")
        ax2.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        accuracy_axis = ax2.twinx()
        accuracy_axis.plot(
            threshold_summary["threshold"],
            threshold_summary["pseudo_accuracy_mean"],
            marker="s",
            linewidth=1.5,
            linestyle="--",
            color="#C55A11",
            label="Pseudo-label accuracy",
        )
        accuracy_axis.set_ylabel("Pseudo-label accuracy")
        accuracy_axis.set_ylim(0.6, 1.02)
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="#D9E2F3", edgecolor="#2F5597"),
            plt.Line2D([0], [0], color="#C55A11", linestyle="--", marker="s"),
        ]
        ax2.legend(
            handles,
            ["Pseudo-label count", "Pseudo-label accuracy"],
            frameon=False,
            loc="lower left",
        )
        fig.tight_layout()
        fig.savefig(args.output_dir / "figure_threshold_sensitivity.png")
        plt.close(fig)

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
