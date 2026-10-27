"""Generate SCI manuscript figures from repeated public benchmark results."""

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
    "cwt_cnn": "CWT 2D-CNN",
    "cwt_gru": "CWT-GRU",
    "cwt_gru_selftrain": "CWT-GRU + calibrated self-training",
}
CLASS_LABELS = ["Healthy", "Outer ring", "Inner ring", "Multiple damage"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
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
        default=OUTPUTS_DIR / "public_paderborn" / "results.csv",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn" / "analysis" / "summary_with_ci.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn" / "predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn" / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results)
    summary = pd.read_csv(args.analysis)
    colors = ["#4D4D4D", "#8C8C8C", "#2F5597", "#C55A11"]

    full = summary[np.isclose(summary["label_fraction"], 1.0)].copy()
    full["method_label"] = full["method"].map(METHOD_LABELS).fillna(full["method"])
    full = full.sort_values("macro_f1_mean")
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(full))
    errors = np.nan_to_num(full["macro_f1_std"].to_numpy())
    ax.bar(x, full["macro_f1_mean"], yerr=errors, capsize=3, color=colors[: len(full)])
    ax.set_xticks(x, full["method_label"], rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro-F1")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.savefig(args.output_dir / "figure_main_comparison.png")
    plt.close(fig)

    budget_methods = [method for method in ("cwt_gru", "cwt_gru_selftrain") if method in set(summary["method"])]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for idx, method in enumerate(budget_methods):
        subset = summary[summary["method"] == method].sort_values("label_fraction")
        ax.errorbar(
            subset["label_fraction"] * 100,
            subset["macro_f1_mean"],
            yerr=np.nan_to_num(subset["macro_f1_std"]),
            marker="o" if idx == 0 else "s",
            capsize=3,
            linewidth=1.6,
            color=colors[2 + idx],
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Labeled training windows per class (%)")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.0)
    ax.grid(color="#E5E5E5", linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(args.output_dir / "figure_label_efficiency.png")
    plt.close(fig)

    calibration_results = results[np.isclose(results["label_fraction"], 1.0)]
    calibration = (
        calibration_results.groupby("method")[["raw_ece", "cal_ece"]]
        .mean()
        .reindex([method for method in METHOD_LABELS if method in set(results["method"])])
    )
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(calibration))
    width = 0.36
    ax.bar(x - width / 2, calibration["raw_ece"], width, label="Before scaling", color="#8C8C8C")
    ax.bar(x + width / 2, calibration["cal_ece"], width, label="Temperature scaled", color="#2F5597")
    ax.set_xticks(x, [METHOD_LABELS[item] for item in calibration.index], rotation=18, ha="right")
    ax.set_ylabel("Expected calibration error")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False)
    fig.savefig(args.output_dir / "figure_calibration.png")
    plt.close(fig)

    prediction_files = sorted(args.predictions.glob("*.npz"))
    if prediction_files:
        preferred = [
            path
            for path in prediction_files
            if "cwt_gru_selftrain" in path.name and "fraction1.00" in path.name
        ]
        selected = preferred[0] if preferred else prediction_files[0]
        with np.load(selected) as prediction:
            logits = prediction["logits"]
            labels = prediction["labels"]
            temperature = float(prediction["temperature"])
        probabilities = np.exp(logits / temperature)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        matrix = confusion_matrix(labels, probabilities.argmax(axis=1), normalize="true")
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ConfusionMatrixDisplay(matrix, display_labels=CLASS_LABELS).plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format=".2f",
        )
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.savefig(args.output_dir / "figure_confusion_matrix.png")
        plt.close(fig)

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
