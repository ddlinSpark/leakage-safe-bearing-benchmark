"""Generate publication figures from canonical CSV results only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..experiments.common import configure_logging, read_results
from ..paths import PROJECT_ROOT


METHOD_LABELS = {
    "raw_cnn": "Raw 1D-CNN",
    "gru_only": "GRU only",
    "cwt_cnn": "CWT 2D-CNN",
    "cwt_gru_no_se": "CWT-GRU w/o SE",
    "cwt_gru": "CWT-GRU",
    "cwt_gru_temp": "CWT-GRU + temp.",
    "cwt_gru_selftrain": "CWT-GRU + temp. + ST",
}

CLASS_LABELS = ["Healthy", "Outer ring", "Inner ring", "Multiple damage"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SCI-ready figures")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "publication",
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required figure input is missing: {path}. Run the experiment and "
            "statistical-analysis commands first."
        )
    return pd.read_csv(path)


def model_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary.loc[summary["method"].isin(METHOD_LABELS)].copy()
    data = data.sort_values("Macro-F1 Mean")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(data))
    ax.bar(
        x,
        data["Macro-F1 Mean"].to_numpy(),
        yerr=data["Macro-F1 Std"].fillna(0).to_numpy(),
        capsize=4,
        color="#315A8A",
    )
    ax.set_xticks(
        x,
        [METHOD_LABELS[value] for value in data["method"]],
        rotation=24,
        ha="right",
    )
    ax.set_ylabel("Macro-F1")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    save(fig, output_dir, "figure_01_model_comparison")


def ablation(ablation_summary: pd.DataFrame, output_dir: Path) -> None:
    data = ablation_summary.sort_values("Macro-F1 Mean")
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    x = np.arange(len(data))
    ax.bar(
        x,
        data["Macro-F1 Mean"].to_numpy(),
        yerr=data["Macro-F1 Std"].fillna(0).to_numpy(),
        capsize=4,
        color="#A75D42",
    )
    ax.set_xticks(
        x,
        [METHOD_LABELS.get(value, value) for value in data["Model"]],
        rotation=24,
        ha="right",
    )
    ax.set_ylabel("Macro-F1")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    save(fig, output_dir, "figure_02_ablation")


def validation_test_gap(domain_gap: pd.DataFrame, output_dir: Path) -> None:
    data = domain_gap.copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(data))
    width = 0.36
    ax.bar(
        x - width / 2,
        data["Validation Macro-F1 Mean"],
        width,
        yerr=data["Validation Macro-F1 Std"].fillna(0),
        capsize=4,
        color="#315A8A",
        label="Validation",
    )
    ax.bar(
        x + width / 2,
        data["Test Macro-F1 Mean"],
        width,
        yerr=data["Test Macro-F1 Std"].fillna(0),
        capsize=4,
        color="#B45F3C",
        label="Test",
    )
    for position, row in data.iterrows():
        ax.text(
            position,
            max(
                row["Validation Macro-F1 Mean"],
                row["Test Macro-F1 Mean"],
            )
            + 0.045,
            f"drop {row['Relative Drop (%)']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_xticks(
        x,
        [METHOD_LABELS[method] for method in data["Model"]],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("Macro-F1")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False)
    save(fig, output_dir, "figure_09_validation_test_gap")


def label_efficiency(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary.loc[
        summary["method"].isin(["cwt_gru", "cwt_gru_selftrain"])
    ].copy()
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    for method, marker, color in (
        ("cwt_gru", "o", "#315A8A"),
        ("cwt_gru_selftrain", "s", "#B45F3C"),
    ):
        subset = data.loc[data["method"].eq(method)].sort_values("label_ratio")
        positions = {
            0.1: 0,
            0.5: 1,
            1.0: 2,
        }
        ax.errorbar(
            [positions[round(float(value), 1)] for value in subset["label_ratio"]],
            subset["Macro-F1 Mean"],
            yerr=subset["Macro-F1 Std"].fillna(0),
            marker=marker,
            capsize=4,
            linewidth=1.8,
            color=color,
            label=METHOD_LABELS[method],
        )
    ax.set_xticks(
        [0, 1, 2],
        [
            "10% balanced budget",
            "50% balanced budget",
            "Full balanced budget",
        ],
    )
    ax.set_xlabel("Label budget")
    ax.set_ylabel("Macro-F1")
    ax.grid(color="#E5E5E5", linewidth=0.6)
    ax.legend(frameon=False)
    save(fig, output_dir, "figure_03_label_efficiency")


def threshold_sensitivity(summary: pd.DataFrame, output_dir: Path) -> None:
    model_column = "method" if "method" in summary.columns else "Model"
    method = summary.loc[
        summary[model_column].eq("cwt_gru_selftrain")
    ].sort_values("threshold")
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(6.3, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )
    ax1.errorbar(
        method["threshold"],
        method["Macro-F1 Mean"],
        yerr=method["Macro-F1 Std"].fillna(0),
        marker="o",
        capsize=4,
        color="#315A8A",
        label="Test Macro-F1",
    )
    ax1.set_ylabel("Macro-F1")
    ax1.grid(color="#E5E5E5", linewidth=0.6)
    ax1.legend(frameon=False)
    ax2.plot(
        method["threshold"],
        method.get("pseudo_accuracy_mean", np.nan),
        marker="s",
        color="#B45F3C",
        linewidth=1.8,
        label="Pseudo-label accuracy",
    )
    ax2.set_xlabel("Confidence threshold")
    ax2.set_ylabel("Pseudo-label accuracy")
    ax2.grid(color="#E5E5E5", linewidth=0.6)
    ax2.legend(frameon=False)
    save(fig, output_dir, "figure_04_threshold_sensitivity")


def calibration(raw: pd.DataFrame, output_dir: Path) -> None:
    methods = ["raw_cnn", "cwt_cnn", "cwt_gru", "cwt_gru_selftrain"]
    data = raw.loc[
        raw["experiment_type"].eq("main")
        & raw["method"].isin(methods)
        & raw["label_ratio"].eq(1.0)
    ]
    grouped = (
        data.groupby("method")[["raw_ece", "ece"]]
        .mean()
        .reindex([value for value in methods if value in set(data["method"])])
    )
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    x = np.arange(len(grouped))
    width = 0.36
    ax.bar(x - width / 2, grouped["raw_ece"], width, label="Uncalibrated", color="#8D8D8D")
    ax.bar(
        x + width / 2,
        grouped["ece"],
        width,
        label="Temperature scaled",
        color="#315A8A",
    )
    ax.set_xticks(
        x,
        [METHOD_LABELS[value] for value in grouped.index],
        rotation=22,
        ha="right",
    )
    ax.set_ylabel("Expected calibration error")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False)
    save(fig, output_dir, "figure_05_calibration")


def confusion(raw: pd.DataFrame, runs_dir: Path, output_dir: Path) -> None:
    methods = ["raw_cnn", "cwt_cnn", "cwt_gru", "cwt_gru_selftrain"]
    selected = raw.loc[
        raw["experiment_type"].eq("main")
        & raw["method"].isin(methods)
        & raw["label_ratio"].eq(1.0)
    ][["run_id", "method"]]
    frames: list[pd.DataFrame] = []
    for record in selected.itertuples(index=False):
        path = runs_dir / record.run_id / "confusion_matrix.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame["method"] = record.method
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No per-run confusion matrices were found")
    combined = pd.concat(frames, ignore_index=True)
    aggregate = (
        combined.groupby(["method", "true_class", "predicted_class"], as_index=False)[
            "count"
        ]
        .sum()
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 9.2),
        constrained_layout=True,
    )
    for number, (ax, method) in enumerate(
        zip(axes.flatten(), methods, strict=True)
    ):
        subset = aggregate.loc[aggregate["method"].eq(method)]
        matrix = np.zeros((4, 4), dtype=np.float64)
        for row in subset.itertuples(index=False):
            matrix[int(row.true_class), int(row.predicted_class)] = row.count
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for true_class in range(4):
            for predicted_class in range(4):
                color = "white" if normalized[true_class, predicted_class] > 0.5 else "black"
                ax.text(
                    predicted_class,
                    true_class,
                    f"{normalized[true_class, predicted_class]:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )
        ax.set_xticks(range(4), CLASS_LABELS, rotation=32, ha="right")
        ax.set_yticks(range(4), CLASS_LABELS)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.set_box_aspect(1)
        if number >= 2:
            ax.set_xlabel("Predicted class")
        if number % 2 == 0:
            ax.set_ylabel("True class")
        ax.set_title(METHOD_LABELS[method])
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.03,
        label="Row-normalized count",
    )
    save(fig, output_dir, "figure_06_confusion_matrices")


def computational_cost(cost: pd.DataFrame, output_dir: Path) -> None:
    data = cost.drop_duplicates("model_key").copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    x = np.arange(len(data))
    labels = [
        METHOD_LABELS.get(method, str(method))
        for method in data["model_key"]
    ]
    ax1.bar(x, data["parameters"] / 1e6, color="#315A8A")
    ax1.set_ylabel("Parameters (million)")
    ax1.set_xticks(x, labels, rotation=25, ha="right")
    ax2.bar(x, data["inference_ms_per_batch_median"], color="#B45F3C")
    ax2.set_ylabel("Inference latency (ms/batch)")
    ax2.set_xticks(x, labels, rotation=25, ha="right")
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.tight_layout()
    save(fig, output_dir, "figure_07_computational_cost")


def reliability_diagrams(raw: pd.DataFrame, results_dir: Path, output_dir: Path) -> None:
    methods = ["raw_cnn", "cwt_cnn", "cwt_gru", "cwt_gru_selftrain"]
    selected = raw.loc[
        raw["experiment_type"].eq("main")
        & raw["method"].isin(methods)
        & raw["label_ratio"].eq(1.0)
    ][["run_id", "method"]]
    frames: list[pd.DataFrame] = []
    for record in selected.itertuples(index=False):
        path = results_dir / "runs" / record.run_id / "calibration_bins.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame["method"] = record.method
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No per-run calibration bins were found")
    bins = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, float | str | int]] = []
    for (method, source, bin_id), group in bins.groupby(
        ["method", "source", "bin"],
        dropna=False,
    ):
        count = group["count"].sum()
        if count <= 0:
            continue
        rows.append(
            {
                "method": method,
                "source": source,
                "bin": int(bin_id),
                "count": int(count),
                "mean_confidence": float(
                    (group["mean_confidence"] * group["count"]).sum() / count
                ),
                "accuracy": float((group["accuracy"] * group["count"]).sum() / count),
            }
        )
    aggregate = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.8), constrained_layout=True)
    for ax, method in zip(axes.flatten(), methods, strict=True):
        subset = aggregate.loc[aggregate["method"].eq(method)]
        ax.plot([0, 1], [0, 1], color="#9A9A9A", linestyle="--", linewidth=1)
        for source, marker, color, label in (
            ("raw", "o", "#727272", "Uncalibrated"),
            ("temperature", "s", "#315A8A", "Temperature scaled"),
        ):
            data = subset.loc[subset["source"].eq(source)].sort_values(
                "mean_confidence"
            )
            ax.plot(
                data["mean_confidence"],
                data["accuracy"],
                marker=marker,
                markersize=4.5,
                linewidth=1.5,
                color=color,
                label=label,
            )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(METHOD_LABELS[method])
        ax.grid(color="#E4E4E4", linewidth=0.6)
        ax.legend(frameon=False, loc="upper left")
    save(fig, output_dir, "figure_08_reliability_diagrams")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    configure_logging(results_dir / "figure_generation.log")
    setup_style()
    summary = read_csv(results_dir / "summary.csv")
    ablation_summary = read_csv(results_dir / "ablation_summary.csv")
    label_summary = read_csv(results_dir / "label_efficiency_summary.csv")
    threshold_summary = read_csv(results_dir / "threshold_summary.csv")
    raw = read_results(results_dir / "raw_runs.csv")
    cost = read_csv(results_dir / "model_complexity.csv")

    model_comparison(summary, output_dir)
    ablation(ablation_summary, output_dir)
    validation_test_gap(
        read_csv(results_dir / "domain_gap_summary.csv"),
        output_dir,
    )
    label_efficiency(label_summary, output_dir)
    threshold_sensitivity(threshold_summary, output_dir)
    calibration(raw, output_dir)
    confusion(raw, results_dir / "runs", output_dir)
    computational_cost(cost, output_dir)
    reliability_diagrams(raw, results_dir, output_dir)


if __name__ == "__main__":
    main()
