"""Post-hoc class-coarsening sensitivity analysis on retained predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from ..evaluation.calibration import probabilities_from_logits
from ..paths import PROJECT_ROOT
from ..visualization.style import (
    GRID_COLOR,
    INK_COLOR,
    PAPER_COLORS,
    apply_paper_style,
)


METHOD_LABELS = {
    "raw_cnn": "Raw 1D-CNN",
    "gru_only": "GRU only",
    "cwt_cnn": "CWT 2D-CNN",
    "cwt_gru_no_se": "CWT-GRU without SE",
    "cwt_gru": "CWT-GRU",
    "cwt_gru_selftrain": "CWT-GRU + self-training",
}
PRINCIPAL_METHODS = list(METHOD_LABELS)


def coarse_targets(four_class: np.ndarray, granularity: str) -> np.ndarray:
    if granularity == "binary":  # healthy vs any fault
        return (four_class != 0).astype(np.int64)
    if granularity == "three_class":  # healthy, single-location, multiple
        return np.select(
            [four_class == 0, np.isin(four_class, [1, 2]), four_class == 3],
            [0, 1, 2],
            default=-1,
        )
    raise ValueError(granularity)


def aggregate_probabilities(
    probabilities: np.ndarray,
    granularity: str,
) -> np.ndarray:
    healthy = probabilities[:, 0]
    if granularity == "binary":
        return np.column_stack([healthy, 1.0 - healthy])
    if granularity == "three_class":
        return np.column_stack(
            [
                healthy,
                probabilities[:, 1] + probabilities[:, 2],
                probabilities[:, 3],
            ]
        )
    raise ValueError(granularity)


def metrics(
    labels: np.ndarray,
    predicted: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    class_ids = np.arange(num_classes)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(
                labels,
                predicted,
                labels=class_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }


def audit_run(results_dir: Path, row: pd.Series) -> list[dict[str, Any]]:
    prediction_path = results_dir / "runs" / str(row["run_id"]) / "predictions.npz"
    with np.load(prediction_path) as prediction:
        logits = prediction["logits"]
        labels = prediction["labels"]
        temperature = float(prediction["temperature"])
    probabilities = probabilities_from_logits(logits, temperature)
    base = {
        "run_id": str(row["run_id"]),
        "method": str(row["method"]),
        "model_label": METHOD_LABELS.get(str(row["method"]), str(row["method"])),
        "fold": int(row["fold"]),
        "seed": int(row["seed"]),
        "runs": 1,
        "sampling_note": "Post-hoc coarsening of the trained four-class outputs",
    }
    outputs: list[dict[str, Any]] = []
    four_predicted = probabilities.argmax(axis=1)
    outputs.append(
        {
            **base,
            "granularity": "four_class",
            "class_definition": "Healthy; outer; inner; multiple",
            "num_classes": 4,
            **metrics(labels, four_predicted, 4),
        }
    )
    for granularity, definition, num_classes in (
        ("binary", "Healthy vs any fault", 2),
        ("three_class", "Healthy; single-location; multiple", 3),
    ):
        targets = coarse_targets(labels, granularity)
        coarse_probabilities = aggregate_probabilities(probabilities, granularity)
        predicted = coarse_probabilities.argmax(axis=1)
        outputs.append(
            {
                **base,
                "granularity": granularity,
                "class_definition": definition,
                "num_classes": num_classes,
                **metrics(targets, predicted, num_classes),
            }
        )
    return outputs


def save_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style(font_size=8.5)
    figure, axis = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    methods = [method for method in PRINCIPAL_METHODS if method in set(summary["method"])]
    granularities = ["four_class", "three_class", "binary"]
    labels = ["Four-class", "Healthy / single / multiple", "Healthy / fault"]
    colors = [PAPER_COLORS[0], PAPER_COLORS[3], PAPER_COLORS[8]]
    x = np.arange(len(methods))
    width = 0.25
    for index, (granularity, label, color) in enumerate(
        zip(granularities, labels, colors, strict=True)
    ):
        selected = (
            summary.loc[summary["granularity"].eq(granularity)]
            .set_index("method")
            .reindex(methods)
        )
        axis.bar(
            x + (index - 1) * width,
            selected["macro_f1_mean"],
            width=width,
            yerr=selected["macro_f1_sd"],
            capsize=2,
            color=color,
            edgecolor=INK_COLOR,
            linewidth=0.45,
            label=label,
        )
    axis.set_xticks(
        x,
        [METHOD_LABELS[method] for method in methods],
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("Macro-F1")
    axis.set_ylim(0, 1.04)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    axis.legend(frameon=False, ncol=3, fontsize=7)
    figure.savefig(
        output_dir / "figure_16_label_granularity.png",
        dpi=320,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "figure_16_label_granularity.pdf",
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
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "publication",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.results_dir / "raw_runs.csv")
    selected = raw.loc[
        raw["experiment_type"].eq("main")
        & raw["method"].isin(PRINCIPAL_METHODS)
    ]
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        rows.extend(audit_run(args.results_dir, row))
    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby(
            ["method", "model_label", "granularity", "class_definition", "num_classes"]
        )
        .agg(
            runs=("run_id", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
        )
        .reset_index()
    )
    per_run.to_csv(
        args.results_dir / "label_granularity_per_run.csv",
        index=False,
        lineterminator="\n",
    )
    summary.to_csv(
        args.results_dir / "label_granularity_sensitivity.csv",
        index=False,
        lineterminator="\n",
    )
    save_figure(summary, args.figures_dir)
    print(
        f"Wrote {len(per_run)} granularity evaluations from "
        f"{selected['run_id'].nunique()} trained four-class runs."
    )


if __name__ == "__main__":
    main()
