"""Per-run and per-fold calibration audit from retained logits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..evaluation.calibration import probabilities_from_logits
from ..evaluation.metrics import metrics_from_logits
from ..paths import PROJECT_ROOT
from ..visualization.style import (
    AUXILIARY_COLORS,
    GRID_COLOR,
    INK_COLOR,
    MUTED_COLOR,
    apply_paper_style,
    method_color,
)


METHOD_LABELS = {
    "raw_cnn": "Raw 1D-CNN",
    "gru_only": "GRU only",
    "cwt_cnn": "CWT 2D-CNN",
    "cwt_gru_no_se": "CWT-GRU without SE",
    "cwt_gru": "CWT-GRU",
    "cwt_gru_temp": "CWT-GRU + temperature",
    "cwt_gru_selftrain": "CWT-GRU + self-training",
}


def calibration_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> dict[str, float]:
    raw = metrics_from_logits(logits, labels, temperature=1.0)
    scaled = metrics_from_logits(logits, labels, temperature=temperature)
    return {
        "ece_before": raw["ece"],
        "ece_after": scaled["ece"],
        "nll_before": raw["nll"],
        "nll_after": scaled["nll"],
        "brier_before": raw["brier"],
        "brier_after": scaled["brier"],
    }


def audit_run(results_dir: Path, row: pd.Series) -> dict[str, Any]:
    run_id = str(row["run_id"])
    prediction_path = results_dir / "runs" / run_id / "predictions.npz"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    with np.load(prediction_path) as prediction:
        logits = prediction["logits"]
        labels = prediction["labels"]
        temperature = float(prediction["temperature"])
        validation_logits = (
            prediction["validation_logits"]
            if "validation_logits" in prediction
            else np.empty((0, logits.shape[1]), dtype=np.float32)
        )
        validation_labels = (
            prediction["validation_labels"]
            if "validation_labels" in prediction
            else np.empty((0,), dtype=np.int64)
        )
    test_metrics = calibration_metrics(logits, labels, temperature)
    validation_available = len(validation_logits) > 0 and len(validation_labels) > 0
    validation_metrics = (
        calibration_metrics(
            validation_logits,
            validation_labels,
            temperature,
        )
        if validation_available
        else {
            key: np.nan for key in test_metrics
        }
    )
    return {
        "model": str(row["method"]),
        "model_label": METHOD_LABELS.get(str(row["method"]), str(row["method"])),
        "experiment_type": str(row["experiment_type"]),
        "fold": int(row["fold"]),
        "seed": int(row["seed"]),
        "run_id": run_id,
        "fitted_temperature": temperature,
        "temperature_at_lower_bound": bool(temperature <= 0.050001),
        "temperature_at_upper_bound": bool(temperature >= 19.999),
        "validation_logits_available": validation_available,
        **{
            key.replace("ece_", "validation_ece_"): value
            for key, value in validation_metrics.items()
            if key.startswith("ece_")
        },
        **{
            key.replace("nll_", "validation_nll_"): value
            for key, value in validation_metrics.items()
            if key.startswith("nll_")
        },
        **{
            key.replace("brier_", "validation_brier_"): value
            for key, value in validation_metrics.items()
            if key.startswith("brier_")
        },
        **{
            key.replace("ece_", "test_ece_"): value
            for key, value in test_metrics.items()
            if key.startswith("ece_")
        },
        **{
            key.replace("nll_", "test_nll_"): value
            for key, value in test_metrics.items()
            if key.startswith("nll_")
        },
        **{
            key.replace("brier_", "test_brier_"): value
            for key, value in test_metrics.items()
            if key.startswith("brier_")
        },
    }


def reliability_bins(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    *,
    n_bins: int = 15,
) -> pd.DataFrame:
    probabilities = probabilities_from_logits(logits, temperature)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (confidence > lower) & (confidence <= upper)
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "midpoint": (lower + upper) / 2,
                "count": int(mask.sum()),
                "confidence": (
                    float(confidence[mask].mean()) if mask.any() else np.nan
                ),
                "accuracy": (
                    float((predicted[mask] == labels[mask]).mean())
                    if mask.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_reliability(
    results_dir: Path,
    raw: pd.DataFrame,
    *,
    method: str,
    folds: list[int],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    selected = raw.loc[
        raw["experiment_type"].eq("main") & raw["method"].eq(method)
    ]
    for fold in folds:
        run_bins: dict[str, list[pd.DataFrame]] = {
            "before": [],
            "after": [],
        }
        for _, row in selected.loc[selected["fold"].eq(fold)].iterrows():
            path = results_dir / "runs" / str(row["run_id"]) / "predictions.npz"
            with np.load(path) as prediction:
                logits = prediction["logits"]
                labels = prediction["labels"]
                temperature = float(prediction["temperature"])
            run_bins["before"].append(
                reliability_bins(logits, labels, 1.0)
            )
            run_bins["after"].append(
                reliability_bins(logits, labels, temperature)
            )
        for source, frames in run_bins.items():
            combined = pd.concat(frames, ignore_index=True)
            count = combined.groupby("bin", as_index=False)["count"].sum()
            weighted_confidence = combined.assign(
                weighted=combined["confidence"].fillna(0) * combined["count"]
            ).groupby("bin", as_index=False)["weighted"].sum()
            weighted_accuracy = combined.assign(
                weighted=combined["accuracy"].fillna(0) * combined["count"]
            ).groupby("bin", as_index=False)["weighted"].sum()
            aggregated = count.merge(weighted_confidence, on="bin").merge(
                weighted_accuracy,
                on="bin",
                suffixes=("_confidence", "_accuracy"),
            )
            aggregated["confidence"] = (
                aggregated["weighted_confidence"]
                / aggregated["count"].replace(0, np.nan)
            )
            aggregated["accuracy"] = (
                aggregated["weighted_accuracy"]
                / aggregated["count"].replace(0, np.nan)
            )
            lower = combined.groupby("bin", as_index=False)["lower"].first()
            upper = combined.groupby("bin", as_index=False)["upper"].first()
            midpoint = combined.groupby("bin", as_index=False)["midpoint"].first()
            aggregated = aggregated.merge(lower, on="bin").merge(
                upper,
                on="bin",
            ).merge(midpoint, on="bin")
            aggregated["method"] = method
            aggregated["fold"] = fold
            aggregated["source"] = source
            aggregated["temperature"] = np.nan
            rows.append(aggregated)
    return pd.concat(rows, ignore_index=True)


def save_figures(
    per_run: pd.DataFrame,
    fold_bins: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style(font_size=8.5)
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    axes = axes.ravel()
    colors = {
        "before": AUXILIARY_COLORS["reference"],
        "after": method_color("cwt_gru_temp"),
    }
    for fold, axis in zip(sorted(fold_bins["fold"].unique()), axes, strict=True):
        axis.plot(
            [0, 1],
            [0, 1],
            color=MUTED_COLOR,
            linewidth=0.8,
            linestyle="--",
        )
        for source in ("before", "after"):
            part = fold_bins.loc[
                fold_bins["fold"].eq(fold) & fold_bins["source"].eq(source)
            ]
            valid = part.loc[part["count"].gt(0)]
            axis.plot(
                valid["confidence"],
                valid["accuracy"],
                marker="o",
                markersize=3.2,
                linewidth=1.0,
                color=colors[source],
                linestyle="-" if source == "before" else "--",
                label="Before" if source == "before" else "After",
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_title(f"Fold {fold}", loc="left")
        axis.set_xlabel("Confidence")
        axis.set_ylabel("Accuracy")
        axis.grid(color=GRID_COLOR, linewidth=0.6)
        axis.legend(frameon=False, fontsize=7)
    png = output_dir / "figure_14_per_fold_reliability.png"
    figure.savefig(png, dpi=320, bbox_inches="tight")
    plt.close(figure)

    selected = per_run.loc[
        per_run["experiment_type"].eq("main")
        & per_run["model"].isin(
            [
                "raw_cnn",
                "cwt_cnn",
                "cwt_gru",
                "cwt_gru_selftrain",
            ]
        )
    ]
    methods = sorted(selected["model"].unique())
    figure, axis = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
    data = [
        selected.loc[selected["model"].eq(method), "fitted_temperature"].to_numpy()
        for method in methods
    ]
    boxplot = axis.boxplot(
        data,
        positions=np.arange(len(methods)),
        widths=0.58,
        showfliers=True,
        patch_artist=True,
        medianprops={"color": INK_COLOR, "linewidth": 1.0},
        whiskerprops={"color": INK_COLOR},
        capprops={"color": INK_COLOR},
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": MUTED_COLOR,
            "markeredgecolor": MUTED_COLOR,
        },
    )
    for patch, method in zip(boxplot["boxes"], methods, strict=True):
        patch.set_facecolor(method_color(method))
        patch.set_edgecolor(INK_COLOR)
        patch.set_linewidth(0.7)
    axis.set_xticks(
        np.arange(len(methods)),
        [METHOD_LABELS[method] for method in methods],
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("Validation-fitted temperature")
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    png = output_dir / "figure_15_temperature_distribution.png"
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
        "--calibration-rerun-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "calibration_recheck",
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
    per_run = pd.DataFrame(
        [
            audit_run(args.results_dir, row)
            for _, row in raw.iterrows()
        ]
    )
    rerun_path = args.calibration_rerun_dir / "raw_runs.csv"
    if rerun_path.is_file():
        rerun = pd.read_csv(rerun_path)
        rerun_audit = pd.DataFrame(
            [
                audit_run(args.calibration_rerun_dir, row)
                for _, row in rerun.iterrows()
            ]
        )
        validation_rows = rerun_audit.loc[
            rerun_audit["validation_logits_available"]
        ].copy()
        validation_rows["source"] = "calibration_recheck"
        validation_columns = [
            "validation_logits_available",
            "validation_ece_before",
            "validation_ece_after",
            "validation_nll_before",
            "validation_nll_after",
            "validation_brier_before",
            "validation_brier_after",
        ]
        validation_by_key = validation_rows.set_index(["model", "fold", "seed"])
        for index, row in per_run.iterrows():
            if row["experiment_type"] != "main":
                continue
            key = (row["model"], int(row["fold"]), int(row["seed"]))
            source_key = (
                ("cwt_gru", key[1], key[2])
                if row["model"] == "cwt_gru_temp"
                else key
            )
            if source_key in validation_by_key.index:
                source = validation_by_key.loc[source_key]
                if isinstance(source, pd.DataFrame):
                    source = source.iloc[0]
                for column in validation_columns:
                    per_run.at[index, column] = source[column]
        canonical_main = raw.loc[
            raw["experiment_type"].eq("main")
            & raw["method"].isin(
                ["raw_cnn", "cwt_cnn", "cwt_gru", "cwt_gru_selftrain"]
            )
        ]
        rerun_main = rerun.loc[
            rerun["experiment_type"].eq("main")
            & rerun["method"].isin(
                ["raw_cnn", "cwt_cnn", "cwt_gru", "cwt_gru_selftrain"]
            )
        ]
        matched = rerun_main.merge(
            canonical_main,
            on=["method", "fold", "seed"],
            suffixes=("_rerun", "_canonical"),
            validate="one_to_one",
        )
        reproduction_rows = []
        for _, match in matched.iterrows():
            metric_rows = []
            for metric in (
                "macro_f1",
                "accuracy",
                "ece",
                "nll",
                "brier",
                "temperature",
                "best_epoch",
            ):
                difference = abs(
                    float(match[f"{metric}_rerun"])
                    - float(match[f"{metric}_canonical"])
                )
                metric_rows.append(
                    {
                        "method": match["method"],
                        "fold": int(match["fold"]),
                        "seed": int(match["seed"]),
                        "metric": metric,
                        "rerun": float(match[f"{metric}_rerun"]),
                        "canonical": float(match[f"{metric}_canonical"]),
                        "absolute_difference": difference,
                        "status": "VERIFIED" if difference <= 1e-9 else "UNVERIFIED",
                    }
                )
            reproduction_rows.extend(metric_rows)
        reproduction = pd.DataFrame(reproduction_rows)
        reproduction.to_csv(
            args.results_dir / "calibration_reproduction_check.csv",
            index=False,
            lineterminator="\n",
        )
    else:
        validation_rows = pd.DataFrame()

    per_run.to_csv(
        args.results_dir / "calibration_per_run.csv",
        index=False,
        lineterminator="\n",
    )
    if not validation_rows.empty:
        validation_rows.to_csv(
            args.results_dir / "calibration_validation_recheck.csv",
            index=False,
            lineterminator="\n",
        )
    fold_bins = aggregate_reliability(
        args.results_dir,
        raw,
        method="cwt_gru",
        folds=[0, 1, 2, 3],
    )
    fold_bins.to_csv(
        args.results_dir / "calibration_per_fold_bins.csv",
        index=False,
        lineterminator="\n",
    )
    summary = (
        per_run.loc[
            per_run["experiment_type"].eq("main")
            & per_run["model"].isin(METHOD_LABELS)
        ]
        .groupby("model")
        .agg(
            runs=("run_id", "count"),
            temperature_mean=("fitted_temperature", "mean"),
            temperature_sd=("fitted_temperature", "std"),
            temperature_min=("fitted_temperature", "min"),
            temperature_max=("fitted_temperature", "max"),
            lower_bound_runs=("temperature_at_lower_bound", "sum"),
            upper_bound_runs=("temperature_at_upper_bound", "sum"),
            test_ece_before=("test_ece_before", "mean"),
            test_ece_after=("test_ece_after", "mean"),
            test_nll_before=("test_nll_before", "mean"),
            test_nll_after=("test_nll_after", "mean"),
            test_brier_before=("test_brier_before", "mean"),
            test_brier_after=("test_brier_after", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(
        args.results_dir / "calibration_summary.csv",
        index=False,
        lineterminator="\n",
    )
    save_figures(per_run, fold_bins, args.figures_dir)
    print(
        f"Audited {len(per_run)} runs; validation logits available for "
        f"{int(per_run['validation_logits_available'].sum())} canonical runs and "
        f"{len(validation_rows)} rerun rows."
    )


if __name__ == "__main__":
    main()
