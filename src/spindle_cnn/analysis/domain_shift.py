"""Audit raw-signal domain shifts without using test labels for fitting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..datasets.splits import load_fold_indices
from ..paths import PROJECT_ROOT
from ..visualization.style import (
    GRID_COLOR,
    INK_COLOR,
    MUTED_COLOR,
    PAPER_COLORS,
    apply_paper_style,
)
from ..public_benchmark.pu_data import (
    CLASS_NAMES,
    CONDITION_TO_LOAD,
    HELDOUT_RECORD_IDS,
    RECORD_ID_TO_BEARING,
)


LOAD_TO_CONDITION = {value: key for key, value in CONDITION_TO_LOAD.items()}
CHANNEL_NAMES = ["vibration_1", "phase_current_1", "phase_current_2"]
SAMPLING_RATE_HZ = 64_000
WINDOW_SIZE = 8_192
BAND_EDGES_HZ = (0, 2_000, 5_000, 10_000, 20_000, 32_000)


def signal_metrics(window: np.ndarray) -> dict[str, float]:
    values = np.asarray(window, dtype=np.float64)
    centered = values - values.mean()
    standard_deviation = float(centered.std())
    rms = float(np.sqrt(np.mean(values**2)))
    peak = float(np.max(np.abs(values)))
    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / SAMPLING_RATE_HZ)
    if len(power) > 1:
        power_no_dc = power[1:]
        frequency_no_dc = frequencies[1:]
    else:
        power_no_dc = power
        frequency_no_dc = frequencies
    total_power = float(power_no_dc.sum())
    if total_power <= 0:
        dominant = 0.0
        centroid = 0.0
        entropy = 0.0
    else:
        weights = power_no_dc / total_power
        dominant = float(frequency_no_dc[int(np.argmax(power_no_dc))])
        centroid = float(np.sum(frequency_no_dc * weights))
        entropy = float(-np.sum(weights * np.log(weights + 1e-15)) / np.log(len(weights)))
    metrics = {
        "mean": float(values.mean()),
        "standard_deviation": standard_deviation,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(values.max() - values.min()),
        "kurtosis": float(stats.kurtosis(centered, fisher=True, bias=False)),
        "skewness": float(stats.skew(centered, bias=False)),
        "crest_factor": peak / max(rms, 1e-12),
        "dominant_frequency_hz": dominant,
        "spectral_centroid_hz": centroid,
        "spectral_entropy": entropy,
    }
    for lower, upper in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:], strict=True):
        mask = (frequencies >= lower) & (frequencies < upper)
        band_power = float(power[mask].sum())
        metrics[f"band_energy_ratio_{lower // 1000}_{upper // 1000}khz"] = (
            band_power / total_power if total_power > 0 else np.nan
        )
    return metrics


def build_window_metrics(raw_cache: Path) -> pd.DataFrame:
    with np.load(raw_cache, allow_pickle=False) as data:
        raw = data["raw"]
        labels = data["labels"]
        loads = data["loads"]
        record_ids = data["record_ids"]
        measurements = data["measurements"]
        starts = data["starts"]
        rows: list[dict[str, Any]] = []
        for index in range(len(raw)):
            bearing = RECORD_ID_TO_BEARING[int(record_ids[index])]
            for channel_id, channel_name in enumerate(CHANNEL_NAMES):
                row = {
                    "window_id": index,
                    "bearing_code": bearing,
                    "class_id": int(labels[index]),
                    "class_name": CLASS_NAMES[int(labels[index])],
                    "condition_id": int(loads[index]),
                    "condition_name": LOAD_TO_CONDITION[int(loads[index])],
                    "measurement": int(measurements[index]),
                    "window_start": int(starts[index]),
                    "channel_id": channel_id,
                    "channel": channel_name,
                }
                row.update(signal_metrics(raw[index, channel_id]))
                rows.append(row)
    return pd.DataFrame(rows)


def assign_splits(
    window_metrics: pd.DataFrame,
    labels: np.ndarray,
    loads: np.ndarray,
    record_ids: np.ndarray,
) -> pd.DataFrame:
    from ..public_benchmark.cwru_data import CWRUDataset

    dummy = CWRUDataset(
        raw=np.empty((len(labels), 1, 1), dtype=np.float32),
        cwt=np.empty((len(labels), 1, 1, 1), dtype=np.float16),
        labels=labels,
        loads=loads,
        record_ids=record_ids,
        starts=np.zeros(len(labels), dtype=np.int32),
    )
    frames = []
    for fold in range(4):
        split = load_fold_indices(dummy, fold, np.asarray(sorted(HELDOUT_RECORD_IDS)))
        split_name = np.full(len(labels), "", dtype=object)
        split_name[split.train] = "train"
        split_name[split.validation] = "validation"
        split_name[split.test] = "test"
        part = window_metrics.copy()
        part["fold"] = fold
        part["split"] = split_name[part["window_id"].to_numpy(dtype=np.int64)]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def aggregate_metrics(
    window_metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    metric_columns: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_scope(
        frame: pd.DataFrame,
        *,
        group_type: str,
        value_column: str,
        fold: int | None = None,
        split: str | None = None,
    ) -> None:
        id_columns = [
            "window_id",
            "bearing_code",
            "class_name",
            "condition_name",
            "channel",
        ]
        if value_column not in id_columns:
            id_columns.append(value_column)
        for (value, channel, metric), group in frame.melt(
            id_vars=id_columns,
            value_vars=list(metric_columns),
            var_name="metric",
            value_name="value",
        ).groupby([value_column, "channel", "metric"], sort=False):
            finite = group["value"].replace([np.inf, -np.inf], np.nan).dropna()
            if finite.empty:
                continue
            rows.append(
                {
                    "aggregation_scope": group_type,
                    "fold": "" if fold is None else fold,
                    "split": "" if split is None else split,
                    "group_value": str(value),
                    "channel": channel,
                    "metric": metric,
                    "n_windows": int(group["window_id"].nunique()),
                    "n_bearings": int(group["bearing_code"].nunique()),
                    "mean": float(finite.mean()),
                    "median": float(finite.median()),
                    "sd": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                    "q25": float(finite.quantile(0.25)),
                    "q75": float(finite.quantile(0.75)),
                }
            )

    for value_column in ("bearing_code", "class_name", "condition_name"):
        add_scope(
            window_metrics,
            group_type=value_column.removesuffix("_code").removesuffix("_name"),
            value_column=value_column,
        )
    for fold, fold_frame in split_metrics.groupby("fold"):
        for split, split_frame in fold_frame.groupby("split"):
            add_scope(
                split_frame,
                group_type="split",
                value_column="split",
                fold=int(fold),
                split=str(split),
            )
    return pd.DataFrame(rows)


def _paired_effect(
    source: pd.Series,
    target: pd.Series,
) -> dict[str, float]:
    source_values = source.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    target_values = target.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(source_values) < 2 or len(target_values) < 2:
        return {
            "source_mean": np.nan,
            "target_mean": np.nan,
            "absolute_difference": np.nan,
            "relative_difference": np.nan,
            "standardized_mean_difference": np.nan,
        }
    pooled = np.sqrt(
        ((len(source_values) - 1) * source_values.var(ddof=1)
         + (len(target_values) - 1) * target_values.var(ddof=1))
        / (len(source_values) + len(target_values) - 2)
    )
    difference = target_values.mean() - source_values.mean()
    return {
        "source_mean": float(source_values.mean()),
        "target_mean": float(target_values.mean()),
        "absolute_difference": float(difference),
        "relative_difference": float(
            difference / abs(source_values.mean())
        )
        if source_values.mean() != 0
        else np.nan,
        "standardized_mean_difference": float(difference / pooled)
        if pooled > 0
        else np.nan,
    }


def build_contrasts(
    split_metrics: pd.DataFrame,
    metric_columns: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, fold_frame in split_metrics.groupby("fold"):
        train = fold_frame.loc[fold_frame["split"].eq("train")]
        for target_split in ("validation", "test"):
            target = fold_frame.loc[fold_frame["split"].eq(target_split)]
            for channel in CHANNEL_NAMES:
                train_channel = train.loc[train["channel"].eq(channel)]
                target_channel = target.loc[target["channel"].eq(channel)]
                for metric in metric_columns:
                    effect = _paired_effect(
                        train_channel[metric],
                        target_channel[metric],
                    )
                    rows.append(
                        {
                            "fold": int(fold),
                            "comparison": f"train_vs_{target_split}",
                            "channel": channel,
                            "metric": metric,
                            "n_train": len(train_channel),
                            "n_target": len(target_channel),
                            **effect,
                        }
                    )
        validation = fold_frame.loc[fold_frame["split"].eq("validation")]
        test = fold_frame.loc[fold_frame["split"].eq("test")]
        for channel in CHANNEL_NAMES:
            validation_channel = validation.loc[validation["channel"].eq(channel)]
            test_channel = test.loc[test["channel"].eq(channel)]
            for metric in metric_columns:
                effect = _paired_effect(
                    validation_channel[metric],
                    test_channel[metric],
                )
                rows.append(
                    {
                        "fold": int(fold),
                        "comparison": "validation_vs_test",
                        "channel": channel,
                        "metric": metric,
                        "n_train": len(validation_channel),
                        "n_target": len(test_channel),
                        **effect,
                    }
                )
    return pd.DataFrame(rows)


def save_plot(
    split_metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style(font_size=8.5)
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    bearing_colors = [
        PAPER_COLORS[0],
        PAPER_COLORS[1],
        PAPER_COLORS[3],
        PAPER_COLORS[8],
    ]
    condition_colors = [
        PAPER_COLORS[0],
        PAPER_COLORS[1],
        PAPER_COLORS[3],
        PAPER_COLORS[4],
    ]

    def color_boxes(axis: plt.Axes, colors: list[str]) -> None:
        for patch, color in zip(axis.patches, colors, strict=True):
            patch.set_facecolor(color)
            patch.set_edgecolor(INK_COLOR)
            patch.set_linewidth(0.6)

    vibration = split_metrics.loc[
        split_metrics["channel"].eq("vibration_1")
    ]
    test_rows = vibration.loc[vibration["split"].eq("test")]
    bearings = sorted(test_rows["bearing_code"].unique())
    positions = np.arange(len(bearings))
    data = [
        test_rows.loc[test_rows["bearing_code"].eq(code), "rms"].to_numpy()
        for code in bearings
    ]
    axes[0, 0].boxplot(
        data,
        positions=positions,
        widths=0.62,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK_COLOR, "linewidth": 0.8},
    )
    color_boxes(axes[0, 0], bearing_colors)
    axes[0, 0].set_xticks(positions, bearings, rotation=35, ha="right")
    axes[0, 0].set_ylabel("RMS (source units)")
    axes[0, 0].set_title("(a) Test-bearing amplitude", loc="left")

    conditions = condition_order = list(CONDITION_TO_LOAD)
    condition_data = [
        vibration.loc[
            vibration["condition_name"].eq(condition),
            "spectral_centroid_hz",
        ].to_numpy()
        for condition in conditions
    ]
    axes[0, 1].boxplot(
        condition_data,
        positions=np.arange(len(conditions)),
        widths=0.62,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK_COLOR, "linewidth": 0.8},
    )
    color_boxes(axes[0, 1], condition_colors)
    axes[0, 1].set_xticks(
        np.arange(len(conditions)),
        [condition.replace("_", "\n") for condition in conditions],
    )
    axes[0, 1].set_ylabel("Spectral centroid (Hz)")
    axes[0, 1].set_title("(b) Operating-condition spectra", loc="left")

    entropy_data = [
        vibration.loc[
            vibration["condition_name"].eq(condition),
            "spectral_entropy",
        ].to_numpy()
        for condition in condition_order
    ]
    axes[1, 0].boxplot(
        entropy_data,
        positions=np.arange(len(conditions)),
        widths=0.62,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK_COLOR, "linewidth": 0.8},
    )
    color_boxes(axes[1, 0], condition_colors)
    axes[1, 0].set_xticks(
        np.arange(len(conditions)),
        [condition.replace("_", "\n") for condition in conditions],
    )
    axes[1, 0].set_ylabel("Normalized spectral entropy")
    axes[1, 0].set_title("(c) Spectral concentration", loc="left")

    selected_metrics = [
        "rms",
        "kurtosis",
        "crest_factor",
        "spectral_centroid_hz",
        "spectral_entropy",
    ]
    selected = contrasts.loc[
        contrasts["channel"].eq("vibration_1")
        & contrasts["comparison"].eq("train_vs_test")
        & contrasts["metric"].isin(selected_metrics)
    ]
    labels = [
        metric.replace("_", " ").replace(" hz", "")
        for metric in selected_metrics
    ]
    values = [
        selected.loc[selected["metric"].eq(metric), "standardized_mean_difference"]
        .to_numpy()
        for metric in selected_metrics
    ]
    axis = axes[1, 1]
    axis.axvline(0, color=MUTED_COLOR, linewidth=0.8, linestyle="--")
    boxplot = axis.boxplot(
        values,
        positions=np.arange(len(selected_metrics)),
        vert=False,
        widths=0.62,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK_COLOR, "linewidth": 0.8},
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor(PAPER_COLORS[8])
        patch.set_edgecolor(INK_COLOR)
        patch.set_linewidth(0.6)
    axis.set_yticks(np.arange(len(selected_metrics)), labels)
    axis.set_xlabel("Standardized train-to-test difference")
    axis.set_title("(d) Shift across four folds", loc="left")

    for axis in axes.flat:
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)
    png_path = output_dir / "figure_13_domain_shift_diagnostics.png"
    pdf_path = output_dir / "figure_13_domain_shift_diagnostics.pdf"
    figure.savefig(png_path, dpi=320, bbox_inches="tight")
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "public"
        / "paderborn"
        / "pu_4class_w8192_raw.npz",
    )
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
    if not args.raw_cache.is_file():
        raise FileNotFoundError(
            f"Raw-amplitude cache not found: {args.raw_cache}. "
            "Run paderborn_raw_windows after extracting the official archives."
        )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    window_metrics = build_window_metrics(args.raw_cache)
    with np.load(args.raw_cache, allow_pickle=False) as data:
        labels = data["labels"]
        loads = data["loads"]
        record_ids = data["record_ids"]
    split_metrics = assign_splits(window_metrics, labels, loads, record_ids)
    metric_columns = [
        column
        for column in window_metrics.columns
        if column
        not in {
            "window_id",
            "bearing_code",
            "class_id",
            "class_name",
            "condition_id",
            "condition_name",
            "measurement",
            "window_start",
            "channel_id",
            "channel",
        }
    ]
    aggregate = aggregate_metrics(window_metrics, split_metrics, metric_columns)
    contrasts = build_contrasts(split_metrics, metric_columns)
    window_metrics.to_csv(
        args.results_dir / "domain_shift_window_metrics.csv",
        index=False,
        lineterminator="\n",
    )
    aggregate.to_csv(
        args.results_dir / "domain_shift_statistics.csv",
        index=False,
        lineterminator="\n",
    )
    contrasts.to_csv(
        args.results_dir / "domain_shift_contrasts.csv",
        index=False,
        lineterminator="\n",
    )
    save_plot(split_metrics, contrasts, args.figures_dir)
    print(
        f"Wrote {len(window_metrics)} channel-window rows, "
        f"{len(aggregate)} aggregates, {len(contrasts)} contrasts."
    )


if __name__ == "__main__":
    main()
