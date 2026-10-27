"""Generate uncertainty, ablation, and paired statistical summaries."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..experiments.common import (
    atomic_write_csv,
    configure_logging,
    read_results,
)
from ..paths import PROJECT_ROOT


ABLATION_COMPONENTS = {
    "cwt_cnn": (1, 0, 1, 0, 0),
    "gru_only": (0, 1, 0, 0, 0),
    "cwt_gru_no_se": (1, 1, 0, 0, 0),
    "cwt_gru": (1, 1, 1, 0, 0),
    "cwt_gru_temp": (1, 1, 1, 1, 0),
    "cwt_gru_selftrain": (1, 1, 1, 1, 1),
}

METHOD_ORDER = [
    "raw_cnn",
    "gru_only",
    "cwt_cnn",
    "cwt_gru_no_se",
    "cwt_gru",
    "cwt_gru_temp",
    "cwt_gru_selftrain",
]

DOMAIN_GAP_METHODS = [
    "raw_cnn",
    "cwt_cnn",
    "cwt_gru",
    "cwt_gru_no_se",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute confidence intervals and paired Wilcoxon tests"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "results" / "raw_runs.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--minimum-pairs",
        type=int,
        default=5,
        help="Do not run Wilcoxon when the paired sample is smaller.",
    )
    return parser.parse_args()


def confidence_interval(
    values: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    half_width = stats.t.ppf(1.0 - alpha / 2.0, len(clean) - 1) * (
        clean.std(ddof=1) / np.sqrt(len(clean))
    )
    return float(clean.mean() - half_width), float(clean.mean() + half_width)


def prepare_analysis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Include the 100% main rows in the label-efficiency view once."""

    main = raw.loc[raw["status"].astype(str).eq("success")].copy()
    label_full = main.loc[
        main["experiment_type"].eq("main")
        & main["label_ratio"].eq(1.0)
        & main["method"].isin(["cwt_gru", "cwt_gru_selftrain"])
    ].copy()
    label_full["experiment_type"] = "label_efficiency"
    label_full["run_id"] = label_full["run_id"].astype(str) + "__label100"
    return pd.concat([main, label_full], ignore_index=True).drop_duplicates("run_id")


def summarize(
    frame: pd.DataFrame,
    confidence: float,
) -> pd.DataFrame:
    group_columns = ["experiment_type", "method", "label_ratio", "threshold"]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        experiment_type, method, label_ratio, threshold = keys
        macro_low, macro_high = confidence_interval(group["macro_f1"], confidence)
        accuracy_low, accuracy_high = confidence_interval(group["accuracy"], confidence)
        ece_low, ece_high = confidence_interval(group["ece"], confidence)
        rows.append(
            {
                "experiment_type": experiment_type,
                "Model": method,
                "method": method,
                "label_ratio": label_ratio,
                "threshold": threshold,
                "Runs": len(group),
                "Macro-F1 Mean": group["macro_f1"].mean(),
                "Macro-F1 Std": group["macro_f1"].std(ddof=1),
                "Macro-F1 CI95 Low": macro_low,
                "Macro-F1 CI95 High": macro_high,
                "Accuracy Mean": group["accuracy"].mean(),
                "Accuracy Std": group["accuracy"].std(ddof=1),
                "Accuracy CI95 Low": accuracy_low,
                "Accuracy CI95 High": accuracy_high,
                "ECE Mean": group["ece"].mean(),
                "ECE Std": group["ece"].std(ddof=1),
                "ECE CI95 Low": ece_low,
                "ECE CI95 High": ece_high,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["experiment_type", "label_ratio", "threshold", "method"],
        na_position="first",
    )


def paired_tests(
    frame: pd.DataFrame,
    *,
    confidence: float,
    minimum_pairs: int,
) -> pd.DataFrame:
    paired_frame = frame.copy()
    paired_frame["_comparison_threshold"] = paired_frame["threshold"].where(
        paired_frame["experiment_type"].eq("threshold"),
        np.nan,
    )
    task_columns = ["experiment_type", "label_ratio", "_comparison_threshold"]
    rows: list[dict[str, object]] = []
    for keys, group in paired_frame.groupby(task_columns, dropna=False):
        experiment_type, label_ratio, threshold = keys
        pivot = group.pivot_table(
            index=["fold", "seed"],
            columns="method",
            values="macro_f1",
            aggfunc="mean",
        )
        present = set(pivot.columns)
        methods = [method for method in METHOD_ORDER if method in present]
        methods.extend(sorted(present - set(methods)))
        for position, reference in enumerate(methods):
            for method in methods[position + 1 :]:
                paired = pivot[[reference, method]].dropna()
                differences = paired[method] - paired[reference]
                n_pairs = len(paired)
                p_value = float("nan")
                test = "not_run"
                if n_pairs >= minimum_pairs:
                    if np.allclose(differences.to_numpy(dtype=float), 0.0):
                        p_value = 1.0
                        test = "all_differences_zero"
                    else:
                        p_value = float(
                            stats.wilcoxon(
                                differences,
                                alternative="two-sided",
                                zero_method="wilcox",
                            ).pvalue
                        )
                        test = "wilcoxon_signed_rank"
                delta_low, delta_high = confidence_interval(differences, confidence)
                rows.append(
                    {
                        "experiment_type": experiment_type,
                        "label_ratio": label_ratio,
                        "threshold": threshold,
                        "reference": reference,
                        "method": method,
                        "paired_runs": n_pairs,
                        "reference_mean_macro_f1": paired[reference].mean(),
                        "method_mean_macro_f1": paired[method].mean(),
                        "mean_paired_delta": differences.mean(),
                        "std_paired_delta": differences.std(ddof=1),
                        "paired_delta_ci95_low": delta_low,
                        "paired_delta_ci95_high": delta_high,
                        "test": test,
                        "p_value": p_value,
                    }
                )
    return pd.DataFrame(rows)


def build_ablation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    main = frame.loc[
        frame["experiment_type"].eq("main")
        & frame["method"].isin(ABLATION_COMPONENTS)
    ].copy()
    rows: list[dict[str, object]] = []
    for method, (cwt, gru, se, calibration, self_training) in (
        ABLATION_COMPONENTS.items()
    ):
        group = main.loc[main["method"].eq(method)]
        if group.empty:
            continue
        rows.append(
            {
                "Model": method,
                "CWT": cwt,
                "GRU": gru,
                "SE": se,
                "Calibration": calibration,
                "Self-training": self_training,
                "Macro-F1 Mean": group["macro_f1"].mean(),
                "Macro-F1 Std": group["macro_f1"].std(ddof=1),
                "Accuracy Mean": group["accuracy"].mean(),
                "Accuracy Std": group["accuracy"].std(ddof=1),
                "Runs": len(group),
            }
        )
    return pd.DataFrame(rows)


def build_domain_gap_summary(raw: pd.DataFrame) -> pd.DataFrame:
    main = raw.loc[
        raw["experiment_type"].eq("main")
        & raw["method"].isin(DOMAIN_GAP_METHODS)
    ].copy()
    rows: list[dict[str, object]] = []
    for method in DOMAIN_GAP_METHODS:
        group = main.loc[main["method"].eq(method)]
        validation = group["best_val_macro_f1"]
        test = group["macro_f1"]
        absolute_gap = validation - test
        relative_drop = 100.0 * absolute_gap / validation.replace(0, np.nan)
        rows.append(
            {
                "Model": method,
                "Validation Macro-F1 Mean": validation.mean(),
                "Validation Macro-F1 Std": validation.std(ddof=1),
                "Test Macro-F1 Mean": test.mean(),
                "Test Macro-F1 Std": test.std(ddof=1),
                "Absolute Gap": absolute_gap.mean(),
                "Absolute Gap Std": absolute_gap.std(ddof=1),
                "Relative Drop (%)": relative_drop.mean(),
                "Relative Drop Std (%)": relative_drop.std(ddof=1),
                "Runs": len(group),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_summary(frame: pd.DataFrame) -> pd.DataFrame:
    threshold = frame.loc[frame["experiment_type"].eq("threshold")].copy()
    if threshold.empty:
        return pd.DataFrame()
    grouped = (
        threshold.groupby(["method", "threshold"], dropna=False)
        .agg(
            **{
                "Macro-F1 Mean": ("macro_f1", "mean"),
                "Macro-F1 Std": ("macro_f1", "std"),
                "Accuracy Mean": ("accuracy", "mean"),
                "Accuracy Std": ("accuracy", "std"),
                "ECE Mean": ("ece", "mean"),
                "ECE Std": ("ece", "std"),
                "pseudo_count_mean": ("pseudo_count", "mean"),
                "pseudo_count_std": ("pseudo_count", "std"),
                "pseudo_coverage_mean": ("pseudo_coverage", "mean"),
                "pseudo_coverage_std": ("pseudo_coverage", "std"),
                "pseudo_accuracy_mean": ("pseudo_accuracy", "mean"),
                "pseudo_accuracy_std": ("pseudo_accuracy", "std"),
                "Runs": ("seed", "count"),
            }
        )
        .reset_index()
        .rename(columns={"method": "Model"})
        .sort_values("threshold")
    )
    return grouped


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    configure_logging(output_dir / "statistical_analysis.log")
    raw = read_results(args.results.resolve())
    if raw.empty:
        raise ValueError(f"No results found in {args.results}")
    frame = prepare_analysis_frame(raw)
    summary = summarize(frame, args.confidence)
    paired = paired_tests(
        frame,
        confidence=args.confidence,
        minimum_pairs=args.minimum_pairs,
    )
    ablation = build_ablation_summary(frame)
    domain_gap = build_domain_gap_summary(raw)
    label_summary = summary.loc[
        summary["experiment_type"].eq("label_efficiency")
    ].copy()
    threshold_summary = build_threshold_summary(frame)

    atomic_write_csv(output_dir / "summary_with_ci.csv", summary)
    atomic_write_csv(
        output_dir / "summary.csv",
        summary.loc[summary["experiment_type"].eq("main")].copy(),
    )
    atomic_write_csv(output_dir / "label_efficiency_summary.csv", label_summary)
    atomic_write_csv(output_dir / "ablation_summary.csv", ablation)
    atomic_write_csv(output_dir / "domain_gap_summary.csv", domain_gap)
    atomic_write_csv(output_dir / "threshold_summary.csv", threshold_summary)
    atomic_write_csv(output_dir / "statistical_tests.csv", paired)
    logging.info("Wrote statistical summaries to %s", output_dir)


if __name__ == "__main__":
    main()
