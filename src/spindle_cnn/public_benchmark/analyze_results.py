"""Summarize repeated public-benchmark runs with uncertainty and paired deltas."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..paths import OUTPUTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总重复实验的均值、置信区间和配对差异")
    parser.add_argument(
        "--results",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn" / "results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "public_paderborn" / "analysis",
    )
    return parser.parse_args()


def confidence_interval(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return float("nan"), float("nan")
    half_width = stats.t.ppf(0.975, len(clean) - 1) * clean.std(ddof=1) / np.sqrt(len(clean))
    return float(clean.mean() - half_width), float(clean.mean() + half_width)


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (method, fraction), group in results.groupby(["method", "label_fraction"]):
        low, high = confidence_interval(group["raw_macro_f1"])
        cal_low, cal_high = confidence_interval(group["cal_ece"])
        rows.append(
            {
                "method": method,
                "label_fraction": fraction,
                "runs": len(group),
                "accuracy_mean": group["raw_accuracy"].mean(),
                "accuracy_std": group["raw_accuracy"].std(ddof=1),
                "macro_f1_mean": group["raw_macro_f1"].mean(),
                "macro_f1_std": group["raw_macro_f1"].std(ddof=1),
                "macro_f1_ci95_low": low,
                "macro_f1_ci95_high": high,
                "calibrated_ece_mean": group["cal_ece"].mean(),
                "calibrated_ece_std": group["cal_ece"].std(ddof=1),
                "calibrated_ece_ci95_low": cal_low,
                "calibrated_ece_ci95_high": cal_high,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["method", "label_fraction"])
    summary.to_csv(args.output_dir / "summary_with_ci.csv", index=False)

    full_results = results[np.isclose(results["label_fraction"], 1.0)]
    pivot = full_results.pivot_table(
        index=["test_condition", "seed"],
        columns="method",
        values="raw_macro_f1",
        aggfunc="mean",
    )
    comparison_rows = []
    methods = sorted(pivot.columns)
    for reference_index, reference in enumerate(methods):
        for method in methods[reference_index + 1 :]:
            paired = pivot[[reference, method]].dropna()
            delta = paired[method] - paired[reference]
            if len(paired) < 2:
                p_value = float("nan")
            else:
                p_value = float(stats.wilcoxon(delta).pvalue)
            comparison_rows.append(
                {
                    "reference": reference,
                    "method": method,
                    "paired_runs": len(paired),
                    "mean_delta_macro_f1": delta.mean(),
                    "std_delta_macro_f1": delta.std(ddof=1),
                    "wilcoxon_p": p_value,
                }
            )
    pd.DataFrame(comparison_rows).to_csv(
        args.output_dir / "paired_comparisons.csv",
        index=False,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
