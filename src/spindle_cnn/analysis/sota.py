"""Summarize same-protocol external baselines without modifying canonical runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..experiments.common import atomic_write_csv, read_results
from ..paths import PROJECT_ROOT


EXTERNAL_METHODS = ("drsn", "tfnet")
INTERNAL_METHODS = (
    "raw_cnn",
    "gru_only",
    "cwt_cnn",
    "cwt_gru_no_se",
    "cwt_gru",
    "cwt_gru_temp",
    "cwt_gru_selftrain",
)
REFERENCE_METHODS = ("raw_cnn", "cwt_cnn", "cwt_gru", "cwt_gru_no_se")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "sota",
    )
    return parser.parse_args()


def ci95(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return float("nan"), float("nan")
    half_width = stats.t.ppf(0.975, len(clean) - 1) * clean.std(ddof=1) / np.sqrt(
        len(clean)
    )
    return float(clean.mean() - half_width), float(clean.mean() + half_width)


def summarize_external(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in EXTERNAL_METHODS:
        group = frame.loc[frame["method"].eq(method)]
        macro_low, macro_high = ci95(group["macro_f1"])
        accuracy_low, accuracy_high = ci95(group["accuracy"])
        rows.append(
            {
                "Method": method,
                "Runs": len(group),
                "Macro-F1 Mean": group["macro_f1"].mean(),
                "Macro-F1 Std": group["macro_f1"].std(ddof=1),
                "Macro-F1 CI95 Low": macro_low,
                "Macro-F1 CI95 High": macro_high,
                "Accuracy Mean": group["accuracy"].mean(),
                "Accuracy Std": group["accuracy"].std(ddof=1),
                "Accuracy CI95 Low": accuracy_low,
                "Accuracy CI95 High": accuracy_high,
            }
        )
    return pd.DataFrame(rows)


def paired_external_tests(
    internal: pd.DataFrame,
    external: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat(
        [
            internal.loc[internal["method"].isin(REFERENCE_METHODS)],
            external,
        ],
        ignore_index=True,
    )
    pivot = combined.pivot_table(
        index=["fold", "seed"],
        columns="method",
        values="macro_f1",
        aggfunc="mean",
    )
    rows = []
    for reference in REFERENCE_METHODS:
        for method in EXTERNAL_METHODS:
            paired = pivot[[reference, method]].dropna()
            differences = paired[method] - paired[reference]
            if np.allclose(differences, 0):
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
            low, high = ci95(differences)
            rows.append(
                {
                    "reference": reference,
                    "method": method,
                    "paired_runs": len(paired),
                    "reference_mean_macro_f1": paired[reference].mean(),
                    "method_mean_macro_f1": paired[method].mean(),
                    "mean_paired_delta": differences.mean(),
                    "std_paired_delta": differences.std(ddof=1),
                    "paired_delta_ci95_low": low,
                    "paired_delta_ci95_high": high,
                    "test": test,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def same_protocol_comparison(
    internal: pd.DataFrame,
    external_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for method in INTERNAL_METHODS:
        group = internal.loc[internal["method"].eq(method)]
        macro_low, macro_high = ci95(group["macro_f1"])
        accuracy_low, accuracy_high = ci95(group["accuracy"])
        rows.append(
            {
                "Method": method,
                "Source": "internal",
                "Runs": len(group),
                "Macro-F1 Mean": group["macro_f1"].mean(),
                "Macro-F1 Std": group["macro_f1"].std(ddof=1),
                "Macro-F1 CI95 Low": macro_low,
                "Macro-F1 CI95 High": macro_high,
                "Accuracy Mean": group["accuracy"].mean(),
                "Accuracy Std": group["accuracy"].std(ddof=1),
                "Accuracy CI95 Low": accuracy_low,
                "Accuracy CI95 High": accuracy_high,
                "Parameters": group["parameters"].median(),
                "GFLOPs/sample": group["gflops_sample"].median(),
                "Training time (s)": group["train_time"].median(),
                "Inference (ms/batch)": group["inference_ms_per_batch"].median(),
            }
        )
    for _, row in external_summary.iterrows():
        external_method = row["Method"]
        rows.append(
            {
                "Method": external_method,
                "Source": "external adaptation",
                "Runs": int(row["Runs"]),
                "Macro-F1 Mean": row["Macro-F1 Mean"],
                "Macro-F1 Std": row["Macro-F1 Std"],
                "Macro-F1 CI95 Low": row["Macro-F1 CI95 Low"],
                "Macro-F1 CI95 High": row["Macro-F1 CI95 High"],
                "Accuracy Mean": row["Accuracy Mean"],
                "Accuracy Std": row["Accuracy Std"],
                "Accuracy CI95 Low": row["Accuracy CI95 Low"],
                "Accuracy CI95 High": row["Accuracy CI95 High"],
                "Parameters": np.nan,
                "GFLOPs/sample": np.nan,
                "Training time (s)": np.nan,
                "Inference (ms/batch)": np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    internal = read_results(args.results_dir / "raw_runs.csv")
    external = read_results(args.external_dir / "raw_runs.csv")
    internal_main = internal.loc[internal["experiment_type"].eq("main")].copy()
    if external["experiment_type"].eq("sota").sum() != len(external):
        raise ValueError("External table contains non-SOTA experiment rows")
    if len(external) != 40 or set(external["method"]) != set(EXTERNAL_METHODS):
        raise ValueError("Expected 20 runs for each external baseline")

    summary = summarize_external(external)
    paired = paired_external_tests(internal_main, external)
    external_complexity = (
        external.groupby(["method", "model_key"], as_index=False)
        .agg(
            parameters=("parameters", "median"),
            gflops_sample=("gflops_sample", "median"),
            train_time_s_median=("train_time", "median"),
            inference_ms_per_batch_median=("inference_ms_per_batch", "median"),
            runs=("seed", "count"),
        )
    )
    comparison = same_protocol_comparison(internal_main, summary)
    complexity_by_method = external_complexity.set_index("method")
    comparison.loc[comparison["Source"].eq("external adaptation")] = comparison.loc[
        comparison["Source"].eq("external adaptation")
    ].assign(
        Parameters=lambda frame: frame["Method"].map(
            complexity_by_method["parameters"]
        ),
        **{
            "GFLOPs/sample": lambda frame: frame["Method"].map(
                complexity_by_method["gflops_sample"]
            ),
            "Training time (s)": lambda frame: frame["Method"].map(
                complexity_by_method["train_time_s_median"]
            ),
            "Inference (ms/batch)": lambda frame: frame["Method"].map(
                complexity_by_method["inference_ms_per_batch_median"]
            ),
        },
    )

    atomic_write_csv(args.external_dir / "summary_with_ci.csv", summary)
    atomic_write_csv(args.external_dir / "statistical_tests.csv", paired)
    atomic_write_csv(args.external_dir / "model_complexity.csv", external_complexity)
    atomic_write_csv(
        args.external_dir / "same_protocol_comparison.csv",
        comparison,
    )
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
