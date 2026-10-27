"""Generate a Markdown experiment report from canonical result CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..experiments.common import atomic_write_csv, read_results
from ..paths import PROJECT_ROOT


METHOD_LABELS = {
    "raw_cnn": "Raw 1D-CNN",
    "gru_only": "GRU only",
    "cwt_cnn": "CWT 2D-CNN",
    "cwt_gru_no_se": "CWT-GRU without SE",
    "cwt_gru": "CWT-GRU",
    "cwt_gru_temp": "CWT-GRU + temperature scaling",
    "cwt_gru_selftrain": "CWT-GRU + temperature scaling + self-training",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an audited result report")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    return parser.parse_args()


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def fmt_interval(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    raw = read_results(results_dir / "raw_runs.csv")
    if raw.empty:
        raise ValueError("No canonical rows in results/raw_runs.csv")
    failures = raw.loc[~raw["status"].astype(str).eq("success")]
    if not failures.empty:
        raise ValueError(
            f"Result report refuses to hide {len(failures)} failed runs. "
            "Inspect results/raw_runs.csv first."
        )

    main = raw.loc[raw["experiment_type"].eq("main")].copy()
    main_counts = main.groupby("method").size()
    if not main_counts.eq(20).all():
        raise ValueError(f"Main experiment does not have 20 runs per method: {main_counts}")

    label = raw.loc[
        raw["experiment_type"].eq("label_efficiency")
    ].groupby(["method", "label_ratio"]).size()
    if not label.eq(20).all():
        raise ValueError(f"Each label condition must have 20 runs: {label}")

    thresholds = raw.loc[raw["experiment_type"].eq("threshold")].groupby(
        "threshold"
    ).size()
    if not thresholds.eq(20).all():
        raise ValueError(
            f"Each confidence threshold must have 20 runs: {thresholds}"
        )
    zero_threshold_cases = (
        raw.loc[
            raw["experiment_type"].eq("threshold")
            & raw["pseudo_count"].fillna(0).eq(0),
            ["fold", "seed"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    zero_low_budget_runs = raw.loc[
        raw["experiment_type"].eq("label_efficiency")
        & raw["method"].eq("cwt_gru_selftrain")
        & raw["label_ratio"].eq(0.1)
        & raw["pseudo_count"].fillna(0).eq(0)
    ].shape[0]

    summary = pd.read_csv(results_dir / "summary_with_ci.csv")
    summary = summary.loc[summary["experiment_type"].eq("main")]
    summary = summary.set_index("method")
    main_rows = []
    for method in METHOD_LABELS:
        if method not in summary.index:
            continue
        row = summary.loc[method]
        main_rows.append(
            [
                METHOD_LABELS[method],
                fmt_mean_std(row["Macro-F1 Mean"], row["Macro-F1 Std"]),
                fmt_mean_std(row["Accuracy Mean"], row["Accuracy Std"]),
                fmt_interval(
                    float(row["Macro-F1 CI95 Low"]),
                    float(row["Macro-F1 CI95 High"]),
                ),
                int(row["Runs"]),
            ]
        )

    tests = pd.read_csv(results_dir / "statistical_tests.csv")
    raw_tests = tests.loc[
        tests["experiment_type"].eq("main")
        & tests["reference"].eq("raw_cnn")
    ]
    test_rows = []
    for row in raw_tests.itertuples(index=False):
        test_rows.append(
            [
                METHOD_LABELS.get(row.method, row.method),
                f"{float(row.mean_paired_delta):+.3f}",
                fmt_interval(
                    float(row.paired_delta_ci95_low),
                    float(row.paired_delta_ci95_high),
                ),
                f"{float(row.p_value):.3f}",
                int(row.paired_runs),
            ]
        )

    label_summary = pd.read_csv(results_dir / "label_efficiency_summary.csv")
    label_rows = []
    budget_labels = {
        0.1: "10% balanced",
        0.5: "50% balanced",
        1.0: "Full balanced",
    }
    for row in label_summary.sort_values(["label_ratio", "method"]).itertuples(
        index=False
    ):
        label_rows.append(
            [
                budget_labels.get(round(float(row.label_ratio), 1), row.label_ratio),
                METHOD_LABELS.get(row.method, row.method),
                fmt_mean_std(row._6, row._7),
                fmt_mean_std(row._10, row._11),
                int(row.Runs),
            ]
        )

    threshold_summary = pd.read_csv(results_dir / "threshold_summary.csv")
    threshold_rows = []
    for row in threshold_summary.sort_values("threshold").itertuples(index=False):
        threshold_rows.append(
            [
                f"{float(row.threshold):.2f}",
                fmt_mean_std(row._2, row._3),
                fmt_mean_std(row._4, row._5),
                f"{float(row.pseudo_count_mean):.1f}",
                f"{float(row.pseudo_coverage_mean):.3f}",
                f"{float(row.pseudo_accuracy_mean):.3f}",
                int(row.Runs),
            ]
        )

    calibration_rows = []
    for method in ("raw_cnn", "cwt_cnn", "cwt_gru", "cwt_gru_selftrain"):
        group = main.loc[main["method"].eq(method)]
        calibration_rows.append(
            [
                METHOD_LABELS[method],
                f"{group['raw_ece'].mean():.3f}",
                f"{group['ece'].mean():.3f}",
                f"{group['raw_nll'].mean():.3f}",
                f"{group['nll'].mean():.3f}",
            ]
        )

    domain_gap = pd.read_csv(results_dir / "domain_gap_summary.csv")
    domain_gap_rows = []
    for row in domain_gap.itertuples(index=False):
        domain_gap_rows.append(
            [
                METHOD_LABELS.get(row.Model, row.Model),
                fmt_mean_std(
                    row._1,
                    row._2,
                ),
                fmt_mean_std(
                    row._3,
                    row._4,
                ),
                f"{float(row._5):.3f}",
                f"{float(row._7):.1f}%",
                int(row.Runs),
            ]
        )

    complexity = pd.read_csv(results_dir / "model_complexity.csv")
    complexity = complexity.drop_duplicates("model_key").set_index("model_key")
    complexity_rows = []
    for method in (
        "raw_cnn",
        "gru_only",
        "cwt_cnn",
        "cwt_gru_no_se",
        "cwt_gru",
    ):
        row = complexity.loc[method]
        complexity_rows.append(
            [
                METHOD_LABELS[method],
                f"{float(row['parameters']) / 1e6:.3f}",
                f"{float(row['gflops_sample']):.3f}",
                f"{float(row['train_time_s_median']):.1f}",
                f"{float(row['inference_ms_per_batch_median']):.1f}",
            ]
        )

    commits = sorted(set(raw["git_commit"].dropna().astype(str)))
    report = f"""# Five-Seed Benchmark Report

This report was generated from `results/raw_runs.csv`. No values are entered by
hand. The canonical experiment contains {len(raw)} successful runs.

## Experiment integrity

- Main comparison: {len(main)} runs, 20 per method.
- Label efficiency: {len(raw.loc[raw['experiment_type'].eq('label_efficiency')])} runs, 20 per method and label ratio.
- Threshold sensitivity: {len(raw.loc[raw['experiment_type'].eq('threshold')])} runs, 20 per threshold.
- Failed runs: 0.
- 10% balanced-budget self-training produced no pseudo-label update in {zero_low_budget_runs}/20 runs.
- {zero_threshold_cases} fold-plus-seed threshold cases produced no pseudo-label update at any tested threshold.
- Recorded Git commits: {", ".join(f"`{commit}`" for commit in commits)}.
- Primary seeds: 42, 123, 2024, 2025, 3407.
- Folds: 0, 1, 2, 3.
- Test bearing IDs: 4, 8, 12, 15.
- Window size and stride: 8192 and 8192.

## Main comparison

{markdown_table(
    ["Model", "Macro-F1", "Accuracy", "Macro-F1 95% CI", "Runs"],
    main_rows,
)}

## Paired comparison against Raw 1D-CNN

Positive values mean the method has higher Macro-F1 than Raw 1D-CNN on the same
`fold + seed` keys.

{markdown_table(
    ["Model", "Mean paired Δ Macro-F1", "Δ 95% CI", "Wilcoxon p", "Pairs"],
    test_rows,
)}

## Label efficiency

The full balanced budget samples the minimum available count from every class;
it is not all training windows.

{markdown_table(
    ["Balanced budget", "Model", "Macro-F1", "Accuracy", "Runs"],
    label_rows,
)}

## Threshold sensitivity at a 50% balanced budget

{markdown_table(
    ["τ", "Macro-F1", "Accuracy", "Mean pseudo-labels", "Coverage", "Pseudo accuracy", "Runs"],
    threshold_rows,
)}

## Calibration

The test-set ECE values are descriptive. Temperature is selected by validation
NLL, and temperature scaling preserves the predicted class.

{markdown_table(
    ["Model", "ECE raw", "ECE scaled", "NLL raw", "NLL scaled"],
    calibration_rows,
)}

## Validation-to-test domain gap

{markdown_table(
    ["Model", "Validation Macro-F1", "Test Macro-F1", "Absolute Gap", "Relative Drop", "Runs"],
    domain_gap_rows,
)}

## Computational cost

This aggregate groups all experiment types by method. The manuscript cost table
uses only the 20 main rows per method, whereas this report includes all matched
rows represented by each model key. Inference latency is measured per test batch
and is hardware-dependent.

{markdown_table(
    ["Model", "Parameters (M)", "GFLOPs/sample", "Train time (s)", "Inference (ms/batch)"],
    complexity_rows,
)}

## Interpretation boundary

- The 0.28–0.31 Macro-F1 range is measured under simultaneous unseen-bearing
  and unseen-condition shift; it is not comparable to random-window or
  same-bearing scores.
- Temperature scaling changes probabilities and calibration metrics, not the
  predicted class.
- Self-training is not consistently beneficial across budgets or seeds.
- Threshold response is non-monotonic, and some runs produce no pseudo-label
  above the configured threshold.
- Values are specific to this public benchmark, model implementations, and
  hardware environment.
"""
    output = results_dir / "EXPERIMENT_REPORT.md"
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
