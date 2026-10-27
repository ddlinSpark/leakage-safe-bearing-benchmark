"""Build the statistical sensitivity and revision audit Markdown reports."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "raw_cnn": "Raw 1D-CNN",
    "gru_only": "GRU-only",
    "cwt_cnn": "CWT 2D-CNN",
    "cwt_gru_no_se": "CWT-GRU without SE",
    "cwt_gru": "CWT-GRU",
    "cwt_gru_temp": "CWT-GRU with temperature scaling",
    "cwt_gru_selftrain": "CWT-GRU self-training",
    "drsn": "DRSN adaptation",
    "tfnet": "TFN adaptation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(low: float, high: float) -> str:
    return f"[{low:+.3f}, {high:+.3f}]"


def p_formatted(value: float) -> str:
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def inference_label(row: pd.Series) -> str:
    if row["hierarchical_ci_low"] > 0:
        result = "Higher mean; interval above zero"
    elif row["hierarchical_ci_high"] < 0:
        result = "Lower mean; interval below zero"
    else:
        result = "No stable direction"
    if bool(row["conclusion_changed"]):
        result += "; changed from unclustered test"
    return result


def comparison_label(row: pd.Series) -> str:
    label = (
        f"{METHOD_LABELS[row['reference']]} vs {METHOD_LABELS[row['method']]}"
    )
    if row["experiment_type"] == "label_efficiency":
        label = f"50% balanced budget: {label}"
    elif row["experiment_type"] == "label_full":
        label = f"Full balanced budget: {label}"
    return label


def comparison_table(statistics: pd.DataFrame) -> str:
    lines = [
        "| Comparison | Mean delta | Unclustered 95% CI | Hierarchical 95% CI | "
        "Unclustered Wilcoxon | Fold sign-flip | Interpretation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in statistics.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        comparison = comparison_label(row_series)
        lines.append(
            "| "
            + " | ".join(
                [
                    comparison,
                    f"{row.mean_delta:+.3f}",
                    interval(row.naive_ci_low, row.naive_ci_high),
                    interval(
                        row.hierarchical_ci_low,
                        row.hierarchical_ci_high,
                    ),
                    p_formatted(row.naive_wilcoxon_p),
                    p_formatted(row.fold_aware_p),
                    inference_label(row_series),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def changed_rows(statistics: pd.DataFrame) -> list[pd.Series]:
    return [
        row
        for _, row in statistics.iterrows()
        if bool(row["conclusion_changed"])
    ]


def write_sensitivity_report(
    root: Path,
    statistics: pd.DataFrame,
) -> None:
    changed = changed_rows(statistics)
    changed_lines = []
    for row in changed:
        comparison = comparison_label(row)
        changed_lines.append(
            f"- {comparison}: the unclustered Wilcoxon result was "
            f"{p_formatted(float(row['naive_wilcoxon_p']))}, whereas the "
            f"hierarchical interval was "
            f"{interval(float(row['hierarchical_ci_low']), float(row['hierarchical_ci_high']))} "
            f"and the fold sign-flip result was "
            f"{p_formatted(float(row['fold_aware_p']))}. The manuscript uses "
            "the conservative conclusion."
        )
    if not changed_lines:
        changed_lines.append("- No comparison changed its inferential conclusion.")

    unchanged = statistics.loc[~statistics["conclusion_changed"]]
    unchanged_lines = []
    for row in unchanged.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        comparison = comparison_label(row_series)
        unchanged_lines.append(
            f"- {comparison}: hierarchical interval "
            f"{interval(row.hierarchical_ci_low, row.hierarchical_ci_high)}; "
            f"{inference_label(row_series).lower()}."
        )

    text = f"""# Statistical Sensitivity Report

## Scope

This report reuses the existing 320 canonical fold-seed runs and 40 external
fold-seed runs. No model was retrained, no seed or fold was removed, and the
leakage-safe split was not changed. The analysis file is
`results/hierarchical_statistics.csv`.

## Why 20 fold-seed evaluations are not 20 independent domains

The principal matrix contains four operating-condition folds and five seeds per
fold. Within a fold, all five seeds use the same train, validation, and test
partitions, including the same held-out bearing codes and operating condition.
Seeds change optimization randomness, not the domain-level test sample.
Treating all 20 fold-seed cells as independent domains would overstate the
effective sample size and can make a p-value look stronger than the design
supports.

## Statistical methods

- Descriptive statistics retain the mean, sample standard deviation, and all 20
  fold-seed evaluations for every method condition.
- The primary paired interval is a two-stage hierarchical cluster bootstrap.
  Operating-condition folds are resampled first; seeds are then resampled within
  each selected fold. Reference and comparator rows remain paired by fold and
  seed. Each comparison uses 20,000 bootstrap replicates with a deterministic
  seed.
- A two-sided exact sign-flip test is applied to the four fold-level mean paired
  differences. Four clusters permit only 16 sign assignments, so the smallest
  attainable nonzero two-sided p-value is 0.125. This test is reported as a
  coarse sensitivity check.
- The previous unclustered t interval and Wilcoxon signed-rank result are
  retained for comparison. They are not treated as the primary evidence.

## Results comparison

{comparison_table(statistics)}

## Conclusions that changed

{chr(10).join(changed_lines)}

## Conclusions that remained unchanged

{chr(10).join(unchanged_lines)}

## Interpretation under four domain-level units

Four held-out operating conditions provide limited inferential resolution. A
fold-aware interval that excludes zero is evidence of a consistent direction
under this protocol, not a precise estimate of generalization across all
bearing systems. Conversely, a nonsignificant fold sign-flip result should not
be described as proof of equivalence. The paper therefore emphasizes effect
direction, hierarchical intervals, and descriptive stability, and keeps
p-values secondary.

The external DRSN and TFN comparisons have intervals below zero, but four
domain-level units cannot support a strong significance claim. The self-training
comparisons have hierarchical intervals that include zero and are described as
showing no stable evidence of improvement.

No exact fold sign-flip p-value in this study can be below 0.125 with four
operating-condition folds, and none was. The manuscript therefore does not
upgrade a fold-aware result to a p < 0.05 claim. Unclustered p-values remain
visible only to show where the previous test was more permissive.

## Recommended manuscript wording

Use "the fold-aware interval included zero" rather than "the methods were
equivalent." Use "the mean was lower" for direction and avoid "significantly
outperformed" unless the fold-aware analysis and its limited cluster count
support that wording. Report the 20 evaluations as a crossed fold-seed matrix,
not as 20 independent datasets.
"""
    audit_dir = root / "docs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "STATISTICAL_SENSITIVITY_REPORT.md").write_text(
        text,
        encoding="utf-8",
    )


def write_statistical_audit(
    root: Path,
    statistics: pd.DataFrame,
) -> None:
    raw_path = root / "results" / "raw_runs.csv"
    external_path = root / "results" / "sota" / "raw_runs.csv"
    backup_path = (
        root
        / "docs"
        / "paper"
        / "archive"
        / "Leakage_Safe_Bearing_Benchmark_SCI_pre_statistics_revision.docx"
    )
    final_path = (
        root
        / "docs"
        / "paper"
        / "Leakage_Safe_Bearing_Benchmark_SCI_final.docx"
    )
    changed = changed_rows(statistics)
    changed_text = "\n".join(
        f"- `{row['comparison']}`: unclustered significance "
        f"{bool(row['naive_significant'])}, hierarchical interval excludes zero "
        f"{bool(row['hierarchical_ci_excludes_zero'])}; the primary conclusion "
        "was changed conservatively."
        for row in changed
    )
    if not changed_text:
        changed_text = "- No conclusion changed."

    backup_line = (
        f"- Backup: `{backup_path.relative_to(root).as_posix()}`; SHA-256 "
        f"`{sha256(backup_path)}`."
        if backup_path.is_file()
        else "- Backup file was not found."
    )
    final_line = (
        f"- Revised manuscript: `{final_path.relative_to(root).as_posix()}`; "
        f"SHA-256 `{sha256(final_path)}`."
        if final_path.is_file()
        else "- Revised manuscript had not yet been generated when this audit was written."
    )

    text = f"""# Final Statistical Audit

## Data preservation

- Canonical result rows: {len(pd.read_csv(raw_path))}
- External-baseline rows: {len(pd.read_csv(external_path))}
- Canonical raw-results SHA-256: `{sha256(raw_path)}`
- External raw-results SHA-256: `{sha256(external_path)}`
{backup_line}
{final_line}

No experimental row was rerun, deleted, relabeled by seed performance, or
filtered by outcome.

## Before revision

The manuscript described 20 matched fold-seed evaluations and used a t interval
plus a two-sided Wilcoxon signed-rank test across those 20 cells. The
fold-within-seed dependence was not represented in the inferential analysis.
Those values remain available in `results/statistical_tests.csv` as sensitivity
analyses.

## After revision

The manuscript now distinguishes descriptive variability from domain-level
inference. Primary paired intervals use a deterministic two-stage cluster
bootstrap with operating-condition folds as clusters and seeds nested within
folds. The exact fold sign-flip test is reported with its minimum attainable
two-sided p-value of 0.125. The report is generated from all 20 matched
fold-seed pairs per comparison; no average-of-seed summary replaces the raw
pairs.

## Changed conclusions

{changed_text}

## Preserved conclusions

- Raw 1D-CNN versus CWT 2D-CNN still has a hierarchical interval that includes
  zero.
- Raw 1D-CNN versus CWT-GRU still has a hierarchical interval that includes
  zero.
- CWT-GRU without SE versus full CWT-GRU still has an interval that includes
  zero; the numerical direction is retained without an SE penalty claim.
- Self-training at 50% balanced budget and Full balanced budget still has no
  stable evidence of improvement.
- DRSN and TFN adaptations have lower mean macro-F1 than Raw 1D-CNN under the
  same protocol; the four-fold p-values remain explicitly coarse.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.analysis.hierarchical `
  --results-dir results `
  --external-dir results/sota `
  --output results/hierarchical_statistics.csv `
  --bootstrap-samples 20000
python scripts/build_statistical_reports.py
```

The comparison table and all numerical statements in the revised manuscript
are loaded from `results/hierarchical_statistics.csv` or the retained canonical
result files by the manuscript builder.
"""
    audit_dir = root / "docs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "FINAL_STATISTICAL_AUDIT.md").write_text(
        text,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    statistics = pd.read_csv(root / "results" / "hierarchical_statistics.csv")
    write_sensitivity_report(root, statistics)
    write_statistical_audit(root, statistics)
    print("Wrote statistical sensitivity and revision audit reports.")


if __name__ == "__main__":
    main()
