"""Fold-aware paired statistics for repeated cross-domain evaluations.

The 20 main evaluations are four operating-condition folds crossed with five
stochastic seeds.  Seeds within a fold share the same test partition, so the
fold, not the fold-seed cell, is the independent domain-level unit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..experiments.common import atomic_write_csv
from ..paths import PROJECT_ROOT


@dataclass(frozen=True)
class Comparison:
    comparison_id: str
    experiment_type: str
    reference: str
    method: str
    label_ratio: float | None = None


COMPARISONS = (
    Comparison("main:raw_cnn:gru_only", "main", "raw_cnn", "gru_only"),
    Comparison("main:raw_cnn:cwt_cnn", "main", "raw_cnn", "cwt_cnn"),
    Comparison(
        "main:raw_cnn:cwt_gru_no_se",
        "main",
        "raw_cnn",
        "cwt_gru_no_se",
    ),
    Comparison("main:raw_cnn:cwt_gru", "main", "raw_cnn", "cwt_gru"),
    Comparison(
        "external:raw_cnn:drsn",
        "external",
        "raw_cnn",
        "drsn",
    ),
    Comparison(
        "external:raw_cnn:tfnet",
        "external",
        "raw_cnn",
        "tfnet",
    ),
    Comparison(
        "main:cwt_cnn:cwt_gru",
        "main",
        "cwt_cnn",
        "cwt_gru",
    ),
    Comparison(
        "main:cwt_gru_no_se:cwt_gru",
        "main",
        "cwt_gru_no_se",
        "cwt_gru",
    ),
    Comparison(
        "label_50:cwt_gru:cwt_gru_selftrain",
        "label_efficiency",
        "cwt_gru",
        "cwt_gru_selftrain",
        label_ratio=0.5,
    ),
    Comparison(
        "label_full:cwt_gru:cwt_gru_selftrain",
        "label_full",
        "cwt_gru",
        "cwt_gru_selftrain",
        label_ratio=1.0,
    ),
    Comparison(
        "main:gru_only:cwt_gru",
        "main",
        "gru_only",
        "cwt_gru",
    ),
    Comparison(
        "external:cwt_cnn:drsn",
        "external",
        "cwt_cnn",
        "drsn",
    ),
    Comparison(
        "external:cwt_cnn:tfnet",
        "external",
        "cwt_cnn",
        "tfnet",
    ),
    Comparison(
        "external:cwt_gru:drsn",
        "external",
        "cwt_gru",
        "drsn",
    ),
    Comparison(
        "external:cwt_gru:tfnet",
        "external",
        "cwt_gru",
        "tfnet",
    ),
    Comparison(
        "external:cwt_gru_no_se:drsn",
        "external",
        "cwt_gru_no_se",
        "drsn",
    ),
    Comparison(
        "external:cwt_gru_no_se:tfnet",
        "external",
        "cwt_gru_no_se",
        "tfnet",
    ),
    Comparison(
        "main:raw_cnn:cwt_gru_temp",
        "main",
        "raw_cnn",
        "cwt_gru_temp",
    ),
    Comparison(
        "main:raw_cnn:cwt_gru_selftrain",
        "main",
        "raw_cnn",
        "cwt_gru_selftrain",
    ),
)


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
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "hierarchical_statistics.csv",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261027)
    return parser.parse_args()


def _t_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if len(clean) < 2:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    half_width = stats.t.ppf(1.0 - alpha / 2.0, len(clean) - 1) * (
        clean.std(ddof=1) / np.sqrt(len(clean))
    )
    return float(clean.mean() - half_width), float(clean.mean() + half_width)


def _hierarchical_bootstrap_interval(
    paired: pd.DataFrame,
    column: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Resample folds and then seeds within each selected fold."""

    folds = sorted(int(value) for value in paired.index.get_level_values("fold").unique())
    values_by_fold = {
        fold: paired.loc[fold, column].to_numpy(dtype=float) for fold in folds
    }
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=float)
    for index in range(samples):
        selected_folds = rng.choice(folds, size=len(folds), replace=True)
        selected_values: list[np.ndarray] = []
        for fold in selected_folds:
            fold_values = values_by_fold[int(fold)]
            seed_indices = rng.integers(0, len(fold_values), size=len(fold_values))
            selected_values.append(fold_values[seed_indices])
        bootstrap_means[index] = np.concatenate(selected_values).mean()
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def _fold_signflip_p_value(paired: pd.DataFrame, column: str) -> float:
    """Exact sign-flip test on fold-level mean paired differences.

    With four folds the attainable two-sided p-value for a nonzero effect is
    at least 0.125.  This coarseness is reported rather than hidden.
    """

    fold_means = paired.groupby(level="fold")[column].mean().to_numpy(dtype=float)
    observed = float(fold_means.mean())
    if np.allclose(fold_means, 0.0, rtol=0.0, atol=1e-15):
        return 1.0
    permuted = []
    for signs in product((-1.0, 1.0), repeat=len(fold_means)):
        permuted.append(float(np.mean(fold_means * np.asarray(signs))))
    exceedances = np.sum(
        np.abs(np.asarray(permuted)) >= abs(observed) - 1e-15
    )
    return float(exceedances / len(permuted))


def _select_rows(
    canonical: pd.DataFrame,
    external: pd.DataFrame,
    comparison: Comparison,
) -> pd.DataFrame:
    if comparison.experiment_type == "main":
        selected = canonical.loc[
            canonical["experiment_type"].eq("main")
            & canonical["method"].isin([comparison.reference, comparison.method])
        ].copy()
    elif comparison.experiment_type == "label_efficiency":
        selected = canonical.loc[
            canonical["experiment_type"].eq("label_efficiency")
            & np.isclose(
                canonical["label_ratio"].to_numpy(dtype=float),
                float(comparison.label_ratio),
            )
            & canonical["method"].isin([comparison.reference, comparison.method])
        ].copy()
    elif comparison.experiment_type == "label_full":
        selected = canonical.loc[
            canonical["experiment_type"].eq("main")
            & np.isclose(canonical["label_ratio"].to_numpy(dtype=float), 1.0)
            & canonical["method"].isin([comparison.reference, comparison.method])
        ].copy()
    elif comparison.experiment_type == "external":
        reference = canonical.loc[
            canonical["experiment_type"].eq("main")
            & canonical["method"].eq(comparison.reference)
        ].copy()
        adapted = external.loc[external["method"].eq(comparison.method)].copy()
        selected = pd.concat([reference, adapted], ignore_index=True)
    else:
        raise ValueError(f"Unknown comparison type: {comparison.experiment_type}")

    if not selected["status"].astype(str).eq("success").all():
        raise ValueError(f"Failed rows found for {comparison.comparison_id}")
    return selected


def _paired_frame(selected: pd.DataFrame, comparison: Comparison) -> pd.DataFrame:
    pivot = selected.pivot_table(
        index=["fold", "seed"],
        columns="method",
        values="macro_f1",
        aggfunc="mean",
    )
    required = [comparison.reference, comparison.method]
    missing = [method for method in required if method not in pivot.columns]
    if missing:
        raise ValueError(
            f"Missing methods for {comparison.comparison_id}: {missing}"
        )
    paired = pivot[required].dropna().copy()
    expected = {
        (fold, seed)
        for fold in (0, 1, 2, 3)
        for seed in (42, 123, 2024, 2025, 3407)
    }
    if set(paired.index) != expected:
        raise ValueError(
            f"Incomplete fold-seed matrix for {comparison.comparison_id}: "
            f"expected {len(expected)} pairs, found {len(paired)}"
        )
    if paired.index.get_level_values("fold").nunique() != 4:
        raise ValueError(f"Expected four folds for {comparison.comparison_id}")
    if set(paired.index.get_level_values("seed")) != {
        42,
        123,
        2024,
        2025,
        3407,
    }:
        raise ValueError(f"Unexpected seeds for {comparison.comparison_id}")
    paired["difference"] = paired[comparison.method] - paired[comparison.reference]
    return paired


def build_hierarchical_statistics(
    canonical: pd.DataFrame,
    external: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, comparison in enumerate(COMPARISONS):
        selected = _select_rows(canonical, external, comparison)
        paired = _paired_frame(selected, comparison)
        differences = paired["difference"].to_numpy(dtype=float)
        naive_low, naive_high = _t_interval(differences)
        hierarchical_low, hierarchical_high = _hierarchical_bootstrap_interval(
            paired,
            "difference",
            samples=bootstrap_samples,
            seed=seed + offset,
        )
        if np.allclose(differences, 0.0, rtol=0.0, atol=1e-15):
            naive_p = 1.0
        else:
            naive_p = float(
                stats.wilcoxon(
                    differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                ).pvalue
            )
        fold_p = _fold_signflip_p_value(paired, "difference")
        naive_significant = naive_p < 0.05
        hierarchical_supported = hierarchical_low > 0.0 or hierarchical_high < 0.0
        rows.append(
            {
                "comparison": comparison.comparison_id,
                "experiment_type": comparison.experiment_type,
                "label_ratio": comparison.label_ratio,
                "reference": comparison.reference,
                "method": comparison.method,
                "n_folds": int(paired.index.get_level_values("fold").nunique()),
                "n_seeds": int(paired.index.get_level_values("seed").nunique()),
                "n_pairs": len(paired),
                "reference_mean_macro_f1": paired[comparison.reference].mean(),
                "method_mean_macro_f1": paired[comparison.method].mean(),
                "mean_delta": float(differences.mean()),
                "mean_paired_delta": float(differences.mean()),
                "std_delta": float(differences.std(ddof=1)),
                "naive_ci_low": naive_low,
                "naive_ci_high": naive_high,
                "hierarchical_ci_low": hierarchical_low,
                "hierarchical_ci_high": hierarchical_high,
                "naive_wilcoxon_p": naive_p,
                "fold_aware_p": fold_p,
                "naive_significant": naive_significant,
                "hierarchical_ci_excludes_zero": hierarchical_supported,
                "conclusion_changed": naive_significant != hierarchical_supported,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": seed + offset,
                "inferential_unit": "operating-condition fold",
                "p_value_note": (
                    "Exact sign-flip test on four fold means; the minimum "
                    "attainable two-sided p-value is 0.125."
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    canonical = pd.read_csv(args.results_dir / "raw_runs.csv")
    external = pd.read_csv(args.external_dir / "raw_runs.csv")
    table = build_hierarchical_statistics(
        canonical,
        external,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    atomic_write_csv(args.output, table)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
