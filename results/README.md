# Results

This directory contains the frozen result registry and the per-run evidence used
by the manuscript. Result CSVs are generated outputs and must not be edited by
hand.

## Canonical status

The authoritative canonical row matrix remains at
`results/raw_runs.csv` for backward compatibility with the verified runner,
analysis code, and manuscript builder. The canonical declaration and immutable
hash index are stored in:

```text
results/canonical/manifest.json
```

The manifest records the exact SHA-256 values for row-level results, summaries,
paired tests, hierarchical intervals, the external block, and
`paper_metrics.json`.

No file such as `final_results.csv`, `new_final.csv`, `latest_results.csv`, or
`debug_results.csv` is an official source.

## Primary experiment

| File or directory | Role |
| --- | --- |
| `raw_runs.csv` | 320 canonical row-level evaluations |
| `summary.csv` | Main model mean/SD table |
| `summary_with_ci.csv` | Descriptive confidence intervals |
| `statistical_tests.csv` | Unclustered sensitivity tests |
| `hierarchical_statistics.csv` | Primary fold-first paired inference |
| `domain_gap_summary.csv` | Validation-to-test gap |
| `ablation_summary.csv` | Component ablation |
| `label_efficiency_summary.csv` | Balanced-label budget analysis |
| `threshold_summary.csv` | Confidence-threshold analysis |
| `model_complexity.csv` | Parameter, FLOP, and timing summaries |
| `runs/<run_id>/` | Configuration, history, predictions, confusion, calibration |
| `splits/fold<k>.json` | Exact frozen fold membership |
| `rechecks/all_runs.json` | Independent metric rechecks |

## External adaptations

`sota/` contains 20 DRSN and 20 TFN same-protocol adaptation rows. These are
adaptations, not reproductions of the original papers' reported performance.

## Sensitivity and post-hoc families

| File prefix or directory | Classification |
| --- | --- |
| `cwt_sensitivity/`, `cwt_sensitivity_*` | CWT preprocessing sensitivity |
| `normalization_sensitivity/`, `normalization_sensitivity_*` | Normalization sensitivity |
| `self_training_sensitivity/`, `self_training_*` | Coverage-controlled self-training |
| `baseline_tuning/`, `external_baseline_tuning*` | Validation-only baseline tuning |
| `calibration_*` | Calibration and reliability audit |
| `training_budget_30/`, `training_budget_*` | Training-opportunity audit |
| `label_granularity_*` | Post-hoc label coarsening |
| `domain_shift_*`, `damage_origin_*` | Raw-signal and confounding diagnostics |
| `feature_embeddings/`, `feature_space_clustering.csv` | Post-hoc embedding analysis |
| `overfit_sanity.csv`, `training_diagnostics*` | Engineering sanity diagnostics |

These files do not alter the primary method, split, seed, or configuration.

## Regeneration

```powershell
python scripts/generate_statistics.py
python scripts/build_paper_metrics.py
python scripts/generate_figures.py --include-domain-shift
python scripts/build_results_catalog.py
```

The results catalog command refreshes
`results/canonical/manifest.json` and `splits/manifest.json` without changing
experimental values.

## Integrity rules

- Retain all completed runs and negative results.
- Do not select seeds using test outcomes.
- Do not replace a primary configuration using a sensitivity result.
- Do not treat unclustered tests as the primary domain-level inference.
- Do not edit a CSV to match the manuscript.
- Run `scripts/check_manuscript_consistency.py` after any derived-table rebuild.
