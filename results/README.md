# Results

This directory is populated by the unified benchmark runner.

Canonical files:

```text
raw_runs.csv                 # row-level experiment results
summary.csv                  # main model comparison
summary_with_ci.csv          # mean, SD, and 95% CI
domain_gap_summary.csv       # validation-to-test generalization gap
ablation_summary.csv         # component-wise ablation matrix
label_efficiency_summary.csv
threshold_summary.csv
statistical_tests.csv
hierarchical_statistics.csv # fold-aware paired comparisons and sensitivity tests
model_complexity.csv         # aggregates every experiment type by method
domain_shift_statistics.csv
domain_shift_contrasts.csv
damage_origin_manifest.csv
label_granularity_sensitivity.csv
calibration_per_run.csv
calibration_summary.csv
training_diagnostics_summary.csv
self_training_sensitivity_summary.csv
cwt_sensitivity_summary.csv
normalization_sensitivity_summary.csv
external_baseline_tuning.csv
experiment_registry.csv
paper_metrics.json
EXPERIMENT_REPORT.md
rechecks/all_runs.json
config_used.json
run_benchmark.log
runs/<run_id>/               # config, history, predictions, diagnostics
splits/fold<k>.json          # exact split membership
sota/                        # 40 external same-protocol retraining records
```

Do not edit result CSVs manually. Regenerate them from `raw_runs.csv` with:

```bash
PYTHONPATH=src python -m spindle_cnn.analysis.statistics \
  --results results/raw_runs.csv \
  --output-dir results

PYTHONPATH=src python -m spindle_cnn.analysis.hierarchical \
  --results-dir results \
  --external-dir results/sota \
  --output results/hierarchical_statistics.csv
```

`runs/*/predictions.npz` are small diagnostic logit archives retained for
independent metric recomputation. Configuration snapshots, compact CSV
summaries, and recheck reports are also retained for audit. The fold-aware file
is the primary paired-inference source; the unclustered t intervals and
Wilcoxon tests remain sensitivity analyses.

The `sota/` directory is intentionally separate from the 320 canonical rows.
It contains DRSN and TFN same-protocol adaptations, their paired tests,
complexity summary, per-run configuration, predictions, and independent
rechecks.

`cwt_sensitivity/`, `normalization_sensitivity/`, and
`self_training_sensitivity/` are separate sensitivity result families. They do
not add rows to the canonical `results/raw_runs.csv` matrix and are never
substituted into the primary comparison based on held-out test score.
