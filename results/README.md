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
model_complexity.csv         # aggregates every experiment type by method
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
```

`runs/*/predictions.npz` are small diagnostic logit archives retained for
independent metric recomputation. Configuration snapshots, compact CSV
summaries, and recheck reports are also retained for audit.

The `sota/` directory is intentionally separate from the 320 canonical rows.
It contains DRSN and TFNet same-protocol adaptations, their paired tests,
complexity summary, per-run configuration, predictions, and independent
rechecks.
