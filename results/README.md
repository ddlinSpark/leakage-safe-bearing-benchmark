# Results

This directory is populated by the unified benchmark runner.

Canonical files:

```text
raw_runs.csv                 # row-level experiment results
summary.csv                  # main model comparison
summary_with_ci.csv          # mean, SD, and 95% CI
ablation_summary.csv         # component-wise ablation matrix
label_efficiency_summary.csv
threshold_summary.csv
statistical_tests.csv
model_complexity.csv
EXPERIMENT_REPORT.md
rechecks/all_runs.json
config_used.json
run_benchmark.log
runs/<run_id>/               # config, history, predictions, diagnostics
splits/fold<k>.json          # exact split membership
```

Do not edit result CSVs manually. Regenerate them from `raw_runs.csv` with:

```bash
PYTHONPATH=src python -m spindle_cnn.analysis.statistics \
  --results results/raw_runs.csv \
  --output-dir results
```

`runs/*/predictions.npz` is local diagnostic data and is ignored by Git. The
configuration snapshots and compact CSV summaries are retained for audit.
