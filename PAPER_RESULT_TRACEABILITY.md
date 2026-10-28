# Paper Result Traceability

Every numerical manuscript item maps to retained code and result rows. `results/paper_metrics.json` contains source hashes and the consolidated values used for machine verification.

| Manuscript item | Derived source | Raw or run-level source | Operation |
| --- | --- | --- | --- |
| Table 5 | results/summary_with_ci.csv | results/raw_runs.csv | main rows; mean and sample SD |
| Table 6 | results/hierarchical_statistics.csv | results/raw_runs.csv | paired fold-seed differences |
| Table 7 | results/label_efficiency_summary.csv | results/raw_runs.csv | label_efficiency rows plus reused main Full row |
| Table 8 | results/threshold_summary.csv | results/raw_runs.csv | threshold rows |
| Table 9 | results/calibration_per_run.csv | results/raw_runs.csv | per-run probability metrics |
| Table 10 | results/summary_with_ci.csv; results/sota/summary_with_ci.csv | results/raw_runs.csv; results/sota/raw_runs.csv | main and external rows |
| Table 11 | results/ablation_summary.csv | results/raw_runs.csv | component matrix and main rows |
| Table 12 | results/model_complexity.csv | results/raw_runs.csv | median complexity |
| Table 13 | results/domain_gap_summary.csv | results/raw_runs.csv | validation minus test Macro-F1 |
| Table 14 | results/domain_shift_contrasts.csv | results/domain_shift_window_metrics.csv | train-to-test standardized contrasts |
| Table 15 | results/label_granularity_sensitivity.csv | results/label_granularity_per_run.csv | post-hoc probability grouping |
| Table 16 | results/cwt_sensitivity_summary.csv | results/cwt_sensitivity/*/raw_runs.csv | matched CWT controls |
| Table 17 | results/self_training_sensitivity_summary.csv | results/self_training_sensitivity/raw_runs.csv | coverage versus fixed threshold |
| Table 18 | results/training_budget_comparison.csv; results/external_baseline_tuning_comparisons.csv | results/training_budget_30/raw_runs.csv; results/baseline_tuning/raw_runs.csv | validation-only audits |
| Figure 13 | figures/publication/figure_13_domain_shift_diagnostics.png | results/domain_shift_window_metrics.csv | analysis.domain_shift |
| Figure 14 | figures/publication/figure_16_label_granularity.png | results/label_granularity_sensitivity.csv | analysis.label_granularity |
| Figure 15 | figures/publication/figure_19_cwt_sensitivity.png | results/cwt_sensitivity_summary.csv | analysis.cwt_sensitivity |
| Figure 16 | figures/publication/figure_17_self_training_sensitivity.png | results/self_training_sensitivity_summary.csv | analysis.self_training |
| Figure 17 | figures/publication/figure_22_feature_space_pca.png | results/feature_embeddings/fold*.npz | analysis.feature_space |
| Figure 18 | figures/publication/figure_23_feature_space_tsne.png | results/feature_embeddings/fold*.npz | analysis.feature_space |
| Figure S1 | figures/publication/figure_14_per_fold_reliability.png | results/calibration_per_fold_bins.csv | analysis.calibration_audit |
| Figure S2 | figures/publication/figure_15_temperature_distribution.png | results/calibration_per_run.csv | analysis.calibration_audit |
| Figure S3 | figures/publication/figure_20_foldwise_paired_differences.png | results/foldwise_paired_differences.csv | analysis.foldwise |
| Figure S4 | figures/publication/figure_21_training_budget.png | results/training_budget_comparison.csv | analysis.training_budget |
| Figure S5 | figures/publication/figure_24_normalization_sensitivity.png | results/normalization_sensitivity_summary.csv | analysis.normalization_sensitivity |

## Verification commands

```powershell
python scripts/build_paper_metrics.py
python scripts/verify_manuscript_numbers.py
python scripts/audit_submission.py --json-output artifacts/local/pre_submission_audit.json
```

## Integrity rules

- Test labels are read for final metrics and post-hoc diagnostics only.
- Pseudo-label and threshold decisions use training candidates and validation data only.
- Baseline and training-budget choices use validation metrics only.
- The four original test folds are never modified.
- Sensitivity results are not substituted into the primary comparison.
