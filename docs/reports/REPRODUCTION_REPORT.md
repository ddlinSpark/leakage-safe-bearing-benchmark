# Reproduction Report

## Verdict

**VERIFIED for the retained result registry and manuscript tables.** Every canonical test metric was recomputed independently from saved logits, all split manifests were regenerated, and all 13 original manuscript tables passed the independent numerical audit.

The status is not a claim that all 320 stochastic training runs were executed again during this revision. Training-time reproduction is reported separately in `docs/reports/CLEAN_REPRODUCTION_REPORT.md`.

## Recorded environment

Python 3.13.5; PyTorch 2.11.0+cu130; CUDA 13.0; NumPy 2.1.3; pandas 2.2.3; device NVIDIA GeForce RTX 5080

## Result inventory

- Canonical rows: 320
- Main rows: 140
- Label-efficiency rows: 80
- Threshold rows: 100
- External rows: 40
- Canonical raw_runs.csv SHA-256: `fc8336c7d44cba399c8b0a2356cc378d80a36c4e368769ff922a3e41f55ab5bf`
- External raw_runs.csv SHA-256: `193bf7ac9a00d8980eb3c5ed807db6ac12a33dc1657f2a6d1eb0c95c58b7881e`
- Dataset hash values: 1
- Split hashes by fold: {0: 1, 1: 1, 2: 1, 3: 1}

## Numerical audit

- Overall independent audit: **PASS**
- Mismatches: 0
- Canonical logits rechecked: 320
- External logits rechecked: 40
- Tables parsed: 18
- Figures parsed: 23

## No-SE interval consistency

| point_estimate | recomputed_ci_low | recomputed_ci_high | point_outside_percentile_interval | status |
| --- | --- | --- | --- | --- |
| -0.0270 | -0.0733 | 0.0221 | False | VERIFIED |

The apparent mismatch between the positive no-SE advantage and an interval whose upper endpoint is smaller comes from a direction label. The stored row is full-minus-no-SE (`-0.027`, interval `[-0.073, +0.022]`). Relabelled as no-SE-minus-full, the estimate and both interval endpoints reverse sign, giving `+0.027` and `[-0.022, +0.073]`; the estimate then lies inside. The conservative no-stable-evidence conclusion remains.

## Main figures

| figure | file | status |
| --- | --- | --- |
| Figure 4 | figure_01_model_comparison.png | PRESENT |
| Figure 5 | figure_03_label_efficiency.png | PRESENT |
| Figure 6 | figure_04_threshold_sensitivity.png | PRESENT |
| Figure 7 | figure_05_calibration.png | PRESENT |
| Figure 8 | figure_08_reliability_diagrams.png | PRESENT |
| Figure 9 | figure_02_ablation.png | PRESENT |
| Figure 10 | figure_06_confusion_matrices.png | PRESENT |
| Figure 11 | figure_07_computational_cost.png | PRESENT |
| Figure 12 | figure_09_validation_test_gap.png | PRESENT |

## Reproduction levels

| Level | Status | Evidence |
| --- | --- | --- |
| Split reconstruction | VERIFIED | Four manifests regenerate with identical index sets and hashes. |
| Metric recomputation | VERIFIED | 320 canonical and 40 external rows reproduce from saved logits. |
| Manuscript tables | VERIFIED | Tables 1-13 pass the independent audit. |
| Primary figures | VERIFIED | Figures 4-12 are retained as publication-resolution PNG files. |
| Full stochastic retraining | SEE CLEAN REPORT | Time-intensive training is evaluated in the isolated clean checkout. |
| Sensitivity experiments | SEE EXPERIMENT REPORT | New runs are marked exploratory where the worktree was dirty. |
