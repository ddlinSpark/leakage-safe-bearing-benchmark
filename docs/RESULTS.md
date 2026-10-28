# Results

## Main benchmark

Values are mean +/- sample standard deviation over four folds and five seeds.

| Method | Test Macro-F1 | Accuracy |
| --- | ---: | ---: |
| Raw 1D-CNN | 0.289 +/- 0.104 | 0.392 +/- 0.094 |
| GRU only | 0.221 +/- 0.030 | 0.242 +/- 0.025 |
| CWT 2D-CNN | 0.291 +/- 0.086 | 0.399 +/- 0.081 |
| CWT-GRU without SE | 0.307 +/- 0.103 | 0.408 +/- 0.097 |
| CWT-GRU | 0.280 +/- 0.107 | 0.383 +/- 0.092 |
| CWT-GRU + temperature scaling + self-training | 0.290 +/- 0.116 | 0.387 +/- 0.099 |

No evaluated representation, recurrent addition, SE block, or self-training
procedure establishes a stable primary advantage.

## Generalization gap

For the four principal models:

- validation Macro-F1: approximately `0.797-0.853`;
- held-out test Macro-F1: approximately `0.280-0.307`.

The gap indicates that development-domain fit is not evidence of transfer to a
new physical bearing and operating condition. Bearing identity,
operating-condition, and damage-origin effects remain coupled.

## Main paired comparisons

| Comparison | Mean paired change | Hierarchical 95% interval | Fold wins/losses |
| --- | ---: | ---: | ---: |
| CWT 2D-CNN minus Raw | +0.002 | [-0.093, +0.135] | 1/3 |
| CWT-GRU minus Raw | -0.010 | [-0.140, +0.131] | 2/2 |
| Full CWT-GRU minus no-SE | -0.027 | [-0.073, +0.022] | 2/2 |

The table reports the stored `method minus reference` direction. No comparison
establishes a stable gain.

## Self-training

- Coverage-controlled 50% budget minus supervised: `+0.034`, interval
  `[-0.020, +0.096]`, three of four fold wins.
- Offline pseudo-label accuracy is approximately `0.987`.
- Full-budget coverage-controlled minus supervised: approximately `0.000`.
- No multiple-damage pseudo-labels were selected in 4 of 20 full-budget runs.

High pseudo-label precision does not establish a transferable end-task gain.

## Label granularity

| Model | Four classes | Healthy/Fault |
| --- | ---: | ---: |
| Raw 1D-CNN | 0.289 | 0.701 |
| CWT-GRU | 0.280 | 0.607 |

This post-hoc analysis indicates partial anomaly-level separability but weaker
fault-location boundary transfer. It is not a replacement primary task.

## Calibration

Temperature scaling changes probability metrics and not argmax classification.
Validation-fitted temperatures do not necessarily improve held-out test
calibration. For CWT-GRU, mean test ECE changes from approximately `0.543` to
`0.391`; for Raw 1D-CNN, mean test ECE changes from approximately `0.388` to
`0.424`.

## Canonical sources

- Row-level primary results: `results/raw_runs.csv`.
- Main summaries: `results/summary.csv`,
  `results/summary_with_ci.csv`.
- Descriptive paired tests: `results/statistical_tests.csv`.
- Fold-aware inference: `results/hierarchical_statistics.csv`.
- Domain gap: `results/domain_gap_summary.csv`.
- External adaptations: `results/sota/raw_runs.csv`.
- Consolidated numerical snapshot: `results/paper_metrics.json`.
- Manuscript source mapping: `docs/paper/TABLE_SOURCES.md` and
  `docs/paper/FIGURE_SOURCES.md`.

The automated consistency report is `docs/reports/CONSISTENCY_REPORT.md`.
