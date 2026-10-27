# Five-Seed Benchmark Report

This report was generated from `results/raw_runs.csv`. No values are entered by
hand. The canonical experiment contains 320 successful runs.

## Experiment integrity

- Main comparison: 140 runs, 20 per method.
- Label efficiency: 80 runs, 20 per method and label ratio.
- Threshold sensitivity: 100 runs, 20 per threshold.
- Failed runs: 0.
- 10% balanced-budget self-training produced no pseudo-label update in 20/20 runs.
- 2 fold-plus-seed threshold cases produced no pseudo-label update at any tested threshold.
- Recorded Git commits: `470df63b4d3ca11fa8e8655d73031c6788014848`, `a146cb5cdff3a1e9c30dada44effd13d6f2fdfc1`, `c9b99120c38358db4e648d0410d414348c25c4cc`.
- Primary seeds: 42, 123, 2024, 2025, 3407.
- Folds: 0, 1, 2, 3.
- Test bearing IDs: 4, 8, 12, 15.
- Window size and stride: 8192 and 8192.

## Main comparison

| Model | Macro-F1 | Accuracy | Macro-F1 95% CI | Runs |
| --- | --- | --- | --- | --- |
| Raw 1D-CNN | 0.289 ± 0.104 | 0.392 ± 0.094 | [0.241, 0.338] | 20 |
| GRU only | 0.221 ± 0.030 | 0.242 ± 0.025 | [0.207, 0.236] | 20 |
| CWT 2D-CNN | 0.291 ± 0.086 | 0.399 ± 0.081 | [0.251, 0.331] | 20 |
| CWT-GRU without SE | 0.307 ± 0.103 | 0.408 ± 0.097 | [0.258, 0.355] | 20 |
| CWT-GRU | 0.280 ± 0.107 | 0.383 ± 0.092 | [0.230, 0.330] | 20 |
| CWT-GRU + temperature scaling | 0.280 ± 0.107 | 0.383 ± 0.092 | [0.230, 0.330] | 20 |
| CWT-GRU + temperature scaling + self-training | 0.290 ± 0.116 | 0.387 ± 0.099 | [0.236, 0.344] | 20 |

## Paired comparison against Raw 1D-CNN

Positive values mean the method has higher Macro-F1 than Raw 1D-CNN on the same
`fold + seed` keys.

| Model | Mean paired Δ Macro-F1 | Δ 95% CI | Wilcoxon p | Pairs |
| --- | --- | --- | --- | --- |
| GRU only | -0.068 | [-0.115, -0.020] | 0.006 | 20 |
| CWT 2D-CNN | +0.002 | [-0.070, 0.074] | 0.596 | 20 |
| CWT-GRU without SE | +0.017 | [-0.063, 0.098] | 0.898 | 20 |
| CWT-GRU | -0.010 | [-0.087, 0.068] | 0.756 | 20 |
| CWT-GRU + temperature scaling | -0.010 | [-0.087, 0.068] | 0.756 | 20 |
| CWT-GRU + temperature scaling + self-training | +0.001 | [-0.078, 0.079] | 0.784 | 20 |

## Label efficiency

The full balanced budget samples the minimum available count from every class;
it is not all training windows.

| Balanced budget | Model | Macro-F1 | Accuracy | Runs |
| --- | --- | --- | --- | --- |
| 10% balanced | CWT-GRU | 0.128 ± 0.025 | 0.250 ± 0.006 | 20 |
| 10% balanced | CWT-GRU + temperature scaling + self-training | 0.128 ± 0.025 | 0.250 ± 0.006 | 20 |
| 50% balanced | CWT-GRU | 0.244 ± 0.098 | 0.351 ± 0.090 | 20 |
| 50% balanced | CWT-GRU + temperature scaling + self-training | 0.283 ± 0.126 | 0.394 ± 0.110 | 20 |
| Full balanced | CWT-GRU | 0.280 ± 0.107 | 0.383 ± 0.092 | 20 |
| Full balanced | CWT-GRU + temperature scaling + self-training | 0.290 ± 0.116 | 0.387 ± 0.099 | 20 |

## Threshold sensitivity at a 50% balanced budget

| τ | Macro-F1 | Accuracy | Mean pseudo-labels | Coverage | Pseudo accuracy | Runs |
| --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 0.270 ± 0.130 | 0.375 ± 0.115 | 757.1 | 0.776 | 0.927 | 20 |
| 0.40 | 0.259 ± 0.124 | 0.365 ± 0.105 | 750.9 | 0.770 | 0.930 | 20 |
| 0.50 | 0.239 ± 0.099 | 0.344 ± 0.089 | 723.9 | 0.742 | 0.942 | 20 |
| 0.60 | 0.241 ± 0.093 | 0.357 ± 0.082 | 671.5 | 0.689 | 0.955 | 20 |
| 0.70 | 0.253 ± 0.106 | 0.372 ± 0.095 | 573.9 | 0.589 | 0.966 | 20 |

## Calibration

The test-set ECE values are descriptive. Temperature is selected by validation
NLL, and temperature scaling preserves the predicted class.

| Model | ECE raw | ECE scaled | NLL raw | NLL scaled |
| --- | --- | --- | --- | --- |
| Raw 1D-CNN | 0.388 | 0.424 | 2.599 | 6.237 |
| CWT 2D-CNN | 0.487 | 0.375 | 3.615 | 2.498 |
| CWT-GRU | 0.543 | 0.391 | 4.528 | 2.304 |
| CWT-GRU + temperature scaling + self-training | 0.522 | 0.397 | 3.969 | 2.482 |

## Validation-to-test domain gap

| Model | Validation Macro-F1 | Test Macro-F1 | Absolute Gap | Relative Drop | Runs |
| --- | --- | --- | --- | --- | --- |
| Raw 1D-CNN | 0.853 ± 0.153 | 0.289 ± 0.104 | 0.564 | 66.4% | 20 |
| CWT 2D-CNN | 0.808 ± 0.112 | 0.291 ± 0.086 | 0.516 | 62.4% | 20 |
| CWT-GRU | 0.797 ± 0.115 | 0.280 ± 0.107 | 0.518 | 63.0% | 20 |
| CWT-GRU without SE | 0.809 ± 0.139 | 0.307 ± 0.103 | 0.503 | 59.6% | 20 |

## Computational cost

This aggregate groups all experiment types by method. The manuscript cost table
uses only the 20 main rows per method, whereas this report includes all matched
rows represented by each model key. Inference latency is measured per test batch
and is hardware-dependent.

| Model | Parameters (M) | GFLOPs/sample | Train time (s) | Inference (ms/batch) |
| --- | --- | --- | --- | --- |
| Raw 1D-CNN | 0.245 | 0.677 | 4.3 | 12.8 |
| GRU only | 0.118 | 1.630 | 23.6 | 43.2 |
| CWT 2D-CNN | 0.885 | 0.185 | 4.2 | 8.3 |
| CWT-GRU without SE | 1.086 | 1.814 | 24.4 | 44.9 |
| CWT-GRU | 1.101 | 1.815 | 14.4 | 45.5 |

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
