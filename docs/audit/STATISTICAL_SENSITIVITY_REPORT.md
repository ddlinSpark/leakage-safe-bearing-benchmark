# Statistical Sensitivity Report

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

| Comparison | Mean delta | Unclustered 95% CI | Hierarchical 95% CI | Unclustered Wilcoxon | Fold sign-flip | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Raw 1D-CNN vs GRU-only | -0.068 | [-0.115, -0.020] | [-0.129, +0.002] | p = 0.006 | p = 0.250 | No stable direction; changed from unclustered test |
| Raw 1D-CNN vs CWT 2D-CNN | +0.002 | [-0.070, +0.074] | [-0.093, +0.135] | p = 0.596 | p = 1.000 | No stable direction |
| Raw 1D-CNN vs CWT-GRU without SE | +0.017 | [-0.063, +0.098] | [-0.096, +0.160] | p = 0.898 | p = 0.875 | No stable direction |
| Raw 1D-CNN vs CWT-GRU | -0.010 | [-0.087, +0.068] | [-0.140, +0.131] | p = 0.756 | p = 1.000 | No stable direction |
| Raw 1D-CNN vs DRSN adaptation | -0.141 | [-0.195, -0.086] | [-0.207, -0.067] | p < 0.001 | p = 0.125 | Lower mean; interval below zero |
| Raw 1D-CNN vs TFN adaptation | -0.114 | [-0.178, -0.051] | [-0.185, -0.016] | p = 0.002 | p = 0.250 | Lower mean; interval below zero |
| CWT 2D-CNN vs CWT-GRU | -0.012 | [-0.053, +0.030] | [-0.071, +0.043] | p = 0.622 | p = 0.875 | No stable direction |
| CWT-GRU without SE vs CWT-GRU | -0.027 | [-0.068, +0.014] | [-0.073, +0.022] | p = 0.189 | p = 0.500 | No stable direction |
| 50% balanced budget: CWT-GRU vs CWT-GRU self-training | +0.039 | [-0.011, +0.089] | [-0.029, +0.112] | p = 0.199 | p = 0.500 | No stable direction |
| Full balanced budget: CWT-GRU vs CWT-GRU self-training | +0.010 | [-0.024, +0.044] | [-0.031, +0.049] | p = 0.475 | p = 0.750 | No stable direction |
| GRU-only vs CWT-GRU | +0.058 | [+0.007, +0.109] | [-0.036, +0.153] | p = 0.044 | p = 0.375 | No stable direction; changed from unclustered test |
| CWT 2D-CNN vs DRSN adaptation | -0.143 | [-0.207, -0.078] | [-0.251, -0.025] | p < 0.001 | p = 0.250 | Lower mean; interval below zero |
| CWT 2D-CNN vs TFN adaptation | -0.116 | [-0.172, -0.060] | [-0.185, -0.053] | p < 0.001 | p = 0.125 | Lower mean; interval below zero |
| CWT-GRU vs DRSN adaptation | -0.131 | [-0.211, -0.051] | [-0.258, +0.034] | p = 0.004 | p = 0.250 | No stable direction; changed from unclustered test |
| CWT-GRU vs TFN adaptation | -0.105 | [-0.160, -0.050] | [-0.189, -0.011] | p < 0.001 | p = 0.250 | Lower mean; interval below zero |
| CWT-GRU without SE vs DRSN adaptation | -0.158 | [-0.231, -0.086] | [-0.280, -0.022] | p < 0.001 | p = 0.250 | Lower mean; interval below zero |
| CWT-GRU without SE vs TFN adaptation | -0.132 | [-0.184, -0.079] | [-0.202, -0.058] | p < 0.001 | p = 0.125 | Lower mean; interval below zero |
| Raw 1D-CNN vs CWT-GRU with temperature scaling | -0.010 | [-0.087, +0.068] | [-0.139, +0.132] | p = 0.756 | p = 1.000 | No stable direction |
| Raw 1D-CNN vs CWT-GRU self-training | +0.001 | [-0.078, +0.079] | [-0.134, +0.128] | p = 0.784 | p = 1.000 | No stable direction |

## Conclusions that changed

- Raw 1D-CNN vs GRU-only: the unclustered Wilcoxon result was p = 0.006, whereas the hierarchical interval was [-0.129, +0.002] and the fold sign-flip result was p = 0.250. The manuscript uses the conservative conclusion.
- GRU-only vs CWT-GRU: the unclustered Wilcoxon result was p = 0.044, whereas the hierarchical interval was [-0.036, +0.153] and the fold sign-flip result was p = 0.375. The manuscript uses the conservative conclusion.
- CWT-GRU vs DRSN adaptation: the unclustered Wilcoxon result was p = 0.004, whereas the hierarchical interval was [-0.258, +0.034] and the fold sign-flip result was p = 0.250. The manuscript uses the conservative conclusion.

## Conclusions that remained unchanged

- Raw 1D-CNN vs CWT 2D-CNN: hierarchical interval [-0.093, +0.135]; no stable direction.
- Raw 1D-CNN vs CWT-GRU without SE: hierarchical interval [-0.096, +0.160]; no stable direction.
- Raw 1D-CNN vs CWT-GRU: hierarchical interval [-0.140, +0.131]; no stable direction.
- Raw 1D-CNN vs DRSN adaptation: hierarchical interval [-0.207, -0.067]; lower mean; interval below zero.
- Raw 1D-CNN vs TFN adaptation: hierarchical interval [-0.185, -0.016]; lower mean; interval below zero.
- CWT 2D-CNN vs CWT-GRU: hierarchical interval [-0.071, +0.043]; no stable direction.
- CWT-GRU without SE vs CWT-GRU: hierarchical interval [-0.073, +0.022]; no stable direction.
- 50% balanced budget: CWT-GRU vs CWT-GRU self-training: hierarchical interval [-0.029, +0.112]; no stable direction.
- Full balanced budget: CWT-GRU vs CWT-GRU self-training: hierarchical interval [-0.031, +0.049]; no stable direction.
- CWT 2D-CNN vs DRSN adaptation: hierarchical interval [-0.251, -0.025]; lower mean; interval below zero.
- CWT 2D-CNN vs TFN adaptation: hierarchical interval [-0.185, -0.053]; lower mean; interval below zero.
- CWT-GRU vs TFN adaptation: hierarchical interval [-0.189, -0.011]; lower mean; interval below zero.
- CWT-GRU without SE vs DRSN adaptation: hierarchical interval [-0.280, -0.022]; lower mean; interval below zero.
- CWT-GRU without SE vs TFN adaptation: hierarchical interval [-0.202, -0.058]; lower mean; interval below zero.
- Raw 1D-CNN vs CWT-GRU with temperature scaling: hierarchical interval [-0.139, +0.132]; no stable direction.
- Raw 1D-CNN vs CWT-GRU self-training: hierarchical interval [-0.134, +0.128]; no stable direction.

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
