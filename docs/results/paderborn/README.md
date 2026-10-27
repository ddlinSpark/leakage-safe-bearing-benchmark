# Paderborn Repeated Experiment Results

Files:

- `results.csv`: row-level outcomes for 64 matched or budget-specific runs.
- `summary_with_ci.csv`: mean, standard deviation, and 95% confidence interval.
- `paired_comparisons.csv`: Wilcoxon signed-rank comparisons on matched folds and seeds.
- `cache_metadata.json`: identifiers, class groups, condition mapping, window settings,
  and final tensor shapes.

The full-label comparison uses four held-out operating conditions and three random
seeds per method (12 runs per method). The 10% and 50% label-budget comparisons use
four held-out conditions with seed 42.

The test-bearing identifiers were 4, 8, 12, and 15, corresponding to K005, KA15,
KI14, and KB27. Training used K001-K004, KA01/KA04/KA08, KI01/KI04/KI08, and
KB23/KB24. No test bearing or test condition appears in training or validation.
