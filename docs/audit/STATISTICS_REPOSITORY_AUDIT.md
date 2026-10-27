# Statistics Repository Audit

## Verdict

**PASS.**

The 20 main evaluations are four operating-condition folds crossed with five
stochastic seeds. The paper retains all 20 mean +/- SD values as descriptive
stochastic-stability summaries, while paired inference resamples fold first and
seed second.

## Primary analysis

- Script: `scripts/hierarchical_statistics.py`.
- Module: `src/spindle_cnn/analysis/hierarchical.py`.
- Output: `results/hierarchical_statistics.csv`.
- Bootstrap: 20,000 paired, two-stage cluster-bootstrap samples.
- Fold sensitivity test: exact sign-flip over four fold-level mean differences.
- Minimum attainable nonzero two-sided sign-flip p-value: 0.125.
- Unclustered t intervals and Wilcoxon tests: retained as sensitivity analyses.

All required comparisons contain four folds, five seeds, and 20 matched pairs:

```text
external:raw_cnn:drsn
external:raw_cnn:tfnet
label_50:cwt_gru:cwt_gru_selftrain
label_full:cwt_gru:cwt_gru_selftrain
main:cwt_cnn:cwt_gru
main:cwt_gru_no_se:cwt_gru
main:raw_cnn:cwt_cnn
main:raw_cnn:cwt_gru
main:raw_cnn:cwt_gru_no_se
main:raw_cnn:gru_only
```

## Conclusions changed by fold-aware analysis

- `main:raw_cnn:gru_only`
- `main:gru_only:cwt_gru`
- `external:cwt_gru:drsn`

No result is described as equivalent merely because an interval includes zero.
No result is described as significant solely from the unclustered 20-cell test.

## Audit issues

- None.
