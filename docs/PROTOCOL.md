# Leakage-Safe Protocol

## Research question

Do commonly reported bearing-fault-diagnosis additions provide a stable
classification advantage when both the physical bearing and operating
condition are unseen?

The paper is a benchmark and evaluation study. It does not propose a new
high-scoring architecture.

## Fold definition

| Fold | Training conditions | Validation condition | Test condition |
| ---: | --- | --- | --- |
| 0 | C2, C3 | C1 | C0 |
| 1 | C0, C3 | C2 | C1 |
| 2 | C0, C1 | C3 | C2 |
| 3 | C1, C2 | C0 | C3 |

Condition names are defined in `docs/DATASET.md`.

For every fold:

- the test consists of unseen bearing codes in the test condition;
- validation uses seen bearing codes in a different, unseen-in-training
  condition;
- training uses the same seen bearing codes in the two remaining conditions;
- train, validation, and test signal ranges cannot overlap.

The exact memberships and hashes are in `results/splits/fold<k>.json`.

## Repetition design

- Folds: `0, 1, 2, 3`.
- Seeds: `42, 123, 2024, 2025, 3407`.
- Main matrix: four folds by five seeds per method.
- Canonical rows: 320.
- External rows: 40.

The same fold-seed key is preserved for every paired comparison.

## Leakage boundaries

The following are prohibited for held-out test data:

- gradient updates;
- validation membership;
- early stopping;
- temperature fitting;
- calibration fitting;
- pseudo-label generation;
- threshold selection;
- learning-rate or architecture selection;
- model or seed selection.

Validation data can be used for early stopping and temperature fitting.
Pseudo-labeled examples are selected only from the unlabeled portion of the
training partition.

The project has undergone many analyses, so the manuscript does not claim that
test outcomes were cognitively unknown to every design decision. The claim is
that test outcomes did not select the primary method, split, seed, threshold,
normalization, or CWT configuration.

## Balanced-label budget

The 10%, 50%, and Full balanced budgets contain the same number of verified
labels from each class. Full means the minimum available per-class training
count, not every training window.

The primary self-training threshold is `0.60`. Threshold sensitivity evaluates
`0.30, 0.40, 0.50, 0.60, 0.70` at the 50% balanced budget.

At 10% budget, no candidate satisfies the predefined confidence threshold.
The procedure therefore returns the initial supervised model; this is a valid
protocol outcome, not an execution failure.

## Statistical inference

Primary paired inference is a two-stage cluster bootstrap:

1. resample the four operating-condition folds;
2. resample the five seeds within each selected fold;
3. preserve the paired fold-seed key;
4. repeat 20,000 times.

The 20 fold-seed evaluations are not treated as 20 independent domains. The
exact four-fold sign-flip test has a minimum attainable non-zero two-sided
p-value of `0.125`. Conclusions prioritize paired effects, intervals, and
fold-wise consistency.

## Analysis status

The four-class task is primary. Label-granularity, feature-space, calibration,
training-budget, normalization, preprocessing, and self-training extensions are
sensitivity or post-hoc diagnostics. They do not replace the primary
configuration after test outcomes are observed.
