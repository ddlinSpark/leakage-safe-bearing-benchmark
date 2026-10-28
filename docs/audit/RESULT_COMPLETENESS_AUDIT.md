# Result Completeness Audit

## Verdict

**PASS.**

- Canonical rows: 320
- Unique canonical run IDs: 320
- External rows: 40
- Unique external run IDs: 40
- Failed canonical rows: 0
- Failed external rows: 0
- Dirty-worktree canonical rows: 0
- Dirty-worktree external rows: 0
- Missing required per-run files: 0
- Canonical saved-logit rechecks passed: 320/320
- External saved-logit rechecks passed: 40/40

## Coverage

- Main: 7 methods x 4 folds x 5 seeds = 140 rows.
- Label efficiency: 2 methods x 2 saved partial budgets x 4 folds x 5 seeds =
  80 rows.
- Threshold sensitivity: 5 thresholds x 4 folds x 5 seeds = 100 rows.
- External: DRSN 20 rows and TFN 20 rows.
- Seeds: 42, 123, 2024, 2025, 3407.
- Folds: 0, 1, 2, 3.

## Integrity checks

- Duplicate canonical run IDs: none.
- Duplicate external run IDs: none.
- Duplicate experiment keys: none.
- Failed rows: none.
- Unexpected seeds or folds: none.
- Missing required metric columns: none.
- Missing core values: canonical {};
  external {}.
- Missing per-run predictions or diagnostic files: none.

## Hashes

- `results/raw_runs.csv`: `fc8336c7d44cba399c8b0a2356cc378d80a36c4e368769ff922a3e41f55ab5bf`
- `results/sota/raw_runs.csv`: `193bf7ac9a00d8980eb3c5ed807db6ac12a33dc1657f2a6d1eb0c95c58b7881e`

## Audit issues

- None.
