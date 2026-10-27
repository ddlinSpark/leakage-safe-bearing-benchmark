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

- `results/raw_runs.csv`: `a047b0f9530f07643ef2c5c57f395a15abe045c7e8640a10119c273e46fbeb10`
- `results/sota/raw_runs.csv`: `c92f2b4a447dbc13f581981d15d76e97ca3382ab024a905f4fbccd6ac1e5b09a`

## Audit issues

- None.
