# Experiment Completeness

Canonical result baseline: `8ac2d75`  
Raw-result SHA-256: `a047b0f9530f07643ef2c5c57f395a15abe045c7e8640a10119c273e46fbeb10`

## Overall status

**PASS**

The canonical matrix contains 320 successful rows. No duplicate run ID,
duplicate experiment key, failed row, missing required metric, missing seed, or
missing fold was found.

## Declared and observed matrix

| Experiment | Declared design | Observed rows | Status |
| --- | ---: | ---: | --- |
| Main | 7 methods x 4 folds x 5 seeds = 140 | 140 | PASS |
| Label efficiency | 2 methods x 2 saved partial budgets x 4 folds x 5 seeds = 80 | 80 | PASS |
| Threshold sensitivity | 5 thresholds x 4 folds x 5 seeds = 100 | 100 | PASS |
| Total | 320 | 320 | PASS |

The external same-protocol DRSN and TFNet matrix is maintained separately under
`results/sota/` and is audited after its retraining completes. It is not mixed
into the 320-row canonical table.

The 100% label-efficiency view reuses the main CWT-GRU and calibrated
self-training rows. It is not counted as another independently trained block.

## Main method groups

Every method contains exactly 20 unique fold-seed evaluations over folds
`0,1,2,3` and seeds `42,123,2024,2025,3407`.

| Method | Runs | Missing combinations |
| --- | ---: | ---: |
| `raw_cnn` | 20 | 0 |
| `gru_only` | 20 | 0 |
| `cwt_cnn` | 20 | 0 |
| `cwt_gru_no_se` | 20 | 0 |
| `cwt_gru` | 20 | 0 |
| `cwt_gru_temp` | 20 | 0 |
| `cwt_gru_selftrain` | 20 | 0 |

`cwt_gru_temp` is a temperature-scaled evaluation of the matched CWT-GRU
weights. Its saved logits are identical to the corresponding CWT-GRU logits
(maximum absolute difference `0.0`); this is expected because temperature
scaling preserves the argmax.

## Label-efficiency groups

| Method | Budget | Runs | Status |
| --- | ---: | ---: | --- |
| CWT-GRU | 10% | 20 | PASS |
| CWT-GRU + calibrated self-training | 10% | 20 | PASS |
| CWT-GRU | 50% | 20 | PASS |
| CWT-GRU + calibrated self-training | 50% | 20 | PASS |

All 20 self-training runs at the 10% budget produced zero pseudo-labels and
returned the initial model.

## Threshold sensitivity

Each of thresholds `0.3,0.4,0.5,0.6,0.7` has 20 runs at the 50% balanced-label
budget. Two of the 20 fold-seed cases produced no pseudo-label at any tested
threshold, so their pseudo-label accuracy fields are correctly recorded as
missing.

The historical threshold run configurations retain the global
`training.pseudo_rounds=2` setting even though the threshold runner performs one
selection-and-retraining update. The effective execution is confirmed by
`self_training_rounds_completed=1` in every threshold row that selected
pseudo-labels. Future configurations now record
`run.effective_self_training_rounds` explicitly.

## Missing-value review

The only missing values are structurally expected:

- `failure_reason`: 320 rows, because all runs succeeded.
- `threshold`: 160 rows, because these are supervised or non-self-training
  conditions.
- `pseudo_accuracy` and `pseudo_macro_f1`: 192 rows each. This comprises
  non-self-training conditions, the 20 zero-update 10% runs, two zero-update
  50% runs, and ten threshold rows from two fold-seed cases.

No required model, split, metric, timing, parameter, or provenance field is
missing.

## Integrity checks

| Check | Result |
| --- | --- |
| Rows | 320 |
| Unique `run_id` | 320 |
| Duplicate `run_id` | 0 |
| Duplicate experiment keys | 0 |
| Failed runs | 0 |
| Rows recorded from dirty worktrees | 0 |
| Dataset hash values | 1 unique hash |
| Split hashes per fold | 1 unique hash per fold |
| Independent metric rechecks | 320/320 passed |
| Maximum metric recheck difference | `3.55e-15` |

The recheck reads saved logits and independently recomputes Macro-F1, accuracy,
ECE, NLL, and Brier score against every recorded run.

## History and anomalous-row review

The canonical `results/raw_runs.csv` grew monotonically in its recorded
commits:

| Commit | Rows | Non-success rows | Rows removed from previous version |
| --- | ---: | ---: | ---: |
| `a146cb5` | 140 | 0 | 0 |
| `c9b9912` | 220 | 0 | 0 |
| `fcf350f` | 320 | 0 | 0 |

No canonical row present in an earlier recorded version disappeared in a later
version. The former three-seed and industrial experiments are separate
superseded protocols; they are not mixed into the current canonical table.

## Code provenance

The rows record three clean training commits:

| Recorded commit | Contributing rows |
| --- | ---: |
| `470df63` | 140 main evaluations |
| `a146cb5` | 80 label-efficiency evaluations |
| `c9b9912` | 100 threshold evaluations |

Comparison with the current code shows no change to model, split, training,
metric, or calibration computation after those commits. The only later
`run_benchmark.py` changes add audit columns and make the derived 100% `run_id`
unique. The obsolete industrial and early public-benchmark modules were removed.

## Figure reproducibility

Running both figure generators from current `main` into a separate directory
reproduced the existing manuscript PNG files byte-for-byte. The added
validation-to-test gap figure is generated from the same raw table.

## External baseline matrix

| Model | Expected keys | Observed successful runs | Status |
| --- | --- | ---: | --- |
| DRSN adaptation | 4 folds x 5 seeds | 20 | PASS |
| TFNet adaptation | 4 folds x 5 seeds | 20 | PASS |
| Total | 40 | 40 | PASS |

Every external run uses the same dataset hash and the same split hash for its
fold. All 40 runs record clean-worktree provenance at commit `673bb99`.
Independent metric recomputation passed for 40/40 runs, with maximum absolute
difference `1.78e-15`.

## Reproduce this check

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_submission.py --json-output artifacts/local/pre_submission_audit.json
python -m pytest -q tests
```
