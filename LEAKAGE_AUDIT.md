# Leakage Audit

Canonical result baseline: `8ac2d75`

## Overall status

**PASS with one study-level warning**

The implemented training and evaluation pipeline does not use held-out bearing
identities, test conditions, test labels, or test statistics for fitting,
normalization, early stopping, calibration, pseudo-label selection, or threshold
selection.

The warning concerns study-level reuse of the same four test folds throughout
the project. The repository proves that the final run code never consults test
data during fitting, but it cannot prove that no human design choice was
influenced by previously observed test summaries.

## Item-by-item audit

| Check | Status | Code and data evidence | Conclusion |
| --- | --- | --- | --- |
| Window overlap | PASS | `public_benchmark/pu_data.py` uses `range(..., stride)`; metadata records window and stride `8192/8192`. Cached starts are all on that stride; minimum distinct-start spacing within each record/condition is 8192. | Windows do not share samples. |
| Bearing identity leakage | PASS | `datasets/splits.py::load_fold_indices` masks held-out record IDs from training and validation. Each split manifest has no index overlap; test records are exactly `4,8,12,15`; train and validation contain none of them. | Unseen-bearing claim is supported. |
| Operating-condition leakage | PASS | `load_fold_indices` assigns two conditions to train, one distinct condition to validation, and one distinct condition to test. Every manifest confirms no condition overlap. | Unseen-condition claim is supported. |
| Train/validation overlap | PASS | `validate_split_isolation` rejects shared indices and conditions. Manifest intersection is empty for all four folds. | No overlap. |
| Train/test overlap | PASS | Same function rejects shared indices and held-out record IDs. Manifest intersection is empty for all four folds. | No overlap. |
| Validation/test overlap | PASS | Same function rejects shared indices and identical conditions. Manifest intersection is empty for all four folds. | No overlap. |
| Test label to pseudo-label leakage | PASS | `run_benchmark.py` creates candidates as `train indices - labeled indices`. `run_self_training` and `select_pseudo_labels` accept only those training candidates. Test labels are first read by `evaluate_model` after fitting. | Test labels do not enter pseudo-label creation or retraining. |
| Pseudo-label audit labels | PASS | True labels are used only to calculate offline `pseudo_accuracy` and `pseudo_macro_f1` after selection. Those values are not used to choose samples or update weights. | Audit does not feed back into training. |
| Early stopping | PASS | `training/runner.py::train_model` computes early-stopping Macro-F1 on `val_indices`; test indices are absent from the function. | No test-based early stopping. |
| Temperature scaling | PASS | `train_model` calls `fit_temperature` with validation logits and labels. Test logits are scaled only during final evaluation. | No test-fitted calibration. |
| Test-set hyperparameter tuning | PASS at run level / WARNING at study level | The pipeline contains no test-driven search. `configs/paderborn_main.json` is unchanged from protocol commit `a390b8b`; all predeclared models are reported. The same fixed test folds were nevertheless reused across development iterations, so unrecorded human decision influence cannot be excluded. | Expand or refresh untouched confirmation data before making a strong first-seen-test claim. |
| Pseudo-label threshold selection | PASS at run level | Main threshold `0.6` was present in the protocol configuration before main results were recorded. The five-threshold sweep was recorded later as a sensitivity analysis, not used to replace the main threshold. | No code path selects the threshold on test labels. |
| Window normalization | PASS | `public_benchmark/cwru_data.py::_standardize_window` computes mean and standard deviation independently for each channel of each window. | No shared train/test normalization statistics. |
| CWT preprocessing | PASS | `public_benchmark/pu_data.py::_cwt_image` computes CWT magnitude, resizes, and per-channel maximum normalization from one window only. | No cross-partition CWT statistic. |
| Validation-to-training leakage | PASS | Training uses `split.train`; validation uses disjoint `split.validation`. Pseudo-label candidates are a subset of train only. | Validation labels affect stopping and temperature only. |
| Test-informed threshold update | PASS at run level | Threshold comparison uses the same folds and seeds after the main configuration was fixed. Main and threshold configs are retained separately. | Sensitivity results are not substituted into the main result. |
| Label-budget selection | PASS | `balanced_indices` samples equal labels per class from `split.train`. Seed/fold/ration combinations are complete. | Label budgets do not draw from validation or test. |

## Split evidence

| Fold | Train conditions | Validation | Test | Train / validation / test windows | Index overlaps |
| ---: | --- | --- | --- | --- | --- |
| 0 | C2,C3 | C1 | C0 | 1488 / 714 / 248 | 0 |
| 1 | C0,C3 | C2 | C1 | 1489 / 744 / 248 | 0 |
| 2 | C0,C1 | C3 | C2 | 1459 / 744 / 248 | 0 |
| 3 | C1,C2 | C0 | C3 | 1458 / 745 / 248 | 0 |

All folds use held-out bearing records `4,8,12,15`. Training and validation
contain only records `0,1,2,3,5,6,7,9,10,11,13,14`.

## Dataset construction evidence

- Cached shape: `3939 x 3 x 8192` for raw windows and `3939 x 3 x 96 x 96` for
  CWT maps.
- 16 bearing records, four operating conditions, and four classes are present.
- Every cached window start is a multiple of the 8192-sample stride.
- The minimum spacing between distinct starts within a record-condition group
  is 8192 samples.

## Warning and claim boundary

This audit supports the statement that the final pipeline is leakage-safe. It
does not turn the benchmark into a preregistered or untouched final test study.
The manuscript should retain the limitation that the same four grouped folds
were used for repeated model-family comparisons and should avoid claims that
would require a permanently untouched external test set.
