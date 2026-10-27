# Final Leakage Check

## Overall verdict

**PASS at run level. WARNING at study level.**

The run-level protocol shows no split, pseudo-label, calibration, normalization,
early-stopping, or evaluation leakage. The fixed test folds were reused during
project development, so unrecorded human decisions cannot be ruled out without
an untouched external confirmation set.

| Check | Status | Evidence |
| --- | --- | --- |
| Window overlap | PASS | Cache metadata records `window_size = stride = 8192`; tests and manifests use non-overlapping starts. |
| Bearing identity leakage | PASS | Held-out record IDs 4, 8, 12, and 15 are absent from train and validation in every manifest. |
| Operating-condition leakage | PASS | Test and validation conditions are excluded from `train_conditions`; validation and test conditions differ. |
| Train/validation contamination | PASS | Stored index sets are pairwise disjoint. |
| Test-index leakage | PASS | Test indices are disjoint from fitting indices and are evaluated after fitting. |
| Pseudo-label candidate leakage | PASS | `run_benchmark.py` forms candidates from training indices minus labeled indices, never from validation or test. |
| Test-label pseudo-label leakage | PASS | Candidate indices are restricted to the training partition. |
| Temperature-scaling leakage | PASS | `training/runner.py` fits temperature from validation logits and labels only. |
| Threshold-selection leakage | PASS | The main threshold remains fixed at 0.60; the sensitivity sweep evaluates fixed thresholds and does not use test performance to alter the main model. |
| Normalization leakage | PASS | Standardization is performed independently inside each window; no global train/test statistic is shared. |
| CWT preprocessing leakage | PASS | CWT and per-channel magnitude normalization are local to each window. |
| Early stopping uses test | PASS | Early stopping reads validation Macro-F1 only. |
| Test-set hyperparameter tuning | WARNING at study level | No test-driven search exists in code, but the same fixed test folds were reused across development. |

## Test labels

Test labels are read only for final metric evaluation. The threshold experiment
reports offline pseudo-label accuracy against labels in the training partition,
not against test labels.
