# External Baseline Retraining Report

## Scope

Two external model families were retrained under the existing leakage-safe
Paderborn protocol without changing any split, window, seed, metric, or test-set
selection rule.

| Item | Value |
| --- | --- |
| Dataset | Paderborn University bearing dataset |
| Inputs | Three raw channels: vibration and two phase currents |
| Window / stride | 8192 / 8192 samples |
| Bearing holdout | Records 4, 8, 12, 15 |
| Operating-condition folds | 0, 1, 2, 3 |
| Seeds | 42, 123, 2024, 2025, 3407 |
| Runs per model | 4 folds x 5 seeds = 20 |
| Label condition | Full balanced budget |
| Primary metric | Macro-F1 |
| Secondary metric | Accuracy |
| Training configuration | Canonical AdamW settings, 12 epochs maximum, patience 4 |

All 40 external runs succeeded. No seed or fold is missing, and the raw table
contains no duplicate `(method, fold, seed)` key. All runs record
`git_dirty=false`.

## Implementations

### DRSN

- Source concept: M. Zhao, S. Zhong, X. Fu, B. Tang, and M. Pecht, "Deep
  residual shrinkage networks for fault diagnosis," IEEE Transactions on
  Industrial Informatics, 2020.
- Code: `src/spindle_cnn/models/drsn.py`.
- Status: **Same-protocol adaptation**. No official training implementation was
  supplied with this project, so the model follows the published residual
  shrinkage concept rather than claiming a line-for-line reproduction.
- Adaptation: four residual shrinkage stages, adaptive channel-wise soft
  thresholds, three waveform channels, and four output classes.
- Measured parameters: 233,040.

### Time-Frequency Network (TFN)

- Source paper: Q. Chen, X. Dong, G. Tu, et al., "TFN: An interpretable neural
  network with time-frequency transform embedded for intelligent fault
  diagnosis," Mechanical Systems and Signal Processing, 2024.
- Public structural source:
  `https://github.com/Bingbingboopoo/MTFC_T/tree/main/comparison_models/TFNet`.
- Reference repository commit inspected: `b503e80`.
- Code: `src/spindle_cnn/models/tfnet.py`.
- Status: **Same-protocol adaptation** of the published comparison model. The
  internal result key remains `tfnet` so historical runs remain auditable; the
  display name is TFN.
- Adaptation: three input channels, four output classes, removal of
  import-time global seed side effects, and a deterministic equivalent of
  adaptive max pooling.
- Measured parameters: 184,900.

### Excluded external candidates

- **MTFC-T:** not retrained. The authors' linked repository contains only a
  placeholder for the proposed model at the inspected commit, so the exact
  implementation could not be recovered from the public source.
- **DRSN-Transformer:** not retrained. No official implementation was recorded
  in this project, and the supplied materials did not provide enough
  architecture and training detail for a defensible reconstruction.
- No accuracy from either original study was inserted to fill these gaps.

## Results

All values below are computed from `results/sota/raw_runs.csv`.

| Model | Macro-F1 | Descriptive t CI | Accuracy | Descriptive t CI | Parameters | GFLOPs/sample | Training time (s) | Inference (ms/batch) | Runs |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DRSN adaptation | 0.149 +/- 0.084 | [0.109, 0.188] | 0.275 +/- 0.079 | [0.238, 0.312] | 233,040 | 0.795 | 6.8 | 32.6 | 20 |
| TFN adaptation | 0.175 +/- 0.078 | [0.139, 0.211] | 0.257 +/- 0.066 | [0.226, 0.289] | 184,900 | 0.388 | 4.5 | 17.0 | 20 |

Training and inference times are medians from the same RTX 5080 environment as
the internal benchmark. Latency is implementation- and hardware-dependent. The
shown confidence intervals are descriptive one-sample t intervals across 20
fold-seed evaluations; they are retained as sensitivity summaries. Paired
inference below resamples operating-condition folds before seeds.

## Paired comparison with internal models

Positive deltas mean that the internal reference has higher Macro-F1 than the
external adaptation on the same fold and seed.

| External model | Reference | Internal minus external Delta | Hierarchical 95% CI | Fold sign-flip p | Unclustered Wilcoxon p | Pairs |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| DRSN | Raw 1D-CNN | +0.141 | [0.067, 0.207] | p = 0.125 | p < 0.001 | 20 |
| TFN | Raw 1D-CNN | +0.114 | [0.016, 0.185] | p = 0.250 | p = 0.002 | 20 |
| DRSN | CWT 2D-CNN | +0.143 | [0.025, 0.251] | p = 0.250 | p < 0.001 | 20 |
| TFN | CWT 2D-CNN | +0.116 | [0.053, 0.185] | p = 0.125 | p < 0.001 | 20 |
| DRSN | CWT-GRU | +0.131 | [-0.034, 0.258] | p = 0.250 | p = 0.004 | 20 |
| TFN | CWT-GRU | +0.105 | [0.011, 0.189] | p = 0.250 | p < 0.001 | 20 |
| DRSN | CWT-GRU without SE | +0.158 | [0.022, 0.280] | p = 0.250 | p < 0.001 | 20 |
| TFN | CWT-GRU without SE | +0.132 | [0.058, 0.202] | p = 0.125 | p < 0.001 | 20 |

The external adaptations had lower mean Macro-F1 than the internal baselines.
Seven of eight hierarchical intervals are below zero. However, there are only
four operating-condition clusters, so the exact fold sign-flip test cannot
produce a nonzero two-sided p-value below 0.125. The unclustered Wilcoxon result
for DRSN against CWT-GRU was significant, but the hierarchical interval
includes zero; the manuscript therefore does not describe that difference as
stable.

## Interpretation boundaries

- These results do not reproduce the original datasets, splits, model
  dimensionalities, or training recipes of DRSN and TFN.
- The comparison is valid only as an adaptation study under one common
  leakage-safe protocol.
- The low external scores do not show that the original methods are ineffective
  under their own evaluation settings.
- DRSN's aggregate confusion matrix contains no predictions for the
  multiple-damage class, which contributes to its low Macro-F1.
- No test result was used to select epochs, thresholds, architectures, or seeds.

## Validation and source files

- `results/sota/raw_runs.csv`: 40 source rows.
- `results/sota/summary.csv`: run-level summary emitted by the runner.
- `results/sota/summary_with_ci.csv`: independent mean, SD, and 95% CI summary.
- `results/sota/statistical_tests.csv`: paired tests against internal models.
- `results/sota/model_complexity.csv`: median parameter, FLOP, and latency data.
- `results/sota/same_protocol_comparison.csv`: table used for the paper.
- `results/hierarchical_statistics.csv`: fold-aware paired intervals and
  sign-flip sensitivity tests.
- `results/sota/rechecks/all_runs.json`: 40/40 independent metric rechecks,
  maximum absolute difference `1.78e-15`.
- `results/sota/runs/<run_id>/`: configuration, logits, confusion matrix,
  calibration bins, and training history for every run.

Reproduce the summary from saved runs:

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.analysis.sota --results-dir results --external-dir results/sota
python -m spindle_cnn.analysis.hierarchical --results-dir results --external-dir results/sota --output results/hierarchical_statistics.csv
```
