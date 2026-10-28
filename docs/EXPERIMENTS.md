# Experiments

## Status definition

"Completed" means successful execution under the frozen protocol and
reproducible recording. It does not imply a positive performance result.

> All reported experiments completed as designed; not all evaluated methods produced positive or statistically stable gains.

| Experiment group | Status | Main outcome |
| --- | --- | --- |
| Main benchmark | Completed | No stable advantage established for the evaluated CWT, recurrent, or SE variants |
| Calibration | Completed | Probability metrics improved for selected CWT models but argmax classification did not change |
| Self-training | Completed | No stable diagnostic improvement established |
| Threshold sensitivity | Completed | Higher pseudo-label accuracy did not guarantee higher held-out Macro-F1 |
| Label efficiency | Completed | Performance depended strongly on the verified-label budget |
| External baselines | Completed | Same-protocol adaptations did not outperform the waveform baseline |
| CWT sensitivity | Completed | Representation settings changed results but no universal CWT advantage emerged |
| Training opportunity | Completed | Additional epochs did not materially change the primary conclusion |
| Embedding diagnostics | Completed | Post-hoc descriptive diagnostics only |

## Canonical matrix

| Group | Runs | Source |
| --- | ---: | --- |
| Main methods | 140 | `results/raw_runs.csv` |
| Label efficiency | 80 | `results/raw_runs.csv` |
| Threshold sensitivity | 100 | `results/raw_runs.csv` |
| External adaptations | 40 | `results/sota/raw_runs.csv` |

The 460 additional sensitivity-training rows remain separate from the 320
canonical rows. The full experiment registry has 820 rows.

## Canonical commands

```powershell
python scripts/run_main_benchmark.py --experiment all
python scripts/run_external_baselines.py
python scripts/generate_statistics.py
python scripts/generate_figures.py --include-domain-shift
```

Individual groups:

```powershell
python scripts/run_label_efficiency.py
python scripts/run_threshold_sensitivity.py
python scripts/run_external_baselines.py
python scripts/run_preprocessing_sensitivity.py --kind all
python scripts/run_training_opportunity.py
```

Training commands write full provenance in each per-run `config.json`.

## External adaptations

DRSN and TFN are re-trained under the same windows, folds, seeds, labels,
optimizer, epoch limit, early-stopping rule, validation split, test split, and
metrics as the primary models.

These results are same-protocol adaptations, not official reproductions of the
original papers. Unavailable or incomplete implementations are not replaced by
invented results.

## Sensitivity analyses

- CWT scale count, image resolution, magnitude transform, normalization, and
  real/imaginary channels;
- window-wise versus training-statistic normalization;
- confidence thresholds and class-conditional coverage;
- coverage-controlled pseudo-labeling;
- 12 versus 30 training epochs;
- validation-only learning-rate alternatives;
- per-fold calibration;
- post-hoc label granularity and penultimate-feature visualization.

Detailed result files are catalogued in `results/README.md` and summarized in
`docs/RESULTS.md`.
