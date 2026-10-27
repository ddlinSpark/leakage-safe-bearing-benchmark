# Scientific Manuscript Revision Log

Date: 2026-10-27

## Revised manuscript

- Restructured the paper around six explicit research questions and one
  leakage-safe cross-domain protocol.
- Replaced the spindle-specific framing with the evidence-supported title
  "Leakage-Safe Benchmarking of Time-Frequency Fusion and Calibrated
  Self-Training for Bearing Fault Diagnosis."
- Reorganized the introduction and related work by method family rather than by
  paper chronology. Each research strand now states what prior work addressed,
  its residual limitation, and the relation to this study.
- Added an overall methodology framework, a CWT-GRU architecture figure, and a
  calibrated self-training pipeline figure.
- Replaced the former narrative model description with implementation details
  verified against the project code: three-channel inputs, 48 Morlet scales,
  96 x 96 magnitude maps, a 192-dimensional image feature, a two-layer
  bidirectional GRU with 64 hidden units per direction, 128-dimensional recurrent
  features, a 320-dimensional fusion input, and validation-only temperature
  scaling.
- Corrected the augmentation description. Circular shifts are applied to the raw
  waveform and the corresponding CWT field; additive noise is applied with 50%
  probability to both inputs.
- Added mean +/- standard deviation, 95% confidence intervals, paired Wilcoxon
  tests, and 20 matched fold-seed evaluations for each main method condition.
- Added a controlled model table that contains only models trained in this
  project under the same dataset, split, windowing, preprocessing, and metric.
- Expanded the ablation analysis to include CWT, GRU, SE, calibration, and
  self-training, including the unfavorable SE and recurrent-fusion results.
- Added class confusion, pseudo-label coverage and accuracy, threshold
  sensitivity, ECE, NLL, Brier score, reliability diagrams, parameter counts,
  FLOPs, training latency, and inference latency.
- Rewrote the discussion and conclusion to distinguish classification
  performance from confidence reliability and to avoid claims of significant
  improvement where the paired tests do not support one.
- Added explicit limitations: the Paderborn test rig is not a complete
  machine-tool spindle assembly, only four held-out bearing codes are used, and
  external SOTA models remain unreproduced.

## Canonical evidence

- Canonical run table: `results/raw_runs.csv`.
- Successful runs: 320.
- Main comparison: 140 runs, 20 per method condition.
- Label efficiency: 80 runs.
- Threshold sensitivity: 100 runs.
- Seeds: 42, 123, 2024, 2025, 3407.
- Folds: 0, 1, 2, 3.
- Window length and stride: 8192 samples.
- Main pseudo-label threshold: 0.60.
- Main self-training limit: two rounds.
- Threshold experiment: one selection-and-retraining round to isolate threshold
  effects.

## Corrected or excluded claims

- Removed claims based on engineering-threshold labels from the former industrial
  pipeline.
- Removed the 99% headline accuracy and any implication that it was comparable
  with the present cross-domain benchmark.
- Removed direct ranking against published numbers obtained under different
  datasets, splits, preprocessing, or evaluation rules.
- Removed the claim that SE improves robustness; no-SE has the higher observed
  mean, and the paired difference is not significant.
- Removed the claim that temporal fusion provides a stable gain; CWT-GRU is not
  significantly better than CWT 2D-CNN or Raw 1D-CNN.
- Removed the claim that self-training significantly improves limited-label
  performance; the 10% balanced budget produces no update, and the 50% balanced
  and full balanced-budget differences are not significant.
- Removed the claim that calibration repairs classification. Temperature scaling
  preserves predicted classes and changes probability quality only.

## Generated deliverables

- Full manuscript:
  `docs/paper/Leakage_Safe_Bearing_Benchmark_SCI.docx`
- Table provenance:
  `docs/paper/TABLE_SOURCES.md`
- Figure provenance:
  `docs/paper/FIGURE_SOURCES.md`
- Reproduction guide: `PAPER_REPRODUCIBILITY.md`
- Manuscript builder: `src/spindle_cnn/paper/build_docx.py`
- Method-figure generator: `src/spindle_cnn/experiments/make_method_figures.py`
- Result-figure generator: `src/spindle_cnn/experiments/make_figures.py`

All manuscript result values are loaded from the canonical CSV files at build
time. The manuscript is not edited by copying numbers from a prior draft.

## Publication checks

- The complete build contains 20 pages, 12 tables, and 11 figures.
- Python compilation passed.
- The repository test suite passed: 14 tests.
- The final DOCX was rendered to 20 page images, and every page was inspected at
  original resolution for clipping, overlap, broken tables, split figure
  captions, malformed equations, and isolated trailing pages.

## Pre-submission reproducibility audit

- Independently recomputed run counts, means, standard deviations, confidence
  intervals, paired differences, and Wilcoxon tests from the 320 canonical run
  rows.
- Reconciled all 12 manuscript tables and 30 key numerical text statements with
  the result files; the mismatch count is zero.
- Reproduced all 11 manuscript PNG figures byte-for-byte from current code.
- Confirmed 320 successful, unique runs and the complete
  4-fold x 5-seed matrix for every main condition.
- Added a code-level leakage audit covering window overlap, bearing and
  condition isolation, normalization, CWT construction, early stopping,
  calibration, pseudo-label selection, and threshold selection.
- Recorded a study-level warning about repeated use of the same test folds and
  outstanding public-release privacy actions in `PRE_SUBMISSION_AUDIT.md`.
