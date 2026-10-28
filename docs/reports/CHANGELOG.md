# Scientific Manuscript Revision Log

Date: 2026-10-28

## Final DOCX-only repository cleanup

- Archived root-level reports under `docs/reports/`.
- Retained the final manuscript as DOCX and removed all tracked PDF figures,
  manuscript exports, supplementary exports, and PDF assembly tooling.
- Kept the publication figures as 24 tracked PNG files with matching hashes.
- Verified README links, commands, paper numbers, figure hashes, DOCX citations,
  and cross-references.

## Final submission packaging

- Added one centralized 11-color publication palette and fixed model-color
  mapping in `src/spindle_cnn/plotting/style.py`; all 12 plotting modules now
  use the same typography, grid, line, error-bar, and white-background style.
- Regenerated all 24 PNG/PDF figure pairs without changing experiment data and
  recorded their byte sizes and SHA-256 values in the tracked
  `figures/publication/figure_hashes.json` manifest.
- Added automated palette and evidence-chain checks in
  `scripts/check_figure_palette.py` and
  `scripts/check_manuscript_consistency.py`.
- Retained compact feature embeddings for deterministic figure-only PCA/t-SNE
  reproduction without retraining.
- Rendered the 30-page main manuscript and assembled a five-page standalone
  supplementary PDF containing Figures S1-S5.
- Rebuilt the final DOCX from the canonical result files, inspected all 30
  rendered pages, and retained zero numerical-table mismatches and zero
  accessibility findings.

## Cross-domain evidence revision

- Restored and checksum-verified all 16 selected Paderborn archives, rebuilt
  3,939 original-amplitude windows with exact measurement filenames, and
  reproduced the canonical raw cache after per-window standardization.
- Added a full protocol audit covering half-open window ranges, bearing and
  condition isolation, manifest hashes, label mappings, source files, and
  damage origin.
- Preserved all 320 canonical and 40 external rows without changing the split,
  selecting favorable seeds, or dropping failed evaluations.
- Added raw time- and frequency-domain diagnostics, post-hoc label-granularity
  sensitivity, CWT representation controls, training-statistic normalization,
  coverage-controlled self-training, baseline learning-rate auditing,
  12-versus-30-epoch checks, per-fold calibration, and feature-space analysis.
- Clarified that training time excludes offline CWT generation and added a
  measured CWT preprocessing cost audit and GRU FLOP estimation note.
- Retitled the manuscript to emphasize cross-domain benchmarking and expanded
  the abstract, contributions, sensitivity results, discussion, and limitations.
- Added independent result traceability, a single-source `paper_metrics.json`,
  reproducibility reports, and a second-dataset compatibility assessment.

## [Unreleased] - v1.0.0 candidate

- Initial paper-release candidate; not yet tagged or published.
- Leakage-safe Paderborn benchmark with simultaneous unseen-bearing and
  unseen-operating-condition evaluation.
- 320 canonical evaluations: 140 main, 80 label-efficiency, and 100
  threshold-sensitivity runs.
- 40 same-protocol external-baseline evaluations: 20 DRSN adaptation runs and
  20 TFN adaptation runs.
- Component-wise ablation, calibration analysis, pseudo-label analysis, and
  validation-to-test domain-gap analysis.
- Descriptive fold-seed summaries, unclustered sensitivity tests, and
  fold-aware hierarchical statistics.
- Reproducible data preparation, split manifests, result rechecks, figures,
  manuscript tables, and release-audit scripts.
- Added `CITATION.cff`, root-level reproduction instructions, provenance
  matrices, split and leakage audits, release readiness, and Zenodo preparation.
- No GitHub Release or Zenodo DOI has been created for this candidate.

## Pre-submission statistical robustness revision

- Preserved the existing leakage-safe split, all 320 canonical rows, and all 40
  external-baseline rows.
- Added `results/hierarchical_statistics.csv` with paired two-stage cluster
  bootstrap intervals that resample operating-condition folds before seeds.
- Added exact four-fold sign-flip tests as a coarse sensitivity analysis and
  retained the unclustered t intervals and Wilcoxon tests as secondary results.
- Downgraded Raw 1D-CNN versus GRU-only and GRU-only versus CWT-GRU after their
  fold-aware intervals crossed zero.
- Updated Table 6, Sections 3.8, 4.1-4.4, 5, 6, 8, and 9 to distinguish
  descriptive variability from domain-level inference.
- Standardized the displayed model name as TFN while retaining the historical
  result key `tfnet`.
- Standardized p-value formatting and the `10% balanced budget`,
  `50% balanced budget`, and `Full balanced budget` terminology.
- Clarified the boundary from prior leakage-aware and few-shot target-support
  studies and completed reference [22] with its DOI.
- Added `docs/audit/STATISTICAL_SENSITIVITY_REPORT.md`,
  `docs/audit/FINAL_STATISTICAL_AUDIT.md`, and
  `docs/audit/FINAL_MANUSCRIPT_AUDIT.md`.

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
  MTFC-T and DRSN-Transformer remain excluded because a fair retraining
  implementation was unavailable.

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
  mean, but the fold-aware interval includes zero.
- Removed the claim that temporal fusion provides a stable gain; the fold-aware
  CWT-GRU intervals include zero against CWT 2D-CNN and Raw 1D-CNN.
- Removed the claim that self-training improves limited-label performance; the
  10% balanced budget produces no update, and the 50% balanced budget and
  Full balanced budget intervals include zero.
- Removed the claim that calibration repairs classification. Temperature scaling
  preserves predicted classes and changes probability quality only.

## Generated deliverables

- Full manuscript:
  `docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx`
- Table provenance:
  `docs/paper/TABLE_SOURCES.md`
- Figure provenance:
  `docs/paper/FIGURE_SOURCES.md`
- Reproduction guide: `docs/paper/PAPER_REPRODUCIBILITY.md`
- Manuscript builder: `src/spindle_cnn/paper/build_docx.py`
- Method-figure generator: `src/spindle_cnn/experiments/make_method_figures.py`
- Result-figure generator: `src/spindle_cnn/experiments/make_figures.py`

All manuscript result values are loaded from the canonical CSV files at build
time. The manuscript is not edited by copying numbers from a prior draft.

## Publication checks

- The complete build contains 21 pages, 13 tables, and 12 figures.
- Python compilation passed.
- The repository test suite passed: 14 tests.
- The final DOCX was rendered to 20 page images, and every page was inspected at
  original resolution for clipping, overlap, broken tables, split figure
  captions, malformed equations, and isolated trailing pages.

## Pre-submission reproducibility audit

- Independently recomputed run counts, means, standard deviations, confidence
  intervals, paired differences, and Wilcoxon tests from the 320 canonical run
  rows.
- Reconciled all 13 manuscript tables and key numerical text statements with
  the result files; the mismatch count is zero.
- Reproduced all 11 manuscript PNG figures byte-for-byte from current code.
- Confirmed 320 successful, unique runs and the complete
  4-fold x 5-seed matrix for every main condition.
- Added a code-level leakage audit covering window overlap, bearing and
  condition isolation, normalization, CWT construction, early stopping,
  calibration, pseudo-label selection, and threshold selection.
- Recorded a study-level warning about repeated use of the same test folds and
  outstanding public-release privacy actions in
  `docs/audit/PRE_SUBMISSION_AUDIT.md`.

## External baselines and domain gap

- Added a separate same-protocol DRSN implementation and a TFN adaptation
  derived from the public comparison-model code.
- Retrained each external model for four folds and five seeds with a clean
  worktree; all 40 runs succeeded.
- Added paired comparisons, confidence intervals, parameter counts, FLOPs,
  training time, and inference latency for the external models.
- Added an automatically generated validation-to-test gap table and figure for
  the four principal internal models.
- Refined the three Introduction contributions to emphasize the comparison
  design, same-protocol external baselines, and transparent retention of
  negative results.
- Replaced ambiguous percentage-only label references with `10% balanced budget`,
  `50% balanced budget`, and `Full balanced budget`.
