# Manuscript Figure Sources

All final figures have editable vector PDF counterparts under
`figures/publication/`. The manuscript embeds the PNG files at publication
resolution.

Every plotting module calls `apply_paper_style()` from
`src/spindle_cnn/plotting/style.py`. The shared 11-color palette and model-color
mapping are verified by `scripts/check_figure_palette.py`. The current SHA-256
manifest is `figures/publication/figure_hashes.json`.

## Figure 1: Overall methodology framework

- Output: `figures/publication/figure_method_01_overall_framework.{png,pdf}`
- Generator: `python scripts/make_method_figures.py`
- Function: `overall_framework()` in
  `src/spindle_cnn/experiments/make_method_figures.py`
- Inputs: protocol in `configs/paderborn_main.json`; split construction in
  `src/spindle_cnn/datasets/splits.py`; self-training logic in
  `src/spindle_cnn/training/runner.py`.

## Figure 2: CWT-GRU architecture

- Output: `figures/publication/figure_method_02_cwt_gru_architecture.{png,pdf}`
- Generator: `python scripts/make_method_figures.py`
- Function: `cwt_gru_architecture()` in
  `src/spindle_cnn/experiments/make_method_figures.py`
- Inputs: `src/spindle_cnn/models/cwt_gru.py`,
  `src/spindle_cnn/models/layers.py`, and
  `src/spindle_cnn/public_benchmark/pu_data.py`.

## Figure 3: Calibrated self-training pipeline

- Output: `figures/publication/figure_method_03_self_training_pipeline.{png,pdf}`
- Generator: `python scripts/make_method_figures.py`
- Function: `self_training_pipeline()` in
  `src/spindle_cnn/experiments/make_method_figures.py`
- Inputs: `configs/paderborn_main.json`,
  `src/spindle_cnn/training/runner.py`, and
  `src/spindle_cnn/evaluation/calibration.py`.

## Figure 4: Full balanced budget model comparison

- Output: `figures/publication/figure_01_model_comparison.{png,pdf}`
- Generator: `python scripts/generate_figures.py`
- Function: `model_comparison()` in
  `src/spindle_cnn/experiments/make_figures.py`
- Input: `results/summary.csv`.

## Figure 5: Label efficiency

- Output: `figures/publication/figure_03_label_efficiency.{png,pdf}`
- Generator: `python scripts/generate_figures.py`
- Function: `label_efficiency()` in
  `src/spindle_cnn/experiments/make_figures.py`
- Input: `results/label_efficiency_summary.csv`.

## Figure 6: Threshold sensitivity

- Output: `figures/publication/figure_04_threshold_sensitivity.{png,pdf}`
- Generator: `python scripts/generate_figures.py`
- Function: `threshold_sensitivity()` in
  `src/spindle_cnn/experiments/make_figures.py`
- Input: `results/threshold_summary.csv`.

## Figure 7: Expected calibration error

- Output: `figures/publication/figure_05_calibration.{png,pdf}`
- Generator: `python scripts/generate_figures.py`
- Function: `calibration()` in
  `src/spindle_cnn/experiments/make_figures.py`
- Inputs: `results/raw_runs.csv`, columns `raw_ece` and `ece`.

## Figure 8: Reliability diagrams

- Output: `figures/publication/figure_08_reliability_diagrams.{png,pdf}`
- Generator: `python scripts/generate_figures.py`
- Function: `reliability_diagrams()` in
  `src/spindle_cnn/experiments/make_figures.py`
- Inputs: `results/raw_runs.csv` and
  `results/runs/<run_id>/calibration_bins.csv`.

## Figure 9: Component ablation

- Output: `figures/publication/figure_02_ablation.{png,pdf}`
- Generator: `python scripts/generate_figures.py`
- Function: `ablation()` in `src/spindle_cnn/experiments/make_figures.py`
- Input: `results/ablation_summary.csv`.

## Figure 10: Confusion matrices

- Output: `figures/publication/figure_06_confusion_matrices.{png,pdf}`
- Generator: `python scripts/generate_figures.py`
- Function: `confusion()` in `src/spindle_cnn/experiments/make_figures.py`
- Inputs: `results/raw_runs.csv` and
  `results/runs/<run_id>/confusion_matrix.csv` for every selected run.

## Figure 11: Computational cost

- Output: `figures/publication/figure_07_computational_cost.{png,pdf}`
- Generator: `python scripts/generate_figures.py`
- Function: `computational_cost()` in
  `src/spindle_cnn/experiments/make_figures.py`
- Input: `results/model_complexity.csv`, derived from
  `results/raw_runs.csv`.

## Figure 12: Validation-to-test generalization gap

- Output: `figures/publication/figure_09_validation_test_gap.{png,pdf}`
- Generator: `python scripts/generate_figures.py`
- Function: `validation_test_gap()` in
  `src/spindle_cnn/experiments/make_figures.py`
- Input: `results/domain_gap_summary.csv`, derived from the validation and test
  Macro-F1 columns in `results/raw_runs.csv`.

## Numerical provenance rule

The figure scripts read the CSV or per-run files listed above. No experimental
value is entered directly in a plotting function.

## Figures 13-18: sensitivity and diagnostics

| Figure | Output | Generator | Input |
| --- | --- | --- | --- |
| Figure 13 | `figure_13_domain_shift_diagnostics.{png,pdf}` | `analysis.domain_shift` | Original-amplitude window metrics and train-test contrasts |
| Figure 14 | `figure_16_label_granularity.{png,pdf}` | `analysis.label_granularity` | Retained four-class test probabilities |
| Figure 15 | `figure_19_cwt_sensitivity.{png,pdf}` | `analysis.cwt_sensitivity` | Six controlled CWT caches and their matched result rows |
| Figure 16 | `figure_17_self_training_sensitivity.{png,pdf}` | `analysis.self_training` | Fixed, supervised, and coverage-controlled matched rows |
| Figure 17 | `figure_22_feature_space_pca.{png,pdf}` | `analysis.feature_space` | CWT-GRU penultimate embeddings |
| Figure 18 | `figure_23_feature_space_tsne.{png,pdf}` | `analysis.feature_space` | CWT-GRU penultimate embeddings |

## Supplementary Figures S1-S5

| Figure | Output | Input |
| --- | --- | --- |
| S1 | `figure_14_per_fold_reliability.{png,pdf}` | Per-run logits and validation-fitted temperatures |
| S2 | `figure_15_temperature_distribution.{png,pdf}` | `calibration_per_run.csv` |
| S3 | `figure_20_foldwise_paired_differences.{png,pdf}` | `foldwise_paired_differences.csv` |
| S4 | `figure_21_training_budget.{png,pdf}` | `training_budget_comparison.csv` |
| S5 | `figure_24_normalization_sensitivity.{png,pdf}` | `normalization_sensitivity_summary.csv` |

Figure-only PCA/t-SNE reproduction can use the retained compact embeddings:

```powershell
python -m spindle_cnn.analysis.feature_space `
  --results-dir results `
  --figures-dir figures/publication `
  --reuse-embeddings
```

`figure_18_baseline_tuning.{png,pdf}` is a retained analysis artifact and is
not embedded as one of the 23 inline manuscript figures.
