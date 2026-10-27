# Manuscript Figure Sources

All final figures have editable vector PDF counterparts under
`figures/publication/`. The manuscript embeds the PNG files at publication
resolution.

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

## Figure 4: Full-label model comparison

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
