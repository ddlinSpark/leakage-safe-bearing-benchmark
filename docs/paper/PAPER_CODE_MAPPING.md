# Paper to Code Mapping

Canonical result baseline: `8ac2d75`

This file maps the manuscript methods, tables, and figures to the code,
configuration, and result files that produce them. Result values are read from
the canonical CSV files and are never maintained as separate constants in this
document.

## Shared experiment path

```text
configs/paderborn_main.json
  -> src/spindle_cnn/experiments/run_benchmark.py
  -> src/spindle_cnn/datasets/splits.py
  -> src/spindle_cnn/training/runner.py
  -> src/spindle_cnn/evaluation/metrics.py
  -> results/raw_runs.csv
  -> results/runs/<run_id>/{config.json,predictions.npz,calibration_bins.csv,confusion_matrix.csv}
  -> src/spindle_cnn/analysis/statistics.py
  -> results/{summary.csv,summary_with_ci.csv,statistical_tests.csv,ablation_summary.csv}
  -> src/spindle_cnn/analysis/hierarchical.py
  -> results/hierarchical_statistics.csv
  -> src/spindle_cnn/experiments/{make_figures.py,make_method_figures.py}
  -> src/spindle_cnn/paper/build_docx.py
  -> docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx
```

## Manuscript methods

| Paper content | Code | Configuration | Result source | Figure / table |
| --- | --- | --- | --- | --- |
| Raw 1D-CNN | `src/spindle_cnn/models/raw_cnn.py::RawCNN`; `models/registry.py::METHODS["raw_cnn"]` | `configs/paderborn_main.json` | `results/raw_runs.csv` where `method=raw_cnn` | Figures 4, 8, 10; Tables 5, 6, 9-12 |
| GRU-only | `src/spindle_cnn/models/gru.py::RawGRU`; `models/registry.py::METHODS["gru_only"]` | Same canonical configuration | `results/raw_runs.csv` where `method=gru_only` | Figures 4, 9, 10; Tables 5-6, 9-12 |
| CWT representation | `src/spindle_cnn/public_benchmark/pu_data.py::_cwt_image` | `window_size=8192`, `stride=8192`, 48 Morlet scales, 96 × 96 images | `results/runs/<run_id>/config.json`; `data/public/paderborn/pu_4class_w8192.json` | Figure 1-2; Tables 2-3 |
| CWT 2D-CNN | `src/spindle_cnn/models/cwt_cnn.py::CWT2DCNN`; `models/registry.py::METHODS["cwt_cnn"]` | Same canonical configuration | `results/raw_runs.csv` where `method=cwt_cnn` | Figures 4, 9, 10; Tables 5-6, 9-12 |
| SE attention | `src/spindle_cnn/models/layers.py::SEBlock`; `ResidualBlock(use_se=True)` in `ImageEncoder` | Selected by `METHODS` model key and `build_model()` | Parameter and metric rows distinguish `cwt_gru_no_se` from `cwt_gru` | Figures 2, 9; Tables 10-12 |
| CWT-GRU without SE | `src/spindle_cnn/models/cwt_gru.py::CWTGRU(use_se=False)` | `method=cwt_gru_no_se` | `results/raw_runs.csv` | Figures 4, 9; Tables 5-6, 10-12 |
| CWT-GRU | `src/spindle_cnn/models/cwt_gru.py::CWTGRU(use_se=True)` | `method=cwt_gru` | `results/raw_runs.csv` | Figures 2, 4, 8-10; Tables 5-6, 9-12 |
| Temperature scaling | `src/spindle_cnn/evaluation/calibration.py::{fit_temperature,probabilities_from_logits}`; fitting call in `training/runner.py::train_model` | Validation logits/labels; method `cwt_gru_temp` reuses CWT-GRU weights in `run_benchmark.py` | `results/raw_runs.csv`: `raw_*` versus calibrated probability columns | Figures 7-8; Tables 6, 9-11 |
| Calibrated self-training | `src/spindle_cnn/training/runner.py::{select_pseudo_labels,run_self_training}`; orchestration in `experiments/run_benchmark.py` | `pseudo_threshold=0.6`, `pseudo_rounds=2`; thresholds `0.3-0.7` in the sensitivity design | `results/raw_runs.csv`; `threshold_summary.csv`; `label_efficiency_summary.csv` | Figures 3, 5-6, 10; Tables 6-8, 10-11 |
| Leakage-safe split construction | `src/spindle_cnn/datasets/splits.py::{load_fold_indices,validate_split_isolation,balanced_indices}` | `heldout_record_ids=[4,8,12,15]`; four condition folds | `results/splits/fold0.json` through `fold3.json` | Figure 1; Table 1 |
| Macro-F1, accuracy, ECE, NLL, Brier | `src/spindle_cnn/evaluation/metrics.py` | Metrics applied after training to held-out test logits | `results/raw_runs.csv`; `results/rechecks/all_runs.json` | Tables 5-12 |
| Mean, SD, CI, paired tests | `src/spindle_cnn/analysis/statistics.py`; `analysis/hierarchical.py` | 20,000-replicate two-stage fold-then-seed bootstrap; exact four-fold sign-flip sensitivity test | `results/summary_with_ci.csv`; `results/statistical_tests.csv`; `results/hierarchical_statistics.csv` | Tables 5-7; `docs/audit/STATISTICAL_SENSITIVITY_REPORT.md` |
| Parameter count, GFLOPs, latency | `src/spindle_cnn/evaluation/efficiency.py` | Canonical batch size and recorded CUDA environment | `results/raw_runs.csv`; `results/model_complexity.csv` | Figure 11; Tables 10, 12 |

## Research-question mapping

| Paper section | Model or experiment | Script | Configuration | Raw result source | Generated output |
| --- | --- | --- | --- | --- | --- |
| RQ1 | Raw 1D-CNN vs CWT 2D-CNN | `experiments/run_benchmark.py --experiment main` | `configs/paderborn_main.json` | `results/runs/main__*/` and `results/raw_runs.csv` | Tables 5-6; Figure 4 |
| RQ2 | CWT-based temporal fusion | Same main runner | Same canonical configuration | `results/raw_runs.csv`; `results/hierarchical_statistics.csv` | Table 6 |
| RQ3 | CWT, GRU, SE, calibration, self-training components | `experiments/run_benchmark.py --experiment ablation` or `run_ablation.py` | Canonical main configuration | Matched `experiment_type=main` rows | Table 11; Figure 9 |
| RQ4 | 10% balanced budget, 50% balanced budget, Full balanced budget | `run_benchmark.py --experiment label_efficiency` | `experiments.label_efficiency` | `results/label_efficiency_summary.csv`; `results/hierarchical_statistics.csv` | Table 7; Figure 5 |
| RQ5 | Confidence-threshold sweep | `run_benchmark.py --experiment threshold` | `experiments.threshold` | `results/threshold_summary.csv`; threshold rows in `results/raw_runs.csv` | Table 8; Figure 6 |
| RQ6 | Temperature scaling and probability metrics | Main runner; `evaluation/calibration.py` | Canonical training configuration | `raw_ece`, `ece`, `raw_nll`, `nll`, `raw_brier`, `brier` in `results/raw_runs.csv` | Table 9; Figures 7-8 |
| External comparison | DRSN adaptation and TFN adaptation | `run_benchmark.py --experiment sota` | `configs/paderborn_sota.json` | `results/sota/raw_runs.csv`; `results/hierarchical_statistics.csv` | Table 10; `SOTA_RETRAINING_REPORT.md` |
| Domain gap | Validation and held-out test Macro-F1 | Main runner; `analysis/statistics.py::build_domain_gap_summary` | Canonical folds and seeds | `results/raw_runs.csv` | Table 13; Figure 12; `DOMAIN_GAP_ANALYSIS.md` |
| Protocol and label audit | Window, bearing, condition, source-file, and damage-origin checks | `analysis/protocol_audit.py` | Canonical cache, raw-amplitude cache, split manifests | `reports/audit_status.csv`; `reports/window_audit.csv`; `reports/label_audit.csv` | `docs/reports/AUDIT_REPORT.md`; Methods and Section 7.6 |
| Domain-shift diagnostics | Raw time- and frequency-domain contrasts | `analysis/domain_shift.py` | Original-amplitude windows and canonical splits | `results/domain_shift_statistics.csv`; `results/domain_shift_contrasts.csv` | Table 14; Figure 13 |
| Label granularity | Post-hoc four-class, three-class, and binary views | `analysis/label_granularity.py` | Retained test probabilities | `results/label_granularity_sensitivity.csv` | Table 15; Figure 14 |
| CWT sensitivity | Scale, resolution, magnitude, maximum normalization, and real/imaginary controls | `public_benchmark/cwt_sensitivity.py`; `experiments/run_cwt_sensitivity.py`; `analysis/cwt_sensitivity.py` | Independent CWT caches; canonical split/seed keys | `results/cwt_sensitivity_summary.csv`; `results/cwt_sensitivity_comparisons.csv` | Table 16; Figure 15 |
| Self-training sensitivity | Per-class top-50% coverage-controlled pseudo-labeling | `training/runner.py`; `analysis/self_training.py` | Fixed fold-seed matrix | `results/self_training_sensitivity_summary.csv`; `results/pseudo_label_diagnostics.csv` | Table 17; Figure 16 |
| Training and baseline audits | 30-epoch opportunity and validation-only learning-rate candidate | `analysis/training_budget.py`; `analysis/baseline_tuning.py` | Predeclared sensitivity configurations | `results/training_budget_comparison.csv`; `results/external_baseline_tuning.csv` | Table 18 |
| Feature-space analysis | CWT-GRU penultimate embeddings, PCA, t-SNE, neighbour agreement | `analysis/feature_space.py` | Train/validation/test embeddings after fitting | `results/feature_space_clustering.csv`; `results/feature_embeddings/fold*.npz` | Figures 17-18 |

## Manuscript tables

| Manuscript table | Builder | Code / configuration evidence | Canonical result source |
| --- | --- | --- | --- |
| Table 1 | `paper/build_docx.py::fold_rows` | `datasets/splits.py`; `configs/paderborn_main.json` | `results/splits/fold*.json` |
| Table 2 | `paper/build_docx.py` | `models/layers.py`; `models/cwt_gru.py`; `public_benchmark/pu_data.py` | Cached-shape metadata and per-run configuration |
| Table 3 | `paper/build_docx.py` | `configs/paderborn_main.json`; `training/runner.py`; `evaluation/calibration.py` | `results/runs/<run_id>/config.json` |
| Table 4 | `paper/build_docx.py` | `configs/paderborn_main.json` | `results/raw_runs.csv` grouped by `experiment_type` |
| Table 5 | `paper/build_docx.py::main_table_rows` | Registry and runner | `results/summary_with_ci.csv`; `results/raw_runs.csv` |
| Table 6 | `paper/build_docx.py::paired_rows` | `analysis/hierarchical.py::{_hierarchical_bootstrap_interval,_fold_signflip_p_value}` | `results/hierarchical_statistics.csv`; `results/raw_runs.csv`; sensitivity values in `results/statistical_tests.csv` |
| Table 7 | `paper/build_docx.py::label_rows` | `run_benchmark.py` label-efficiency path | `results/label_efficiency_summary.csv`; `results/raw_runs.csv` |
| Table 8 | `paper/build_docx.py::threshold_rows` | `run_benchmark.py::run_threshold_experiment` | `results/threshold_summary.csv`; `results/raw_runs.csv` |
| Table 9 | `paper/build_docx.py::calibration_rows` | Calibration metrics in `evaluation/metrics.py` | `results/raw_runs.csv` |
| Table 10 | `paper/build_docx.py::sota_rows` | `models/registry.py`; architecture files | `results/summary_with_ci.csv`; `results/raw_runs.csv`; `results/sota/` |
| Table 11 | `paper/build_docx.py::ablation_rows` | `models/registry.py`; `analysis/statistics.py::ABLATION_COMPONENTS` | `results/ablation_summary.csv`; `results/raw_runs.csv` |
| Table 12 | `paper/build_docx.py::complexity_rows` | `evaluation/efficiency.py`; `experiments/run_benchmark.py::summarize_results` | Main rows in `results/raw_runs.csv` |
| Table 13 | `paper/build_docx.py::domain_gap_rows` | `analysis/statistics.py::build_domain_gap_summary` | `results/domain_gap_summary.csv`; `results/raw_runs.csv` |

## Manuscript figures

| Figure | Generator | Input |
| --- | --- | --- |
| Figure 1 | `experiments/make_method_figures.py::overall_framework` | Protocol and split code |
| Figure 2 | `experiments/make_method_figures.py::cwt_gru_architecture` | CWT/model source |
| Figure 3 | `experiments/make_method_figures.py::self_training_pipeline` | Self-training source and canonical configuration |
| Figure 4 | `experiments/make_figures.py::model_comparison` | `results/summary.csv` |
| Figure 5 | `experiments/make_figures.py::label_efficiency` | `results/label_efficiency_summary.csv` |
| Figure 6 | `experiments/make_figures.py::threshold_sensitivity` | `results/threshold_summary.csv` |
| Figure 7 | `experiments/make_figures.py::calibration` | `results/raw_runs.csv` |
| Figure 8 | `experiments/make_figures.py::reliability_diagrams` | Calibration bins under `results/runs/<run_id>/` |
| Figure 9 | `experiments/make_figures.py::ablation` | `results/ablation_summary.csv` |
| Figure 10 | `experiments/make_figures.py::confusion` | Confusion matrices under `results/runs/<run_id>/` |
| Figure 11 | `experiments/make_figures.py::computational_cost` | `results/model_complexity.csv` |
| Figure 12 | `experiments/make_figures.py::validation_test_gap` | `results/domain_gap_summary.csv` |

The detailed table/figure provenance is also recorded in
`docs/paper/TABLE_SOURCES.md` and `docs/paper/FIGURE_SOURCES.md`.

## Baseline and same-protocol eligibility

| Model | Source / origin | Official or project code | Input | Training and split | Parameters | Metric status |
| --- | --- | --- | --- | --- | --- | --- |
| Raw 1D-CNN | Project-defined supervised baseline | `models/raw_cnn.py` | Raw waveform, 3 × 8192 | Canonical configuration; four bearing-code/condition-disjoint folds × five seeds | Recorded in `raw_runs.csv` | 20 same-protocol runs; Macro-F1 and accuracy |
| GRU-only | Project-defined recurrent baseline using the GRU concept [8] | `models/gru.py` | Raw waveform, 3 × 8192 | Same protocol | Recorded in `raw_runs.csv` | 20 same-protocol runs |
| CWT 2D-CNN | Project-defined CWT baseline using CWT/CNN/SE components [4-6] | `models/cwt_cnn.py`; `models/layers.py` | Magnitude CWT, 3 × 96 × 96 | Same protocol | Recorded in `raw_runs.csv` | 20 same-protocol runs |
| CWT-GRU without SE | Project-defined ablation | `models/cwt_gru.py` with `use_se=False` | Raw + CWT | Same protocol | 1,086,244, measured from code and records | 20 same-protocol runs |
| CWT-GRU | Project-defined fused model | `models/cwt_gru.py` | Raw + CWT | Same protocol | 1,101,012, measured from code and records | 20 same-protocol runs |
| CWT-GRU + temperature scaling | Project-defined calibration condition on the same weights | `evaluation/calibration.py`; `run_benchmark.py` reuse branch | Raw + CWT | Same protocol and matched base runs | Same as CWT-GRU | 20 matched evaluations |
| CWT-GRU + temperature scaling + self-training | Project-defined semi-supervised condition | `training/runner.py`; `run_benchmark.py` | Raw + CWT | Same protocol; pseudo-labels restricted to unlabeled training windows | Same as CWT-GRU | 20 same-protocol runs |
| DRSN adaptation | Zhao et al., deep residual shrinkage networks [16]; three-channel adaptation | `models/drsn.py`; `models/registry.py` | Raw waveform | Same protocol | 233,040, measured from code | 20 completed runs; Macro-F1 and accuracy |
| TFN adaptation | Chen et al. (2024), TFN; public comparison-model source `https://github.com/Bingbingboopoo/MTFC_T/tree/main/comparison_models/TFNet` | `models/tfnet.py`; internal historical key `tfnet`, official comparison-model structure adapted locally | Raw waveform | Same protocol | 184,900, measured from code | 20 completed runs; Macro-F1 and accuracy |

No original-study performance number is inserted into Table 10. The manuscript
states that reported values from other datasets or splits are not used for
ranking.

### Explicitly unavailable external implementations

| Name in code registry | Source-paper result in manuscript | Official code / reproduction status | Numerical comparison |
| --- | --- | --- | --- |
| `cnn_lstm` | Not used as a numeric comparator | Not reproduced | Excluded |
| `drsn_transformer` | Background discussion only | Not reproduced | Excluded |
| `mtfc_t` | Discussed as methodological reference [14] | Original authors link `https://github.com/Bingbingboopoo/MTFC_T/tree/main/our_model`; not adapted to this protocol | Excluded |

The placeholder registry
`src/spindle_cnn/models/baselines/registry.py` is intentionally disconnected
from the executable model factory. No paper table cell depends on these
placeholders.

## Numerical audit

Run:

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_submission.py --json-output artifacts/local/pre_submission_audit.json
```

The audit independently recomputes mean, SD, run counts, paired differences,
fold-aware intervals, and sensitivity tests from `results/raw_runs.csv` and
`results/sota/raw_runs.csv`, then compares them with all derived result files and
all 13 manuscript tables. It does not call the production statistics functions
to establish expected values.
