# Figure Provenance

All 12 manuscript figures have tracked PNG outputs under
`figures/publication/`. Methodology diagrams are code-generated diagrams;
Figures 4-12 are data-derived and contain no manually entered result values.

| Figure | Type | Generator | Input |
| --- | --- | --- | --- |
| Figure 1 | methodology diagram | `src/spindle_cnn/experiments/make_method_figures.py::overall_framework` | `configs/paderborn_main.json`; split and self-training code |
| Figure 2 | methodology diagram | `src/spindle_cnn/experiments/make_method_figures.py::cwt_gru_architecture` | model source and cache metadata |
| Figure 3 | methodology diagram | `src/spindle_cnn/experiments/make_method_figures.py::self_training_pipeline` | `configs/paderborn_main.json`; runner and calibration code |
| Figure 4 | data-derived | `src/spindle_cnn/experiments/make_figures.py::model_comparison` | `results/summary.csv` |
| Figure 5 | data-derived | `src/spindle_cnn/experiments/make_figures.py::label_efficiency` | `results/label_efficiency_summary.csv` |
| Figure 6 | data-derived | `src/spindle_cnn/experiments/make_figures.py::threshold_sensitivity` | `results/threshold_summary.csv` |
| Figure 7 | data-derived | `src/spindle_cnn/experiments/make_figures.py::calibration` | `results/raw_runs.csv` |
| Figure 8 | data-derived | `src/spindle_cnn/experiments/make_figures.py::reliability_diagrams` | per-run `calibration_bins.csv` |
| Figure 9 | data-derived | `src/spindle_cnn/experiments/make_figures.py::ablation` | `results/ablation_summary.csv` |
| Figure 10 | data-derived | `src/spindle_cnn/experiments/make_figures.py::confusion` | per-run `confusion_matrix.csv` |
| Figure 11 | data-derived | `src/spindle_cnn/experiments/make_figures.py::computational_cost` | `results/model_complexity.csv` |
| Figure 12 | data-derived | `src/spindle_cnn/experiments/make_figures.py::validation_test_gap` | `results/domain_gap_summary.csv` |

## Reproduction commands

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.experiments.make_method_figures --output-dir figures/publication
python -m spindle_cnn.experiments.make_figures --results-dir results --output-dir figures/publication
```

The current revision regenerated all 12 PNG files byte-identically in a
separate output directory. No PDF figure output is tracked. No Excel,
Photoshop, or manual numeric overlays are used for data-derived figures.
