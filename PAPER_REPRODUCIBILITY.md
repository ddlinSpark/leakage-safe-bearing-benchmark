# Paper Reproducibility

This guide reproduces the SCI manuscript from the canonical five-seed benchmark
results. In this repository, "100% labels" means 100% of the class-balanced
training budget, not all training windows.

## Recorded experiment

| Item | Recorded value |
| --- | --- |
| Dataset | Paderborn University Bearing Dataset |
| Inputs | Vibration and two phase-current channels |
| Window / stride | 8192 / 8192 samples |
| Folds | 0, 1, 2, 3 |
| Seeds | 42, 123, 2024, 2025, 3407 |
| Main runs | 140, or 20 per method condition |
| Label-efficiency runs | 80 |
| Threshold runs | 100 |
| Successful canonical rows | 320 |
| Main threshold | 0.60 |
| Main self-training rounds | 2 |
| Threshold-experiment rounds | 1 |

The canonical run matrix is `results/raw_runs.csv`. Each row contains the
dataset, split, and configuration hashes, the Git commit used for training, the
complete path to the per-run configuration, and the success status.

## Environment

The recorded principal environment is:

- Python 3.13.5
- PyTorch 2.11.0 with CUDA 13.0
- NumPy 2.1.3
- pandas 2.2.3
- NVIDIA GeForce RTX 5080
- Windows 11

Exact training and inference times depend on hardware. Environments from each
run are retained in `results/runs/<run_id>/config.json`.

Install the Python package after selecting a PyTorch build compatible with the
local CUDA or CPU environment:

```powershell
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
```

## Verify the canonical inputs

```powershell
python -m pytest -q tests
python -c "import pandas as pd; d=pd.read_csv('results/raw_runs.csv'); assert len(d)==320; assert d.status.eq('success').all(); print(d.groupby('experiment_type').size())"
```

The manuscript data loader also enforces:

- exactly 320 successful, unique canonical runs;
- 20 main runs for every reported method;
- 20 threshold runs at every tested threshold;
- the presence of every required split manifest and confusion matrix.

## Recompute derived statistics

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.analysis.statistics --results results/raw_runs.csv --output-dir results
python -m spindle_cnn.analysis.report --results-dir results
python -m spindle_cnn.analysis.recheck_run --results-dir results --all
```

The statistics command regenerates `summary.csv`, `summary_with_ci.csv`,
`statistical_tests.csv`, `ablation_summary.csv`,
`label_efficiency_summary.csv`, `threshold_summary.csv`, and
`model_complexity.csv`.

## Regenerate every figure

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.experiments.make_method_figures --output-dir figures/publication
python -m spindle_cnn.experiments.make_figures --results-dir results --output-dir figures/publication
```

The mapping from manuscript figure number to function and source file is in
`docs/paper/FIGURE_SOURCES.md`.

## Rebuild the manuscript

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.paper.build_docx `
  --results-dir results `
  --figures-dir figures/publication `
  --output docs/paper/Leakage_Safe_Bearing_Benchmark_SCI.docx
```

The builder reads all result values from the canonical files. It loads method
dimensions from the model code, training settings from
`configs/paderborn_main.json`, fold membership from `results/splits/`, and the
recorded device name from a per-run configuration.

## Optional full rerun

A complete rerun trains the canonical matrix and should be performed only after
the official Paderborn archives have been prepared as described in
`data/README.md`.

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.experiments.run_benchmark --config configs/paderborn_main.json --experiment all
python -m spindle_cnn.analysis.statistics --results results/raw_runs.csv --output-dir results
python -m spindle_cnn.analysis.report --results-dir results
python -m spindle_cnn.experiments.make_method_figures --output-dir figures/publication
python -m spindle_cnn.experiments.make_figures --results-dir results --output-dir figures/publication
python -m spindle_cnn.paper.build_docx --results-dir results --figures-dir figures/publication --output docs/paper/Leakage_Safe_Bearing_Benchmark_SCI.docx
```

The runner refuses an uncommitted working tree by default. During exploratory
development, use `--allow-dirty`, then treat those rows as exploratory. Do not
merge exploratory rows into the canonical table.

## Numerical traceability

- Table-by-table inputs: `docs/paper/TABLE_SOURCES.md`.
- Figure-by-figure scripts: `docs/paper/FIGURE_SOURCES.md`.
- Row-level audit: `results/raw_runs.csv` and `results/runs/<run_id>/`.
- Split audit: `results/splits/fold0.json` through `fold3.json`.
- Independent metric recomputation: `results/rechecks/all_runs.json`.
- Human-readable result summary: `results/EXPERIMENT_REPORT.md`.

## Scientific boundaries

- The Paderborn test rig is a bearing test bench, not a complete machine-tool
  spindle assembly.
- Published model accuracies from different datasets, splits, preprocessing
  pipelines, or metrics are not entered as fair SOTA rankings.
- External models listed in `src/spindle_cnn/models/baselines/registry.py` are
  placeholders. They are excluded until their architecture and training recipe
  are reproduced and trained on the same split.
- Calibration can improve ECE, NLL, and Brier score without changing the
  predicted class.
- Non-significant self-training and ablation differences are reported as
  non-significant.
