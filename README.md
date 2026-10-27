# Leakage-Safe Bearing Fault Diagnosis Benchmark

This repository contains the reproducible experiment code for the manuscript
**“Leakage-Safe Benchmarking of Time-Frequency Fusion and Calibrated
Self-Training for Bearing Fault Diagnosis.”**

The primary experiment uses the public Paderborn University bearing dataset and
tests simultaneous transfer to:

- an unseen bearing code;
- an unseen operating condition;
- `8192`-sample windows with stride `8192` (no overlap);
- pseudo-labels drawn only from the unlabeled training partition.

The benchmark compares Raw 1D-CNN, GRU-only, CWT 2D-CNN, CWT-GRU without SE,
full CWT-GRU, temperature-scaled CWT-GRU, and calibrated self-training.

The historical industrial-data experiment is no longer present in the working
tree. Its engineering-threshold labels, overlapping windows, repeated station
groups, and test relabeling do not support a confirmatory diagnostic-accuracy
claim. The old code remains recoverable from Git history only.

## Repository layout

```text
AUDIT_REPORT.md                  # audit before refactoring
CLEANUP_REPORT.md                # removed and retained local artifacts
configs/                         # canonical experiment configuration
data/README.md                   # official sources and preprocessing definition
docs/                            # protocol, reproducibility, and audit notes
figures/                         # generated publication figures
results/                         # canonical CSV and per-run audit records
scripts/                         # thin executable wrappers
src/spindle_cnn/
├── models/                      # BaseModel and registered architectures
│   └── baselines/               # not-yet-reproduced SOTA placeholders
├── datasets/                    # group-safe splits and balanced budgets
├── training/                    # deterministic training and self-training
├── evaluation/                  # metrics, calibration, FLOPs, latency
├── experiments/                 # unified runners and figures
├── analysis/                    # CIs, paired tests, ablation summaries
└── public_benchmark/            # public data preparation
tests/                           # split, model, metric, and registry tests
```

## Installation

Python 3.10 or newer is required. Install a PyTorch build matching the local
CUDA/CPU environment, then install the package:

```bash
python -m pip install -e ".[dev]"
```

An optional Conda environment specification is available in
`environment.yml`. Exact numerical latency depends on hardware.

## Data preparation

Raw data and caches are not redistributed. Download the official archives and
build the Paderborn cache as documented in
[data/README.md](data/README.md). The canonical cache path is:

```text
data/public/paderborn/pu_4class_w8192.npz
```

The selected training and test bearing codes, operating conditions, sampling
rate, window size, stride, and split rule are explicitly listed in that file.

## Reproduce the full experiment

The canonical configuration uses:

```text
seeds = 42, 123, 2024, 2025, 3407
folds = 0, 1, 2, 3
primary comparison = 4 folds x 5 seeds = 20 runs per method
```

From the repository root:

```bash
PYTHONPATH=src python -m spindle_cnn.experiments.run_benchmark \
  --config configs/paderborn_main.json \
  --experiment all

PYTHONPATH=src python -m spindle_cnn.analysis.statistics \
  --results results/raw_runs.csv \
  --output-dir results

PYTHONPATH=src python -m spindle_cnn.experiments.make_figures \
  --results-dir results \
  --output-dir figures/publication

PYTHONPATH=src python -m spindle_cnn.analysis.recheck_run \
  --results-dir results --all
```

On PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.experiments.run_benchmark --config configs/paderborn_main.json --experiment all
python -m spindle_cnn.analysis.statistics --results results/raw_runs.csv --output-dir results
python -m spindle_cnn.experiments.make_figures --results-dir results --output-dir figures/publication
python -m spindle_cnn.analysis.recheck_run --results-dir results --all
```

The runner refuses a dirty worktree unless `--allow-dirty` is passed. Such runs
are marked exploratory in every row. Resume is allowed only for an exact
`run_id + config_hash` match.

### Label budget semantics

The `100%` setting means 100% of the class-balanced label budget. It samples the
same number of windows from every class, using the size of the smallest training
class. It does **not** mean that every training window is labeled. The exact
labeled and candidate counts are stored in every run.

### Self-training isolation

Validation data are used for early stopping and temperature fitting. Test data
are used only after fitting is complete. Pseudo-label candidates are constructed
as `training indices - labeled indices`; validation and test indices are never
passed to candidate selection. Pseudo-label accuracy is recorded only as an
offline audit when labels are available.

## Canonical result files

- `results/raw_runs.csv`: one row per fold, seed, method, budget, and threshold.
- `results/summary.csv`: requested model-level macro-F1 and accuracy summaries.
- `results/statistical_tests.csv`: paired differences and Wilcoxon tests.
- `results/ablation_summary.csv`: component matrix and mean ± standard deviation.
- `results/label_efficiency_summary.csv`.
- `results/threshold_summary.csv`.
- `results/model_complexity.csv`.
- `results/rechecks/all_runs.json`: independent metric recomputation from saved
  logits for every canonical row.
- `results/runs/<run_id>/config.json`: per-run configuration, environment,
  dataset hash, split hash, and Git commit.
- `results/splits/fold<k>.json`: exact train/validation/test indices and hashes.

Every successful row records training time, test inference time, parameter count,
estimated GFLOPs, train/validation loss, temperature, Macro-F1, accuracy, ECE,
NLL, Brier score, configuration hash, split hash, dataset hash, and Git commit.

The generated five-seed report is
[results/EXPERIMENT_REPORT.md](results/EXPERIMENT_REPORT.md), and the
publication-ready figures are under
[figures/publication](figures/publication/). The superseded three-seed result
files were removed from the working tree and are available only in Git history.

## Tests

```bash
PYTHONPATH=src python -m pytest -q tests
```

## Model and data licensing

Code is released under the MIT License. Dataset files remain subject to their
providers' terms and are not included in this repository.
