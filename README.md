# Leakage-Safe Bearing Fault Diagnosis Benchmark

This repository contains the reproducible experiment code for the manuscript
**“Leakage-Safe Cross-Domain Benchmarking for Bearing Fault Diagnosis:
Time-Frequency Fusion, Calibration, and Self-Training.”**

The repository and Python distribution are named
`leakage-safe-bearing-benchmark`. The import namespace remains `spindle_cnn`
for compatibility with the recorded experiment code, commands, and tests.

The primary experiment uses the public Paderborn University bearing dataset and
tests simultaneous transfer to:

- an unseen bearing code;
- an unseen operating condition;
- `8192`-sample windows with stride `8192` (no overlap);
- pseudo-labels drawn only from the unlabeled training partition.

The benchmark compares Raw 1D-CNN, GRU-only, CWT 2D-CNN, CWT-GRU without SE,
full CWT-GRU, temperature-scaled CWT-GRU, and calibrated self-training.
DRSN and the Time-Frequency Network (TFN) are also retrained as same-protocol
external adaptations in a separate `results/sota/` block. They are adaptations
to the common three-channel, four-class Paderborn setup, not exact reproductions
of the original papers' training recipes.

The benchmark is designed to test a negative and reproducibility-oriented
research question: do time-frequency fusion, recurrent modeling, calibration,
and self-training provide stable gains when both bearing identity and operating
condition are unseen? The manuscript reports the observed negative and
statistically non-significant outcomes rather than selecting a favorable seed
or importing accuracy values from incompatible published splits.

Published values such as 99% accuracy are not used as same-protocol rankings
because different bearings, preprocessing, window overlap, class definitions,
and train/test splits can produce a different estimand. Only models retrained
under the protocol in `configs/` enter the numerical comparison.

The primary 320-row matrix and 40-row external adaptation block remain fixed.
New analyses are explicitly separated from model selection:

- raw-signal domain-shift statistics;
- damage-origin and label manifests;
- post-hoc coarse-label sensitivity;
- CWT magnitude, scale, resolution, and normalization controls;
- validation-only baseline learning-rate tuning;
- 12- versus 30-epoch training-opportunity checks;
- per-fold calibration and reliability analysis;
- coverage-controlled pseudo-labeling;
- penultimate-feature PCA/t-SNE diagnostics.

No sensitivity result replaces a primary configuration after its test score is
observed.

The historical industrial-data experiment is no longer present in the working
tree. Its engineering-threshold labels, overlapping windows, repeated station
groups, and test relabeling do not support a confirmatory diagnostic-accuracy
claim. The old code remains recoverable from Git history only.

## Repository layout

```text
configs/                         # canonical experiment configuration
data/README.md                   # official sources and preprocessing definition
docs/
├── audit/                       # release, leakage, provenance, and privacy audits
└── paper/                       # paper mapping, analysis, and reproducibility notes
figures/                         # generated publication figures
manuscript/                      # rendered manuscript and supplementary PDFs
results/                         # canonical CSV and per-run audit records
scripts/                         # thin executable wrappers
src/spindle_cnn/
├── models/                      # BaseModel and registered architectures
│   └── baselines/               # unavailable external-model placeholders
├── datasets/                    # group-safe splits and balanced budgets
├── training/                    # deterministic training and self-training
├── evaluation/                  # metrics, calibration, FLOPs, latency
├── experiments/                 # unified runners and figures
├── analysis/                    # CIs, paired tests, ablation summaries
├── plotting/                    # shared journal palette and visual style
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
`environment.yml`. Models are trained from scratch; pretrained weights are not
required. Reported training time, inference latency, and hardware-sensitive
measurements come from an NVIDIA GeForce RTX 5080 environment and are not
expected to transfer unchanged to other GPUs or CPUs.

The recorded principal runtime is Python 3.13.5, PyTorch 2.11.0 with CUDA 13.0,
NumPy 2.1.3, pandas 2.2.3, and an NVIDIA GeForce RTX 5080. Exact environment
snapshots are retained in every canonical `results/runs/<run_id>/config.json`.

When the package is launched outside the repository, set
`BEARING_BENCHMARK_ROOT` to the repository path. The legacy
`SPINDLE_CNN_ROOT` variable remains accepted for compatibility.

## Data preparation

Raw data and caches are not redistributed. Download the official archives and
build the Paderborn cache as documented in
[data/README.md](data/README.md). The canonical cache path is:

```text
data/public/paderborn/pu_4class_w8192.npz
```

The selected training and test bearing codes, operating conditions, sampling
rate, window size, stride, and split rule are explicitly listed in that file.
The exact selected-archive sizes and SHA-256 digests are recorded in
`data/manifests/paderborn_archives_sha256.csv`. After extraction, the raw
amplitude audit cache is built and checked against the canonical cache with:

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.public_benchmark.paderborn_raw_windows `
  --data-root data/public/paderborn/extracted `
  --output data/public/paderborn/pu_4class_w8192_raw.npz `
  --validate
```

The checked cache contains original amplitudes, exact measurement numbers, and
source filenames. Re-standardizing its windows reproduces the canonical cache
elementwise; `pu_4class_w8192_raw_validation.json` records the maximum
difference and field-equality checks.

## Run one experiment

After preparing the data cache, a single Raw 1D-CNN fold-seed run can be
executed with:

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_main.json `
  --experiment main `
  --methods raw_cnn `
  --folds 0 `
  --seeds 42 `
  --output-dir artifacts/local/smoke-main `
  --no-resume
```

The runner refuses a dirty worktree by default and writes the selected split
manifest, per-run configuration, predictions, diagnostics, and result row.
Additional single-model, label-budget, threshold, ablation, and external-model
commands are listed in [REPRODUCE.md](REPRODUCE.md).

## Reproduce the full experiment

The commands below use the runner's default resume behavior and therefore
verify or complete the checked-in experiment matrix. For an isolated
from-scratch rerun in a separate output directory, follow
[REPRODUCE.md](REPRODUCE.md).

The canonical configuration uses:

```text
seeds = 42, 123, 2024, 2025, 3407
folds = 0, 1, 2, 3
primary comparison = 4 folds x 5 seeds = 20 runs per method
canonical results = 320 runs
external results = 2 adaptations x 4 folds x 5 seeds = 40 runs
```

The stochastic seeds are exactly `42, 123, 2024, 2025, 3407`.

The 320 canonical rows comprise 140 main, 80 label-efficiency, and 100
threshold-sensitivity evaluations. The complete from-scratch workflow,
including data preparation and single-experiment commands, is in
[REPRODUCE.md](REPRODUCE.md).

From the repository root:

```bash
PYTHONPATH=src python -m spindle_cnn.experiments.run_benchmark \
  --config configs/paderborn_main.json \
  --experiment all

PYTHONPATH=src python -m spindle_cnn.experiments.run_benchmark \
  --config configs/paderborn_sota.json \
  --experiment sota

PYTHONPATH=src python -m spindle_cnn.analysis.statistics \
  --results results/raw_runs.csv \
  --output-dir results

PYTHONPATH=src python -m spindle_cnn.analysis.hierarchical \
  --results-dir results \
  --external-dir results/sota \
  --output results/hierarchical_statistics.csv
python scripts/build_statistical_reports.py

PYTHONPATH=src python -m spindle_cnn.analysis.report --results-dir results

PYTHONPATH=src python -m spindle_cnn.analysis.sota \
  --results-dir results \
  --external-dir results/sota

PYTHONPATH=src python -m spindle_cnn.experiments.make_method_figures \
  --output-dir figures/publication

PYTHONPATH=src python -m spindle_cnn.experiments.make_figures \
  --results-dir results \
  --output-dir figures/publication

PYTHONPATH=src python scripts/hash_publication_figures.py
PYTHONPATH=src python scripts/check_figure_palette.py

PYTHONPATH=src python -m spindle_cnn.analysis.recheck_run \
  --results-dir results --all

PYTHONPATH=src python -m spindle_cnn.analysis.recheck_run \
  --results-dir results/sota --all

PYTHONPATH=src python -m spindle_cnn.paper.build_docx \
  --results-dir results \
  --figures-dir figures/publication \
  --output docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx
```

On PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.experiments.run_benchmark --config configs/paderborn_main.json --experiment all
python -m spindle_cnn.experiments.run_benchmark --config configs/paderborn_sota.json --experiment sota
python -m spindle_cnn.analysis.statistics --results results/raw_runs.csv --output-dir results
python -m spindle_cnn.analysis.hierarchical --results-dir results --external-dir results/sota --output results/hierarchical_statistics.csv
python scripts/build_statistical_reports.py
python -m spindle_cnn.analysis.report --results-dir results
python -m spindle_cnn.analysis.sota --results-dir results --external-dir results/sota
python -m spindle_cnn.experiments.make_method_figures --output-dir figures/publication
python -m spindle_cnn.experiments.make_figures --results-dir results --output-dir figures/publication
python scripts/hash_publication_figures.py
python scripts/check_figure_palette.py
python -m spindle_cnn.analysis.recheck_run --results-dir results --all
python -m spindle_cnn.analysis.recheck_run --results-dir results/sota --all
python -m spindle_cnn.paper.build_docx --results-dir results --figures-dir figures/publication --output docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx
```

All plotting modules import `src/spindle_cnn/plotting/style.py`. This file is the
single source of the prescribed 11-color palette, fixed model colors, typography,
grid treatment, and white-background export settings. Figure hashes are retained
in `figures/publication/figure_hashes.json`.

The manuscript workflow, source mapping, and validation steps are documented in
[docs/paper/PAPER_REPRODUCIBILITY.md](docs/paper/PAPER_REPRODUCIBILITY.md).
Table provenance is in
[docs/paper/TABLE_SOURCES.md](docs/paper/TABLE_SOURCES.md), figure provenance is
in [docs/paper/FIGURE_SOURCES.md](docs/paper/FIGURE_SOURCES.md), and the
scientific revision record is in [CHANGELOG.md](CHANGELOG.md).

External retraining details are in the
[external-baseline report](docs/paper/SOTA_RETRAINING_REPORT.md). The
validation-to-test analysis is in
[docs/paper/DOMAIN_GAP_ANALYSIS.md](docs/paper/DOMAIN_GAP_ANALYSIS.md), and the
final contribution text is in
[docs/paper/UPDATED_CONTRIBUTIONS.md](docs/paper/UPDATED_CONTRIBUTIONS.md).

## Pre-submission audit

Run the independent numerical and manuscript audit with:

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_submission.py `
  --json-output artifacts/local/pre_submission_audit.json
python scripts/verify_manuscript_numbers.py
python scripts/check_manuscript_consistency.py
python scripts/check_figure_palette.py
python scripts/audit_release_repository.py
python scripts/audit_final_manuscript.py
```

The current reports are:

- [PAPER_CODE_MAPPING.md](docs/paper/PAPER_CODE_MAPPING.md)
- [EXPERIMENT_COMPLETENESS.md](docs/audit/EXPERIMENT_COMPLETENESS.md)
- [LEAKAGE_AUDIT.md](docs/audit/LEAKAGE_AUDIT.md)
- [PRE_SUBMISSION_AUDIT.md](docs/audit/PRE_SUBMISSION_AUDIT.md)
- [STATISTICAL_SENSITIVITY_REPORT.md](docs/audit/STATISTICAL_SENSITIVITY_REPORT.md)
- [FINAL_STATISTICAL_AUDIT.md](docs/audit/FINAL_STATISTICAL_AUDIT.md)
- [FINAL_MANUSCRIPT_AUDIT.md](docs/audit/FINAL_MANUSCRIPT_AUDIT.md)
- [CONSISTENCY_REPORT.md](CONSISTENCY_REPORT.md)
- [CLEAN_REPRODUCTION_REPORT.md](CLEAN_REPRODUCTION_REPORT.md)
- [FINAL_RELEASE_REPORT.md](FINAL_RELEASE_REPORT.md)

After rendering the final DOCX, assemble the submission PDFs with:

```powershell
python scripts/build_submission_pdfs.py `
  --manuscript-pdf artifacts/local/render/final_release/Leakage_Safe_Bearing_Benchmark_SCI_final.pdf `
  --figures-dir figures/publication `
  --output-dir manuscript
```

The submission files are `manuscript/manuscript.pdf` and
`manuscript/supplementary.pdf`.

The runner refuses a dirty worktree unless `--allow-dirty` is passed. Such runs
are marked exploratory in every row. Resume is allowed only for an exact
`run_id + config_hash` match.

### Label budget semantics

The balanced-budget labels are `10% balanced`, `50% balanced`, and
`Full balanced budget`. The Full balanced budget samples the minimum available count from
every class. It does **not** mean that every training window is labeled. The
exact labeled and candidate counts are stored in every run.

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
- `results/domain_gap_summary.csv`.
- `results/sota/`: 40 external-baseline runs, summaries, paired tests, and
  per-run audit records.
- `results/rechecks/all_runs.json`: independent metric recomputation from saved
  logits for every canonical row.
- `results/runs/<run_id>/config.json`: per-run configuration, environment,
  dataset hash, split hash, and Git commit.
- `results/splits/fold<k>.json`: exact train/validation/test indices and hashes.
- `results/damage_origin_manifest.csv`: fold-condition split by physical bearing
  and damage origin.
- `results/domain_shift_statistics.csv`: raw-signal statistics by bearing,
  class, condition, split, channel, and fold.
- `results/label_granularity_sensitivity.csv`: four-class, three-class, and
  binary views from the same trained predictions.
- `results/calibration_per_run.csv`: per-run temperature and test probability
  metrics; validation logits are retained by the calibration recheck block.
- `results/experiment_registry.csv`: one-row audit index across canonical and
  sensitivity experiments.
- `results/paper_metrics.json`: one numerical source for manuscript checks.

Every successful row records training time, test inference time, parameter count,
estimated GFLOPs, train/validation loss, temperature, Macro-F1, accuracy, ECE,
NLL, Brier score, configuration hash, split hash, dataset hash, and Git commit.

The generated five-seed report is
[results/EXPERIMENT_REPORT.md](results/EXPERIMENT_REPORT.md), and the
publication-ready figures are under
[figures/publication](figures/publication/). The superseded three-seed result
files were removed from the working tree and are available only in Git history.

The manuscript tables are generated by the tracked DOCX builder described in
[docs/paper/TABLE_PROVENANCE.md](docs/paper/TABLE_PROVENANCE.md). Figure
provenance for Figures 1-12 is recorded in
[docs/paper/FIGURE_PROVENANCE.md](docs/paper/FIGURE_PROVENANCE.md).

## Tests

```bash
PYTHONPATH=src python -m pytest -q tests
```

## Protocol and sensitivity audits

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.analysis.protocol_audit --run-numeric-audit
python -m spindle_cnn.analysis.domain_shift
python -m spindle_cnn.analysis.label_granularity
python -m spindle_cnn.analysis.calibration_audit
python -m spindle_cnn.analysis.training_sanity
python -m spindle_cnn.analysis.foldwise
python scripts/build_experiment_registry.py
python scripts/build_paper_metrics.py
python scripts/verify_manuscript_numbers.py
```

The full sensitivity commands are listed in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Test data are never used to select a
normalization, CWT setting, architecture, confidence threshold, pseudo-label
policy, or training budget.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). Update the
preferred citation after journal acceptance and archival DOI assignment.

## Release audit

The final repository review is indexed by:

- [REPOSITORY_AUDIT.md](docs/audit/REPOSITORY_AUDIT.md)
- [REPRODUCE.md](REPRODUCE.md)
- [SPLIT_REPRODUCIBILITY_AUDIT.md](docs/audit/SPLIT_REPRODUCIBILITY_AUDIT.md)
- [LEAKAGE_FINAL_CHECK.md](docs/audit/LEAKAGE_FINAL_CHECK.md)
- [RESULT_COMPLETENESS_AUDIT.md](docs/audit/RESULT_COMPLETENESS_AUDIT.md)
- [STATISTICS_REPOSITORY_AUDIT.md](docs/audit/STATISTICS_REPOSITORY_AUDIT.md)
- [FIGURE_PROVENANCE.md](docs/paper/FIGURE_PROVENANCE.md)
- [TABLE_PROVENANCE.md](docs/paper/TABLE_PROVENANCE.md)
- [PAPER_CODE_MAPPING.md](docs/paper/PAPER_CODE_MAPPING.md)
- [PAPER_RESULT_CONSISTENCY.md](docs/audit/PAPER_RESULT_CONSISTENCY.md)
- [GIT_HISTORY_PRIVACY_AUDIT.md](docs/audit/GIT_HISTORY_PRIVACY_AUDIT.md)
- [DOUBLE_BLIND_RISK.md](docs/audit/DOUBLE_BLIND_RISK.md)
- [RELEASE_READINESS.md](docs/audit/RELEASE_READINESS.md)
- [ZENODO_PREPARATION.md](docs/audit/ZENODO_PREPARATION.md)
- [AUDIT_REPORT.md](AUDIT_REPORT.md)
- [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md)
- [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)
- [RESULT_CHANGELOG.md](RESULT_CHANGELOG.md)
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- [PAPER_REVISION_REPORT.md](PAPER_REVISION_REPORT.md)
- [PAPER_RESULT_TRACEABILITY.md](PAPER_RESULT_TRACEABILITY.md)
- [DAMAGE_ORIGIN_ANALYSIS.md](DAMAGE_ORIGIN_ANALYSIS.md)
- [SECOND_DATASET_ASSESSMENT.md](SECOND_DATASET_ASSESSMENT.md)

## Model and data licensing

Code is released under the [MIT License](LICENSE). Dataset files remain subject
to their providers' terms and are not included in this repository.
