# Reproducibility Guide

## Scope

This guide reproduces the 320-row canonical Paderborn benchmark, the 40-row
same-protocol external adaptation block, the raw-data audit, and all retained
sensitivity analyses. Raw data, large caches, and model weights are not
redistributed.

## Recorded environment

| Component | Recorded value |
| --- | --- |
| Operating system | Windows 11 |
| Python | 3.13.5 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |
| PyWavelets | 1.8.0 |
| GPU | NVIDIA GeForce RTX 5080, 16 GB |

CPU execution reproduces the statistical calculations, but hardware-sensitive
timing values will differ.

## 1. Restore the environment

```powershell
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
python scripts/capture_environment.py
```

Use a PyTorch wheel compatible with the local CUDA or CPU platform. The project
does not pin a CUDA wheel because the correct build depends on the host.

## 2. Download and verify Paderborn archives

The exact archive URLs, byte sizes, and SHA-256 values are in
`data/manifests/paderborn_archives_sha256.csv`.

```powershell
python -m spindle_cnn.public_benchmark.prepare_paderborn `
  --archive-dir data/public/paderborn/rars `
  --extract-dir data/public/paderborn/extracted `
  --workers 4 `
  --seven-zip bsdtar
```

Verify downloaded archive hashes before preprocessing:

```powershell
python scripts/verify_paderborn_archives.py
```

## 3. Recreate the canonical cache

```powershell
python -m spindle_cnn.public_benchmark.pu_data `
  --data-root data/public/paderborn/extracted `
  --output data/public/paderborn/pu_4class_w8192.npz `
  --window-size 8192 `
  --stride 8192 `
  --scales 48 `
  --image-size 96 `
  --max-measurements 2
```

The prepared cache must contain 3,939 windows and shape
`3939 x 3 x 8192` for raw input and `3939 x 3 x 96 x 96` for CWT input.

One official file, `N15_M01_F10_KA08_2.mat`, is truncated. The canonical
preprocessing cache excludes the same unreadable measurement as the raw-window
audit. This is documented in the cache metadata and audit report.

## 4. Build the original-amplitude audit cache

```powershell
python -m spindle_cnn.public_benchmark.paderborn_raw_windows `
  --data-root data/public/paderborn/extracted `
  --output data/public/paderborn/pu_4class_w8192_raw.npz `
  --validate `
  --workers 4
```

The validation step standardizes each restored raw window and compares it with
the canonical cache. A successful run reports field equality for labels, loads,
record IDs, and starts, with zero maximum standardization difference at the
recorded tolerance.

## 5. Run the canonical benchmark

```powershell
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_main.json `
  --experiment all

python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_sota.json `
  --experiment sota
```

The canonical matrix has:

- 140 main evaluations: 7 method conditions x 4 folds x 5 seeds;
- 80 label-efficiency evaluations: 2 methods x 3 budgets x 4 folds x 5 seeds,
  with the Full balanced view reusing exact main rows;
- 100 threshold evaluations: 5 thresholds x 4 folds x 5 seeds;
- 40 external adaptation rows: DRSN and TFN x 4 folds x 5 seeds.

The runner marks any dirty-worktree run exploratory. A confirmatory result must
record `git_dirty=false`.

Retained sensitivity training adds 120 CWT-control runs, 60 normalization runs,
40 coverage-controlled self-training runs, 80 baseline-tuning runs, 80
training-budget runs, and 80 calibration-recheck runs. These 460 rows are kept
separate from the 360 canonical and external rows. The threshold diagnostics
reuse 20 fitted base models to evaluate five thresholds and therefore add 100
selection cells, not 100 new training runs.

## 6. Rebuild canonical statistics

```powershell
python -m spindle_cnn.analysis.statistics `
  --results results/raw_runs.csv `
  --output-dir results

python -m spindle_cnn.analysis.hierarchical `
  --results-dir results `
  --external-dir results/sota `
  --output results/hierarchical_statistics.csv `
  --bootstrap-samples 20000

python scripts/build_statistical_reports.py
python -m spindle_cnn.analysis.report --results-dir results
python -m spindle_cnn.analysis.sota --results-dir results --external-dir results/sota
python -m spindle_cnn.analysis.recheck_run --results-dir results --all
python -m spindle_cnn.analysis.recheck_run --results-dir results/sota --all
```

The primary interval resamples four operating-condition folds first and seeds
within each selected fold. It does not treat the 20 fold-seed evaluations as 20
independent domains. The exact four-fold sign-flip test has a minimum nonzero
two-sided p-value of 0.125.

## 7. Rebuild domain and label diagnostics

```powershell
python -m spindle_cnn.analysis.protocol_audit --run-numeric-audit
python -m spindle_cnn.analysis.domain_shift
python -m spindle_cnn.analysis.damage_origin
python -m spindle_cnn.analysis.label_granularity
python -m spindle_cnn.analysis.calibration_audit
python -m spindle_cnn.analysis.training_sanity
python -m spindle_cnn.analysis.foldwise
python -m spindle_cnn.analysis.threshold_diagnostics
python -m spindle_cnn.analysis.cost_audit
```

Post-hoc class labels are used only for metrics and visualization after model
fitting. They do not enter preprocessing, pseudo-label selection, calibration,
epoch selection, or hyperparameter choice.

## 8. Run the CWT sensitivity controls

```powershell
python -m spindle_cnn.public_benchmark.cwt_sensitivity `
  --variants no_max_norm,log_magnitude,resolution_128,scales_32,scales_64,real_imag `
  --workers 4
python scripts/build_cwt_cache_manifest.py

python -m spindle_cnn.experiments.run_cwt_sensitivity `
  --variants no_max_norm,log_magnitude,resolution_128,scales_32,scales_64,real_imag `
  --allow-dirty

python -m spindle_cnn.analysis.cwt_sensitivity
```

The canonical 48-scale, 96 x 96 magnitude CWT remains the primary
representation. Controls are reported in full; the best test score does not
replace the primary representation.

## 9. Run the training-statistic normalization sensitivity

The experiment recomputes channel-wise mean and standard deviation from each
fold's training partition, applies those statistics to all partitions, and
rebuilds the CWT maps. It does not use validation or test statistics.

```powershell
python -m spindle_cnn.experiments.run_normalization_sensitivity --allow-dirty
python -m spindle_cnn.analysis.normalization_sensitivity
```

## 10. Run the stronger self-training baseline

```powershell
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_self_training_sensitivity.json `
  --experiment label_efficiency `
  --allow-dirty

python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_self_training_sensitivity.json `
  --experiment main `
  --methods cwt_gru_selftrain_coverage `
  --allow-dirty

python -m spindle_cnn.analysis.self_training
```

The stronger baseline chooses the top 50% highest-confidence samples within each
predicted class. It saves per-class selection counts and offline pseudo-label
accuracy. Test labels never enter selection.

## 11. Run baseline fairness and training-budget audits

```powershell
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_baseline_tuning_lr001.json `
  --experiment main `
  --output-dir results/baseline_tuning `
  --allow-dirty

python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_training_budget_30.json `
  --experiment main `
  --output-dir results/training_budget_30 `
  --allow-dirty

python -m spindle_cnn.analysis.baseline_tuning
python -m spindle_cnn.analysis.training_budget
```

Baseline configurations are selected by mean validation Macro-F1 within each
fold. Training-budget selection compares the validation outcome of 12 and 30
epochs over matched seeds.

## 12. Calibration and feature-space replication

```powershell
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_main.json `
  --experiment main `
  --methods raw_cnn,cwt_cnn,cwt_gru,cwt_gru_selftrain `
  --output-dir results/calibration_recheck `
  --allow-dirty

python -m spindle_cnn.analysis.calibration_audit
python -m spindle_cnn.analysis.feature_space
```

The calibration recheck retains validation logits that were not present in the
original per-run archives. The feature analysis trains one CWT-GRU per fold with
seed 42 and applies PCA/t-SNE only after fitting.

## 13. Build all audit artifacts

```powershell
python -m spindle_cnn.experiments.make_method_figures --output-dir figures/publication
python -m spindle_cnn.experiments.make_figures --results-dir results --output-dir figures/publication
python scripts/build_experiment_registry.py
python scripts/audit_sensitivity_results.py
python scripts/hash_publication_figures.py
python scripts/build_paper_metrics.py
python -m spindle_cnn.paper.build_docx `
  --results-dir results `
  --figures-dir figures/publication `
  --output docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx
python scripts/audit_submission.py --json-output artifacts/local/pre_submission_audit.json
python scripts/verify_manuscript_numbers.py
python scripts/audit_final_manuscript.py
python -m pytest -q tests
```

## Output integrity rules

- Keep every failed run; do not delete stochastic outliers.
- Do not select seeds, folds, thresholds, normalization, or CWT settings using
  test metrics.
- Preserve the exact `(fold, seed)` key when forming paired comparisons.
- Keep `320 + 40` canonical rows distinct from sensitivity runs.
- Retain hashes for every figure, table source, and manuscript result.

## 14. Clean source-snapshot smoke test

When the repository cannot be pushed to the public remote before final testing,
the following command creates an isolated snapshot from the exact current
working tree. It copies only non-ignored source and result files, plus the
canonical cache when requested.

```powershell
python scripts/prepare_clean_snapshot.py `
  --output artifacts/local/clean_source `
  --include-canonical-cache

Set-Location artifacts/local/clean_source
git init
git add --all
git -c user.name="Clean Snapshot" `
  -c user.email="clean-snapshot@users.noreply.github.com" `
  commit -m "Clean source snapshot"
Set-Location ../..

git -c core.longpaths=true clone `
  --no-hardlinks artifacts/local/clean_source `
  artifacts/local/clean_clone
Set-Location artifacts/local/clean_clone
New-Item -ItemType Directory -Force `
  -Path data\public\paderborn | Out-Null
Copy-Item `
  -LiteralPath ..\clean_source\data\public\paderborn\pu_4class_w8192.npz `
  -Destination data\public\paderborn\pu_4class_w8192.npz
Copy-Item `
  -LiteralPath ..\clean_source\data\public\paderborn\pu_4class_w8192.json `
  -Destination data\public\paderborn\pu_4class_w8192.json
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
python -m pytest -q tests
python scripts/build_paper_metrics.py
python -m spindle_cnn.paper.build_docx `
  --results-dir results `
  --figures-dir figures/publication `
  --output docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx
python scripts/verify_manuscript_numbers.py
```

Use `core.longpaths=true` on Windows when the clone path is deep. The final
DOCX is intentionally ignored as a local paper binary, so a clean clone must
rebuild it from the retained results before the manuscript audit. The
pre-revision backup and author-side rendered pages are optional archival
evidence for `audit_final_manuscript.py`; they are not required for the
independent numerical audit above.

The final public-repository check must additionally start from a fresh clone of
the pushed commit. A local snapshot validates the exact files but cannot verify
that the remote contains those files.
