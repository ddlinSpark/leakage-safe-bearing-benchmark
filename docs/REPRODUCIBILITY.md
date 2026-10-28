# Reproducibility

## Fixed protocol

The confirmatory Paderborn experiment uses the configuration in
`configs/paderborn_main.json`:

- Seeds: `42, 123, 2024, 2025, 3407`.
- Operating-condition folds: `0, 1, 2, 3`.
- Test bearing IDs: `4, 8, 12, 15`.
- Window length and stride: `8192` samples.
- Primary matrix: four folds by five seeds per method.
- Full balanced budget setting: the minimum available per-class training count,
  not all training windows.
- Pseudo-label threshold in the main self-training experiment: `0.60`.
- Threshold sensitivity: `0.30, 0.40, 0.50, 0.60, 0.70` at a 50% balanced
  budget.

## Formal entry points

Stable commands include `prepare_data.py`, `run_main_benchmark.py`,
`run_label_efficiency.py`, `run_threshold_sensitivity.py`,
`run_external_baselines.py`, `run_preprocessing_sensitivity.py`,
`run_training_opportunity.py`, `generate_statistics.py`,
`generate_figures.py`, `build_paper_metrics.py`, `reproduce_all.py`,
`smoke_test.py`, `audit_internal_links.py`, and the consistency/audit scripts
under `scripts/`.

`scripts/reproduce_all.py` validates retained results and rebuilds derived
statistics, figures, DOCX, and audits. Add `--run-benchmark` for a from-scratch
matrix and `--include-domain-shift` when the raw-amplitude cache is available.

## Expected outputs

| Output | Expected evidence |
| --- | --- |
| Split manifests | Four frozen fold JSON files with hashes |
| Primary rows | 320 successful canonical rows |
| External rows | 40 successful adaptation rows |
| Main summary | Seven method conditions x 20 matched runs |
| Hierarchical table | Fold-first intervals and exact sign-flip tests |
| Figures | 24 PNG/PDF pairs and a SHA-256 manifest |
| Manuscript | DOCX and 30-page PDF |
| Supplement | Five-page Figures S1-S5 PDF |
| Consistency report | Zero table/manuscript mismatches |
| Internal-link report | Resolved citations, cross-references, and PDF outline |
| Smoke test | Config, split, model, metric, and style checks pass |

## Commands

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

PYTHONPATH=src python -m spindle_cnn.analysis.sota \
  --results-dir results \
  --external-dir results/sota

PYTHONPATH=src python -m spindle_cnn.experiments.make_figures \
  --results-dir results \
  --output-dir figures/publication

PYTHONPATH=src python -m spindle_cnn.analysis.recheck_run \
  --results-dir results \
  --all

PYTHONPATH=src python -m spindle_cnn.analysis.recheck_run \
  --results-dir results/sota \
  --all
```

The `--experiment` choices are `main`, `ablation`, `label_efficiency`,
`threshold`, and `all`.

```bash
python scripts/run_main_benchmark.py --experiment all
python scripts/run_external_baselines.py
python scripts/generate_statistics.py
python scripts/generate_figures.py --include-domain-shift
python scripts/build_results_catalog.py
python scripts/audit_internal_links.py
```

The internal-link audit reads the final DOCX and PDF, extracts every citation
and cross-reference, resolves every PDF link annotation to a target page, and
writes `REFERENCE_LINK_REPORT.md`.

## Provenance

Each run writes:

```text
results/runs/<run_id>/
├── config.json
├── training_history.csv
├── predictions.npz
├── confusion_matrix.csv
└── calibration_bins.csv
```

`config.json` includes the command-line configuration, architecture key,
training settings, seed, fold, label budget, threshold, dataset SHA-256, split
SHA-256, Python/Torch/CUDA versions, Git commit, and dirty-worktree flag.

Exact split membership is stored in `results/splits/fold<k>.json`. Resume is
permitted only when both `run_id` and `config_hash` match. A configuration
change creates a separate row rather than silently pooling incompatible runs.

`results/runs/<run_id>/predictions.npz` contains test logits, labels, final
decisions, and the fitted temperature. The independent recheck command
recomputes all recorded probability metrics and fails if any value differs by
more than `1e-8`.

`results/canonical/manifest.json` and `splits/manifest.json` record the
authoritative result paths and split hashes. Refresh both indexes without
changing an experimental value:

```powershell
python scripts/build_results_catalog.py
```

## Determinism boundary

The runner seeds Python, NumPy, PyTorch, and CUDA; disables cuDNN autotuning;
requests deterministic kernels; and uses an explicit DataLoader generator. GPU
model, driver, CUDA, PyTorch version, and implementation differences can still
change floating-point results and latency. The environment snapshot for every
run is retained for this reason.

## Leakage controls

1. Held-out bearing IDs never enter training or validation.
2. The test operating condition is excluded from training and validation.
3. Windows do not overlap (`stride == window length`).
4. Validation labels are used only for model selection and temperature fitting.
5. Test labels are not used for training, early stopping, calibration, or
   pseudo-label selection.
6. Pseudo-label candidates are drawn only from the unlabeled training partition.
7. Test outcomes do not select the main confidence threshold.

## Legacy industrial pipeline

The industrial-data preprocessing, training, calibration, prediction, and
reporting modules were removed from the current working tree. They remain
recoverable from pre-cleanup Git history but are not part of the confirmatory
public benchmark. The former test-relabeling workflow must not be used to
produce SCI accuracy claims because it relabeled train, validation, and test
samples.
