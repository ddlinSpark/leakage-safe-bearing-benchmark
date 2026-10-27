# Code and Experiment Audit

> Historical audit snapshot. Current release status is maintained in
> `docs/audit/REPOSITORY_AUDIT.md` and `docs/audit/RELEASE_READINESS.md`.

Date: 2026-10-27  
Repository: `spindle_cnn`  
Audited branch: `codex/experiment-code`  
Audited commit: `7a3c63a`  
Scope: leakage-safe Paderborn benchmark plus legacy industrial-data pipeline

## 1. Executive Summary

The active Paderborn benchmark has a sound high-level split: test bearing
identifiers are excluded from training and validation, each fold withholds an
operating condition, the tested windows are non-overlapping, and pseudo-label
candidates are restricted to the unlabeled portion of the training partition.
No direct use of test data for fitting, early stopping, temperature fitting, or
pseudo-label generation was found.

The project is not yet ready to satisfy the requested SCI reproducibility
standard. The main comparison contains four folds and three seeds
(`12 runs/model`), not four folds and five seeds (`20 runs/model`). The 10% and
50% label-budget results also contain only one seed per fold, and the threshold
sweep contains only one seed. Per-run values for training loss, elapsed training
time, inference latency, parameter count, configuration, environment, and Git
commit are not stored. There is no model interface, no separate testing entry
point, no baseline registry, no automated statistical-analysis CLI, and no test
suite.

The principal experiment numbers in the current manuscript are traceable to
`docs/results/paderborn/results_extended.csv` and its generated summaries.
However, the exact main-run configuration cannot be reconstructed from a
snapshot, and the command in `docs/PUBLIC_BENCHMARK.md` uses different defaults
from the runs that produced the manuscript. The current evidence is therefore
numerically consistent but not fully reproducible from the repository alone.

The legacy industrial pipeline must remain historical. It uses threshold-derived
labels, overlapping windows, and the same station-level operating identifiers in
all splits; one legacy self-training script also relabels train, validation, and
test data. Those results must not re-enter the SCI confirmatory benchmark.

## 2. Current Repository Structure

Tracked and relevant local structure:

```text
spindle_cnn/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── docs/audit/AUDIT_REPORT.md
├── artifacts/                    # ignored local archives and review assets
├── data/                         # ignored raw and cached public data
├── dataset_final/                # ignored legacy industrial-data products
├── docs/
│   ├── PUBLIC_BENCHMARK.md
│   ├── REPRODUCIBILITY.md
│   ├── SCIENTIFIC_REVISION.md
│   ├── DATASET.md
│   ├── EXPERIMENTS.md
│   ├── ARTIFACTS.md
│   ├── paper/                    # DOCX ignored; README tracked
│   ├── references/               # PDFs ignored; README tracked
│   └── results/paderborn/        # compact audited CSV/JSON outputs
├── outputs/                      # ignored runtime outputs
├── scripts/                      # legacy miscellaneous scripts
└── src/spindle_cnn/
    ├── data/                     # legacy raw-data cleaning and windowing
    ├── final/                    # legacy industrial training/calibration
    ├── public_benchmark/         # active CWRU/Paderborn benchmark
    ├── baselines/                # legacy exploratory baselines
    ├── reporting/                # legacy report composition
    ├── document_tools/           # manuscript generation
    ├── environment/              # GPU/environment checks
    └── paths.py
```

The active benchmark implementation is concentrated in
`src/spindle_cnn/public_benchmark/`. The folders `data/`, `final/`,
`baselines/`, `reporting/`, and `document_tools/` support legacy experiments or
manuscript assembly and are not part of the leakage-safe public-data result.

## 3. Requested Component Map

| Item | Current location | Assessment |
|---|---|---|
| Data preprocessing | `public_benchmark/pu_data.py`, `public_benchmark/cwru_data.py`, `public_benchmark/prepare_paderborn.py` | Paderborn and CWRU caches include standardization and CWT generation. |
| Dataset splitting | `public_benchmark/train_cwru.py::load_fold_indices`; PU identifiers in `pu_data.py` | Correct high-level grouping, but no persisted split manifest per run. |
| Raw 1D-CNN | `public_benchmark/models.py::RawCNN` | Available in the unified model factory. |
| CWT 2D-CNN | `public_benchmark/models.py::CWT2DCNN` | Available in the unified model factory. |
| GRU-only | `public_benchmark/models.py::RawGRU` | Available in the unified model factory. |
| CWT-GRU | `public_benchmark/models.py::CWTGRU` | Available in the unified model factory. |
| SE attention | `public_benchmark/models.py::SEBlock`; inserted by `ResidualBlock` and `ImageEncoder` | Can be disabled with `use_se=False`. |
| Temperature scaling | `public_benchmark/train_cwru.py::fit_temperature`, `metrics_from_logits` | Implemented and fitted only on validation logits. It is a metric/calibration condition, not a separate registered method. |
| Pseudo-label self-training | `public_benchmark/train_cwru.py::select_pseudo_labels`, `run_self_training` | Restricted to the unlabeled training partition. |
| Training entry | `python -m spindle_cnn.public_benchmark.train_cwru` | Single entry handles model construction, training, testing, and result writing. |
| Testing entry | `train_cwru.py::evaluate_model` | No independent CLI or reusable evaluator module. |
| Evaluation metrics | `train_cwru.py::metrics_from_logits`, `expected_calibration_error` | Accuracy, macro precision/recall/F1, and ECE are present. |
| Random seeds | `train_cwru.py::set_seed`; CLI default `42,43,44` | Python, NumPy, PyTorch, and CUDA seeds are set. Current default does not meet the five-seed requirement. |
| Logging | Console output and incremental CSV append | No durable text log, per-epoch history, effective config snapshot, or commit/environment record in the main runner. |
| Result storage | `outputs/<run>/results.csv`, `summary.csv`; compact copies in `docs/results/paderborn/` | Works for current artifacts but does not match the requested canonical `results/raw_runs.csv` layout. |
| Figure generation | `public_benchmark/make_figures.py` | Main comparison, label efficiency, calibration, confusion matrix, and threshold figures. No dedicated ablation or computational-cost figure. |
| Complexity analysis | `public_benchmark/benchmark_models.py` | Parameters, FLOPs, training-step latency, and inference latency are measured separately. |
| Statistics | `public_benchmark/analyze_results.py` | Mean, standard deviation, 95% CI, paired deltas, and Wilcoxon tests exist, but only as an informal script and only for selected Full balanced budget comparisons. |
| Environment configuration | `pyproject.toml`, `requirements.txt` | No pinned lockfile, no `environment.yml`, and no recorded runtime environment per run. |

## 4. Existing Capabilities

- Public Paderborn and CWRU preprocessing is scriptable and has explicit
  metadata for class and bearing mappings.
- The Paderborn cache stores raw waveform, CWT tensor, label, operating
  condition, bearing-code-derived record ID, and window start index.
- The benchmark contains six implemented models: Raw 1D-CNN, GRU-only,
  CWT 2D-CNN, CWT-GRU without SE, CWT-GRU with SE, and CWT-GRU with
  calibrated self-training.
- Training and validation selection use validation macro-F1 and early stopping.
- Temperature is fitted from validation logits by minimizing cross-entropy.
- Self-training uses only candidates from `train_indices - labeled_indices`.
  Neither validation nor test data is passed to `select_pseudo_labels`.
- Every prediction is saved as an NPZ file containing logits, labels, and
  temperature.
- The project computes macro-F1, accuracy, macro precision/recall, and ECE.
- Paired outputs can be formed on identical `fold + seed` keys.
- Compact result CSV files are tracked while raw data, weights, large caches, and
  paper binaries are excluded from Git.

## 5. Data-Leakage Assessment

### 5.1 Public Paderborn benchmark

| Check | Finding | Status |
|---|---|---|
| Overlapping windows | `window_size=8192`, `stride=8192` in the audited cache. | Pass |
| Bearing identity across train/test | Test IDs `4,8,12,15` map to `K005, KA15, KI14, KB27`; these IDs are removed before train/validation construction. | Pass |
| Operating condition across train/test | The test condition equals the fold ID and is excluded from train/validation. | Pass |
| Validation/test used for pseudo-labels | Candidate indices are taken only from the unlabeled training partition. | Pass |
| Test data used for early stopping or temperature | Validation indices and validation labels are used; test indices are not passed to either operation. | Pass |
| Test data used for threshold choice | The main threshold is fixed at `0.60`. The threshold sweep evaluates test results across fixed thresholds but does not retroactively tune the main result. | Pass with documentation caveat |
| All models use the same split | All methods use the same fold construction and the same `seed + 1000 * fold` label-sampling RNG. | Pass at current code level |
| Split membership persisted | No CSV/JSON manifest of the exact train/validation/test and labeled/unlabeled indices is saved per run. | Gap |

Important labels are still used to audit pseudo-label accuracy in the threshold
sweep. This is valid only as post-hoc evaluation and must never feed model
selection, threshold choice, or training. The current report identifies this
distinction, but the code should enforce it structurally.

### 5.2 CWRU benchmark

The CWRU experiment holds out load conditions and uses non-overlapping windows.
It does not hold out physical bearing identity because the official files are
not organized as the same bearing-unit grouping used by Paderborn. Therefore,
CWRU must remain a saturation or pilot check and must not be described as a
bearing-code-disjoint confirmation study.

### 5.3 Legacy industrial pipeline

The following high-risk behaviors remain in legacy scripts:

- `src/spindle_cnn/data/build_dataset_from_raw.py` uses a default stride of 500
  for a window size of 1000, producing 50% overlap.
- The same station-level `op` values appear in train, validation, and test.
- Labels are assigned from engineering thresholds and normal/abnormal ratios.
- `src/spindle_cnn/final/self_train_relabel.py` performs inference on all
  `train`, `val`, and `test` splits and constructs replacement labels across all
  three splits.
- The historical self-training result therefore contains circular-label and
  test-label exposure risks and cannot support a confirmatory accuracy claim.

No finding indicates that the active Paderborn benchmark calls these legacy
relabeling paths.

## 6. Repetition and Experimental-Design Audit

Current artifact counts:

| Experiment | Current design | Required minimum | Status |
|---|---:|---:|---|
| Full balanced budget main comparison | 4 folds × 3 seeds = 12 runs/model | 4 × 5 = 20 | Incomplete |
| GRU-only and no-SE ablations | 4 × 3 = 12 | 20 | Incomplete |
| 10% label budget | 4 folds × 1 seed = 4 | 20 | Incomplete |
| 50% label budget | 4 × 1 = 4 | 20 | Incomplete |
| Threshold sweep | 4 folds × 1 seed × 5 thresholds = 20 conditions; each threshold has 4 runs | 5 seeds per threshold, preferably 20 runs | Incomplete |

The five requested seeds should be:

```text
42,123,2024,2025,3407
```

The data split must remain unchanged while seeds are added. New runs must not
replace or reinterpret the existing 12-run evidence. Existing and new results
should be kept distinguishable, with the five-seed set serving as the revised
confirmatory experiment.

### 6.1 Meaning of the Full balanced budget

The current 100% setting is not all training windows. It means:

1. Find the smallest class in the training partition.
2. Sample `ceil(smallest_class_count × fraction)` windows from every class.
3. Use these balanced windows as the labeled set.
4. Keep all remaining training windows as pseudo-label candidates.

For Paderborn, the resulting Full balanced budget set contains about 992 or 996 windows,
not the entire training partition. The manuscript and future CSV schema should
call this **100% of the class-balanced label budget**, or report the exact number
of labeled windows and the total training-window count. Calling it “all labels”
would be inaccurate.

### 6.2 Temperature scaling

Temperature scaling is implemented correctly as a monotonic probability
transformation, but it is not an independently selectable model in the current
runner. Every method has calibrated metrics computed after validation fitting.
The requested condition “CWT-GRU + Temperature Scaling” should be represented
as an explicit evaluation condition while reusing the same trained logits,
without retraining a fake model.

### 6.3 Training and evaluation isolation

- Early stopping uses validation macro-F1.
- Temperature is fitted after early stopping on the same validation split.
- The validation split is not disjoint from the early-stopping data. This is not
  test leakage, but calibration may be optimistic and this should be reported.
- The primary threshold `0.60` is fixed and is not selected from test results.
- Pseudo-label accuracy in threshold analysis is an audit metric only.
- Test labels are read after all fitting is complete.

## 7. Paper-to-Artifact Numerical Consistency

The following manuscript values match the current compact result artifacts:

| Method | Accuracy | Macro-F1 | Macro-F1 SD |
|---|---:|---:|---:|
| Raw 1D-CNN | 0.409 | 0.308 | 0.072 |
| CWT 2D-CNN | 0.400 | 0.291 | 0.088 |
| CWT-GRU | 0.391 | 0.284 | 0.106 |
| CWT-GRU + calibrated self-training | 0.350 | 0.240 | 0.103 |

The validation macro-F1 values in the manuscript
(`0.861`, `0.816`, `0.823`) and the paired comparison
`CWT-GRU - Raw CNN = -0.024`, `p = 0.622` also match the CSV.

The threshold means match:

| Threshold | Macro-F1 |
|---:|---:|
| 0.3 | 0.293 |
| 0.4 | 0.345 |
| 0.5 | 0.290 |
| 0.6 | 0.257 |
| 0.7 | 0.242 |

The model complexity values also match the tracked CSV.

These checks show that the manuscript does not appear to contain hand-invented
values for the main table. They do not prove end-to-end regeneration: the main
output directory lacks a `run_config.json`, and the documented command does not
state the epoch and patience values used by the existing runs.

## 8. Reproducibility Gaps

1. The main final-output directory has no config snapshot.
2. Results are keyed only by dataset, fold, method, fraction, and seed. A changed
   learning rate or architecture can therefore be skipped or pooled with
   incompatible runs during resume.
3. No commit hash, dependency version, command, host/device, or config hash is
   stored with each result.
4. No exact labeled/unlabeled/test manifest is persisted.
5. No per-epoch train loss is saved in the public benchmark runner.
6. No per-run training duration or inference duration is saved.
7. No per-run parameter count is saved; complexity is measured separately.
8. No random seed is stored in the threshold output summary apart from raw rows.
9. No automated confidence-interval and paired-test CLI is exposed.
10. No `LICENSE`, `environment.yml`, or `data/README.md` exists.
11. No tests validate non-overlap, exclusion of held-out groups, or the absence
    of validation/test indices from pseudo-label candidates.
12. No Git remote is configured, so a public GitHub URL cannot yet be created.

## 9. Required Experiments Before the Revised Paper Is Final

1. Re-run all six Full balanced budget architectures with
   `4 folds × 5 seeds = 20 runs/model`.
2. Re-run 10%, 50%, and 100% balanced-budget supervised CWT-GRU and calibrated
   self-training with the same 20 run keys.
3. Re-run every threshold (`0.3–0.7`) with all five seeds and the same folds.
4. Preserve the current 12-run results as legacy/exploratory artifacts; do not
   overwrite them silently.
5. Generate component-wise ablation summaries from the new 20-run set.
6. Generate all confidence intervals and paired tests from the 20 keys.
7. Recompute calibration ECE and decide whether NLL or Brier score should be
   added as a proper scoring rule.
8. Only add a SOTA baseline after its architecture, preprocessing, optimization,
   and split can be reproduced under the same protocol.
9. Regenerate paper tables and figures only from the new canonical CSV files.

## 10. Recommended Refactor

The existing public benchmark should be retained as the numerical reference and
refactored incrementally rather than replaced with unreviewed model code.

1. Introduce a small `models/baselines/` package with a common `BaseModel`
   contract and register the six existing architectures.
2. Move metric and calibration logic into `evaluation/`.
3. Separate split construction into `datasets/splits.py`; persist an immutable
   split manifest per fold.
4. Move training into `training/runner.py` and preserve the current model
   definitions exactly.
5. Add a canonical raw-run schema and unique key:
   `dataset + protocol + split_version + fold + method + label_ratio + seed`.
6. Add per-run `config.json`, `command.txt`, `environment.json`, `split.json`,
   `metrics.json`, predictions, and a config/commit hash.
7. Refuse resume when the current config hash or split version differs from
   the existing result rows.
8. Add controlled ablation and threshold runners that call the same training
   and evaluation functions.
9. Add `scripts/statistical_analysis.py` with explicit paired keys, CI, and
   Wilcoxon reporting.
10. Add tests for non-overlap, bearing/condition isolation, label-budget
    semantics, pseudo-label provenance, and deterministic seed reuse.
11. Add GitHub-ready `LICENSE`, `environment.yml`, `data/README.md`, configs,
    and reproducible command documentation.
12. Do not modify the manuscript before the new five-seed results and
    statistical summaries pass audit.

## 11. Audit Decision

- **Leakage-safe public benchmark claim:** supported by the current code and
  cache metadata.
- **Current manuscript numbers traceable to CSV:** supported.
- **Current manuscript numbers fully reproducible from documented commands:**
  not yet supported.
- **Five independent repetitions:** not yet satisfied.
- **Complete model/ablation/label-budget/threshold experiment matrix:** not yet
  satisfied.
- **Public GitHub reproducibility package:** not yet complete; no remote is
  configured.

## 12. Post-Audit Remediation Status

The code and experiment gaps identified above have been addressed through a
unified framework and a new confirmatory run.

- Canonical runner: `configs/paderborn_main.json` plus
  `spindle_cnn.experiments.run_benchmark`.
- Main comparison: 20 successful runs for each of seven registered method
  conditions, including the temperature-scaling evaluation condition.
- Label efficiency: 20 runs for each supervised/self-training ×
  `10%/50%/100%` condition, with the 100% view reusing the exact main
  `fold + seed` rows.
- Threshold sensitivity: 20 runs for each threshold from `0.30` through `0.70`.
- Total canonical successful rows: 320.
- Failed rows: 0 in the completed matrix.
- Per-run provenance: configuration, environment, dataset hash, split hash,
  Git commit, training history, calibration bins, and confusion matrix.
- Automated outputs: `results/EXPERIMENT_REPORT.md`, CSVs with mean ± standard
  deviation and paired tests, and seven figures under `figures/publication/`.

The new five-seed results supersede the manuscript's old three-seed numbers.
The manuscript has deliberately not been rewritten in this phase; its tables and
text must be updated from the canonical CSV files rather than from the previous
draft. A Git remote is still absent, so a public GitHub push cannot be completed
from this checkout until a repository URL is supplied.

After finalization, the superseded working-tree artifacts listed in this audit
were removed: the legacy industrial code/data, old model outputs, exploratory
archives, old figures, and the old three-seed CSV directory. The audit record is
retained, and the superseded files remain recoverable from Git history.

