# Clean Reproduction Report

## Verdict

The final publication package was cloned from the named public GitHub remote
and rerun in an isolated directory. The tested commit is
`9006d56085773acc2a56d60cb77933cbbf4d73d1`.

- Fresh GitHub clone: **PASS**
- Clean checkout before execution: **PASS**
- Import and full test suite: **PASS**, 33 tests
- Canonical config, split manifests, and model initialization: **PASS**
- Paderborn preprocessing from original measurements: **PASS**
- Canonical cache equality: **PASS**, exact SHA-256 and array equality
- One complete fold-seed training run: **PASS**
- Statistics rebuild: **PASS**, seven key CSV files byte-identical
- Figure rebuild: **PASS**, 24 PNG files byte-identical
- Vector PDF figure rendering: **PASS**, 24 of 24 visual matches
- DOCX rebuild: **PASS**, 41 of 41 package entries content-identical
- Manuscript table audit: **PASS**, 0 mismatches
- Main and supplementary PDF rendering: **PASS**, 30 and 5 A4 pages
- Full 820-row retraining: **NOT REPEATED**

The final status is a clean-room validation of the source, preprocessing,
training path, statistics, figure generation, manuscript generation, and PDF
assembly. It is not a second full run of every stochastic sensitivity result.

## Clone provenance

| Item | Value |
| --- | --- |
| Remote | `https://github.com/ddlinSpark/leakage-safe-bearing-benchmark.git` |
| Branch | `main` |
| Tested commit | `9006d56085773acc2a56d60cb77933cbbf4d73d1` |
| Isolated directory | `D:\lsbbl_clean_9006d56` |
| Clone command | `git -c core.longpaths=true clone <remote> <directory>` |
| Checkout state before execution | clean |
| Recorded smoke-run `git_dirty` | `False` |

The final report documents this tested code revision. The subsequent
documentation and line-ending metadata commit does not change the experimental
code, result files, figures, or manuscript content.

The repository does not redistribute the Paderborn archives or canonical
605 MB cache. The preprocessing audit used the checksum-verified official
archives restored outside Git. Raw data, cache files, weights, temporary
render pages, and local clean-clone directories remain ignored.

## Environment

| Component | Recorded value |
| --- | --- |
| Operating system | Windows 11 |
| Python | 3.13.5 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| cuDNN | 91900 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |
| PyWavelets | 1.8.0 |
| Device | NVIDIA GeForce RTX 5080 |
| LibreOffice | 26.8.1.1 |

## 1. Source and load validation

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
python scripts/check_figure_palette.py
python scripts/check_manuscript_consistency.py
```

Results:

```text
33 passed
Palette/figure integrity: PASS
Evidence chain: 15/15 checks passed
```

The canonical configuration loads four folds, seeds
`42, 123, 2024, 2025, 3407`, and seven main methods. All four split manifests
exclude held-out record IDs `4, 8, 12, 15` from train and validation. All seven
model families initialize at the expected parameter counts.

## 2. Preprocessing

The clean clone ran:

```powershell
python -m spindle_cnn.public_benchmark.pu_data `
  --data-root <official-extracted-data> `
  --output data/public/paderborn/pu_4class_w8192_generated.npz `
  --window-size 8192 `
  --stride 8192 `
  --scales 48 `
  --image-size 96 `
  --max-measurements 2 `
  --workers 4
```

The run produced 3,939 windows. The generated NPZ had the same SHA-256 as the
recorded canonical cache:

```text
a898abcb63e17ee68f2014d19b4d0f675b6c4d19a9abc89376031a78c3a23df4
```

The arrays `raw`, `cwt`, `labels`, `loads`, `record_ids`, and `starts` were all
exactly equal, including shape and dtype. The known truncated official file
`N15_M01_F10_KA08_2.mat` was skipped by the same preprocessing path documented
in the audit.

## 3. Training smoke

```powershell
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_main.json `
  --experiment main `
  --output-dir artifacts/local/clean_smoke `
  --seeds 42 `
  --folds 0 `
  --methods raw_cnn `
  --no-resume
```

The clean run loaded 3,939 windows, used commit `9006d56`, recorded
`git_dirty=false`, and completed with test Macro-F1
`0.3205054825882636`.

All selected canonical and clean numeric fields agree within
`8.32667268468867e-17`. Only three CSV decimal representations differ at
machine precision: `raw_ece`, `raw_brier`, and `best_val_loss`. Data, split,
model, selected epoch, and predicted test metrics agree.

## 4. Statistics rebuild

The clean clone regenerated canonical summaries and the 20,000-sample
fold-first hierarchical bootstrap. These files were byte-identical to the
repository versions:

- `summary.csv`
- `label_efficiency_summary.csv`
- `ablation_summary.csv`
- `domain_gap_summary.csv`
- `threshold_summary.csv`
- `statistical_tests.csv`
- `hierarchical_statistics.csv`

Each comparison retains four folds, five seeds, 20 matched fold-seed pairs, and
the minimum attainable non-zero two-sided sign-flip p-value of `0.125`.

## 5. Figure rebuild

All figure generators ran from retained result files. The checks showed:

- 24 of 24 PNG files were byte-identical to the tracked publication figures.
- 24 of 24 PDF files produced visually identical first-page renders at 96 DPI.
- The retained feature embeddings rebuilt PCA/t-SNE without retraining.
- The shared palette check found no local hexadecimal data colors outside the
  centralized 11-color palette.

PDF container hashes are expected to vary because PDF metadata includes
creation time; visual equality and PNG equality provide the figure-content
checks.

## 6. Manuscript and PDF rebuild

The clean clone rebuilt the DOCX from committed result files and publication
figures:

```text
TRACKED_SIZE 6,682,738 bytes
REBUILT_SIZE 6,682,738 bytes
ENTRY_COUNT 41
CONTENT_MATCH 41 of 41
```

The independent manuscript audit on the rebuilt DOCX returned:

```text
status = PASS
mismatches = 0
tables = 18
figures = 23
```

The rebuilt DOCX rendered to 30 A4 pages. All 30 rendered page PNGs were
byte-identical to the author-workspace visual QA pages. The standalone
supplementary material rendered to five A4 pages containing Figures S1-S5.

## Reproduction scope

The clean-room run did not repeat all 820 retained training rows. It repeated
the complete preprocessing, test suite, model/configuration loading, one full
fold-seed training run, statistics, figures, DOCX, and PDFs. The numerical
audits independently recompute logged metrics for all 320 canonical and 40
external rows.

## Remaining release considerations

The repository remains public while the manuscript is anonymous for review.
Existing Git history contains non-anonymous author metadata and retained run
configurations contain workstation hostname fields. Resolving those privacy
items requires an explicit repository/history decision and is not solved by a
numerical reproduction run.
