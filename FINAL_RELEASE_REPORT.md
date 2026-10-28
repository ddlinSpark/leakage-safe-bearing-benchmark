# Final Release Report

## 1. Changes made

- Added `src/spindle_cnn/plotting/style.py` as the single publication palette
  and style definition and routed all 12 figure-generation modules through it.
- Regenerated every publication figure from retained CSV, JSON, prediction, or
  compact embedding files without changing an experimental CSV value or
  selecting a configuration using test performance.
- Added the tracked `figures/publication/figure_hashes.json` manifest and
  automated palette, DPI, PNG/PDF pairing, and hash checks.
- Added `scripts/check_manuscript_consistency.py`, which checks 320 canonical
  rows, 40 external rows, the 820-row experiment registry, model matrices,
  means, parameter counts, folds, seeds, hierarchical intervals, and the DOCX
  table audit.
- Rebuilt the final DOCX and rendered the 30-page A4 manuscript PDF. Added a
  five-page standalone supplementary PDF containing Figures S1-S5.
- Retained four compact feature-embedding files for deterministic PCA/t-SNE
  reconstruction without another model training run.
- Added `.gitattributes` binary handling for document/image artifacts and LF
  handling for text, CSV, and JSON outputs.
- Updated README, reproducibility, figure-source, and data/code availability
  documentation with the actual commands and output locations.

## 2. Figure update

All 24 PNG/PDF pairs under `figures/publication/` were regenerated:

| Figure group | Files |
| --- | --- |
| Method diagrams 1-3 | `figure_method_01_overall_framework`, `figure_method_02_cwt_gru_architecture`, `figure_method_03_self_training_pipeline` |
| Main and diagnostic figures 4-12 | `figure_01_model_comparison`, `figure_03_label_efficiency`, `figure_04_threshold_sensitivity`, `figure_05_calibration`, `figure_08_reliability_diagrams`, `figure_02_ablation`, `figure_06_confusion_matrices`, `figure_07_computational_cost`, `figure_09_validation_test_gap` |
| Additional manuscript figures 13-18 | `figure_13_domain_shift_diagnostics`, `figure_16_label_granularity`, `figure_19_cwt_sensitivity`, `figure_17_self_training_sensitivity`, `figure_22_feature_space_pca`, `figure_23_feature_space_tsne` |
| Supplementary figures S1-S5 | `figure_14_per_fold_reliability`, `figure_15_temperature_distribution`, `figure_20_foldwise_paired_differences`, `figure_21_training_budget`, `figure_24_normalization_sensitivity` |
| Retained analysis figure | `figure_18_baseline_tuning` |

The manuscript embeds 23 figures: Figures 1-18 and S1-S5. The baseline-tuning
diagnostic is retained as an analysis artifact but is not one of the 23 inline
figures. Existing descriptive filenames were preserved to avoid breaking
manuscript, audit, and reproduction paths; manuscript-number mapping remains in
`docs/paper/FIGURE_SOURCES.md`.

## 3. Palette verification

**PASS.**

`src/spindle_cnn/plotting/style.py` defines exactly:

```text
#7B95C6  #49C2D9  #A1D8E8  #67A583  #A2C986  #D0E2C0
#FDED95  #FFC1A6  #F59C7C  #F47254  #C85E62
```

Fixed model colors:

| Model | Color |
| --- | --- |
| Raw 1D-CNN | `#7B95C6` |
| GRU only | `#49C2D9` |
| CWT 2D-CNN | `#67A583` |
| CWT-GRU without SE | `#A2C986` |
| CWT-GRU | `#FDED95` |
| CWT-GRU + temperature | `#FFC1A6` |
| CWT-GRU + self-training | `#F47254` |
| DRSN adaptation | `#C85E62` |
| TFN adaptation | `#F59C7C` |
| Auxiliary/reference | `#A1D8E8`, `#D0E2C0` |

Checks:

- 12 of 12 plotting modules call the shared style.
- No plotting module contains a local hexadecimal data color outside the
  approved palette.
- 24 PNG/PDF pairs exist.
- All 24 PNGs have DPI metadata of at least 300.
- 48 of 48 figure files match their tracked SHA-256 manifest.
- All 30 manuscript pages and all 24 figure pairs were visually inspected.

## 4. Result consistency

**PASS.**

- Canonical rows: 320 (`140` main, `80` label-efficiency, `100` threshold).
- External adaptation rows: 40 (`20` DRSN, `20` TFN).
- Consolidated experiment registry: 820 rows.
- Main matrix: seven methods x four folds x five seeds.
- Seeds: `42, 123, 2024, 2025, 3407`.
- Independent DOCX/table audit: 0 mismatches across 18 tables and 23 figures.
- Evidence-chain check: 15 of 15 checks passed.
- Figure-only rebuild: 24 of 24 PNGs byte-identical.
- Vector figure rendering: 24 of 24 PDFs visually identical.

The principal result remains a leakage-safe negative/boundary finding rather
than a new high-scoring architecture:

| Model | Test Macro-F1 |
| --- | ---: |
| Raw 1D-CNN | 0.289 |
| CWT 2D-CNN | 0.291 |
| CWT-GRU without SE | 0.307 |
| CWT-GRU | 0.280 |

No sensitivity result replaces the primary configuration after observation of
test data.

## 5. Reproduction status

A fresh clone from the named GitHub remote at commit
`9006d56085773acc2a56d60cb77933cbbf4d73d1` passed:

- 33 automated tests;
- configuration, split-manifest, and all-model initialization checks;
- full 3,939-window preprocessing from official raw measurements;
- exact canonical NPZ SHA-256
  `a898abcb63e17ee68f2014d19b4d0f675b6c4d19a9abc89376031a78c3a23df4`;
- Raw fold 0 / seed 42 training at test Macro-F1 `0.3205054825882636`;
- seven byte-identical statistics outputs, including the 20,000-sample
  hierarchical bootstrap;
- byte-identical reconstruction of all 24 PNG figures;
- content-identical reconstruction of the 41-entry final DOCX;
- a 30-page manuscript PDF and five-page supplementary PDF.

The full 820-row stochastic matrix was not rerun during the clean-room pass.
Logged metrics for all 320 canonical and 40 external rows were independently
recomputed from predictions. Details are in `CLEAN_REPRODUCTION_REPORT.md`.

## 6. Repository structure

```text
leakage-safe-bearing-benchmark/
├── .gitattributes
├── .gitignore
├── README.md
├── REPRODUCIBILITY.md
├── CLEAN_REPRODUCTION_REPORT.md
├── CONSISTENCY_REPORT.md
├── FINAL_RELEASE_REPORT.md
├── configs/
├── data/manifests/
├── docs/
│   ├── audit/
│   └── paper/
├── figures/publication/
│   ├── 24 PNG figures
│   ├── 24 vector PDF figures
│   └── figure_hashes.json
├── manuscript/
│   ├── manuscript.pdf
│   ├── supplementary.pdf
│   └── README.md
├── results/
│   ├── canonical result CSV/JSON
│   ├── split manifests and per-run evidence
│   ├── feature_embeddings/
│   └── sensitivity result families
├── scripts/
├── src/spindle_cnn/
│   ├── analysis/
│   ├── experiments/
│   ├── models/
│   ├── paper/
│   ├── plotting/
│   ├── public_benchmark/
│   └── training/
└── tests/
```

The existing import namespace and result layout were preserved so that the
final packaging pass would not break run identifiers, configuration paths, or
the manuscript builder.

## 7. Files excluded from Git

- Paderborn raw archives and extracted `.mat` files.
- Canonical and original-amplitude NPZ caches.
- CWT sensitivity and normalization caches.
- Model weights and checkpoints.
- Local clean-clone directories.
- Page-render PNGs and accessibility/audit scratch JSON under
  `artifacts/local/`.
- Temporary logs, Python bytecode, pytest caches, and IDE files.
- Pre-revision or superseded local manuscript binaries in the archive.

The exact official archive names, byte sizes, and SHA-256 values are tracked in
`data/manifests/paderborn_archives_sha256.csv`.

## 8. Remaining warnings

- The empirical benchmark covers one Paderborn dataset. It does not establish
  universal transferability across all bearing datasets.
- Bearing identity, operating condition, and damage origin remain coupled in
  the strict holdout. No causal decomposition is claimed.
- The clean-room pass fully reran one fold-seed training job, not all 820
  stochastic rows.
- PDF container hashes vary with creation metadata; visual renders and PNG
  hashes are used for figure-content equality.
- The public repository currently exposes named ownership while the manuscript
  is anonymous for review. Existing history also contains non-anonymous author
  metadata and retained workstation hostname fields. Resolving these privacy
  items requires explicit decisions about repository visibility and history.
