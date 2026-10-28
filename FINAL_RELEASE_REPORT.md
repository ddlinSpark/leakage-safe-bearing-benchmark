# Final Release Report

## 1. Repository restructuring

The repository was converted from a development-oriented layout into a formal
paper reproducibility package without moving the trained result matrices or
changing the import namespace used by the verified runner.

The main changes are:

- a 250-line README focused on the benchmark question, key findings, protocol,
  experiment status, quick start, and limitations;
- dedicated `docs/DATASET.md`, `docs/PROTOCOL.md`, `docs/EXPERIMENTS.md`,
  `docs/RESULTS.md`, and `docs/REPRODUCIBILITY.md` pages;
- stable entry points under `scripts/` for data preparation, main/label/
  threshold/external/preprocessing/training-opportunity experiments,
  statistics, figures, full reproduction, smoke testing, and consistency;
- `results/canonical/manifest.json` as the formal canonical-result index;
- `splits/manifest.json` as a root-level index of the frozen split manifests;
- `src/spindle_cnn/visualization/style.py` as the shared plotting style;
- a GitHub Actions workflow for tests, smoke checks, manuscript consistency,
  figure integrity, and catalog stability.

Existing result and run paths were retained for compatibility. The canonical
manifest points to those verified paths rather than duplicating large result
families.

## 2. README changes

README was reduced from 435 lines to 250 lines.

It now provides:

- the official paper title and benchmark purpose;
- the simultaneous unseen-bearing and unseen-condition question;
- the required leakage-safe protocol constraints;
- a six-row Key Findings table matching canonical results;
- the validation-to-test generalization gap;
- an Experiment Status table using "completed" only for execution status;
- the explicit statement that not all methods produced positive or stable
  gains;
- concise repository structure and Quick Start commands;
- data availability and large-file exclusion rules;
- the exact reproducibility evidence locations;
- coupled-domain and cognitive-exposure limitations;
- citation and license information.

README no longer contains the full command atlas, per-sensitivity details, or
development history.

## 3. Files moved

The actual publication style implementation moved from:

```text
src/spindle_cnn/plotting/style.py
```

to:

```text
src/spindle_cnn/visualization/style.py
```

All plotting imports now target the visualization package.
`src/spindle_cnn/plotting/style.py` remains as a backward-compatible re-export
shim, so existing external imports do not break.

No canonical CSV, run directory, split manifest, figure, manuscript, or model
source file was moved.

## 4. Files deleted

No tracked experimental result, failed run, negative result, or split manifest
was deleted.

The obsolete root-level `scripts/generate_figures.py` wrapper was replaced
in-place by a full repository orchestration entry point. No tracked manuscript
version was removed by this release pass.

Local caches, old ignored drafts, logs, and temporary render evidence remain
outside the Git index.

## 5. Figure regeneration

All 24 PNG/PDF publication pairs were regenerated from retained files:

| Group | Figures |
| --- | --- |
| Method diagrams | Figure 1 framework, Figure 2 architecture, Figure 3 self-training pipeline |
| Main benchmark | Figures 4-12 |
| Sensitivity/diagnostics | Figures 13-18 |
| Supplementary | S1-S5 |
| Retained analysis figure | External-baseline tuning diagnostic |

The manuscript embeds 23 figures. The baseline-tuning diagnostic is retained as
an analysis artifact and is not one of the 23 inline manuscript figures.

The figure workflow can be run with:

```powershell
python scripts/generate_figures.py --include-domain-shift
```

Repeated complete generation produced 48 of 48 byte-identical PNG/PDF files.

## 6. Palette verification

**PASS.**

The exact 11-color palette is centralized in
`src/spindle_cnn/visualization/style.py`:

```text
#7B95C6 #49C2D9 #A1D8E8 #67A583 #A2C986 #D0E2C0
#FDED95 #FFC1A6 #F59C7C #F47254 #C85E62
```

Fixed model colors include:

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

Checks:

- 12 of 12 plotting modules use the shared style;
- no local hexadecimal data color exists outside the approved palette;
- 24 PNG/PDF pairs exist;
- every PNG has DPI metadata of at least 300;
- all 48 figure files match `figure_hashes.json`.

## 7. Result consistency verification

**PASS.**

- canonical rows: 320;
- main rows: 140;
- label-efficiency rows: 80;
- threshold rows: 100;
- external same-protocol rows: 40;
- consolidated registry: 820 rows;
- folds: `0, 1, 2, 3`;
- seeds: `42, 123, 2024, 2025, 3407`;
- manuscript tables: 18;
- inline figures: 23;
- independent manuscript/table mismatches: 0;
- automated evidence checks: 18 of 18 passed.

README Key Findings are checked directly against canonical means and sample
standard deviations. Text-file source hashes are normalized to LF so Windows
and Linux checkouts produce the same content hash.

No experiment value was modified to match the manuscript.

## Internal reference and cross-reference navigation

The manuscript author is Huadong Lin in the DOCX properties, title page, and
`CITATION.cff`.

```yaml
reference_hyperlink_status: PASS
figure_cross_reference_status: PASS
table_cross_reference_status: PASS
equation_cross_reference_status: PASS
pdf_hyperlink_preservation: PASS
pdf_bookmarks: PASS
```

Verified values:

- 22 reference entries;
- 36 in-text citation occurrences;
- all 22 reference numbers cited and linked;
- 0 missing reference targets;
- 0 unused references;
- 0 broken DOCX anchors;
- 8 in-text Figure links;
- 9 in-text Table links;
- 1 in-text Section link;
- 0 in-text Equation references, with all six equation targets bookmarked;
- 114 unique DOCX bookmarks;
- 65 internal DOCX hyperlinks;
- 65 PDF link annotations;
- 0 unresolved PDF destinations;
- 45 PDF outline entries, including all required top-level sections,
  Appendix A, Data and Code Availability, and References.

PDF click tests resolved:

| Display text | Source page | Target page |
| --- | ---: | ---: |
| `[1]` | 2 | 29 |
| `[14]` | 2 | 30 |
| `Figure 4` | 9 | 10 |
| `Table 13` | 19 | 19 |

The four ranges `[1-3]`, `[11-13]`, `[14,16-18]`, and `[16-18]` retain their
IEEE numeric display. Visible range endpoints have independent targets; an
intermediate number hidden inside a hyphenated range cannot have a separate
clickable glyph without changing the displayed citation.

The complete machine-verified result is in `REFERENCE_LINK_REPORT.md`.

## 8. Reproduction checks

The latest source candidate was cloned from GitHub commit
`489f0a0edcf2c2f2449b72e7e98371a86315cecb`.

Observed results:

- 38 tests passed;
- smoke test passed;
- canonical and split catalogs matched;
- 18 of 18 consistency checks passed;
- internal-link audit passed with 22 reference targets, 65 PDF links, and 0
  unresolved destinations;
- palette and figure manifest checks passed;
- Raw fold 0/seed 42 training reproduced Macro-F1
  `0.3205054825882636` with `git_dirty=false`;
- the documented `scripts/reproduce_all.py --include-domain-shift` workflow
  completed statistics, figures, DOCX, and audits from retained results.

The full 820-row stochastic matrix was not repeated in this final repository
cleanup. The distinction between pipeline reproduction and full training
replication remains explicit in `CLEAN_REPRODUCTION_REPORT.md`.

## 9. Smoke test status

**PASS.**

`scripts/smoke_test.py` verifies:

- canonical config loading;
- four frozen split manifests;
- held-out bearing/condition isolation;
- dataset indexing;
- balanced label selection;
- metric execution;
- visualization style import;
- initialization and one forward pass for all seven model families.

The smoke test requires neither the full dataset cache nor the 320-run matrix.

## 10. Remaining warnings

- The primary evidence still covers one Paderborn dataset and four held-out
  domain folds.
- Bearing identity, operating condition, and damage origin remain coupled.
- The repository has undergone post-primary diagnostic analyses, so it does not
  claim complete cognitive isolation from test behavior.
- DRSN and TFN remain same-protocol adaptations.
- The full 820-row stochastic matrix was not rerun during the final directory
  cleanup.
- Existing Git history contains non-anonymous author metadata.
- Retained canonical run configurations contain workstation hostname fields.
- The manuscript and citation metadata identify Huadong Lin. If the target
  venue requires double-blind review, use an approved anonymous mirror.

The Git history/privacy items require an explicit publication decision and
history rewrite or anonymous mirror. They were not altered silently.

## 11. Final repository tree

```text
leakage-safe-bearing-benchmark/
├── .github/workflows/ci.yml
├── .gitattributes
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── REPRODUCIBILITY.md
├── CLEAN_REPRODUCTION_REPORT.md
├── CONSISTENCY_REPORT.md
├── REFERENCE_LINK_REPORT.md
├── FINAL_RELEASE_REPORT.md
├── configs/
│   ├── paderborn_main.json
│   ├── paderborn_sota.json
│   └── sensitivity configurations
├── data/
│   ├── README.md
│   └── manifests/
├── docs/
│   ├── DATASET.md
│   ├── PROTOCOL.md
│   ├── EXPERIMENTS.md
│   ├── RESULTS.md
│   ├── REPRODUCIBILITY.md
│   ├── audit/
│   ├── paper/
│   └── references/
├── figures/publication/
│   ├── 24 PNG figures
│   ├── 24 PDF figures
│   └── figure_hashes.json
├── manuscript/
│   ├── manuscript.pdf
│   ├── supplementary.pdf
│   └── README.md
├── results/
│   ├── README.md
│   ├── canonical/
│   ├── feature_embeddings/
│   ├── runs/
│   ├── splits/
│   ├── sota/
│   └── sensitivity result families
├── scripts/
│   ├── prepare_data.py
│   ├── run_main_benchmark.py
│   ├── run_label_efficiency.py
│   ├── run_threshold_sensitivity.py
│   ├── run_external_baselines.py
│   ├── run_preprocessing_sensitivity.py
│   ├── run_training_opportunity.py
│   ├── generate_statistics.py
│   ├── generate_figures.py
│   ├── reproduce_all.py
│   ├── smoke_test.py
│   ├── audit_internal_links.py
│   └── audit/consistency scripts
├── splits/
│   ├── README.md
│   └── manifest.json
├── src/spindle_cnn/
│   ├── analysis/
│   ├── datasets/
│   ├── evaluation/
│   ├── experiments/
│   ├── models/
│   ├── paper/
│   ├── plotting/       # compatibility alias
│   ├── public_benchmark/
│   ├── training/
│   └── visualization/
└── tests/
```

The complete tracked-file inventory is
`docs/audit/REPOSITORY_FILE_INVENTORY.txt`.

## 12. Recommended v1.0.0 release commit

The verified code baseline is:

```text
489f0a0edcf2c2f2449b72e7e98371a86315cecb
```

The final release commit is the documentation-only successor containing this
report. Verify it with:

```powershell
git rev-parse HEAD
git status --short
python -m pytest -q
python scripts/smoke_test.py
python scripts/check_manuscript_consistency.py
python scripts/check_figure_palette.py
```

Do not create the public `v1.0.0` tag until the privacy metadata issue and the
need for a conditional anonymous mirror are resolved. Once resolved, the
recommended release tag is:

```text
v1.0.0
```

with release title:

```text
Leakage-Safe Bearing Benchmark - Paper Release v1.0.0
```
