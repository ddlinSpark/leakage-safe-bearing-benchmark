# Clean Reproduction Report

## Verdict

The formal repository was cloned from the named public GitHub remote and
validated outside the development checkout.

- Tested code commit: `489f0a0edcf2c2f2449b72e7e98371a86315cecb`
- Test clone: `D:\lsbbl_links_489f0a0`
- Fresh clone checkout: clean
- Tests: **PASS**, 38 tests
- Smoke test: **PASS**
- Result catalogs: **PASS**
- Manuscript/result consistency: **PASS**, 18 of 18 checks
- Palette/figure integrity: **PASS**
- Manuscript table audit: **PASS**, 0 mismatches
- Internal links and PDF navigation: **PASS**
- Full 820-row retraining: **NOT REPEATED**

The final report commit is documentation-only relative to the tested code
commit. All subsequent changes are in Markdown documentation and line-ending
metadata.

## Remote provenance

```text
remote: https://github.com/ddlinSpark/leakage-safe-bearing-benchmark.git
branch: main
tested commit: 489f0a0edcf2c2f2449b72e7e98371a86315cecb
clone method: git -c core.longpaths=true clone
```

The clone contained 4,598 tracked files and no uncommitted checkout changes.

## Environment

| Component | Value |
| --- | --- |
| OS | Windows 11 |
| Python | 3.13.5 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |
| PyWavelets | 1.8.0 |
| GPU | NVIDIA GeForce RTX 5080 |
| LibreOffice | 26.8.1.1 |

## Executed checks

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
python scripts/smoke_test.py
python scripts/check_manuscript_consistency.py
python scripts/check_figure_palette.py
python scripts/audit_internal_links.py
python scripts/build_paper_metrics.py
python scripts/build_results_catalog.py
```

Observed results:

```text
38 passed
SMOKE TEST PASS
PASS: 18/18 checks passed
Palette/figure integrity: PASS
Canonical and split catalogs: PASS
Internal links and PDF navigation: PASS
```

## Preprocessing and training

The canonical preprocessing path was previously rerun from the official
extracted measurements in a clean clone:

- 3,939 windows produced;
- generated canonical NPZ SHA-256
  `a898abcb63e17ee68f2014d19b4d0f675b6c4d19a9abc89376031a78c3a23df4`;
- `raw`, `cwt`, `labels`, `loads`, `record_ids`, and `starts` exactly equal to
  the recorded cache;
- known truncated source measurement skipped by both paths.

The cloned repository completed Raw 1D-CNN fold `0`, seed `42` training with:

```text
Macro-F1 = 0.3205054825882636
accuracy = 0.4717741935483871
git_dirty = false
```

The run loaded 3,939 windows and recorded the expected dataset and split hashes.

## Statistics and figures

The clean workflow regenerated canonical summaries, paired tests, and the
20,000-sample hierarchical fold-first bootstrap. Key canonical files matched
their recorded content hashes.

The figure workflow regenerated:

- 24 PNG publication figures;
- the figure SHA-256 manifest.

A repeated complete figure generation produced 24 of 24 byte-identical PNG
files. `figure_hashes.json` points to files inside the selected output
directory.

## Manuscript

The DOCX builder uses a temporary file and deterministic OOXML ZIP timestamps.
Two builds from identical results and figures produced the same SHA-256:

```text
35b1c138f83069799c14508360805fd43542a5f9194c09db46b24ef2aee23366
```

The document contains 18 tables and 23 inline figures. The independent audit
found no numerical, table, terminology, or figure mismatch. The final document
is 30 A4 pages, and all pages were visually inspected after the final figure
palette update.

A temporary PDF render, which is intentionally excluded from Git, was used to
confirm 65 actionable internal links and a 45-entry navigation outline.

## Reproduction scope

The clean-room process verified:

- repository import and package layout;
- frozen configuration and all four split manifests;
- all seven model initializations and forward passes;
- one full fold-seed training run;
- statistics regeneration from retained result files;
- every publication figure from retained results and compact embeddings;
- deterministic DOCX generation;
- temporary manuscript PDF link and outline validation without tracking a PDF;
- manuscript/result consistency and palette integrity.

It did not rerun all 820 training rows. This distinction is retained so that a
successful pipeline check is not misreported as a full stochastic replication.
