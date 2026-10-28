# Final Release Report

## Repository Status

```yaml
README: PASS
Root directory cleanup: PASS
README links: PASS
Quick verification: PASS
Canonical result consistency: PASS
Figure consistency: PASS
Reference hyperlinks: PASS
Manuscript format: PASS
Manuscript-GitHub consistency: PASS
CITATION.cff: PASS
LICENSE: PASS
Author affiliation: WARNING
Manuscript release: WARNING
v1.0.0 readiness: NO
```

The numerical, manuscript, figure, and internal-link checks pass. A public
`v1.0.0` tag is not recommended yet because privacy metadata remains in Git
history and retained run configurations. Author affiliation and
corresponding-author details also require completion before journal submission.

## 1. Repository restructuring

- Root reports were moved under `docs/reports/`; root Markdown is limited to
  `README.md` and `FINAL_RELEASE_REPORT.md`.
- The canonical results and compatibility paths under `results/` were not
  moved.
- `results/canonical/manifest.json` declares the formal result files and their
  content hashes.
- `splits/manifest.json` indexes the four frozen fold manifests.
- `src/spindle_cnn/visualization/` is the primary plotting namespace;
  `src/spindle_cnn/plotting/` remains a compatibility alias.
- A CI workflow runs tests, smoke checks, manuscript consistency, palette
  checks, README checks, and tracked catalog checks.
- No experimental CSV, split, model result, or negative result was modified.

## 2. README changes

README is 245 lines and now separates:

- benchmark purpose and leakage-safe rationale;
- five concise Key Findings;
- Generalization Gap with the final Figure 12 PNG;
- the complete seven-condition Main Results table;
- Experiment Status with execution and performance clearly distinguished;
- leakage-safe protocol and statistical inference;
- Quick Verification;
- Full Reproduction;
- Result Reproduction versus Full Experimental Reproduction;
- hardware expectations and limitations;
- data availability and evidence links.

The README contains six badges only: Python, PyTorch, MIT License, Reproducible
Benchmark, Leakage-Safe Protocol, and DOCX Paper. No SOTA, best-accuracy, or
promotional badge is present.

The embedded generalization-gap image uses the existing final manuscript
figure:

```text
figures/publication/figure_09_validation_test_gap.png
```

## 3. Files moved

Moved from the repository root to `docs/reports/`:

- `AUDIT_REPORT.md`
- `CHANGELOG.md`
- `CLEAN_REPRODUCTION_REPORT.md`
- `CONSISTENCY_REPORT.md`
- `DAMAGE_ORIGIN_ANALYSIS.md`
- `EXPERIMENT_REPORT.md`
- `PAPER_RESULT_TRACEABILITY.md`
- `PAPER_REVISION_REPORT.md`
- `REFERENCE_LINK_REPORT.md`
- `REPRODUCE.md`
- `REPRODUCIBILITY.md`
- `REPRODUCTION_REPORT.md`
- `RESULT_CHANGELOG.md`
- `SECOND_DATASET_ASSESSMENT.md`

All scripts that generate these reports now write to `docs/reports/`.

## 4. Files deleted

No formal result, split manifest, failed run, or negative result was deleted.

Per the author's final format instruction, these tracked PDF artifacts were
removed from Git:

- 24 vector figure PDFs;
- `manuscript/manuscript.pdf`;
- `manuscript/supplementary.pdf`;
- the PDF assembly helper and `manuscript/README.md`.

PDFs are now ignored. A temporary PDF in `artifacts/local/` was used only to
verify internal PDF links and bookmarks and is not delivered.

## 5. Figure regeneration

All 24 tracked publication figures are PNG files. They were regenerated from
retained result files:

| Group | Figures |
| --- | --- |
| Method figures | 1-3 |
| Main benchmark | 4-12 |
| Sensitivity and diagnostics | 13-18 |
| Supplementary figures | S1-S5 |
| Retained analysis figure | External-baseline tuning |

The formatting pass regenerated only the six figures affected by notation or
font-size changes: the three method diagrams, Figure 16, Figure 17, and
Figure 18. The other 18 frozen PNG files were retained byte-for-byte. All 24
files match the current hash manifest; no PDF figure is tracked.

## 6. Palette verification

**PASS.**

`src/spindle_cnn/visualization/style.py` defines the required 11 colors:

```text
#7B95C6 #49C2D9 #A1D8E8 #67A583 #A2C986 #D0E2C0
#FDED95 #FFC1A6 #F59C7C #F47254 #C85E62
```

All 12 plotting modules use the shared style. No local hexadecimal data color
outside the palette was found. All 24 PNG files have DPI metadata of at least
300 and match `figures/publication/figure_hashes.json`.

## 7. Result consistency verification

**PASS.**

- Canonical evaluations: 320.
- Main evaluations: 140.
- Label-efficiency evaluations: 80.
- Threshold evaluations: 100.
- External same-protocol evaluations: 40.
- Consolidated registry: 820 rows.
- Folds: `0, 1, 2, 3`.
- Seeds: `42, 123, 2024, 2025, 3407`.
- README Key Findings and Main Results match canonical means and sample
  standard deviations.
- Independent manuscript/table mismatches: 0.
- Evidence-chain checks: 18 of 18 passed.

## Final manuscript format pass

The final DOCX is
`docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx`, SHA-256
`2998693d7f86b4dc88ec64a6dd653ada1e075234dc374243b648e6b755b3d969`.

- Converted 105 `+/-` expressions to `±`, 27 numeric `x` expressions to `×`,
  and 11 plain `tau` labels to `τ`.
- Preserved implementation names such as `Conv5x5` and normalized display text
  without changing canonical values.
- Compacted Tables 10 and 11 and clarified the Table 18 audit headers.
- Replaced the Appendix A heading with `Supplementary Material` and retained
  Figure S1-S5 numbering and bookmarks.
- Added the verified GitHub URL to Data and Code Availability.
- The author name is verified, but affiliation and corresponding-author details
  require manual completion before submission.

The complete record is `docs/reports/FINAL_MANUSCRIPT_FORMAT_REPORT.md`.
The manuscript-to-repository comparison is
`docs/reports/MANUSCRIPT_GITHUB_CONSISTENCY_REPORT.md`.

## Internal reference and cross-reference navigation

The manuscript author is Huadong Lin in the DOCX properties, title page, and
`CITATION.cff`.

```yaml
reference_hyperlink_status: PASS
figure_cross_reference_status: PASS
table_cross_reference_status: PASS
section_cross_reference_status: PASS
equation_cross_reference_status: PASS
pdf_hyperlink_preservation: PASS
pdf_bookmarks: PASS
```

- Reference entries: 22.
- In-text citation occurrences: 36.
- Unique cited references: 22.
- Missing reference targets: 0.
- Unused references: 0.
- Broken DOCX anchors: 0.
- DOCX bookmarks: 114.
- DOCX internal hyperlinks: 66.
- Figure targets: 23 of 23.
- Table targets: 18 of 18.
- Equation targets: 6 of 6.
- In-text links: 8 Figure, 10 Table, 1 Section.
- In-text Equation references: 0; all equation targets are bookmarked for
  future use.
- Temporary PDF link annotations: 66.
- Unresolved PDF destinations: 0.
- Temporary PDF outline entries: 46.

Temporary PDF click-path checks:

| Display text | Source page | Target page |
| --- | ---: | ---: |
| `[1]` | 2 | 29 |
| `[11]` | 3 | 30 |
| `[14]` | 2 | 30 |
| `[21]` | 29 | 30 |
| `[22]` | 14 | 30 |
| `Figure 1` | 4 | 4 |
| `Figure 4` | 9 | 10 |
| `Figure 12` | 19 | 19 |
| `Table 5` | 9 | 9 |
| `Table 11` | 11 | 16 |
| `Table 13` | 19 | 19 |
| `Section 6` | 11 | 15 |

Hyphenated ranges such as `[1-3]` and `[11-13]` retain their IEEE display.
Visible endpoints have independent targets; a hidden intermediate number cannot
have its own clickable glyph without changing the displayed citation.

The complete audit is `docs/reports/REFERENCE_LINK_REPORT.md`.
The final format and manuscript-to-repository audits are recorded in
`docs/reports/FINAL_MANUSCRIPT_FORMAT_REPORT.md` and
`docs/reports/MANUSCRIPT_GITHUB_CONSISTENCY_REPORT.md`.

## 8. Reproduction checks

The prior repository cleanup was pushed to GitHub and verified from a fresh
clone at:

```text
8f1bbfea20b27162c3fc92ff5eb389326472ccd5
```

The final formatting pass then rebuilt the manuscript and passed all 41 tests,
the smoke test, 25 README link targets, 15 command checks, 18 of 18 manuscript
consistency checks, numerical traceability, the 24-file figure manifest, and
the DOCX citation/cross-reference audit. No PDF was tracked.

The earlier clean-room source used for the main reproduction check was:

```text
489f0a0edcf2c2f2449b72e7e98371a86315cecb
```

- Tests: 41 passed.
- Smoke test: PASS.
- Canonical and split catalogs: PASS.
- Result/manuscript consistency: 18 of 18 PASS.
- Figure palette and PNG hashes: PASS.
- DOCX rebuild: byte-identical across repeated builds.
- Internal citation and cross-reference links: PASS.
- Temporary PDF links and bookmarks: PASS.
- Full 820-row stochastic matrix: NOT REPEATED.

The full experimental reproduction requires substantially more compute than
regenerating tables and figures from the committed result registry.

## 9. Smoke test status

**PASS.**

`scripts/smoke_test.py` verifies:

- canonical configuration loading;
- all four split manifests;
- held-out bearing and condition isolation;
- balanced label indexing;
- metric execution;
- statistics imports;
- shared plotting-style imports;
- initialization and one forward pass for all seven model families.

The smoke test does not require the full dataset cache.

## 10. Remaining warnings

- The primary evidence covers one Paderborn dataset and four domain folds.
- Bearing identity, operating condition, and damage origin are coupled.
- Findings are configuration-specific and do not support universal claims
  about CWT, GRU, SE, or self-training.
- DRSN and TFN remain same-protocol adaptations.
- The full 820-row stochastic matrix was not rerun during final cleanup.
- Existing Git history contains non-anonymous author metadata.
- Retained canonical run configurations contain workstation hostname fields.
- The manuscript identifies Huadong Lin; a venue requiring double-blind review
  needs an approved anonymous mirror.

## 11. Final repository tree

```text
leakage-safe-bearing-benchmark/
├── .github/workflows/ci.yml
├── .gitattributes
├── .gitignore
├── CITATION.cff
├── FINAL_RELEASE_REPORT.md
├── LICENSE
├── README.md
├── configs/
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
│   ├── references/
│   └── reports/
├── figures/publication/
│   ├── 24 PNG figures
│   └── figure_hashes.json
├── results/
│   ├── canonical/
│   ├── feature_embeddings/
│   ├── runs/
│   ├── splits/
│   ├── sota/
│   └── sensitivity result families
├── scripts/
├── splits/
├── src/spindle_cnn/
│   ├── analysis/
│   ├── datasets/
│   ├── evaluation/
│   ├── experiments/
│   ├── models/
│   ├── paper/
│   ├── plotting/
│   ├── public_benchmark/
│   ├── training/
│   └── visualization/
├── tests/
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

The complete tracked-file inventory is
`docs/audit/REPOSITORY_FILE_INVENTORY.txt`.

## 12. Recommended v1.0.0 release commit

After committing this report, the recommended release target is the current
`main` HEAD:

```powershell
git rev-parse HEAD
```

The code baseline used for the main clean-room test is
`489f0a0edcf2c2f2449b72e7e98371a86315cecb`. The final commit contains only
repository cleanup, report moves, removals of tracked PDFs, and documentation
updates.

Create `v1.0.0` only after the privacy decision is resolved:

```text
Tag: v1.0.0
Release: Leakage-Safe Bearing Benchmark - Paper Release v1.0.0
```
