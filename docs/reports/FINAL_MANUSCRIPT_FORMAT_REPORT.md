# Final Manuscript Format Report

Date: 2026-10-28

## Verdict

The final manuscript passes the requested format, numerical consistency,
internal-link, and temporary PDF-render checks. Author affiliation and
journal-specific submission metadata remain manual completion items.

```yaml
plus_minus_conversion: PASS
multiplication_symbol_conversion: PASS
tau_symbol_conversion: PASS
canonical_completed_evaluations_wording: PASS
table_11_cleanup: PASS
table_18_cleanup: PASS
supplementary_naming: PASS
author_metadata: WARNING
github_url_in_data_availability: PASS
github_manuscript_numeric_consistency: PASS
citation_hyperlinks: PASS
figure_hyperlinks: PASS
table_hyperlinks: PASS
section_hyperlinks: PASS
pdf_hyperlink_preservation: PASS
figure_readability: PASS
reference_consistency: PASS
```

## Manuscript

- Final DOCX:
  `docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx`
- Final DOCX SHA-256:
  `2998693d7f86b4dc88ec64a6dd653ada1e075234dc374243b648e6b755b3d969`
- Author: Huadong Lin
- Rendered pages: 30
- Tables: 18
- Inline figures: 23, comprising Figures 1-18 and Supplementary Figures S1-S5
- PDF QA render: temporary and non-tracked

The DOCX was rebuilt from the canonical result registry and figure files.
No experiment CSV, seed, fold, threshold, model parameter, statistical
inference rule, or negative result was changed.

## Symbol normalization

| Item | Before | After | Status |
| --- | ---: | ---: | --- |
| `+/-` | 105 | 0 | PASS |
| `±` | 0 | 105 | PASS |
| Numeric or fold-count `x` | 27 | 0 | PASS |
| `×` | 0 | 27 | PASS |
| Plain `tau` | 11 | 0 | PASS |
| `τ` | 0 | 11 | PASS |

The counts include body text, tables, and captions. Method diagram labels and
the Figure 16 threshold label were updated separately in the figure source.
Implementation identifiers such as `Conv5x5` and `Conv3x3` were preserved.
Numeric negative values use the mathematical minus sign `−`; hyphens in
compound words, filenames, model names, and citation ranges were not changed.

`Macro-F1` capitalization and `Full balanced budget` terminology were also
normalized throughout the rendered manuscript.

## Table revisions

Table 4 now uses `Canonical completed evaluations`, matching the repository's
completed-run terminology without implying positive performance.

Table 10 retains all canonical values while shortening the architecture column:

- `Two-layer BiGRU + linear head`
- `Residual CNN + SE`
- `Residual CNN + BiGRU`
- `Residual CNN + SE + BiGRU`
- `DRSN backbone`
- `TFN backbone`

Table 11 now uses `Cal.` and `ST`, displays component presence as `✓` or `–`,
and uses abbreviated temperature/self-training row labels. Its caption defines
all abbreviations. No score or run count changed.

Table 18 now reports `Held-out test ΔMacro-F1 [95% CI]` and
`Validation-improved runs` instead of the ambiguous `Improvement` heading. The
values are read from the retained training-budget and baseline-tuning
comparison tables. No source statistic was recalculated or overwritten.

## Supplementary material

The manuscript now uses:

```text
Supplementary Material
Additional Diagnostic Figures
```

Figure captions remain `Figure S1` through `Figure S5`. The former
`Appendix A` heading and bookmark were removed, and the PDF outline contains
both new supplementary headings.

## Author and availability

No affiliation, department, city, country, email, corresponding-author marker,
or ORCID is present in the project source. The manuscript therefore retains
only the verified author name.

`Author affiliation requires manual completion before submission.`

Data and Code Availability now cites the verified public repository:

```text
https://github.com/ddlinSpark/leakage-safe-bearing-benchmark
```

The statement identifies code, fixed splits, configurations, the per-run result
registry, statistics scripts, sensitivity outputs, and figure-generation
scripts. Raw signals, large generated caches, and model weights remain excluded.

## GitHub and manuscript consistency

`MANUSCRIPT_GITHUB_CONSISTENCY_REPORT.md` records the seven-condition comparison
between Table 5, README Main Results, and canonical summary files. All
Macro-F1, accuracy, and run-count checks pass with zero mismatches.

The Abstract, Table 13, Figure 12, and README consistently report:

```text
Validation Macro-F1 approximately 0.797–0.853
Held-out test Macro-F1 approximately 0.280–0.307
```

The rendered manuscript uses en dashes for these numeric ranges.

## Internal links and PDF

- Reference entries: 22
- In-text citation occurrences: 36
- Unique cited references: 22
- Missing or unused reference targets: 0
- DOCX bookmarks: 114
- DOCX internal hyperlinks: 66
- Figure targets: 23 of 23
- Table targets: 18 of 18
- Equation targets: 6 of 6
- PDF link annotations: 66
- Unresolved PDF destinations: 0
- PDF outline entries: 46
- Fonts used by the PDF: 16, all embedded
- Minimum effective figure resolution: 299.7 DPI

Required PDF click paths passed for `[1]`, `[11]`, `[14]`, `[21]`, `[22]`,
Figure 1, Figure 4, Figure 12, Table 5, Table 11, Table 13, and Section 6.

All 30 rendered pages were visually inspected at full resolution. Tables,
equations, captions, page numbers, and supplementary figures are unclipped and
unbroken. Figures 8, 10, 13, 17, and 18 remain readable at manuscript width;
Figure 17 and Figure 18 legends were enlarged without changing data or palette.

## Reference consistency

References are numbered 1-22 without duplicates or gaps. All 22 entries are
cited, and no citation lacks a target. DOI presentation uses a consistent
`doi: 10.xxxx/...` form where a DOI is present. Journal and conference entries
remain in IEEE numeric style.

## Remaining manual work

- Add the author affiliation and corresponding-author details required by the
  selected journal.
- Confirm whether the target venue requires double-blind review. If so, create
  a separate anonymous manuscript without replacing this named author version.
- Apply the journal's final template, reference style, and declaration forms.
- Do not create a public `v1.0.0` tag until the existing Git-history email and
  workstation-hostname privacy blockers are resolved.

## Release recommendation

The research content and repository are suitable for a paper-release freeze.
A public `v1.0.0` tag is not recommended yet because the established privacy
blockers remain outside this formatting change.
